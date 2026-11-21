"""Token-budgeted FP32 CPU/CUDA trainer with resumable, atomic checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.budget import canonical_config_json, config_hash
from src.optim import build_optimizer


METRIC_COLUMNS = [
    "run_id", "protocol_id", "config_id", "family", "cell", "seed", "split",
    "checkpoint_id", "trained_target_tokens", "eval_target_tokens", "loss_nats",
    "status", "source_kind",
]


# 역할: JSON 값의 키 순서에 영향을 받지 않는 SHA-256 식별값을 만든다.
def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# 역할: 파일을 나누어 읽으며 내용 확인용 SHA-256을 계산한다.
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 역할: 임시 파일에 쓰기를 마친 뒤 교체해 불완전한 파일이 원본을 덮지 않게 한다.
def _atomic_bytes(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write_fn(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


# 역할: 딕셔너리를 JSON으로 변환해 임시 파일 교체 방식으로 저장한다.
def _atomic_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_bytes(path, lambda stream: stream.write(data))


# 역할: 중첩된 상태 안의 텐서와 실수에 NaN이나 무한대가 있는지 검사한다.
def _tree_is_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tree_is_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


# 역할: 모델 상태를 계산 그래프에서 분리한 CPU 복사본으로 반환한다.
def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


# 역할: label이 -100이 아닌 위치를 세어 학습에 쓰는 정답 토큰 수를 구한다.
def _target_count(batch: dict) -> int:
    if "labels" not in batch:
        raise ValueError("training batch is missing pre-shifted labels")
    labels = torch.as_tensor(batch["labels"])
    if labels.ndim != 2:
        raise ValueError("labels must have shape [B,T]")
    return int((labels != -100).sum().item())


# 역할: 입력·정답·마스크를 지정 장치의 텐서로 바꾸고 모양이 같은지 검사한다.
def _tensor_batch(batch: dict, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    required = ("input_ids", "labels", "attention_mask")
    if any(key not in batch for key in required):
        raise ValueError(f"batch must contain {', '.join(required)}")
    input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device)
    labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(batch["attention_mask"], dtype=torch.bool, device=device)
    if input_ids.ndim != 2 or labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids, labels, and attention_mask must share [B,T] shape")
    return input_ids, labels, attention_mask


# 역할: 처리한 정답 토큰 수에 따라 학습률을 올린 뒤 cosine 곡선으로 낮춘다.
class TokenWarmupCosine:
    """Learning-rate schedule whose progress is measured in valid targets."""

    # 역할: 학습률·warmup·토큰 예산을 검사하고 진행량을 초기화한다.
    def __init__(self, peak_lr: float, warmup_tokens: int, total_tokens: int, min_lr_ratio: float = 0.1):
        if peak_lr <= 0 or warmup_tokens < 0 or total_tokens <= 0:
            raise ValueError("invalid token schedule configuration")
        if warmup_tokens > total_tokens or not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("warmup must fit the budget and min_lr_ratio must be in [0,1]")
        self.peak_lr = float(peak_lr)
        self.warmup_tokens = int(warmup_tokens)
        self.total_tokens = int(total_tokens)
        self.min_lr_ratio = float(min_lr_ratio)
        self.tokens_seen = 0

    # 역할: 다음 step의 토큰 수까지 고려해 사용할 학습률을 계산한다.
    def lr_for_next_step(self, step_valid_targets: int) -> float:
        endpoint = min(self.total_tokens, self.tokens_seen + step_valid_targets)
        if self.warmup_tokens and endpoint < self.warmup_tokens:
            ratio = endpoint / self.warmup_tokens
        elif self.warmup_tokens and self.tokens_seen < self.warmup_tokens <= endpoint:
            ratio = 1.0
        else:
            decay_tokens = max(1, self.total_tokens - self.warmup_tokens)
            progress = min(1.0, max(0.0, (endpoint - self.warmup_tokens) / decay_tokens))
            ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr * ratio

    # 역할: 완료한 정답 토큰 수를 학습률 일정의 진행량에 더한다.
    def step(self, valid_targets: int) -> None:
        if type(valid_targets) is not int or valid_targets < 0:
            raise ValueError("scheduler token increment must be a nonnegative integer")
        self.tokens_seen += valid_targets

    # 역할: 학습률 설정과 처리한 토큰 수를 저장용으로 반환한다.
    def state_dict(self) -> dict[str, Any]:
        return {
            "peak_lr": self.peak_lr, "warmup_tokens": self.warmup_tokens,
            "total_tokens": self.total_tokens, "min_lr_ratio": self.min_lr_ratio,
            "tokens_seen": self.tokens_seen,
        }

    # 역할: 학습률 설정이 일치하는지 확인하고 토큰 진행량을 복원한다.
    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.peak_lr, self.warmup_tokens, self.total_tokens, self.min_lr_ratio)
        actual = (state.get("peak_lr"), state.get("warmup_tokens"), state.get("total_tokens"), state.get("min_lr_ratio"))
        if actual != expected:
            raise ValueError("scheduler configuration mismatch")
        tokens_seen = state.get("tokens_seen")
        if type(tokens_seen) is not int or tokens_seen < 0:
            raise ValueError("invalid scheduler token cursor")
        self.tokens_seen = tokens_seen


# 역할: 생략된 설정에 기본값을 채우고 optimizer 이름과 주요 옵션을 검사한다.
def _default_training_config(config: dict[str, Any]) -> dict[str, Any]:
    values = {
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "total_target_tokens": 1024,
        "warmup_target_tokens": 64,
        "min_lr_ratio": 0.1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "router_aux_coefficient": 0.01,
        "checkpoint_interval_steps": 1,
        "recipe_id": "adamw-token-cosine-v1",
        "optimizer": "adamw",
        "router_balance_scope": "microbatch",
        "idle_expert_policy": "zero_grad",
        "spectron_momentum": 0.95,
        "spectron_ns_steps": 5,
        "spectron_power_steps": 1,
        "spectron_weight_decay": 0.0,
    }
    values.update(config)
    if values["router_balance_scope"] not in {"microbatch", "optimizer_step"}:
        raise ValueError("router_balance_scope must be microbatch or optimizer_step")
    if values["idle_expert_policy"] not in {"zero_grad", "skip"}:
        raise ValueError("idle_expert_policy must be zero_grad or skip")
    if values["optimizer"] not in {"adamw", "spectron"}:
        raise ValueError("optimizer must be adamw or spectron")
    if values["optimizer"] == "spectron":
        if "recipe_id" not in config:
            values["recipe_id"] = "spectron-paper-factors-adamw-v1"
        elif "spectron" not in values["recipe_id"].lower():
            raise ValueError("Spectron optimizer requires an explicit spectron recipe_id")
    if values["total_target_tokens"] <= 0 or values["gradient_accumulation_steps"] <= 0:
        raise ValueError("target token budget and accumulation steps must be positive")
    if values["checkpoint_interval_steps"] <= 0 or values["max_grad_norm"] <= 0:
        raise ValueError("checkpoint interval and max_grad_norm must be positive")
    if not 0 <= values["warmup_target_tokens"] <= values["total_target_tokens"]:
        raise ValueError("warmup_target_tokens must fit within total_target_tokens")
    if len(values["betas"]) != 2 or any(not 0 <= beta < 1 for beta in values["betas"]):
        raise ValueError("AdamW betas must be a pair in [0,1)")
    if values["learning_rate"] <= 0 or values["eps"] <= 0:
        raise ValueError("AdamW learning_rate and eps must be positive")
    if not values["recipe_id"]:
        raise ValueError("recipe_id is required")
    return values


# 역할: 한 번의 학습에서 모델 갱신, 저장·복구, 실행 기록을 관리한다.
class TrainingSession:
    """One resumable training attempt. Supports FP32 CPU and CUDA execution."""

    # 역할: 모델·데이터·식별 정보를 확인하고 optimizer와 학습 진행 상태를 준비한다.
    def __init__(
        self,
        model: nn.Module,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        train_batches: Sequence[dict],
        identity: dict[str, str],
        output_dir: str | Path,
        *,
        device: str = "cpu",
        run_id: str | None = None,
        audit_bundle: str | Path | None = None,
    ) -> None:
        chosen_device = torch.device(device)
        if chosen_device.type not in {'cpu', 'cuda'}:
            raise ValueError('supported devices are cpu and cuda')
        if chosen_device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this PyTorch environment')
        if not train_batches:
            raise ValueError("training requires at least one batch")
        self.device = chosen_device
        self.model = model.to(self.device)
        self.model_config = dict(model_config)
        self.canonical_config = canonical_config_json(self.model_config)
        self.config_hash = config_hash(self.model_config)
        self.training_config = _default_training_config(training_config)
        if self.training_config["router_balance_scope"] == "optimizer_step":
            from src.model import MoETransformer
            if not isinstance(model, MoETransformer):
                raise ValueError("optimizer_step router balance requires MoETransformer")
        self.training_config_hash = _json_hash(self.training_config)
        self.train_batches = list(train_batches)
        self.batch_target_counts = [_target_count(batch) for batch in self.train_batches]
        if not any(self.batch_target_counts):
            raise ValueError("training data contains no valid targets")
        source_kind = identity["source_kind"]
        audited_purpose = None
        if source_kind == "measured":
            if audit_bundle is None:
                raise ValueError('M9 must provide retained audited evaluation target/label provenance')
            from src.data import load_audited_bundle
            audited = load_audited_bundle(audit_bundle)
            audited_purpose = audited['spec']['purpose']
            if identity['tokenizer_revision'] != audited['spec']['manifest']['tokenizer']['revision']['value']:
                raise ValueError('tokenizer provenance mismatch')
            if identity['data_manifest_hash'] != audited['bundle_hash']:
                raise ValueError('data manifest provenance mismatch')
            if _json_hash(self.train_batches) != _json_hash(audited['batches']['train']):
                raise ValueError('training batch provenance mismatch')
            if identity.get('eval_revision') != _json_hash(audited['batches']['dev']):
                raise ValueError('evaluation target/label provenance mismatch')
            if model_config['vocab_size'] != audited['spec']['vocab_size'] or model_config['context_length'] != audited['spec']['context_length']:
                raise ValueError('model and audited data dimensions mismatch')
        elif source_kind != 'synthetic_fixture':
            raise ValueError('source_kind must be measured or synthetic_fixture')
        self.purpose = audited_purpose if source_kind == 'measured' else 'synthetic_fixture'
        if self.purpose not in {'pipeline_smoke_only', 'mini_exploration_only', 'synthetic_fixture'}:
            raise ValueError('unsupported training purpose')
        self.data_order = list(range(len(self.train_batches)))
        self.data_order_hash = _json_hash(self.data_order)
        self.identity = {
            "protocol_id": identity.get("protocol_id", "unspecified-protocol"),
            "source_revision": identity.get("source_revision", "unavailable"),
            "data_manifest_hash": identity["data_manifest_hash"],
            "tokenizer_revision": identity["tokenizer_revision"],
            "eval_revision": identity.get("eval_revision", identity["tokenizer_revision"]),
            "source_kind": source_kind,
        }
        if self.identity["source_kind"] not in {"measured", "synthetic_fixture"}:
            raise ValueError("source_kind must be measured or synthetic_fixture")
        if not isinstance(self.identity["data_manifest_hash"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.identity["data_manifest_hash"]
        ):
            raise ValueError("data_manifest_hash must be a SHA-256 hex digest")
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or str(uuid.uuid4())

        self.optimizer = build_optimizer(self.model, self.training_config)
        self.scheduler = TokenWarmupCosine(
            peak_lr=self.training_config["learning_rate"],
            warmup_tokens=self.training_config["warmup_target_tokens"],
            total_tokens=self.training_config["total_target_tokens"],
            min_lr_ratio=self.training_config["min_lr_ratio"],
        )
        self.checkpoint_seconds = 0.0
        self.scaler = None
        self.epoch = 0
        self.cursor = 0
        self.trained_target_tokens = 0
        self.optimizer_step = 0
        self.latest_verified_checkpoint_id: str | None = None
        self.failed_reason: str | None = None
        self.last_step_diagnostics: dict[str, Any] = {}

    # 역할: 모델·optimizer·난수·데이터 위치 등 재개에 필요한 상태를 모은다.
    def _checkpoint_payload(self, checkpoint_id: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "execution_device": self.device.type,
            "checkpoint_id": checkpoint_id,
            "latest_verified_checkpoint_id": checkpoint_id,
            "last_finite_checkpoint_id": checkpoint_id,
            "run_id": self.run_id,
            "config": self.model_config,
            "canonical_config_json": self.canonical_config,
            "config_hash": self.config_hash,
            "training_config": self.training_config,
            "training_config_hash": self.training_config_hash,
            "source_revision": self.identity["source_revision"],
            "data_manifest_hash": self.identity["data_manifest_hash"],
            "tokenizer_revision": self.identity["tokenizer_revision"],
            "eval_revision": self.identity["eval_revision"],
            "protocol_id": self.identity["protocol_id"],
            "source_kind": self.identity["source_kind"],
            "purpose": self.purpose,
            "model_state_dict": _cpu_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_step_diagnostics": self.last_step_diagnostics,
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state": None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().clone(),
                "used_device_rng_states": {
                    "cpu": torch.get_rng_state().clone(),
                    **({"cuda": torch.cuda.get_rng_state(self.device).cpu()} if self.device.type == "cuda" else {}),
                },
            },
            "data_order": self.data_order,
            "data_order_hash": self.data_order_hash,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "trained_target_tokens": self.trained_target_tokens,
            "optimizer_step": self.optimizer_step,
        }

    # 역할: checkpoint의 필수 항목, 실험 일치 여부, 진행량과 수치 상태를 검사한다.
    def _validate_checkpoint_payload(self, payload: Any) -> None:
        required = {
            "checkpoint_id", "latest_verified_checkpoint_id", "last_finite_checkpoint_id", "run_id",
            "canonical_config_json", "config_hash", "training_config_hash", "source_revision",
            "data_manifest_hash", "tokenizer_revision", "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "scaler_state", "rng_state", "data_order", "data_order_hash",
            "epoch", "cursor", "trained_target_tokens", "optimizer_step",
        }
        if not isinstance(payload, dict) or not required.issubset(payload):
            missing = sorted(required - set(payload if isinstance(payload, dict) else {}))
            raise ValueError(f"checkpoint is missing required state: {missing}")
        checks = {
            "canonical_config_json": self.canonical_config,
            "config_hash": self.config_hash,
            "training_config_hash": self.training_config_hash,
            "source_revision": self.identity["source_revision"],
            "data_manifest_hash": self.identity["data_manifest_hash"],
            "tokenizer_revision": self.identity["tokenizer_revision"],
            "eval_revision": self.identity["eval_revision"],
            "protocol_id": self.identity["protocol_id"],
            "source_kind": self.identity["source_kind"],
            "purpose": self.purpose,
            "eval_revision": self.identity["eval_revision"],
            "protocol_id": self.identity["protocol_id"],
            "source_kind": self.identity["source_kind"],
            "purpose": self.purpose,
            "data_order_hash": self.data_order_hash,
            "data_order": self.data_order,
        }
        for field, expected in checks.items():
            if payload[field] != expected:
                raise ValueError(f"checkpoint identity mismatch: {field}")
        if payload.get('execution_device', 'cpu') != self.device.type:
            raise ValueError('checkpoint execution device mismatch')
        rng = payload["rng_state"]
        if not isinstance(rng, dict) or not {"python", "numpy", "torch_cpu", "used_device_rng_states"}.issubset(rng):
            raise ValueError("checkpoint is missing RNG state")
        if not isinstance(rng["used_device_rng_states"], dict) or "cpu" not in rng["used_device_rng_states"]:
            raise ValueError("checkpoint is missing used-device RNG state")
        if self.device.type == "cuda" and "cuda" not in rng["used_device_rng_states"]:
            raise ValueError('checkpoint is missing CUDA RNG state')
        if not isinstance(payload["optimizer_state_dict"], dict):
            raise ValueError("checkpoint optimizer state is invalid")
        cursor, epoch = payload["cursor"], payload["epoch"]
        if type(cursor) is not int or not 0 <= cursor <= len(self.train_batches):
            raise ValueError("checkpoint data cursor is invalid")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("checkpoint epoch is invalid")
        for field in ("trained_target_tokens", "optimizer_step"):
            if type(payload[field]) is not int or payload[field] < 0:
                raise ValueError(f"checkpoint {field} is invalid")
        if payload["checkpoint_id"] != payload["latest_verified_checkpoint_id"]:
            raise ValueError("checkpoint latest verified ID is inconsistent")
        if payload["checkpoint_id"] != payload["last_finite_checkpoint_id"]:
            raise ValueError("checkpoint last finite ID is inconsistent")
        if payload["scheduler_state_dict"].get("tokens_seen") != payload["trained_target_tokens"]:
            raise ValueError("checkpoint scheduler and trained-target cursors disagree")
        if not _tree_is_finite(payload["model_state_dict"]) or not _tree_is_finite(payload["optimizer_state_dict"]):
            raise ValueError("checkpoint contains non-finite model or optimizer state")

    # 역할: checkpoint를 저장하고 CUDA 대기를 포함한 저장 시간을 누적한다.
    def _save_checkpoint(self) -> dict[str, str]:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            return self._save_checkpoint_impl()
        finally:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.checkpoint_seconds += time.perf_counter() - started

    # 역할: 상태를 저장하고 다시 읽어 검증한 뒤 최신 checkpoint 포인터를 갱신한다.
    def _save_checkpoint_impl(self) -> dict[str, str]:
        if not _tree_is_finite(self.model.state_dict()) or not _tree_is_finite(self.optimizer.state_dict()):
            raise ValueError("refusing to checkpoint non-finite training state")
        checkpoint_id = uuid.uuid4().hex
        payload = self._checkpoint_payload(checkpoint_id)
        path = self.checkpoint_dir / f"checkpoint-{checkpoint_id}.pt"
        _atomic_bytes(path, lambda stream: torch.save(payload, stream))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_payload(saved)
        if saved["checkpoint_id"] != checkpoint_id:
            raise ValueError("checkpoint round-trip ID mismatch")
        digest = _sha256_file(path)
        pointer = {"filename": path.name, "checkpoint_id": checkpoint_id, "sha256": digest}
        _atomic_json(self.checkpoint_dir / "latest.json", pointer)
        self.latest_verified_checkpoint_id = checkpoint_id
        return {**pointer, "path": str(path)}

    # 역할: 최신 checkpoint의 무결성을 검사하고 모델·optimizer·난수·학습 위치를 복원한다.
    def restore_latest(self) -> None:
        pointer_path = self.checkpoint_dir / "latest.json"
        if not pointer_path.exists():
            raise FileNotFoundError("no checkpoint pointer exists")
        pointer = json.loads(pointer_path.read_text())
        filename = pointer.get("filename", "")
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError("checkpoint pointer filename is invalid")
        path = self.checkpoint_dir / filename
        if not path.is_file() or _sha256_file(path) != pointer.get("sha256"):
            raise ValueError("checkpoint integrity hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_payload(payload)
        if payload["checkpoint_id"] != pointer.get("checkpoint_id"):
            raise ValueError("checkpoint pointer ID mismatch")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload["scaler_state"] is not None:
            raise ValueError("this CPU trainer cannot restore a non-null scaler state")
        self.run_id = payload["run_id"]
        self.epoch = payload["epoch"]
        self.cursor = payload["cursor"]
        self.trained_target_tokens = payload["trained_target_tokens"]
        self.optimizer_step = payload["optimizer_step"]
        self.last_step_diagnostics = payload.get("last_step_diagnostics", {})
        self.latest_verified_checkpoint_id = payload["latest_verified_checkpoint_id"]
        rng = payload["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        torch.set_rng_state(rng["used_device_rng_states"]["cpu"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(rng["used_device_rng_states"]["cuda"], self.device)

    # 역할: 실행 식별 정보와 상태를 기록하고 같은 run_id의 기존 기록은 갱신한다.
    def _write_run_manifest(self, status: str, reason: str) -> None:
        config_id = self.model_config.get("config_id", "unidentified")
        record = {
            "run_id": self.run_id,
            "protocol_id": self.identity["protocol_id"],
            "config_id": config_id,
            "config_hash": self.config_hash,
            "family": self.model_config["family"],
            "cell": self.model_config.get("cell", "unidentified"),
            "seed": self.model_config["seed"],
            "recipe_id": self.training_config["recipe_id"],
            "training_config_hash": self.training_config_hash,
            "optimizer": self.training_config["optimizer"],
            "router_balance_scope": self.training_config["router_balance_scope"],
            "idle_expert_policy": self.training_config["idle_expert_policy"],
            "data_manifest_hash": self.identity["data_manifest_hash"],
            "eval_revision": self.identity["eval_revision"],
            "tokenizer_revision": self.identity["tokenizer_revision"],
            "source_commit": self.identity["source_revision"],
            "checkpoint_id": self.latest_verified_checkpoint_id,
            "checkpoint_step": self.optimizer_step,
            "trained_target_tokens": self.trained_target_tokens,
            "status": status,
            "reason": reason,
            "source_kind": self.identity["source_kind"],
            "purpose": self.purpose,
        }
        path = self.output_dir / "run_manifest.jsonl"
        existing = []
        if path.exists():
            existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        matches = [i for i, item in enumerate(existing) if item.get("run_id") == self.run_id]
        if len(matches) > 1:
            raise ValueError("run manifest already contains duplicate run_id records")
        if matches:
            if existing[matches[0]].get("status") == "failed" and status != "failed":
                raise ValueError("failed run_id is terminal and cannot be reopened")
            existing[matches[0]] = record
        else:
            existing.append(record)
        data = "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in existing).encode()
        _atomic_bytes(path, lambda stream: stream.write(data))

    # 역할: 다음 step에 gradient를 누적할 작은 배치 묶음을 꺼내고 데이터 위치를 옮긴다.
    def _next_group(self) -> tuple[list[dict], list[int], int, int]:
        if self.cursor >= len(self.train_batches):
            self.epoch += 1
            self.cursor = 0
        start = self.cursor
        end = min(start + self.training_config["gradient_accumulation_steps"], len(self.train_batches))
        self.cursor = end
        return self.train_batches[start:end], self.batch_target_counts[start:end], start, end

    # 역할: gradient 계산 전에 배치 묶음 전체의 층별 expert 선택 비율을 집계한다.
    @torch.no_grad()
    def _group_router_fractions(self, group):
        """Collect detached global assignments before the gradient pass.

        The current model has no dropout or mutable forward buffers. This extra
        forward pass trades compute for bounded memory and accumulation-invariant
        balancing. Its time must be included in measured training throughput.
        """
        # 한 optimizer step에 묶인 작은 배치들의 expert 선택 횟수를 먼저 합친다.
        # 계산 그래프를 저장하지 않고 다시 forward하므로 메모리는 절약되지만 시간이 더 든다.
        # 현재 모델에는 dropout이 없어 두 forward에서 같은 선택을 얻는다.
        counts = None
        tokens = 0
        for batch in group:
            ids, _, mask = _tensor_batch(batch, self.device)
            _, _, stats = self.model(ids, mask)
            if not stats["layers"]:
                return None, 0
            current = torch.tensor(stats["selected_count"], device=self.device, dtype=torch.float32)
            counts = current if counts is None else counts + current
            tokens += stats["valid_token_count"]
        if counts is None or tokens == 0:
            raise ValueError("router balancing group contains no valid input tokens")
        # 토큰마다 top_k번 선택하므로 전체 선택 수는 top_k × 유효 입력 토큰 수다.
        return list(counts / (self.model_config["top_k"] * tokens)), tokens

    # 역할: 토큰·step 한도까지 학습하고 기록하며 실패 시 마지막 검증 checkpoint로 복구한다.
    def fit(self, *, max_optimizer_steps: int | None = None) -> dict[str, Any]:
        if max_optimizer_steps is not None and (type(max_optimizer_steps) is not int or max_optimizer_steps <= 0):
            raise ValueError("max_optimizer_steps must be a positive integer")
        manifest_path = self.output_dir / "run_manifest.jsonl"
        if manifest_path.exists():
            records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
            same_run = [row for row in records if row.get("run_id") == self.run_id]
            if len(same_run) > 1:
                raise ValueError("run manifest already contains duplicate run_id records")
            if same_run and same_run[0].get("status") == "failed":
                self.failed_reason = same_run[0].get("reason", "previously_failed")
                return self.result("failed")
        if self.latest_verified_checkpoint_id is None:
            if not _tree_is_finite(self.model.state_dict()):
                self.failed_reason = "nonfinite_initial_model"
                self._write_run_manifest("failed", self.failed_reason)
                return self.result("failed")
            try:
                self._save_checkpoint()
            except TimeoutError:
                raise  # Do not swallow the controller deadline in recovery I/O.
            except Exception as exc:
                self.failed_reason = f"checkpoint_save_failed:{type(exc).__name__}"
                self._write_run_manifest("failed", self.failed_reason)
                return self.result("failed")
        self._write_run_manifest("incomplete", "training in progress")
        starting_step = self.optimizer_step
        failed_reason = None

        while self.trained_target_tokens < self.training_config["total_target_tokens"]:
            if max_optimizer_steps is not None and self.optimizer_step - starting_step >= max_optimizer_steps:
                break
            group, target_counts, start_cursor, _ = self._next_group()
            group_targets = sum(target_counts)
            if group_targets == 0:
                if self.cursor >= len(self.train_batches) and not any(self.batch_target_counts):
                    failed_reason = "no_valid_target_tokens"
                    break
                continue
            if self.trained_target_tokens + group_targets > self.training_config["total_target_tokens"]:
                self.cursor = start_cursor
                break

            self.optimizer.zero_grad(set_to_none=True)
            ce_total, aux_total = 0.0, 0.0
            fractions, router_tokens = None, 0
            global_balance = self.training_config["router_balance_scope"] == "optimizer_step"
            if global_balance and self.training_config["router_aux_coefficient"] != 0:
                fractions, router_tokens = self._group_router_fractions(group)
            # 예측 손실(CE)은 정답이 있는 토큰 수, router 손실은 유효 입력 토큰 수로 가중한다.
            # 문서 끝처럼 입력은 있지만 정답 label이 -100인 위치가 있어 두 수는 다를 수 있다.
            for batch, valid_targets in zip(group, target_counts):
                # 정답이 없는 작은 배치도 전체 router 균형 계산에는 참여해야 한다.
                if valid_targets == 0 and fractions is None:
                    continue
                input_ids, labels, attention_mask = _tensor_batch(batch, self.device)
                if fractions is None:
                    logits, aux_loss, _ = self.model(input_ids, attention_mask)
                    aux_weight = valid_targets / group_targets
                else:
                    logits, aux_loss, _ = self.model(input_ids, attention_mask, router_balance_fractions=fractions)
                    aux_weight = int(attention_mask.sum().item()) / router_tokens
                ce_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
                # 각 작은 배치의 기여를 전체 step 기준으로 나눈 뒤 gradient를 누적한다.
                objective = ce_sum / group_targets + (
                    self.training_config["router_aux_coefficient"] * aux_loss * aux_weight
                )
                if not torch.isfinite(objective).all():
                    failed_reason = "nonfinite_loss"
                    break
                ce_total += float(ce_sum.detach().item())
                aux_total += float(aux_loss.detach().item()) * aux_weight
                objective.backward()
            if failed_reason:
                self.optimizer.zero_grad(set_to_none=True)
                break
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                failed_reason = "nonfinite_gradient"
                self.optimizer.zero_grad(set_to_none=True)
                break
            idle_parameters = sum(p.requires_grad and p.grad is None for p in self.model.parameters())
            # grad=None이면 optimizer는 momentum과 decay까지 건너뛴다.
            # 0 gradient를 넣으면 선택되지 않은 expert도 이전 momentum과 decay가 반영된다.
            if self.training_config["idle_expert_policy"] == "zero_grad":
                for parameter in self.model.parameters():
                    if parameter.requires_grad and parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config["max_grad_norm"])
            if not torch.isfinite(grad_norm).all():
                failed_reason = "nonfinite_gradient_norm"
                self.optimizer.zero_grad(set_to_none=True)
                break
            lr = self.scheduler.lr_for_next_step(group_targets)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = lr
            self.optimizer.step()
            if not _tree_is_finite(self.model.state_dict()) or not _tree_is_finite(self.optimizer.state_dict()):
                failed_reason = "nonfinite_parameter_or_optimizer_state"
                break
            self.scheduler.step(group_targets)
            self.trained_target_tokens += group_targets
            self.optimizer_step += 1
            # 손실 급증의 원인을 추적할 수 있도록 갱신이 끝날 때마다 수치를 기록한다.
            # clip_grad_norm_의 반환값은 clipping 전 gradient 크기다.
            self.last_step_diagnostics = {
                "run_id": self.run_id, "optimizer_step": self.optimizer_step,
                "trained_target_tokens": self.trained_target_tokens,
                "training_config_hash": self.training_config_hash,
                "ce_loss_nats": ce_total / group_targets,
                "router_balance_loss": aux_total,
                "gradient_norm_before_clip": float(grad_norm.item()),
                "learning_rate": lr, "idle_parameter_tensors": idle_parameters,
                "router_balance_scope": self.training_config["router_balance_scope"],
            }
            # 아래 값은 factor별 추정치다. 합성 행렬 W의 실제 변화량 상한은 아니다.
            spectral = getattr(self.optimizer, "spectron", None)
            if spectral is not None:
                for key in ("factor_spectral_estimate", "update_frobenius_norm", "rho"):
                    values = [state[key] for state in spectral.state.values() if key in state]
                    self.last_step_diagnostics[key + "_max"] = float(torch.stack(values).max().item()) if values else 0.
            try:
                # checkpoint보다 먼저 기록되므로 복구 후에는 복구 step 이후 기록을 제외한다.
                _atomic_json(self.output_dir / "diagnostics" / self.run_id / f"step-{self.optimizer_step:08d}.json",
                             self.last_step_diagnostics)
            except TimeoutError:
                raise  # Do not swallow the controller deadline in recovery I/O.
            except Exception as exc:
                failed_reason = f"diagnostic_save_failed:{type(exc).__name__}"
                break
            interval = self.training_config["checkpoint_interval_steps"]
            if self.optimizer_step % interval == 0:
                try:
                    self._save_checkpoint()
                except TimeoutError:
                    raise  # Do not swallow the controller deadline in recovery I/O.
                except Exception as exc:
                    failed_reason = f"checkpoint_save_failed:{type(exc).__name__}"
                    break

        if failed_reason:
            # Discard any unsaved/post-step state. The manifest and returned
            # counters must refer to the latest atomic finite checkpoint.
            if (self.checkpoint_dir / "latest.json").exists():
                self.restore_latest()
            self.failed_reason = failed_reason
            self._write_run_manifest("failed", failed_reason)
            return self.result("failed")
        complete = self.trained_target_tokens >= self.training_config["total_target_tokens"]
        if self.latest_verified_checkpoint_id is None or self.optimizer_step % self.training_config["checkpoint_interval_steps"] != 0:
            try:
                self._save_checkpoint()
            except TimeoutError:
                raise  # Do not swallow the controller deadline in recovery I/O.
            except Exception as exc:
                failed_reason = f"checkpoint_save_failed:{type(exc).__name__}"
                if self.latest_verified_checkpoint_id is not None:
                    self.restore_latest()
                self.failed_reason = failed_reason
                self._write_run_manifest("failed", failed_reason)
                return self.result("failed")
        status = "completed" if complete else "incomplete"
        reason = "target token budget reached" if complete else "stopped at an optimizer-step or data-group boundary"
        self._write_run_manifest(status, reason)
        return self.result(status)

    # 역할: 실행 상태, 처리한 토큰 수, 데이터 위치와 checkpoint ID를 요약해 반환한다.
    def result(self, status: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": status,
            "reason": self.failed_reason or ("target token budget reached" if status == "completed" else "incomplete"),
            "optimizer_step": self.optimizer_step,
            "trained_target_tokens": self.trained_target_tokens,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "checkpoint_id": self.latest_verified_checkpoint_id,
        }


# 역할: 평가 지표의 필수 값과 형식을 검사한 뒤 CSV에 기록한다.
def write_metrics_csv(path: str | Path, row: dict[str, Any]) -> None:
    missing = set(METRIC_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"metrics row missing columns: {sorted(missing)}")
    enum_values = {
        "family": {"LR", "DW", "SHARED_ONLY"},
        "split": {"train", "dev", "test"},
        "status": {"completed", "failed", "incomplete"},
        "source_kind": {"measured", "synthetic_fixture"},
    }
    for field, allowed in enum_values.items():
        if not isinstance(row[field], str) or row[field] not in allowed:
            raise ValueError(f"metrics {field} must be one of {sorted(allowed)}")
    for field in ("run_id", "protocol_id", "config_id", "cell", "checkpoint_id"):
        if row[field] is None or str(row[field]).strip() == "":
            raise ValueError(f"metrics {field} must be nonempty")

    # 역할: 지표 값을 정수로 읽고 0 이상인지, 필요하면 0보다 큰지 검사한다.
    def nonnegative_integer(field: str, *, positive: bool = False) -> int:
        value = row[field]
        if isinstance(value, bool):
            raise ValueError(f"metrics {field} must be an integer")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
        else:
            raise ValueError(f"metrics {field} must be an integer")
        if parsed < 0 or (positive and parsed == 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"metrics {field} must be {qualifier}")
        return parsed

    seed = nonnegative_integer("seed")
    trained_tokens = nonnegative_integer("trained_target_tokens")
    eval_tokens = nonnegative_integer("eval_target_tokens")
    completed = row["status"] == "completed"
    if completed:
        if trained_tokens <= 0:
            raise ValueError("completed metrics require positive trained_target_tokens")
        if eval_tokens <= 0:
            raise ValueError("completed metrics require positive eval_target_tokens")
    loss = row["loss_nats"]
    if loss in (None, ""):
        if completed:
            raise ValueError("completed metrics require finite nonnegative loss_nats")
    else:
        try:
            parsed_loss = float(loss)
        except (TypeError, ValueError) as exc:
            raise ValueError("metrics loss_nats must be numeric when present") from exc
        if not math.isfinite(parsed_loss) or parsed_loss < 0:
            raise ValueError("metrics loss_nats must be finite and nonnegative")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    key = tuple(str(row[field]) for field in ("run_id", "split", "checkpoint_id"))
    if any(not value for value in key):
        raise ValueError("metric primary-key fields run_id, split, and checkpoint_id must be nonempty")
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != METRIC_COLUMNS:
                raise ValueError(f"metrics CSV header must exactly match {METRIC_COLUMNS}")
            seen = set()
            for existing_row in reader:
                existing_key = tuple(existing_row.get(field, "") for field in ("run_id", "split", "checkpoint_id"))
                if existing_key in seen:
                    raise ValueError(f"metrics CSV already contains duplicate primary key {existing_key}")
                seen.add(existing_key)
            if key in seen:
                raise ValueError(f"duplicate metrics primary key {key}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in METRIC_COLUMNS})
