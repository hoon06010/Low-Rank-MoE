"""Bounded FP32 runner for the four audited Colab mini-exploration cells."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data import load_audited_bundle
from src.evaluate import evaluate
from src.model import MoETransformer
from src.train import _default_training_config, TrainingSession, _atomic_json, _json_hash, _sha256_file, _target_count, write_metrics_csv

ROOT = Path(__file__).resolve().parents[1]
PURPOSE = "mini_exploration_only"
GLOBAL_BATCH = 4096
ROW_TARGETS = 512
CHUNK_STEPS = 32
EVAL_INTERVAL = 1_048_576
GRID_CELLS = {"LR-HS": (64, 256, 2), "LR-HL": (64, 256, 5),
              "LR-LS": (32, 256, 2), "LR-LL": (32, 256, 5)}


# 역할: JSON 설정을 읽고 mini 실험의 구조·예산·학습 정책 조건을 검사한다.
def _config(path: Path) -> dict:
    config = json.loads(path.read_text())
    model, training = config["model"], config["training"]
    if config.get("protocol_id") not in {"colab-free-mini-v2", "colab-free-mini-stability-v1"}:
        raise ValueError("unsupported mini protocol")
    _default_training_config(training)  # 실행 전에 optimizer 이름 오타와 잘못된 설정을 잡는다.
    # 안정성 대조 실험은 두 optimizer에 같은 router 집계·idle expert 정책을 적용한다.
    if config["protocol_id"] == "colab-free-mini-stability-v1":
        if training.get("optimizer") not in {"adamw", "spectron"}:
            raise ValueError("stability protocol requires an explicit optimizer")
        if training.get("router_balance_scope") != "optimizer_step" or training.get("idle_expert_policy") != "zero_grad":
            raise ValueError("stability protocol requires global balancing and zero-gradient idle updates")
    if model.get("config_id") not in {"LR-HS", "LR-HL", "LR-LS", "LR-LL"} or model.get("family") != "LR":
        raise ValueError("only LR-HS, LR-HL, LR-LS, and LR-LL are supported")
    seed_repeat = False
    if Path(path).resolve().is_relative_to(ROOT / "configs/seed_repeat_c"):
        from scripts.seed_repeat_policy import validate_config
        seed_repeat = validate_config(path, config)
    if not seed_repeat and model.get("seed") != 17:
        raise ValueError("mini cells use seed 17")
    if config.get("global_target_batch") != GLOBAL_BATCH or training.get("gradient_accumulation_steps") != 8:
        raise ValueError("mini cells require 512-target rows and accumulation of 8")
    if config.get("eval_interval_tokens") != EVAL_INTERVAL:
        raise ValueError("mini evaluation interval must be 1 Mi target tokens")
    if training.get("total_target_tokens", 0) <= 0 or training["total_target_tokens"] % GLOBAL_BATCH:
        raise ValueError("token budget must be a positive multiple of the global target batch")
    if training.get("checkpoint_interval_steps") != CHUNK_STEPS:
        raise ValueError("checkpoint interval must be 32 optimizer steps")
    if training.get("learning_rate") != 3e-4:
        raise ValueError("mini learning rate is fixed at 3e-4")
    if training.get("warmup_target_tokens") != int(training["total_target_tokens"] * .05):
        raise ValueError("mini warmup must be 5% of the token budget")
    return config


# 역할: 네 실험 설정이 정해진 구조와 공통 학습 조건을 따르는지 검사한다.
def _validate_grid(path: Path, selected: dict) -> None:
    configs = {selected["model"]["config_id"]: selected}
    for cell in GRID_CELLS:
        if cell not in configs:
            sibling = path.parent / f"{cell}.json"
            if not sibling.is_file():
                raise ValueError(f"mini grid is incomplete; missing {sibling.name}")
            configs[cell] = _config(sibling)
    for cell, config in configs.items():
        model = config["model"]
        if (model["rank"], model["shared_width"], model.get("shared_experts")) != GRID_CELLS[cell]:
            raise ValueError(f"{cell} architecture differs from the fixed mini grid")
        for field in ("layers", "d_model", "heads", "routed_experts", "top_k", "routed_width", "vocab_size", "context_length", "seed"):
            if model[field] != selected["model"][field]:
                raise ValueError(f"mini grid model mismatch: {field}")
        if (config["protocol_id"] != selected["protocol_id"] or config["training"] != selected["training"]
                or config["global_target_batch"] != selected["global_target_batch"]
                or config["eval_interval_tokens"] != selected["eval_interval_tokens"]):
            raise ValueError("all four mini cells must share protocol, budget, schedule, and evaluation cadence")


# 역할: 개발 데이터로 평가하고 같은 checkpoint의 중복 기록을 피하며 지표를 저장한다.
def _eval_metric(session, batches, config, result, out, device):
    evaluation = evaluate(session.model, batches, device=device)
    row = {
        "run_id": result["run_id"], "protocol_id": config["protocol_id"],
        "config_id": config["model"]["config_id"], "family": "LR",
        "cell": config["model"]["cell"], "seed": config["model"]["seed"], "split": "dev",
        "checkpoint_id": result["checkpoint_id"],
        "trained_target_tokens": result["trained_target_tokens"],
        "eval_target_tokens": evaluation["eval_target_tokens"],
        "loss_nats": evaluation["loss_nats"], "status": result["status"],
        "source_kind": "measured",
    }
    metrics_path = out / "metrics.csv"
    existing = list(csv.DictReader(metrics_path.open(encoding="utf-8", newline=""))) if metrics_path.exists() else []
    key = (row["run_id"], "dev", row["checkpoint_id"])
    if not any((r["run_id"], r["split"], r["checkpoint_id"]) == key for r in existing):
        write_metrics_csv(metrics_path, row)
    return evaluation


# 역할: checkpoint마다 첫 개발 배치의 routing 진단을 중복 없이 기록한다.
def append_routing_diagnostic(path, result, diagnostic):
    """Append one first-dev-batch routing record per verified checkpoint."""
    record = {
        "run_id": result["run_id"],
        "checkpoint_id": result["checkpoint_id"],
        "trained_target_tokens": result["trained_target_tokens"],
        "scope": "first_dev_batch_only",
        "stats": diagnostic,
    }
    path = Path(path)
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    key = (record["run_id"], record["checkpoint_id"])
    if any((item.get("run_id"), item.get("checkpoint_id")) == key for item in existing):
        return False
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return True


# 역할: 설정과 데이터를 검증하고 제한된 학습·평가·저장 후 실행 보고서를 반환한다.
def run_colab_mini(config_path, bundle_path, output_dir, *, device="cuda", resume=False,
                   max_steps=32, max_seconds=None, pilot=False, boundary_hook=None):
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("supported devices are cpu and cuda")
    if selected.type == "cpu" and max_steps is None and not pilot:
        raise ValueError("CPU is reserved for bounded tests; specify --max-steps")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no training was started")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if pilot and resume:
        raise ValueError("pilot requires a fresh run and cannot resume")

    config_path, bundle_path, out = Path(config_path).resolve(), Path(bundle_path).resolve(), Path(output_dir).resolve()
    config = _config(config_path)
    _validate_grid(config_path, config)
    audited = load_audited_bundle(bundle_path)
    if audited["spec"]["purpose"] != PURPOSE:
        raise ValueError(f"bundle purpose must be {PURPOSE}")
    if audited["spec"]["vocab_size"] != config["model"]["vocab_size"] or audited["spec"]["context_length"] != config["model"]["context_length"]:
        raise ValueError("model and mini bundle dimensions differ")
    batches = audited["batches"]["train"]
    counts = [_target_count(row) for row in batches]
    if not counts or any(count != ROW_TARGETS for count in counts):
        raise ValueError("every training row must contain exactly 512 valid targets")
    available = sum(counts)
    target = config["training"]["total_target_tokens"]
    if available < target:
        raise ValueError(f"mini bundle has {available} unique packed targets; budget requires {target}; cycling is forbidden")
    if target % GLOBAL_BATCH:
        raise ValueError("token budget is not aligned to the global target batch")

    if out.exists():
        if not resume and any(out.iterdir()):
            raise ValueError("output directory is not empty; use --resume or choose a fresh output")
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "run_manifest.jsonl"
    if resume:
        if not (out / "checkpoints/latest.json").is_file() or not manifest.is_file():
            raise ValueError("resume requires a checkpoint and run manifest")
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not rows or any(row.get("status") == "failed" for row in rows):
            raise ValueError("failed run manifests cannot be resumed")
        if any(row.get("status") == "completed" for row in rows):
            raise ValueError("completed runs cannot be resumed; choose a fresh output")
        if len({row.get("run_id") for row in rows}) != len(rows):
            raise ValueError("run manifest contains duplicate run IDs")
    elif (out / "checkpoints/latest.json").exists():
        raise ValueError("output already has a checkpoint; use --resume or a fresh output")

    source_paths = [Path(__file__).resolve(), *sorted((Path(__file__).resolve().parents[1] / "src").glob("*.py"))]
    source_hashes = {str(path.relative_to(Path(__file__).resolve().parents[1])): _sha256_file(path) for path in source_paths}
    if config_path.is_relative_to(ROOT / "configs/seed_repeat_c"):
        from scripts.seed_repeat_policy import source_hashes as repeat_source_hashes
        source_hashes = repeat_source_hashes()
    random.seed(config["model"]["seed"]); np.random.seed(config["model"]["seed"]); torch.manual_seed(config["model"]["seed"])
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(config["model"]["seed"])
    identity = {
        "protocol_id": config["protocol_id"], "source_revision": _json_hash(source_hashes),
        "data_manifest_hash": audited["bundle_hash"],
        "tokenizer_revision": audited["spec"]["manifest"]["tokenizer"]["revision"]["value"],
        "eval_revision": _json_hash(audited["batches"]["dev"]), "source_kind": "measured",
    }
    session = TrainingSession(MoETransformer(config["model"]), config["model"], config["training"], batches,
                              identity, out, device=device, audit_bundle=bundle_path)
    if resume:
        session.restore_latest()
    elif boundary_hook is not None:
        # An overnight run needs a durable zero-step generation before training.
        session._save_checkpoint()
        session._write_run_manifest("incomplete", "initial checkpoint")
        if boundary_hook(session, "initial") is False:
            raise ValueError("overnight preflight rejected the initial checkpoint")

    # 역할: CUDA를 사용할 때 작업 완료를 기다려 실행 시간을 정확히 잴 수 있게 한다.
    def sync():
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)

    start_step = session.optimizer_step
    invocation_checkpoint_seconds = session.checkpoint_seconds
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    elapsed_train = elapsed_eval = 0.0
    evaluation = None
    routing_diagnostic = None
    measured_tokens = 0
    warmup_tokens = 0
    warmup_elapsed = 0.0
    sync()
    invocation_started = time.perf_counter()
    if pilot:
        warm_start = session.optimizer_step
        warm_tokens_start = session.trained_target_tokens
        sync(); warm_started = time.perf_counter()
        warmup = session.fit(max_optimizer_steps=5)
        sync(); warmup_elapsed = time.perf_counter() - warm_started
        warmup_tokens = session.trained_target_tokens - warm_tokens_start
        if warmup["status"] == "failed":
            result = warmup
            _atomic_json(out / "mini_report.json", {**warmup, "purpose": PURPOSE, "phase": "pilot_warmup_failed"})
            return {**warmup, "purpose": PURPOSE}
        if session.optimizer_step - warm_start < 5:
            raise ValueError("token budget ended before pilot warmup completed")

    measured_start_step = session.optimizer_step
    invocation_checkpoint_seconds = session.checkpoint_seconds
    sync()
    cap = 100 if pilot else (max_steps if max_steps is not None else target // GLOBAL_BATCH)
    while session.trained_target_tokens < target and session.optimizer_step - measured_start_step < cap:
        if boundary_hook is not None and boundary_hook(session, "before_chunk") is False:
            break
        if max_seconds is not None and time.perf_counter() - invocation_started >= max_seconds:
            break
        previous_tokens = session.trained_target_tokens
        count = min(CHUNK_STEPS, cap - (session.optimizer_step - measured_start_step))
        if not pilot:
            until_eval = EVAL_INTERVAL - (previous_tokens % EVAL_INTERVAL)
            count = min(count, max(1, until_eval // GLOBAL_BATCH))
        sync()
        chunk_started = time.perf_counter()
        result = session.fit(max_optimizer_steps=count)
        sync(); elapsed_train += time.perf_counter() - chunk_started
        if result["status"] == "failed":
            break
        measured_tokens += session.trained_target_tokens - previous_tokens
        crossed = session.trained_target_tokens // EVAL_INTERVAL > previous_tokens // EVAL_INTERVAL
        final = result["status"] == "completed"
        if crossed or final or pilot and session.optimizer_step - measured_start_step >= 100:
            sync(); eval_started = time.perf_counter()
            evaluation = _eval_metric(session, audited["batches"]["dev"], config, result, out, device)
            dev_batch = audited["batches"]["dev"][0]
            with torch.no_grad():
                ids = torch.as_tensor(dev_batch["input_ids"], dtype=torch.long, device=selected)
                mask = torch.as_tensor(dev_batch["attention_mask"], dtype=torch.bool, device=selected)
                _, _, routing_diagnostic = session.model(ids, mask)
            append_routing_diagnostic(out / "routing_diagnostics.jsonl", result, routing_diagnostic)
            sync(); elapsed_eval += time.perf_counter() - eval_started
        if boundary_hook is not None and boundary_hook(session, "after_chunk") is False:
            break
        print(json.dumps({"status": result["status"], "optimizer_step": result["optimizer_step"],
                          "trained_target_tokens": result["trained_target_tokens"]}, separators=(",", ":")), flush=True)
        if result["status"] in {"completed", "failed"}:
            break

    sync()
    if "result" not in locals():
        result = session.result("incomplete")
    measured_elapsed = elapsed_train
    checkpoint_seconds = session.checkpoint_seconds - invocation_checkpoint_seconds
    allocated = torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(selected) if selected.type == "cuda" else None
    report = {
        **result, "purpose": PURPOSE, "source_kind": "measured", "device": str(selected), "precision": "float32",
        "model": config["model"], "training": session.training_config,
        "last_step_diagnostics": session.last_step_diagnostics, "data_manifest_hash": audited["bundle_hash"],
        "source_file_hashes": source_hashes, "unique_train_target_tokens_available": available,
        "eval_revision": identity["eval_revision"], "actual_tokens_this_invocation": measured_tokens,
        "actual_tokens_total": session.trained_target_tokens, "optimizer_steps_this_invocation": session.optimizer_step - start_step,
        "elapsed_training_seconds_including_checkpoint": measured_elapsed,
        "checkpoint_seconds_this_invocation": checkpoint_seconds,
        "tokens_per_second_including_checkpoint": measured_tokens / measured_elapsed if measured_elapsed else None,
        "tokens_per_second_excluding_checkpoint": measured_tokens / (measured_elapsed - checkpoint_seconds) if measured_elapsed > checkpoint_seconds else None,
        "evaluation_seconds_this_invocation": elapsed_eval, "eval_target_tokens": evaluation["eval_target_tokens"] if evaluation else None,
        "loss_nats": evaluation["loss_nats"] if evaluation else None,
        "perplexity": math.exp(evaluation["loss_nats"]) if evaluation and evaluation["loss_nats"] < math.log(float.fromhex("0x1.fffffffffffffp+1023")) else None,
        "parameter_count": sum(parameter.numel() for parameter in session.model.parameters()),
        "routing_diagnostic_scope": "first_dev_batch_only" if evaluation else None,
        "routing_diagnostic": routing_diagnostic if evaluation else None,
        "gpu_memory_allocated_peak_bytes": allocated,
        "gpu_memory_reserved_peak_bytes": reserved, "hardware": torch.cuda.get_device_name(selected) if selected.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "python_version": platform.python_version(),
        "pilot": pilot, "pilot_warmup_steps": 5 if pilot else 0, "pilot_warmup_tokens": warmup_tokens,
        "pilot_warmup_seconds": warmup_elapsed if pilot else None,
        "pilot_measured_steps": session.optimizer_step - measured_start_step if pilot else None,
        "pilot_measurement_complete": (session.optimizer_step - measured_start_step >= 100) if pilot else None,
        "pilot_usable": bool(pilot and session.optimizer_step - measured_start_step >= 100 and result["status"] != "failed" and measured_elapsed > 0 and measured_tokens > 0),
        "pilot_elapsed_including_warmup_seconds": warmup_elapsed + elapsed_train if pilot else None,
        "analysis_boundary": "exploratory only; excluded from confirmatory analysis; single-seed results do not establish hypotheses or efficiency",
    }
    _atomic_json(out / "mini_report.json", report)
    if boundary_hook is not None:
        boundary_hook(session, "after_report")
    return report


# 역할: 명령행 옵션으로 실험을 실행하고 보고서와 성공·실패 종료 코드를 출력한다.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    report = run_colab_mini(args.config, args.bundle, args.output, device=args.device, resume=args.resume,
                            max_steps=args.max_steps, max_seconds=args.max_seconds, pilot=args.pilot)
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
