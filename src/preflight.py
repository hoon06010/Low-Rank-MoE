"""Small CPU-only synthetic training preflight; no external data is fetched."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import torch

from src.data import build_split_batches
from src.evaluate import evaluate
from src.model import MoETransformer
from src.train import TrainingSession, write_metrics_csv


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def _encoder(text: str) -> list[int]:
    try:
        return [int(token) for token in text.split()]
    except ValueError as exc:
        raise ValueError("toy fixture text must contain whitespace-separated token IDs") from exc


def run_preflight(config_path: str | Path, output_dir: str | Path, *, allow_large: bool = False) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("kind") != "toy_cpu" and not allow_large:
        raise ValueError("non-toy configs require explicit --allow-large; only toy CPU was verified by M5")
    if config.get("device", "cpu") != "cpu":
        raise ValueError("this preflight implementation supports CPU only; accelerator execution is unimplemented")
    model_config = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    docs = data_cfg["documents"]
    train_batches = build_split_batches(
        docs, split="train", encode=_encoder, eos_token_id=data_cfg["eos_token_id"],
        context_length=model_config["context_length"], pad_token_id=data_cfg["pad_token_id"], seed=data_cfg["seed"],
    )
    dev_batches = build_split_batches(
        docs, split="dev", encode=_encoder, eos_token_id=data_cfg["eos_token_id"],
        context_length=model_config["context_length"], pad_token_id=data_cfg["pad_token_id"], seed=data_cfg["seed"],
    )
    manifest = {
        "source_kind": "synthetic_fixture",
        "documents": docs,
        "train_batches": train_batches,
        "dev_batches": dev_batches,
    }
    data_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(model_config["seed"])
    model = MoETransformer(model_config)
    session = TrainingSession(
        model, model_config, train_cfg, train_batches,
        {
            "protocol_id": config["protocol_id"],
            "source_revision": config.get("source_revision", "unavailable"),
            "data_manifest_hash": data_hash,
            "tokenizer_revision": config["tokenizer_revision"],
            "eval_revision": config["eval_revision"],
            "source_kind": "synthetic_fixture",
        }, out,
    )
    start = time.perf_counter()
    result = session.fit()
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(rss if os.uname().sysname == "Darwin" else rss * 1024)
    tokens_per_second = result["trained_target_tokens"] / elapsed if elapsed > 0 else None
    evaluation = evaluate(model, dev_batches) if result["status"] == "completed" else None
    if evaluation is not None:
        metric = {
            "run_id": result["run_id"], "protocol_id": config["protocol_id"],
            "config_id": model_config["config_id"], "family": model_config["family"],
            "cell": model_config["cell"], "seed": model_config["seed"], "split": "dev",
            "checkpoint_id": result["checkpoint_id"], "trained_target_tokens": result["trained_target_tokens"],
            "eval_target_tokens": evaluation["eval_target_tokens"], "loss_nats": evaluation["loss_nats"],
            "status": result["status"], "source_kind": "synthetic_fixture",
        }
        write_metrics_csv(out / "metrics.csv", metric)
    report = {
        "status": result["status"], "source_kind": "synthetic_fixture", "device": "cpu",
        "accelerator_support": "unimplemented and unverified",
        "run_id": result["run_id"], "config_hash": session.config_hash,
        "data_manifest_hash": data_hash, "checkpoint_id": result["checkpoint_id"],
        "reason": result["reason"],
        "trained_target_tokens": result["trained_target_tokens"], "optimizer_step": result["optimizer_step"],
        "eval_target_tokens": evaluation["eval_target_tokens"] if evaluation else None,
        "loss_nats": evaluation["loss_nats"] if evaluation else None,
        "elapsed_seconds": elapsed, "tokens_per_second": tokens_per_second,
        "peak_process_rss_bytes": peak_rss_bytes, "gpu_memory_bytes": None,
        "gpu_measurement_status": "not_measured_cpu_only",
    }
    _atomic_json(out / "preflight.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/toy_cpu.json")
    parser.add_argument("--output", default="results/preflight/toy_cpu")
    parser.add_argument("--allow-large", action="store_true", help="explicitly permit non-toy config; CPU-only, unverified")
    args = parser.parse_args(argv)
    report = run_preflight(args.config, args.output, allow_large=args.allow_large)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
