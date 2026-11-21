"""Verified, bounded checkpoint snapshots for a mounted Colab Drive directory."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import math
import signal
import threading
import sys
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import uuid



@contextmanager
def deadline(seconds: float):
    """POSIX main-thread deadline; nested operations cannot extend their caller."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError("no remaining operation time")
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal,"setitimer"):
        raise RuntimeError("POSIX main-thread deadlines required")
    handler=signal.getsignal(signal.SIGALRM)
    timer=signal.getitimer(signal.ITIMER_REAL)
    start=time.monotonic()
    def expired(*_):
        raise TimeoutError("overnight operation deadline exceeded")
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,min(seconds,timer[0]) if timer[0]>0 else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,handler)
        if timer[0]>0:
            remaining=timer[0]-(time.monotonic()-start)
            if remaining>0:
                signal.setitimer(signal.ITIMER_REAL,remaining,timer[1])
            elif sys.exc_info()[0] is None:
                # Do not rearm an expired signal during handler restoration.
                raise TimeoutError("enclosing operation deadline exceeded")

def publish_snapshot(run: Path, drive: Path, identity: dict, *, step: int,
                     keep_generations: int = 2, timeout_seconds: float = 300) -> dict:
    with deadline(timeout_seconds):
        return _publish_snapshot(run,drive,identity,step=step,keep_generations=keep_generations)

def restore_snapshot(drive: Path, output: Path, identity: dict, *, timeout_seconds: float = 300) -> dict:
    with deadline(timeout_seconds):
        return _restore_snapshot(drive,output,identity)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _checkpoint(run: Path) -> tuple[dict, Path]:
    pointer = json.loads((run / "checkpoints/latest.json").read_text())
    name = pointer.get("filename", "")
    if not name or Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("invalid checkpoint pointer")
    path = run / "checkpoints" / name
    if path.is_symlink() or not path.is_file() or digest(path) != pointer.get("sha256"):
        raise ValueError("checkpoint integrity mismatch")
    return pointer, path


def _publish_snapshot(run: Path, drive: Path, identity: dict, *, step: int,
                      keep_generations: int = 2) -> dict:
    """Copy one immutable generation, read it back, then publish its manifest."""
    run, drive = Path(run), Path(drive)
    if step < 0 or keep_generations < 2 or not identity:
        raise ValueError("invalid snapshot settings")
    pointer, checkpoint = _checkpoint(run)
    manifest = run / "run_manifest.jsonl"
    if not manifest.is_file():
        raise ValueError("missing run manifest")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0].get("checkpoint_id") != pointer.get("checkpoint_id") or rows[0].get("checkpoint_step") != step:
        raise ValueError("run manifest and checkpoint step disagree")
    files = [checkpoint, run / "checkpoints/latest.json", manifest]
    files += [run / name for name in ("metrics.csv", "routing_diagnostics.jsonl", "mini_report.json") if (run / name).is_file()]
    files += sorted(path for path in (run / "diagnostics").rglob("*.json") if path.is_file()) if (run / "diagnostics").exists() else []
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("snapshot contains symlink or missing file")
    drive.mkdir(parents=True, exist_ok=True)
    generation = f"generation-{time.time_ns():020d}-{uuid.uuid4().hex}"
    archive_name = generation + ".tar"
    archive = drive / archive_name
    with tempfile.TemporaryDirectory(prefix="overnight-snapshot-") as temp:
        local = Path(temp) / archive_name
        required = sum(path.stat().st_size for path in files)
        if shutil.disk_usage(temp).free < required * 2 + 1024**3:
            raise ValueError("insufficient local snapshot space")
        with tarfile.open(local, "w") as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(run)), recursive=False)
        expected = digest(local)
        partial = drive / (archive_name + ".partial")
        shutil.copyfile(local, partial)
        if partial.stat().st_size != local.stat().st_size or digest(partial) != expected:
            raise ValueError("Drive copy integrity mismatch")
        os.replace(partial, archive)
        if archive.stat().st_size != local.stat().st_size or digest(archive) != expected:
            raise ValueError("Drive readback integrity mismatch")
    record = {"generation": generation, "archive": archive_name, "sha256": expected,
              "bytes": archive.stat().st_size, "step": step, "identity": identity,
              "checkpoint_sha256": pointer["sha256"], "checkpoint_id": pointer["checkpoint_id"]}
    _atomic_json(drive / (generation + ".json"), record)
    _atomic_json(drive / "latest.json", {"generation": generation})
    records = sorted(drive.glob("generation-*.json"), reverse=True)
    for old in records[keep_generations:]:
        old_record = json.loads(old.read_text())
        old_archive = drive / old_record["archive"]
        old.unlink()
        old_archive.unlink(missing_ok=True)
    return record


def _trim_after_checkpoint(output: Path, tokens: int) -> None:
    metrics = output / "metrics.csv"
    if metrics.exists():
        with metrics.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row["trained_target_tokens"]) <= tokens]
        with metrics.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    diagnostics = output / "routing_diagnostics.jsonl"
    if diagnostics.exists():
        kept = [line for line in diagnostics.read_text().splitlines() if line.strip() and
                int(json.loads(line)["trained_target_tokens"]) <= tokens]
        diagnostics.write_text("".join(line + "\n" for line in kept))
    per_step = output / "diagnostics"
    if per_step.exists():
        for path in per_step.rglob("*.json"):
            if int(json.loads(path.read_text())["trained_target_tokens"]) > tokens:
                path.unlink()


def _restore_snapshot(drive: Path, output: Path, identity: dict) -> dict:
    """Restore the newest valid of the two retained generations to a fresh path."""
    drive, output = Path(drive), Path(output)
    if output.exists():
        raise ValueError("restore output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((drive / "latest.json").read_text())
    records = sorted(drive.glob("generation-*.json"), reverse=True)
    if not records or not any(path.stem == pointer.get("generation") for path in records):
        raise ValueError("snapshot pointer mismatch")
    records = [path for path in records if path.stem <= pointer["generation"]]
    last_error = None
    for path in records[:2]:
        try:
            record = json.loads(path.read_text())
            if record["identity"] != identity:
                raise ValueError("snapshot identity mismatch")
            archive = drive / record["archive"]
            if (not archive.is_file() or archive.stat().st_size != record["bytes"] or
                    digest(archive) != record["sha256"]):
                raise ValueError("snapshot archive integrity mismatch")
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                names = [item.name for item in members]
                if (len(names) != len(set(names)) or
                        any(not item.isfile() or not (output / item.name).resolve().is_relative_to(output.resolve()) for item in members)):
                    raise ValueError("unsafe snapshot archive")
                if shutil.disk_usage(output.parent).free < sum(item.size for item in members) + 1024**3:
                    raise ValueError("insufficient restore space")
                output.mkdir(parents=True)
                try:
                    tar.extractall(output, filter="data")
                    actual, _ = _checkpoint(output)
                    if actual["sha256"] != record["checkpoint_sha256"] or actual["checkpoint_id"] != record["checkpoint_id"]:
                        raise ValueError("restored checkpoint integrity mismatch")
                    rows = [json.loads(line) for line in (output / "run_manifest.jsonl").read_text().splitlines() if line.strip()]
                    if len(rows) != 1 or rows[0]["checkpoint_step"] != record["step"]:
                        raise ValueError("restored manifest step mismatch")
                    _trim_after_checkpoint(output, int(rows[0]["trained_target_tokens"]))
                    if rows[0]["status"] != "completed":
                        (output / "mini_report.json").unlink(missing_ok=True)
                    return record
                except Exception:
                    shutil.rmtree(output)
                    raise
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"no verified snapshot: {last_error}")
