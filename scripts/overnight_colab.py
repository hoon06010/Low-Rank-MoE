"""Notebook-started, fail-closed free-Colab B/C controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.overnight_store import digest, publish_snapshot, restore_snapshot, deadline, _atomic_json
from scripts.run_colab_mini import run_colab_mini

CELLS = ("HL", "HS", "LS", "LL")
STEPS = {"B": 768, "C": 2048}
BASE_SECONDS = {"B": 8379.23, "C": 21698.26}
PRESERVE_STEPS = 256
IO_TIMEOUT_SECONDS = 300
FINALIZE_SECONDS = 60
BUNDLE = ROOT / "data/prepared/colab_free_mini_v1/bundle.json"


def _runtime_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _environment() -> dict:
    import torch
    ready = json.loads((ROOT / "longer_environment_ready.json").read_text())
    if ready.get("status") != "ready" or ready.get("gpu") != "Tesla T4" or not torch.cuda.is_available():
        raise ValueError("validated Tesla T4 environment is required")
    if torch.cuda.get_device_name(0) != "Tesla T4" or ready.get("precision") != "float32":
        raise ValueError("GPU or precision differs from the measured pilot")
    if (torch.__version__ != ready.get("torch") or torch.version.cuda != ready.get("cuda") or
            platform.python_version() != ready.get("python")):
        raise ValueError("library version changed after bootstrap")
    if digest(BUNDLE) != ready.get("bundle_sha256"):
        raise ValueError("bundle file changed after bootstrap")
    package = json.loads(Path("/content/longer_payload_manifest.json").read_text())
    if package.get("payload_sha256") != ready.get("payload_sha256"):
        raise ValueError("payload manifest changed after bootstrap")
    for item in package["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"payload file changed after bootstrap: {item['path']}")
    return ready


def _identity(stage: str, cell: str, ready: dict) -> dict:
    config = ROOT / f"configs/longer_training_{stage.lower()}/LR-{cell}.json"
    source_paths = [ROOT / "scripts" / name for name in
                    ("run_colab_mini.py", "overnight_colab.py", "overnight_store.py")]
    source_paths += sorted((ROOT / "src").glob("*.py"))
    return {"stage": stage, "cell": cell, "seed": 17, "config_sha256": digest(config),
            "bundle_sha256": digest(BUNDLE), "payload_sha256": ready["payload_sha256"],
            "source_sha256": hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                                                sort_keys=True).encode()).hexdigest(),
            "notebook_template_sha256": digest(ROOT / "overnight_free_colab.ipynb"),
            "gpu": ready["gpu"], "torch": ready["torch"], "cuda": ready["cuda"],
            "python": ready["python"], "precision": "float32"}


def _trim_local(run: Path, keep: int = 2) -> None:
    pointer = json.loads((run / "checkpoints/latest.json").read_text())
    files = sorted((run / "checkpoints").glob("checkpoint-*.pt"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    retained = {pointer["filename"], *(path.name for path in files[:keep])}
    for path in files:
        if path.name not in retained:
            path.unlink()


def _rehearsal_path(drive: Path) -> Path:
    return drive / "overnight_rehearsal.json"


def _verified_rehearsal(drive: Path, ready: dict) -> dict:
    path = _rehearsal_path(drive)
    if not path.is_file():
        raise ValueError("GPU Drive round-trip rehearsal has not been verified")
    record = json.loads(path.read_text())
    if (record.get("status") != "verified" or record.get("payload_sha256") != ready["payload_sha256"] or
            record.get("gpu") != ready["gpu"] or record.get("restored_step") != 512 or
            record.get("advanced_step") != 513 or record.get("snapshot_seconds_max", 0) <= 0 or
            record.get("snapshot_bytes_max", 0) <= 0):
        raise ValueError("rehearsal does not match this payload and GPU")
    return record


def _budget_gate(stage: str, total_seconds: float, session_seconds: float, rehearsal: dict) -> None:
    if not (math.isfinite(total_seconds) and math.isfinite(session_seconds) and
            0 < session_seconds <= total_seconds):
        raise ValueError("invalid total/session time limits")
    generations = 4 * (STEPS[stage] // PRESERVE_STEPS + 1)
    estimated = BASE_SECONDS[stage] + generations * rehearsal["snapshot_seconds_max"] + rehearsal["bootstrap_seconds"]
    if estimated > total_seconds:
        raise ValueError(f"stage estimate {estimated:.1f}s exceeds declared {total_seconds:.1f}s limit")
    if session_seconds < 900:
        raise ValueError("session time limit leaves no recovery reserve")



def _drive_space_gate(available_bytes: int, rehearsal: dict) -> dict:
    measured=rehearsal.get("snapshot_bytes_max")
    if type(available_bytes) is not int or available_bytes<=0:
        raise ValueError("record manually verified Drive available bytes")
    if type(measured) is not int or measured<=0:
        raise ValueError("missing measured mature snapshot size")
    bound=(measured*11+9)//10
    required=11*bound+1024**3
    local_required=12*bound+2*1024**3
    if available_bytes<required:
        raise ValueError("Drive free space below conservative stage reservation")
    if shutil.disk_usage(ROOT).free<local_required:
        raise ValueError("local free space below stage/temp reservation")
    return {"measured_snapshot_bytes_max":measured,"snapshot_bound_bytes":bound,
            "required_peak_bytes":required,"required_local_free_bytes":local_required,
            "quota_source":"manual account storage UI; conservative byte estimate"}

def _write_progress(drive: Path, stage: str, cumulative: float, *,
                    reserved_until: float | None = None, **details) -> None:
    with deadline(30):
        path=drive/stage/"progress.json"
        old=json.loads(path.read_text()) if path.exists() else {}
        cumulative=max(cumulative,float(old.get("cumulative_elapsed_seconds",0)))
        reservation=float(old.get("reserved_until_seconds",cumulative)) if reserved_until is None else reserved_until
        _atomic_json(path,{"stage":stage,**details,"cumulative_elapsed_seconds":cumulative,
            "reserved_until_seconds":max(cumulative,reservation),"session_runtime_id":_runtime_id()})

def _resume_elapsed(drive: Path, stage: str) -> float:
    record=json.loads((drive/stage/"progress.json").read_text())
    if "reserved_until_seconds" not in record:
        raise ValueError("legacy ledger lacks bounded in-flight reservation")
    values=[float(record["cumulative_elapsed_seconds"]),float(record["reserved_until_seconds"])]
    summary=drive/stage/"overnight_summary.json"
    if summary.exists():
        values.append(float(json.loads(summary.read_text())["cumulative_elapsed_seconds"]))
    if any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError("invalid time ledger")
    return max(values)


def _protocol(stage: str, drive: Path, ready: dict, total_seconds: float,
              session_seconds: float, rehearsal: dict, *, create: bool, space: dict | None = None) -> dict:
    fixed = {"stage": stage, "cells": list(CELLS), "seed": 17,
             "accounting_policy": "reserve-session-until-clean-exit-v1",
             "io_timeout_seconds": IO_TIMEOUT_SECONDS, "space": space,
             "steps_per_cell": STEPS[stage], "preserve_every_steps": PRESERVE_STEPS,
             "retained_drive_generations": 2, "retained_local_checkpoints": 2,
             "total_seconds": total_seconds, "session_seconds": session_seconds,
             "rehearsal_sha256": digest(_rehearsal_path(drive)),
             "identity": {cell: _identity(stage, cell, ready) for cell in CELLS}}
    path = drive / stage / "overnight_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != fixed:
            raise ValueError("frozen overnight protocol differs from current inputs")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fixed, sort_keys=True, indent=2) + "\n")
    else:
        raise ValueError("run requires a preflight-frozen protocol")
    return fixed


def _run_one(stage: str, cell: str, drive: Path, ready: dict, *, max_steps: int | None,
             started: float, total_seconds: float, session_seconds: float,
             snapshot_times: list[float], resume: bool = False, restored: bool = False,
             rehearsal: bool = False, prior_elapsed: float = 0.0) -> dict:
    scope = "rehearsal" if rehearsal else stage
    run = ROOT / "runs/overnight" / scope / cell
    saved = drive / scope / cell
    identity = _identity(stage, cell, ready)
    if resume:
        if not restored:
            restore_snapshot(saved, run, identity, timeout_seconds=IO_TIMEOUT_SECONDS)
    elif run.exists() and any(run.iterdir()):
        raise ValueError(f"existing run requires explicit restore: {run}")

    def boundary(session, phase: str):
        elapsed = time.monotonic() - started
        remaining = min(total_seconds - prior_elapsed - elapsed, session_seconds - elapsed)
        pointer = json.loads((run / "checkpoints/latest.json").read_text())
        latest = run / "checkpoints" / pointer["filename"]
        needed = latest.stat().st_size * 4 + 2 * 1024**3
        if shutil.disk_usage(ROOT).free < needed:
            raise ValueError("insufficient Colab local space for next checkpoint and snapshot")
        if phase == "before_chunk":
            # Allow a bounded 32-step chunk, evaluation, save, and recovery reserve.
            chunk_reserve = 120 if max_steps == 1 else 900
            if remaining < chunk_reserve + max(120, 2 * max(snapshot_times, default=0)):
                return False
        if phase == "initial" or phase == "after_report" and session.trained_target_tokens >= session.training_config["total_target_tokens"] or (
                phase == "after_chunk" and session.optimizer_step % PRESERVE_STEPS == 0 and
                session.optimizer_step > 0 and
                session.trained_target_tokens < session.training_config["total_target_tokens"]):
            if remaining < max(120, 2 * max(snapshot_times, default=0)):
                raise ValueError("insufficient time to verify Drive snapshot")
            snap_start = time.monotonic()
            publish_snapshot(run, saved, identity, step=session.optimizer_step,
                             timeout_seconds=min(IO_TIMEOUT_SECONDS, remaining))
            snapshot_times.append(time.monotonic() - snap_start)
            _trim_local(run)
        elif phase == "after_chunk" and saved.joinpath("latest.json").exists():
            _trim_local(run)
        if phase in {"initial", "after_chunk", "after_report"}:
            _write_progress(drive, scope, prior_elapsed + time.monotonic() - started,
                            cell=cell, step=session.optimizer_step)
        return True

    report = run_colab_mini(ROOT / f"configs/longer_training_{stage.lower()}/LR-{cell}.json", BUNDLE,
                            run, device="cuda", resume=resume, max_steps=max_steps,
                            boundary_hook=boundary)
    if report["status"] == "failed":
        raise ValueError(f"training failed for {stage}/{cell}: {report['reason']}")
    if report["status"] == "completed" and (report["optimizer_step"] != STEPS[stage] or
                                             report.get("loss_nats") is None or
                                             not saved.joinpath("latest.json").is_file()):
        raise ValueError(f"completed {stage}/{cell} lacks final endpoint, dev CE, or Drive generation")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rehearse", "verify-rehearsal", "preflight", "run"), required=True)
    parser.add_argument("--stage", choices=("B", "C"), default="B")
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--total-seconds", type=float)
    parser.add_argument("--session-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--drive-available-bytes", type=int)
    args = parser.parse_args(argv)
    if os.environ.get("COLAB_NOTEBOOK_UI") != "1":
        raise ValueError("start this controller explicitly from the Colab notebook UI")
    drive = args.drive.resolve()
    if not drive.is_dir() or not str(drive).startswith("/content/drive/"):
        raise ValueError("a mounted Google Drive directory is required")
    ready = _environment()
    started = float(os.environ.get("OVERNIGHT_SESSION_STARTED", "nan"))
    if not (0 < started <= time.monotonic()):
        raise ValueError("notebook session start time is missing")
    snapshots = []
    if args.mode == "rehearse":
        if args.stage != "B" or args.resume:
            raise ValueError("rehearsal starts fresh with B/HL")
        if _rehearsal_path(drive).exists() or (drive / "rehearsal/HL/latest.json").exists():
            raise ValueError("rehearsal already exists; inspect it before any new attempt")
        with deadline(10800 - (time.monotonic() - started) - FINALIZE_SECONDS):
            report = _run_one("B", "HL", drive, ready, max_steps=512, started=started,
                              total_seconds=10800, session_seconds=10800, snapshot_times=snapshots,
                              rehearsal=True)
        if report["optimizer_step"] != 512 or len(snapshots) < 3:
            raise ValueError("rehearsal did not cross two preservation boundaries")
        record = {"status": "awaiting_new_runtime", "payload_sha256": ready["payload_sha256"],
                  "gpu": ready["gpu"], "runtime_id": _runtime_id(),
                  "snapshot_seconds_max": max(snapshots), "bootstrap_seconds": ready["bootstrap_seconds"],
                  "snapshot_bytes_max": max(json.loads(p.read_text())["bytes"]
                      for p in (drive / "rehearsal/HL").glob("generation-*.json"))}
        _rehearsal_path(drive).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2)); return
    if args.mode == "verify-rehearsal":
        prior = json.loads(_rehearsal_path(drive).read_text())
        if prior.get("status") != "awaiting_new_runtime" or prior.get("runtime_id") == _runtime_id():
            raise ValueError("rehearsal recovery requires a different Colab runtime")
        with deadline(1800 - (time.monotonic() - started) - FINALIZE_SECONDS):
            report = _run_one("B", "HL", drive, ready, max_steps=1, started=started,
                              total_seconds=1800, session_seconds=1800, snapshot_times=snapshots,
                              resume=True, rehearsal=True)
        if report["optimizer_step"] != 513:
            raise ValueError("rehearsal did not advance restored step 512 to 513")
        prior.update(status="verified", restored_step=512, advanced_step=513)
        _rehearsal_path(drive).write_text(json.dumps(prior, indent=2) + "\n")
        print(json.dumps(prior, indent=2)); return
    rehearsal = _verified_rehearsal(drive, ready)
    if args.total_seconds is None or args.session_seconds is None:
        raise ValueError("declare both total and session time limits")
    _budget_gate(args.stage, args.total_seconds, args.session_seconds, rehearsal)
    space = _drive_space_gate(args.drive_available_bytes, rehearsal)
    with deadline(30):
        _atomic_json(drive / args.stage / "quota_observation.json",
                     {"available_bytes": args.drive_available_bytes, "checked_at_unix": time.time(), **space})
    if args.mode == "preflight":
        _protocol(args.stage, drive, ready, args.total_seconds, args.session_seconds, rehearsal, create=True, space=space)
        print(json.dumps({"status": "ready_to_start", "stage": args.stage,
                          "total_seconds": args.total_seconds, "session_seconds": args.session_seconds,
                          "payload_sha256": ready["payload_sha256"]}, indent=2))
        return
    _protocol(args.stage, drive, ready, args.total_seconds, args.session_seconds, rehearsal, create=False, space=space)
    progress_path = drive / args.stage / "progress.json"
    if args.resume:
        if not progress_path.is_file():
            raise ValueError("resume requires a recorded stage time ledger")
        prior_elapsed = _resume_elapsed(drive, args.stage)
    else:
        if progress_path.exists():
            raise ValueError("stage already has progress; resume explicitly")
        prior_elapsed = 0.0
    if prior_elapsed + (time.monotonic() - started) + 900 >= args.total_seconds:
        raise ValueError("declared stage total time is exhausted")
    elapsed = time.monotonic() - started
    remaining = min(args.total_seconds - prior_elapsed - elapsed, args.session_seconds - elapsed)
    if remaining <= FINALIZE_SECONDS:
        raise ValueError("declared session time is exhausted")
    _write_progress(drive, args.stage, prior_elapsed + elapsed,
                    reserved_until=prior_elapsed + elapsed + remaining, phase="session_reserved")
    results = {}
    error = None
    try:
        with deadline(min(args.total_seconds - prior_elapsed - (time.monotonic() - started),
                          args.session_seconds - (time.monotonic() - started)) - FINALIZE_SECONDS):
            for cell in CELLS:
                scope = drive / args.stage / cell
                run = ROOT / "runs/overnight" / args.stage / cell
                existing = scope / "latest.json"
                should_resume = args.resume and existing.exists()
                if should_resume:
                    identity = _identity(args.stage, cell, ready)
                    restored = restore_snapshot(scope, run, identity, timeout_seconds=IO_TIMEOUT_SECONDS)
                    rows = [json.loads(line) for line in (run / "run_manifest.jsonl").read_text().splitlines() if line.strip()]
                    if rows[0]["status"] == "completed" and restored["step"] == STEPS[args.stage]:
                        report_path = run / "mini_report.json"
                        if not report_path.is_file() or json.loads(report_path.read_text()).get("loss_nats") is None:
                            raise ValueError("restored completed run lacks final dev CE")
                        results[cell] = json.loads(report_path.read_text())
                        continue
                results[cell] = _run_one(args.stage, cell, drive, ready, max_steps=None, started=started,
                                         total_seconds=args.total_seconds, session_seconds=args.session_seconds,
                                         snapshot_times=snapshots, resume=should_resume, restored=should_resume,
                                         prior_elapsed=prior_elapsed)
                if results[cell]["status"] != "completed":
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    status = ("failed" if error else "completed" if len(results) == 4 and
              all(row["status"] == "completed" for row in results.values()) else "incomplete")
    summary = {"status": status, "stage": args.stage, "cells": results, "error": error,
               "session_elapsed_seconds": time.monotonic() - started,
               "cumulative_elapsed_seconds": prior_elapsed + time.monotonic() - started,
               "snapshot_seconds": snapshots}
    with deadline(30):
        _atomic_json(drive / args.stage / "overnight_summary.json", summary)
    # Charge the bounded final ledger transaction conservatively.
    final_elapsed = prior_elapsed + time.monotonic() - started + 30
    _write_progress(drive, args.stage, final_elapsed,
                    reserved_until=final_elapsed, phase="terminal", status=status)
    print(json.dumps(summary, indent=2))
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
