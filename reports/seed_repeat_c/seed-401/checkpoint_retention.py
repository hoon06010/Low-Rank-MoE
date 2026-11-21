"""Fail-closed, explicitly enrolled seed401 local checkpoint retention.

Remote evidence is a trusted operator-produced artifact, not proof invented here.
Register a missing output before training; default inspection never deletes files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXPERIMENT = "stage-c-init-seed-repeat-v1"
HEX = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"checkpoint-[0-9a-f]{32}\.pt")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_run(root, output):
    root = Path(root).resolve()
    output = Path(output)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("output must be a relative seed401 path")
    parts = output.parts
    valid = (len(parts) == 4 and parts[:2] == ("reports", "seed_repeat_c")
             and parts[2] == "seed-401")
    valid = valid or (len(parts) == 6 and parts[:3] == ("reports", "seed_repeat_c", "preflight")
                     and parts[3] in ("pilot", "recovery") and parts[4] == "seed-401")
    if not valid or parts[-1] not in ("LR-HL", "LR-HS", "LR-LS", "LR-LL"):
        raise ValueError("only exact seed401 experiment output paths allowed")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("run path ancestor is a symlink")
    return current


def registration_path(registry, output):
    registry = Path(registry)
    if registry.is_symlink():
        raise ValueError("registration directory symlink")
    return registry / (hashlib.sha256(str(output).encode()).hexdigest() + ".json")


def register_new(output, *, root=ROOT, registry=None):
    run = safe_run(root, output)
    if run.exists():
        raise ValueError("new run enrollment requires absent output")
    registry = registry or HERE / "retention-enrollments"
    path = registration_path(registry, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": 1, "seed": 401, "output": str(output), "registered_at_ns": time.time_ns(),
              "root": str(Path(root).resolve())}
    with path.open("x") as stream:
        json.dump(record, stream, sort_keys=True)
    return record


def regular(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("symlink, hardlink or nonregular file rejected")
    return path


def load(path):
    regular(Path(path))
    return json.loads(Path(path).read_text())


def inspect(output, receipt, evidence, *, root=ROOT, registry=None):
    run = safe_run(root, output)
    registry = registry or HERE / "retention-enrollments"
    enrollment = load(registration_path(registry, output))
    if (enrollment.get("seed") != 401 or enrollment.get("output") != str(output)
            or enrollment.get("root") != str(Path(root).resolve())
            or type(enrollment.get("registered_at_ns")) is not int):
        raise ValueError("new-run enrollment mismatch")
    manifest_path = regular(run / "run_manifest.jsonl")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("exactly one run manifest required")
    manifest = rows[0]
    identity = receipt.get("identity", {})
    expected_phase = Path(output).parts[3] if len(Path(output).parts) == 6 else "main"
    if (identity.get("seed") != 401 or identity.get("experiment_id") != EXPERIMENT
            or identity.get("phase") != expected_phase or identity.get("output") != str(output)
            or identity.get("cell") != Path(output).name[3:]
            or not identity.get("run_id") or identity.get("run_id") != manifest.get("run_id")
            or manifest.get("seed") != 401 or manifest.get("cell") != identity["cell"]):
        raise ValueError("foreign seed or run identity")
    for key, field in (("source_revision", "execution_source_revision"),
                       ("data_manifest_hash", "data_manifest_hash"),
                       ("eval_revision", "eval_revision"),
                       ("config_file_sha256", "config_file_sha256")):
        if not HEX.fullmatch(str(identity.get(key, ""))) or manifest.get(field) != identity[key]:
            raise ValueError("source/data/config identity mismatch")
    if evidence.get("identity") != identity or evidence.get("status") != "remote-publish-verified":
        raise ValueError("independent remote verification missing")
    if receipt.get("evidence_kind") != "direct-drive-independent-download-sha256":
        raise ValueError("independent readback evidence required")
    for left, right in (("sha256", "archive_sha256"), ("bytes", "archive_bytes"),
                        ("checkpoint_id", "checkpoint_id"), ("checkpoint_sha256", "checkpoint_sha256"),
                        ("archive_id", "archive_id"), ("generation", "generation")):
        if not receipt.get(left) or receipt[left] != evidence.get(right):
            raise ValueError("remote archive/checkpoint readback mismatch")
    if not HEX.fullmatch(str(receipt["sha256"])) or not HEX.fullmatch(str(receipt["checkpoint_sha256"])):
        raise ValueError("invalid verified hashes")
    if type(receipt["bytes"]) is not int or receipt["bytes"] <= 0:
        raise ValueError("invalid archive byte count")
    checkpoints = run / "checkpoints"
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        raise ValueError("checkpoint directory symlink or absent")
    pointer = load(checkpoints / "latest.json")
    latest_name = pointer.get("filename", "")
    if not NAME.fullmatch(latest_name):
        raise ValueError("invalid latest filename")
    if (pointer.get("checkpoint_id") != receipt["checkpoint_id"]
            or pointer.get("sha256") != receipt["checkpoint_sha256"]
            or manifest.get("checkpoint_id") != receipt["checkpoint_id"]
            or manifest.get("checkpoint_step") != receipt.get("step")):
        raise ValueError("latest pointer identity mismatch")
    # Managed, enrolled trainer checkpoints are trusted local torch artifacts.
    # Never point this helper at downloaded/untrusted pickle files.
    import torch
    files = []
    for path in checkpoints.iterdir():
        regular(path)
        if path.name == "latest.json":
            continue
        if not NAME.fullmatch(path.name):
            raise ValueError("unknown checkpoint directory content")
        stat = path.stat()
        if stat.st_mtime_ns < enrollment["registered_at_ns"] or stat.st_ctime_ns < enrollment["registered_at_ns"]:
            raise ValueError("checkpoint predates new-run enrollment")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (not isinstance(payload, dict) or payload.get("config", {}).get("seed") != 401
                or payload.get("config", {}).get("cell") != identity["cell"]
                or payload.get("run_id") != identity["run_id"]
                or payload.get("source_revision") != identity["source_revision"]
                or payload.get("data_manifest_hash") != identity["data_manifest_hash"]
                or payload.get("eval_revision") != identity["eval_revision"]):
            raise ValueError("foreign checkpoint payload identity")
        del payload
        files.append(path)
    if not files or latest_name not in [p.name for p in files]:
        raise ValueError("latest checkpoint missing")
    latest = checkpoints / latest_name
    if sha(latest) != pointer["sha256"]:
        raise ValueError("latest checkpoint hash mismatch")
    state_path = run / "execution_state.json"
    if state_path.exists():
        if load(state_path).get("status") != "verified":
            raise ValueError("run is not at a verified stopped boundary")
    # Always retain latest even if clock timestamps rank it older.
    ordered = sorted(files, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    keep = [latest]
    keep += [p for p in ordered if p != latest][:1]
    candidates = [p for p in files if p not in keep]
    return {"run": str(run), "run_id": identity["run_id"], "pointer": pointer,
            "keep": [p.name for p in keep], "delete": [
                {"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p),
                 "mtime_ns": p.stat().st_mtime_ns} for p in candidates],
            "reclaim_bytes": sum(p.stat().st_size for p in candidates), "dry_run": True}


def retain(output, receipt, evidence, *, apply=False, root=ROOT, registry=None):
    plan = inspect(output, receipt, evidence, root=root, registry=registry)
    if not apply:
        return plan
    # Validate everything again before starting mutations; no partial uncertain plan.
    if inspect(output, receipt, evidence, root=root, registry=registry) != plan:
        raise ValueError("run changed before retention")
    run = Path(plan["run"])
    for item in plan["delete"]:
        path = regular(run / "checkpoints" / item["name"])
        if load(run / "checkpoints/latest.json") != plan["pointer"]:
            raise ValueError("latest pointer changed during retention")
        if path.stat().st_mtime_ns != item["mtime_ns"] or sha(path) != item["sha256"]:
            raise ValueError("checkpoint changed during retention")
        path.unlink()
    plan["dry_run"] = False
    return plan


def estimate_peak(mature_checkpoint_bytes, *, cells=4, local_keep=2, remote_keep=2,
                  reserve_bytes=1024**3, package_bytes=0):
    """Uncompressed bound: keep+new checkpoint and tar/stage/readback overlap.

    Reports/data runtime baseline must be measured and added separately.
    Retention requires independent readback before dropping older files.
    """
    values = (mature_checkpoint_bytes, cells, local_keep, remote_keep, reserve_bytes, package_bytes)
    if any(type(v) is not int or v < 0 for v in values) or mature_checkpoint_bytes == 0 or cells == 0:
        raise ValueError("byte budgets must be nonnegative integers, positive size/cells")
    if local_keep < 2 or remote_keep < 2:
        raise ValueError("at least two verified generations required")
    size = mature_checkpoint_bytes
    return {"local_checkpoint_bytes": cells * local_keep * size,
            "remote_checkpoint_bytes": cells * remote_keep * size,
            "local_peak_bytes": (cells * local_keep + 4) * size + reserve_bytes + package_bytes,
            "remote_peak_bytes": (cells * remote_keep + 1) * size + reserve_bytes,
            "assumptions": "one concurrent new checkpoint + local tar + staged copy + independent download; reports/data baseline extra"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--register-new", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.register_new:
        if args.receipt or args.evidence or args.apply:
            parser.error("enrollment cannot also prune")
        result = register_new(args.output)
    else:
        if not args.receipt or not args.evidence:
            parser.error("--receipt and --evidence required")
        result = retain(args.output, load(args.receipt), load(args.evidence), apply=args.apply)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
