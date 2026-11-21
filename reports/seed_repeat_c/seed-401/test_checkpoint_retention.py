import importlib.util
from pathlib import Path
import hashlib
import json
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("checkpoint_retention", Path(__file__).with_name("checkpoint_retention.py"))
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)


@pytest.fixture
def fixture(tmp_path):
    output = "reports/seed_repeat_c/preflight/pilot/seed-401/LR-HL"
    registry = tmp_path / "registry"
    retention.register_new(output, root=tmp_path, registry=registry)
    run = tmp_path / output
    directory = run / "checkpoints"
    directory.mkdir(parents=True)
    files = []
    for i in range(4):
        path = directory / ("checkpoint-" + str(i) * 32 + ".pt")
        torch.save({"config":{"seed":401,"cell":"HL"}, "run_id":"new-seed401-run",
                    "source_revision":"a"*64,"data_manifest_hash":"b"*64,
                    "eval_revision":"c"*64,"dummy":torch.tensor([i])}, path)
        files.append(path)
    latest = files[-1]
    pointer = {"filename": latest.name, "checkpoint_id": "c4", "sha256": retention.sha(latest)}
    (directory / "latest.json").write_text(json.dumps(pointer))
    identity = {"seed":401, "experiment_id":retention.EXPERIMENT, "phase":"pilot",
                "cell":"HL", "output":output, "run_id":"new-seed401-run",
                "source_revision":"a"*64, "data_manifest_hash":"b"*64,
                "eval_revision":"c"*64, "config_file_sha256":"d"*64}
    manifest = {"run_id":identity["run_id"], "seed":401, "cell":"HL",
                "execution_source_revision":"a"*64, "data_manifest_hash":"b"*64,
                "eval_revision":"c"*64, "config_file_sha256":"d"*64,
                "checkpoint_id":"c4", "checkpoint_step":105}
    (run / "run_manifest.jsonl").write_text(json.dumps(manifest)+"\n")
    receipt = {"identity":identity, "evidence_kind":"direct-drive-independent-download-sha256",
               "sha256":"e"*64, "bytes":100, "checkpoint_id":"c4",
               "checkpoint_sha256":pointer["sha256"], "archive_id":"remote-archive",
               "generation":"generation-001", "step":105}
    evidence = {"identity":identity.copy(), "status":"remote-publish-verified",
                "archive_sha256":"e"*64, "archive_bytes":100, "checkpoint_id":"c4",
                "checkpoint_sha256":pointer["sha256"], "archive_id":"remote-archive",
                "generation":"generation-001"}
    return tmp_path, output, registry, run, files, receipt, evidence


def call(fixture, apply=False):
    root, output, registry, run, files, receipt, evidence = fixture
    return retention.retain(output, receipt, evidence, root=root, registry=registry, apply=apply)


def test_dry_run_and_apply(fixture):
    plan = call(fixture)
    assert plan["dry_run"] and len(plan["delete"]) == 2 and plan["reclaim_bytes"] == sum(p.stat().st_size for p in fixture[4][:2])
    assert all(p.exists() for p in fixture[4])
    applied = call(fixture, apply=True)
    assert not applied["dry_run"]
    assert [p.exists() for p in fixture[4]] == [False, False, True, True]


def test_latest_protected_even_when_oldest(fixture):
    import os
    latest = fixture[4][-1]
    marker = json.loads(next(fixture[2].glob("*.json")).read_text())
    # Registered latest remains eligible but has older mtime than other checkpoints.
    os.utime(latest, ns=(marker["registered_at_ns"], marker["registered_at_ns"]))
    plan = call(fixture)
    assert latest.name in plan["keep"]
    call(fixture, apply=True)
    assert latest.exists()


@pytest.mark.parametrize("mutation", ["foreign_seed", "foreign_run", "source", "hash", "bytes", "status", "kind", "pointer"])
def test_evidence_identity_refusals(fixture, mutation):
    receipt, evidence = fixture[-2:]
    if mutation == "foreign_seed":
        receipt["identity"]["seed"] = 402
    elif mutation == "foreign_run":
        receipt["identity"]["run_id"] = "old-run"
    elif mutation == "source":
        receipt["identity"]["source_revision"] = "f"*64
    elif mutation == "hash":
        evidence["checkpoint_sha256"] = "f"*64
    elif mutation == "bytes":
        evidence["archive_bytes"] += 1
    elif mutation == "status":
        evidence["status"] = "pending"
    elif mutation == "kind":
        receipt["evidence_kind"] = "cache-readback"
    else:
        pointer_path = fixture[3] / "checkpoints/latest.json"
        pointer = json.loads(pointer_path.read_text())
        pointer["sha256"] = "f"*64
        pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError):
        call(fixture, apply=True)
    assert all(p.exists() for p in fixture[4])


@pytest.mark.parametrize("target", ["root", "checkpoint_dir", "checkpoint", "pointer"])
def test_symlink_refusals(fixture, target):
    run = fixture[3]
    if target == "root":
        moved = run.with_name("moved")
        run.rename(moved)
        run.symlink_to(moved, target_is_directory=True)
    elif target == "checkpoint_dir":
        directory = run / "checkpoints"
        moved = run / "moved"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    else:
        path = fixture[4][0] if target == "checkpoint" else run / "checkpoints/latest.json"
        moved = path.with_name("moved")
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises(ValueError):
        call(fixture, apply=True)


def test_no_enrollment_of_existing_output(fixture):
    with pytest.raises(ValueError, match="absent"):
        retention.register_new(fixture[1], root=fixture[0], registry=fixture[2])


def test_foreign_paths_refused(tmp_path):
    for output in ("reports/seed_repeat_c/seed-402/LR-HL",
                   "reports/colab_free_mini/train/LR-HL", "../old", "/absolute"):
        with pytest.raises(ValueError):
            retention.register_new(output, root=tmp_path, registry=tmp_path/"registry")


def test_predating_registration_and_active_run_refused(fixture):
    import os
    old = fixture[4][0]
    os.utime(old, ns=(1,1))
    with pytest.raises(ValueError, match="predates"):
        call(fixture, apply=True)
    assert all(p.exists() for p in fixture[4])


def test_active_run_refused(fixture):
    (fixture[3]/"execution_state.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="stopped"):
        call(fixture, apply=True)


def test_peak_estimate_accounts_overlap():
    budget = retention.estimate_peak(508461500, package_bytes=47323796)
    assert budget["local_checkpoint_bytes"] == 4067692000
    assert budget["local_peak_bytes"] == 12*508461500+1024**3+47323796
    assert budget["remote_peak_bytes"] == 9*508461500+1024**3
    with pytest.raises(ValueError):
        retention.estimate_peak(1, local_keep=1)
    with pytest.raises(ValueError):
        retention.estimate_peak(True)

def test_foreign_checkpoint_payload_refused(fixture):
    candidate = fixture[4][0]
    payload = torch.load(candidate, weights_only=False)
    payload["config"]["seed"] = 17
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match="payload identity"):
        call(fixture, apply=True)
    assert all(p.exists() for p in fixture[4])
