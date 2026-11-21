"""Integrity and recovery tests for overnight Colab snapshots."""

import json
from pathlib import Path

import pytest

from scripts import overnight_store
from scripts.overnight_store import publish_snapshot, restore_snapshot


def _run(path: Path, step: int = 32):
    checkpoint = path / "checkpoints"
    checkpoint.mkdir(parents=True)
    payload = b"checkpoint state at step " + str(step).encode()
    (checkpoint / "checkpoint-a.pt").write_bytes(payload)
    import hashlib
    (checkpoint / "latest.json").write_text(json.dumps({
        "filename": "checkpoint-a.pt", "checkpoint_id": "a",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }))
    (path / "run_manifest.jsonl").write_text(json.dumps({
        "run_id": "run-a", "checkpoint_id": "a", "checkpoint_step": step,
        "trained_target_tokens": step * 4096, "status": "incomplete",
    }) + "\n")
    (path / "metrics.csv").write_text("trained_target_tokens,loss_nats\n131072,4.0\n262144,3.0\n")
    (path / "routing_diagnostics.jsonl").write_text(
        json.dumps({"trained_target_tokens": 131072}) + "\n" +
        json.dumps({"trained_target_tokens": 262144}) + "\n"
    )


def test_snapshot_round_trip_discards_records_after_checkpoint(tmp_path):
    run, drive = tmp_path / "run", tmp_path / "drive"
    _run(run)
    identity = {"source": "abc", "config": "def", "data": "ghi"}
    publish_snapshot(run, drive, identity, step=32, keep_generations=2)
    restored = tmp_path / "new" / "nested" / "restored"
    state = restore_snapshot(drive, restored, identity)
    assert state["step"] == 32
    assert (restored / "checkpoints/checkpoint-a.pt").read_bytes().startswith(b"checkpoint state")
    assert (restored / "metrics.csv").read_text().count("\n") == 2
    assert len((restored / "routing_diagnostics.jsonl").read_text().splitlines()) == 1


def test_snapshot_rejects_corrupted_archive_without_creating_output(tmp_path):
    run, drive = tmp_path / "run", tmp_path / "drive"
    _run(run)
    identity = {"source": "abc"}
    record = publish_snapshot(run, drive, identity, step=32, keep_generations=2)
    (drive / record["archive"]).write_bytes(b"partial upload")
    with pytest.raises(ValueError, match="integrity"):
        restore_snapshot(drive, tmp_path / "restored", identity)
    assert not (tmp_path / "restored").exists()


def test_snapshot_rejects_identity_mismatch(tmp_path):
    run, drive = tmp_path / "run", tmp_path / "drive"
    _run(run)
    publish_snapshot(run, drive, {"source": "abc"}, step=32, keep_generations=2)
    with pytest.raises(ValueError, match="identity"):
        restore_snapshot(drive, tmp_path / "restored", {"source": "different"})


def test_snapshot_falls_back_to_prior_verified_generation(tmp_path):
    run, drive = tmp_path / "run", tmp_path / "drive"
    _run(run)
    identity = {"source": "abc"}
    first = publish_snapshot(run, drive, identity, step=32, keep_generations=2)
    second = publish_snapshot(run, drive, identity, step=32, keep_generations=2)
    (drive / second["archive"]).write_bytes(b"interrupted copy")
    restored = restore_snapshot(drive, tmp_path / "restored", identity)
    assert restored["generation"] == first["generation"]


def test_snapshot_rejects_uncommitted_pointer(tmp_path):
    run, drive = tmp_path / "run", tmp_path / "drive"
    _run(run)
    publish_snapshot(run, drive, {"source": "abc"}, step=32, keep_generations=2)
    (drive / "latest.json").write_text('{"generation":"missing"}')
    with pytest.raises(ValueError, match="pointer mismatch"):
        restore_snapshot(drive, tmp_path / "restored", {"source": "abc"})


def test_interrupted_copy_does_not_publish_new_generation(tmp_path, monkeypatch):
    run, drive = tmp_path / "run", tmp_path / "drive"
    _run(run)
    identity = {"source": "abc"}
    original = publish_snapshot(run, drive, identity, step=32, keep_generations=2)

    def interrupted(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated Drive quota or transfer failure")

    monkeypatch.setattr(overnight_store.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="simulated"):
        publish_snapshot(run, drive, identity, step=32, keep_generations=2)
    assert json.loads((drive / "latest.json").read_text())["generation"] == original["generation"]
    assert restore_snapshot(drive, tmp_path / "restored", identity)["generation"] == original["generation"]

def test_snapshot_operation_deadline(tmp_path, monkeypatch):
    import time
    run, drive = tmp_path / "run", tmp_path / "drive"
    _run(run)
    first = publish_snapshot(run, drive, {"source": "abc"}, step=32)
    monkeypatch.setattr(overnight_store.shutil, "copyfile", lambda *args: time.sleep(2))
    with pytest.raises(TimeoutError):
        publish_snapshot(run, drive, {"source": "abc"}, step=32, timeout_seconds=.05)
    assert json.loads((drive / "latest.json").read_text())["generation"] == first["generation"]
    assert restore_snapshot(drive, tmp_path / "restored", {"source": "abc"})["step"] == 32

def test_nested_operation_deadline():
    import time
    with pytest.raises(TimeoutError):
        with overnight_store.deadline(.03):
            with overnight_store.deadline(2):
                time.sleep(.2)
