"""Bounded controller gates without launching GPU training."""

import json

import pytest

from scripts import overnight_colab as controller


def test_b_budget_uses_measured_snapshot_cost():
    rehearsal = {"snapshot_seconds_max": 120, "bootstrap_seconds": 60}
    with pytest.raises(ValueError, match="exceeds declared"):
        controller._budget_gate("B", 9000, 9000, rehearsal)
    controller._budget_gate("B", 12000, 12000, rehearsal)


def test_protocol_is_frozen_after_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "_identity", lambda stage, cell, ready: {"cell": cell})
    (tmp_path / "overnight_rehearsal.json").write_text(json.dumps({"status": "verified"}))
    first = controller._protocol("B", tmp_path, {}, 14400, 14400, {}, create=True)
    assert controller._protocol("B", tmp_path, {}, 14400, 14400, {}, create=False) == first
    with pytest.raises(ValueError, match="frozen"):
        controller._protocol("B", tmp_path, {}, 15000, 14400, {}, create=False)


def test_restored_run_is_not_restored_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "_identity", lambda *args: {"fixed": True})
    run = tmp_path / "runs/overnight/B/HL"
    run.mkdir(parents=True)
    (run / "restored_marker").write_text("verified")
    monkeypatch.setattr(controller, "restore_snapshot", lambda *args: (_ for _ in ()).throw(
        AssertionError("duplicate restore")))
    monkeypatch.setattr(controller, "run_colab_mini", lambda *args, **kwargs: {
        "status": "incomplete", "optimizer_step": 32})
    report = controller._run_one("B", "HL", tmp_path / "drive", {}, max_steps=None,
                                 started=0, total_seconds=10000, session_seconds=10000,
                                 snapshot_times=[], resume=True, restored=True)
    assert report["optimizer_step"] == 32

def test_measured_space_gate(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(controller.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    r = {"snapshot_bytes_max": 500_000_000}
    required = 11 * 550_000_000 + 1024**3
    with pytest.raises(ValueError, match="Drive"):
        controller._drive_space_gate(required-1, r)
    assert controller._drive_space_gate(required, r)["required_peak_bytes"] == required
    with pytest.raises(ValueError):
        controller._drive_space_gate(None, r)
    monkeypatch.setattr(controller.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="local"):
        controller._drive_space_gate(10**12, r)

def test_crash_time_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "_runtime_id", lambda: "runtime")
    controller._write_progress(tmp_path, "B", 100, reserved_until=1000)
    assert controller._resume_elapsed(tmp_path, "B") == 1000
    controller._write_progress(tmp_path, "B", 120, reserved_until=120)
    controller._write_progress(tmp_path, "B", 50)
    assert controller._resume_elapsed(tmp_path, "B") == 120
    (tmp_path/"B/overnight_summary.json").write_text(json.dumps({"cumulative_elapsed_seconds":200}))
    assert controller._resume_elapsed(tmp_path, "B") == 200

@pytest.mark.parametrize("rehearsal", [False, True])
def test_after_snapshot_time_and_rehearsal_scope(tmp_path, monkeypatch, rehearsal):
    from types import SimpleNamespace
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "_identity", lambda *args: {"fixed": True})
    monkeypatch.setattr(controller, "_runtime_id", lambda: "runtime")
    monkeypatch.setattr(controller, "_trim_local", lambda *args: None)
    monkeypatch.setattr(controller.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    now = [100.0]
    monkeypatch.setattr(controller.time, "monotonic", lambda: now[0])
    scope = "rehearsal" if rehearsal else "B"
    run = tmp_path / "runs/overnight" / scope / "HL"
    def runner(*args, boundary_hook, **kwargs):
        (run/"checkpoints").mkdir(parents=True)
        (run/"checkpoints/one.pt").write_bytes(b"state")
        (run/"checkpoints/latest.json").write_text(json.dumps({"filename":"one.pt"}))
        boundary_hook(SimpleNamespace(optimizer_step=0,trained_target_tokens=0,
            training_config={"total_target_tokens":3145728}), "initial")
        return {"status":"incomplete","optimizer_step":0}
    def snapshot(*args, **kwargs):
        now[0] += 60
    monkeypatch.setattr(controller, "run_colab_mini", runner)
    monkeypatch.setattr(controller, "publish_snapshot", snapshot)
    drive=tmp_path/"drive"
    controller._run_one("B","HL",drive,{},max_steps=None,started=1,total_seconds=10000,
        session_seconds=5000,snapshot_times=[],rehearsal=rehearsal)
    assert json.loads((drive/scope/"progress.json").read_text())["cumulative_elapsed_seconds"]==159
    if rehearsal:
        assert not (drive/"B/progress.json").exists()
