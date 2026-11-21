"""Bounded interface and provenance tests for the mini Colab path."""
import copy
import json

import pytest

from scripts import run_colab_mini as runner
from src.data import load_audited_bundle
from src.model import MoETransformer
from src.train import TrainingSession, _json_hash
from test_real_data import bundle
from test_training_smoke import CONFIG, TRAINING, _identity


def _mini_rows(count=8):
    return [
        {"input_ids": [[1] * 512], "labels": [[2] * 512], "attention_mask": [[1] * 512],
         "valid_target_count": 512, "split": "train"}
        for _ in range(count)
    ]


def _runner_config(path, budget=4096):
    model = {"config_id":"LR-HS","family":"LR","cell":"HS","layers":6,"d_model":384,
             "heads":6,"routed_experts":8,"top_k":2,"routed_width":768,"rank":64,
             "shared_width":256,"shared_experts":2,"vocab_size":50257,"context_length":512,"seed":17}
    training = {"learning_rate":3e-4,"weight_decay":.1,"betas":[.9,.95],"eps":1e-8,
                "total_target_tokens":budget,"warmup_target_tokens":int(budget*.05),"min_lr_ratio":.1,
                "gradient_accumulation_steps":8,"max_grad_norm":1.,"router_aux_coefficient":.01,
                "checkpoint_interval_steps":32,"recipe_id":"colab-mini-adamw-token-cosine-v1"}
    folder = path.parent / "configs"
    folder.mkdir(exist_ok=True)
    for config_id, (rank, shared_width, shared_experts) in runner.GRID_CELLS.items():
        cell_model = dict(model, config_id=config_id, cell=config_id.split("-")[1], rank=rank,
                          shared_width=shared_width, shared_experts=shared_experts)
        item = {"protocol_id":"colab-free-mini-v2","model":cell_model,"training":training,
                "global_target_batch":4096,"eval_interval_tokens":1048576}
        (folder / f"{config_id}.json").write_text(json.dumps(item))
    return folder / "LR-HS.json"


def test_mini_grid_uses_fixed_width_shared_expert_counts(tmp_path):
    path = _runner_config(tmp_path / "config.json")
    selected = runner._config(path)
    runner._validate_grid(path, selected)
    for name, (rank, width, count) in runner.GRID_CELLS.items():
        model_config = json.loads((path.parent / f"{name}.json").read_text())["model"]
        assert (model_config["rank"], model_config["shared_width"], model_config["shared_experts"]) == (rank, width, count)
    bad = json.loads(path.read_text())
    bad["model"]["shared_experts"] = 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="fixed mini grid"):
        runner._validate_grid(path, runner._config(path))


def test_audited_loader_accepts_only_the_two_documented_purposes(tmp_path):
    path, spec, _ = bundle(tmp_path)
    spec["purpose"] = "mini_exploration_only"
    path.write_text(json.dumps(spec))
    assert load_audited_bundle(path)["spec"]["purpose"] == "mini_exploration_only"
    spec["purpose"] = "confirmatory_research"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unsupported bundle"):
        load_audited_bundle(path)


def test_measured_session_propagates_mini_purpose_and_checkpoint_identity(tmp_path):
    path, spec, packed = bundle(tmp_path)
    spec["purpose"] = "mini_exploration_only"
    path.write_text(json.dumps(spec))
    audited = load_audited_bundle(path)
    identity = _identity()
    identity.update(source_kind="measured", data_manifest_hash=audited["bundle_hash"],
                    eval_revision=_json_hash(packed["dev"]),
                    tokenizer_revision=spec["manifest"]["tokenizer"]["revision"]["value"])
    session = TrainingSession(MoETransformer(CONFIG), CONFIG, TRAINING, packed["train"], identity,
                              tmp_path / "mini-run", audit_bundle=path)
    assert session.purpose == "mini_exploration_only"
    assert session._checkpoint_payload("fixture")["purpose"] == "mini_exploration_only"


def test_runner_rejects_smoke_bundle_and_inexact_training_rows(tmp_path, monkeypatch):
    config = _runner_config(tmp_path / "config.json")
    monkeypatch.setattr(runner, "load_audited_bundle", lambda _: {"spec":{"purpose":"pipeline_smoke_only"}})
    with pytest.raises(ValueError, match="bundle purpose"):
        runner.run_colab_mini(config, tmp_path / "bundle", tmp_path / "run", device="cpu", max_steps=1)

    invalid_rows = _mini_rows()
    invalid_rows[0]["labels"] = [[2] * 511]
    fake = {"spec":{"purpose":"mini_exploration_only","vocab_size":50257,"context_length":512},
            "bundle_hash":"a"*64,"batches":{"train":invalid_rows,"dev":[]}}
    monkeypatch.setattr(runner, "load_audited_bundle", lambda _: fake)
    with pytest.raises(ValueError, match="exactly 512"):
        runner.run_colab_mini(config, tmp_path / "bundle", tmp_path / "run", device="cpu", max_steps=1)


def test_runner_rejects_insufficient_unique_targets_before_training(tmp_path, monkeypatch):
    config = _runner_config(tmp_path / "config.json")
    fake = {"spec":{"purpose":"mini_exploration_only","vocab_size":50257,"context_length":512},
            "bundle_hash":"a"*64,"batches":{"train":_mini_rows(7),"dev":[]}}
    monkeypatch.setattr(runner, "load_audited_bundle", lambda _: fake)
    with pytest.raises(ValueError, match="cycling is forbidden"):
        runner.run_colab_mini(config, tmp_path / "bundle", tmp_path / "run", device="cpu", max_steps=1)


def test_routing_diagnostics_append_once_per_checkpoint(tmp_path):
    path = tmp_path / "routing_diagnostics.jsonl"
    stats = {"valid_token_count": 512, "selected_count": [128] * 4}
    for checkpoint, tokens in (("checkpoint-a", 4096), ("checkpoint-b", 8192), ("checkpoint-a", 4096)):
        runner.append_routing_diagnostic(
            path,
            {"run_id": "run-1", "checkpoint_id": checkpoint, "trained_target_tokens": tokens},
            stats,
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert [(row["checkpoint_id"], row["trained_target_tokens"]) for row in rows] == [
        ("checkpoint-a", 4096), ("checkpoint-b", 8192)
    ]
    assert all(row["scope"] == "first_dev_batch_only" and row["stats"] == stats for row in rows)


def test_overnight_hook_saves_initial_state_and_can_stop_before_training(tmp_path, monkeypatch):
    import torch
    config = {"protocol_id": "colab-free-mini-stability-v1",
              "model": {"config_id": "LR-HL", "seed": 17, "vocab_size": 50257, "context_length": 512},
              "training": {"total_target_tokens": 8192}}
    fake_bundle = {"spec": {"purpose": runner.PURPOSE, "vocab_size": 50257,
                             "context_length": 512, "manifest": {"tokenizer": {"revision": {"value": "fixed"}}}},
                   "bundle_hash": "a" * 64, "batches": {"train": _mini_rows(16), "dev": []}}
    monkeypatch.setattr(runner, "_config", lambda _: config)
    monkeypatch.setattr(runner, "_validate_grid", lambda *_: None)
    monkeypatch.setattr(runner, "load_audited_bundle", lambda _: fake_bundle)
    monkeypatch.setattr(runner, "MoETransformer", lambda _: torch.nn.Linear(1, 1))
    events = []

    class FakeSession:
        def __init__(self, model, model_config, training, batches, identity, output, **kwargs):
            self.model, self.optimizer_step, self.trained_target_tokens = model, 0, 0
            self.checkpoint_seconds, self.latest_verified_checkpoint_id = 0, None
            self.training_config, self.last_step_diagnostics = training, {}

        def _save_checkpoint(self):
            events.append("initial_saved")
            self.latest_verified_checkpoint_id = "initial"

        def _write_run_manifest(self, status, reason):
            events.append("manifest_written")

        def fit(self, **kwargs):
            raise AssertionError("training must not start after boundary stop")

        def result(self, status):
            return {"status": status, "run_id": "fixture", "checkpoint_id": "initial",
                    "optimizer_step": 0, "trained_target_tokens": 0}

    monkeypatch.setattr(runner, "TrainingSession", FakeSession)

    def hook(session, phase):
        events.append(phase)
        return phase != "before_chunk"

    report = runner.run_colab_mini(tmp_path / "config.json", tmp_path / "bundle.json", tmp_path / "run",
                                   device="cpu", max_steps=1, boundary_hook=hook)
    assert report["status"] == "incomplete"
    assert events == ["initial_saved", "manifest_written", "initial", "before_chunk", "after_report"]
