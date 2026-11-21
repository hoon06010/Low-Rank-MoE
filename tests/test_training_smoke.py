import hashlib
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import src.preflight as preflight
from src.evaluate import evaluate
from src.model import MoETransformer
from src.train import METRIC_COLUMNS, TokenWarmupCosine, TrainingSession, write_metrics_csv


CONFIG = {
    "config_id": "smoke-toy", "family": "LR", "cell": "toy", "layers": 1,
    "d_model": 8, "heads": 2, "routed_experts": 2, "top_k": 2,
    "routed_width": 6, "rank": 2, "shared_width": 4, "vocab_size": 32,
    "context_length": 4, "seed": 17,
}
TRAINING = {
    "learning_rate": .002, "weight_decay": .01, "total_target_tokens": 4,
    "warmup_target_tokens": 2, "gradient_accumulation_steps": 2,
    "checkpoint_interval_steps": 1, "max_grad_norm": 1.,
}
BATCHES = [
    {"input_ids": [[1, 2, 3, 4]], "labels": [[2, 3, -100, -100]], "attention_mask": [[1, 1, 1, 1]]},
    {"input_ids": [[2, 3, 4, 5]], "labels": [[3, 4, -100, -100]], "attention_mask": [[1, 1, 1, 1]]},
]


def _identity():
    return {
        "protocol_id": "smoke-v1", "source_revision": "unavailable",
        "data_manifest_hash": hashlib.sha256(b"smoke synthetic fixture").hexdigest(),
        "tokenizer_revision": "fixture-tokenizer-v1", "eval_revision": "fixture-eval-v1",
        "source_kind": "synthetic_fixture",
    }


def test_toy_training_evaluation_schedule_and_manifest(tmp_path):
    schedule = TokenWarmupCosine(peak_lr=1., warmup_tokens=4, total_tokens=12, min_lr_ratio=.1)
    first_lr = schedule.lr_for_next_step(2)
    schedule.step(2)
    assert first_lr == pytest.approx(.5)
    assert schedule.lr_for_next_step(2) == pytest.approx(1.)

    model = MoETransformer(CONFIG)
    session = TrainingSession(model, CONFIG, TRAINING, BATCHES, _identity(), tmp_path)
    result = session.fit()
    assert result["status"] == "completed"
    assert result["trained_target_tokens"] == 4
    assert result["optimizer_step"] == 1  # two microbatches accumulated by valid-token sum
    metrics = evaluate(model, BATCHES)
    assert metrics["eval_target_tokens"] == 4
    assert metrics["loss_nats"] >= 0
    manifest = json.loads((tmp_path / "run_manifest.jsonl").read_text().splitlines()[0])
    assert manifest["source_kind"] == "synthetic_fixture"
    assert manifest["status"] == "completed"
    assert manifest["trained_target_tokens"] == 4


def test_nonfinite_attempt_keeps_last_finite_checkpoint(tmp_path):
    model = MoETransformer(CONFIG)
    session = TrainingSession(model, CONFIG, TRAINING, BATCHES, _identity(), tmp_path)
    original_forward = model.forward

    def nonfinite_forward(input_ids, attention_mask=None):
        logits = torch.full((*input_ids.shape, CONFIG["vocab_size"]), float("nan"))
        return logits, torch.tensor(0.), {}

    model.forward = nonfinite_forward
    result = session.fit()
    model.forward = original_forward
    assert result["status"] == "failed"
    assert result["reason"] == "nonfinite_loss"
    pointer = json.loads((tmp_path / "checkpoints/latest.json").read_text())
    assert pointer["checkpoint_id"] == result["checkpoint_id"]
    checkpoint = torch.load(tmp_path / "checkpoints" / pointer["filename"], map_location="cpu", weights_only=False)
    assert checkpoint["trained_target_tokens"] == 0
    assert checkpoint["last_finite_checkpoint_id"] == pointer["checkpoint_id"]
    record = json.loads((tmp_path / "run_manifest.jsonl").read_text().splitlines()[0])
    assert record["status"] == "failed"
    assert record["reason"] == "nonfinite_loss"


@pytest.mark.parametrize("failure_mode", ["post_step_nonfinite", "checkpoint_save"])
def test_post_step_or_checkpoint_failure_rolls_back_and_is_terminal(tmp_path, failure_mode):
    model = MoETransformer(CONFIG)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    session = TrainingSession(model, CONFIG, TRAINING, BATCHES, _identity(), tmp_path)

    if failure_mode == "post_step_nonfinite":
        original_step = session.optimizer.step

        def corrupt_after_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            next(session.model.parameters()).data.fill_(float("nan"))
            return result

        session.optimizer.step = corrupt_after_step
    else:
        original_save = session._save_checkpoint
        save_calls = 0

        def fail_after_initial_checkpoint():
            nonlocal save_calls
            save_calls += 1
            if save_calls == 2:
                raise OSError("injected checkpoint write failure")
            return original_save()

        session._save_checkpoint = fail_after_initial_checkpoint

    result = session.fit()
    expected_reason = "nonfinite_parameter_or_optimizer_state" if failure_mode == "post_step_nonfinite" else "checkpoint_save_failed:OSError"
    assert result["status"] == "failed" and result["reason"] == expected_reason
    pointer_before = json.loads((tmp_path / "checkpoints/latest.json").read_text())
    assert result["checkpoint_id"] == pointer_before["checkpoint_id"]
    assert result["trained_target_tokens"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, initial_state[key]), key

    manifest_path = tmp_path / "run_manifest.jsonl"
    failed_manifest = manifest_path.read_text()
    retried = session.fit()
    assert retried["status"] == "failed" and retried["reason"] == expected_reason
    assert manifest_path.read_text() == failed_manifest
    resumed = TrainingSession(MoETransformer(CONFIG), CONFIG, TRAINING, BATCHES, _identity(), tmp_path)
    resumed.restore_latest()
    retried_restored = resumed.fit()
    assert retried_restored["status"] == "failed" and retried_restored["reason"] == expected_reason
    assert manifest_path.read_text() == failed_manifest
    assert json.loads((tmp_path / "checkpoints/latest.json").read_text()) == pointer_before


def test_initial_checkpoint_failure_records_terminal_attempt_without_pointer(tmp_path):
    session = TrainingSession(MoETransformer(CONFIG), CONFIG, TRAINING, BATCHES, _identity(), tmp_path)

    def fail_initial_save():
        raise OSError("injected initial checkpoint failure")

    session._save_checkpoint = fail_initial_save
    result = session.fit()
    assert result["status"] == "failed"
    assert result["reason"] == "checkpoint_save_failed:OSError"
    assert not (tmp_path / "checkpoints/latest.json").exists()
    record = json.loads((tmp_path / "run_manifest.jsonl").read_text().splitlines()[0])
    assert record["status"] == "failed" and record["checkpoint_id"] is None
    assert session.fit()["status"] == "failed"


def test_final_checkpoint_failure_rolls_back_and_records_terminal_attempt(tmp_path):
    final_only_save_config = {**TRAINING, "checkpoint_interval_steps": 2}
    model = MoETransformer(CONFIG)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    session = TrainingSession(model, CONFIG, final_only_save_config, BATCHES, _identity(), tmp_path)
    original_save = session._save_checkpoint
    save_calls = 0

    def fail_final_save():
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise OSError("injected final checkpoint failure")
        return original_save()

    session._save_checkpoint = fail_final_save
    result = session.fit()
    assert result["status"] == "failed"
    assert result["reason"] == "checkpoint_save_failed:OSError"
    pointer = json.loads((tmp_path / "checkpoints/latest.json").read_text())
    assert pointer["checkpoint_id"] == result["checkpoint_id"]
    assert result["trained_target_tokens"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, initial_state[key]), key
    record = json.loads((tmp_path / "run_manifest.jsonl").read_text().splitlines()[0])
    assert record["status"] == "failed" and record["checkpoint_id"] == pointer["checkpoint_id"]


def test_identity_requires_explicit_source_kind(tmp_path):
    model = MoETransformer(CONFIG)
    identity = _identity()
    identity.pop("source_kind")
    with pytest.raises(KeyError):
        TrainingSession(model, CONFIG, TRAINING, BATCHES, identity, tmp_path)


@pytest.mark.parametrize("invalid_hash", ["z" * 64, "A" * 64, "a" * 63])
def test_data_manifest_hash_requires_lowercase_sha256_hex(tmp_path, invalid_hash):
    identity = _identity()
    identity["data_manifest_hash"] = invalid_hash
    with pytest.raises(ValueError, match="data_manifest_hash must be a SHA-256 hex digest"):
        TrainingSession(MoETransformer(CONFIG), CONFIG, TRAINING, BATCHES, identity, tmp_path)


def test_measured_training_requires_m9_audited_eval_provenance(tmp_path):
    identity = _identity()
    identity["source_kind"] = "measured"
    identity["eval_revision"] = "a" * 64
    with pytest.raises(ValueError, match="M9 must provide retained audited evaluation target/label provenance"):
        TrainingSession(MoETransformer(CONFIG), CONFIG, TRAINING, BATCHES, identity, tmp_path)


class _WeightedAuxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(.25))
        self.aux_scale = nn.Parameter(torch.tensor(.7))

    def forward(self, input_ids, attention_mask=None):
        class_one = F.one_hot(torch.ones_like(input_ids), num_classes=8).to(torch.float32)
        logits = class_one * self.logit_scale
        aux = self.aux_scale * input_ids[0, 0].to(torch.float32)
        return logits, aux, {}


def test_auxiliary_loss_uses_valid_target_weighted_accumulation(tmp_path):
    batches = [
        {"input_ids": [[1, 0, 0, 0]], "labels": [[1, -100, -100, -100]], "attention_mask": [[1, 0, 0, 0]]},
        {"input_ids": [[2, 3, 4, 5]], "labels": [[0, 1, 0, -100]], "attention_mask": [[1, 1, 1, 0]]},
    ]
    cfg = {"family": "LR", "cell": "weighted-aux", "seed": 3}
    training = {
        "learning_rate": .001, "total_target_tokens": 4, "warmup_target_tokens": 0,
        "gradient_accumulation_steps": 2, "router_aux_coefficient": .01,
    }
    actual_model = _WeightedAuxModel()
    reference_model = _WeightedAuxModel()
    reference_model.load_state_dict(actual_model.state_dict())
    session = TrainingSession(actual_model, cfg, training, batches, _identity(), tmp_path)

    expected_targets = 4
    for batch, valid_targets in zip(batches, (1, 3)):
        ids = torch.tensor(batch["input_ids"])
        labels = torch.tensor(batch["labels"])
        logits, aux, _ = reference_model(ids, torch.tensor(batch["attention_mask"]))
        ce_sum = F.cross_entropy(logits.reshape(-1, 8), labels.reshape(-1), ignore_index=-100, reduction="sum")
        objective = ce_sum / expected_targets + .01 * aux * (valid_targets / expected_targets)
        objective.backward()
    expected_grads = {name: parameter.grad.detach().clone() for name, parameter in reference_model.named_parameters()}

    observed_grads = {}
    original_step = session.optimizer.step

    def capture_gradients(*args, **kwargs):
        observed_grads.update({name: parameter.grad.detach().clone() for name, parameter in session.model.named_parameters()})
        return original_step(*args, **kwargs)

    session.optimizer.step = capture_gradients
    result = session.fit()
    assert result["status"] == "completed" and result["trained_target_tokens"] == 4
    assert set(observed_grads) == set(expected_grads)
    for name in expected_grads:
        assert torch.allclose(observed_grads[name], expected_grads[name], atol=1e-7, rtol=1e-6), name
    assert observed_grads["aux_scale"].item() == pytest.approx(.01 * 1.75)


def test_failed_preflight_does_not_write_completed_metric(tmp_path, monkeypatch):
    from pathlib import Path

    toy = json.loads(Path("configs/toy_cpu.json").read_text())
    # Force the training objective to encounter NaN while keeping preflight data valid.
    def fail_fit(self):
        self.latest_verified_checkpoint_id = "last-finite"
        self.failed_reason = "nonfinite_loss"
        self._write_run_manifest("failed", "nonfinite_loss")
        return self.result("failed")

    monkeypatch.setattr(TrainingSession, "fit", fail_fit)
    config_path = tmp_path / "toy.json"
    config_path.write_text(json.dumps(toy))
    report = preflight.run_preflight(config_path, tmp_path / "out")
    assert report["status"] == "failed"
    assert report["reason"] == "nonfinite_loss"
    assert report["eval_target_tokens"] is None
    assert not (tmp_path / "out/metrics.csv").exists()
    saved = json.loads((tmp_path / "out/preflight.json").read_text())
    assert saved["status"] == "failed" and saved["loss_nats"] is None


def test_metrics_csv_rejects_duplicate_key_and_invalid_header(tmp_path):
    row = {
        "run_id": "run-1", "protocol_id": "p", "config_id": "c", "family": "LR", "cell": "HS",
        "seed": 1, "split": "dev", "checkpoint_id": "ckpt-1", "trained_target_tokens": 10,
        "eval_target_tokens": 5, "loss_nats": 1.0, "status": "completed", "source_kind": "synthetic_fixture",
    }
    path = tmp_path / "metrics.csv"
    write_metrics_csv(path, row)
    assert path.read_text().splitlines()[0].split(",") == METRIC_COLUMNS
    with pytest.raises(ValueError, match="duplicate metrics primary key"):
        write_metrics_csv(path, row)

    broken = tmp_path / "broken.csv"
    broken.write_text("run_id,wrong_header\n")
    with pytest.raises(ValueError, match="header"):
        write_metrics_csv(broken, row)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("eval_target_tokens", 0, "positive eval_target_tokens"),
    ("eval_target_tokens", -1, "nonnegative"),
    ("loss_nats", float("nan"), "finite and nonnegative"),
    ("loss_nats", float("inf"), "finite and nonnegative"),
    ("loss_nats", -0.01, "finite and nonnegative"),
    ("trained_target_tokens", 0, "positive trained_target_tokens"),
    ("trained_target_tokens", "1.5", "must be an integer"),
    ("seed", -1, "must be nonnegative"),
    ("seed", 1.5, "must be an integer"),
    ("status", "success", "metrics status"),
    ("source_kind", "fixture", "metrics source_kind"),
    ("split", "validation", "metrics split"),
    ("family", "OTHER", "metrics family"),
])
def test_metrics_csv_rejects_contract_invalid_completed_rows(tmp_path, field, value, message):
    row = {
        "run_id": "run-invalid", "protocol_id": "p", "config_id": "c", "family": "LR", "cell": "HS",
        "seed": 1, "split": "dev", "checkpoint_id": "ckpt-1", "trained_target_tokens": 10,
        "eval_target_tokens": 5, "loss_nats": 1.0, "status": "completed", "source_kind": "measured",
    }
    row[field] = value
    with pytest.raises(ValueError, match=message):
        write_metrics_csv(tmp_path / "invalid.csv", row)
    assert not (tmp_path / "invalid.csv").exists()


def test_noncompleted_metric_may_omit_eval_loss_but_enum_and_counts_still_validate(tmp_path):
    row = {
        "run_id": "run-failed", "protocol_id": "p", "config_id": "c", "family": "LR", "cell": "HS",
        "seed": 1, "split": "dev", "checkpoint_id": "last-finite", "trained_target_tokens": 0,
        "eval_target_tokens": 0, "loss_nats": "", "status": "failed", "source_kind": "synthetic_fixture",
    }
    write_metrics_csv(tmp_path / "failed.csv", row)

@pytest.mark.parametrize("where", ["initial", "periodic", "final", "diagnostic"])
def test_deadline_timeout_escapes_training_cleanup(tmp_path, monkeypatch, where):
    import src.train as train_module
    config = {**TRAINING, "checkpoint_interval_steps": 2 if where=="final" else 1}
    session = TrainingSession(MoETransformer(CONFIG), CONFIG, config, BATCHES, _identity(), tmp_path)
    original_save=session._save_checkpoint
    calls=0
    def save():
        nonlocal calls
        calls+=1
        if calls==(1 if where=="initial" else 2) and where!="diagnostic":
            raise TimeoutError("operation deadline")
        return original_save()
    monkeypatch.setattr(session,"_save_checkpoint",save)
    original_json=train_module._atomic_json
    def write_json(path,value):
        if where=="diagnostic" and "diagnostics" in path.parts:
            raise TimeoutError("operation deadline")
        return original_json(path,value)
    monkeypatch.setattr(train_module,"_atomic_json",write_json)
    with pytest.raises(TimeoutError,match="deadline"):
        session.fit()
