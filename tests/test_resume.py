import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from src.model import MoETransformer
from src.train import TrainingSession


CONFIG = {
    "config_id": "resume-toy", "family": "LR", "cell": "toy", "layers": 1,
    "d_model": 8, "heads": 2, "routed_experts": 2, "top_k": 2,
    "routed_width": 6, "rank": 2, "shared_width": 4, "vocab_size": 32,
    "context_length": 4, "seed": 11,
}
TRAINING = {
    "learning_rate": .003, "weight_decay": .01, "total_target_tokens": 8,
    "warmup_target_tokens": 2, "gradient_accumulation_steps": 1,
    "checkpoint_interval_steps": 1, "max_grad_norm": 1.,
}
BATCHES = [
    {"input_ids": [[1, 2, 3, 4]], "labels": [[2, 3, -100, -100]], "attention_mask": [[1, 1, 1, 1]]},
    {"input_ids": [[2, 3, 4, 5]], "labels": [[3, 4, -100, -100]], "attention_mask": [[1, 1, 1, 1]]},
    {"input_ids": [[3, 4, 5, 6]], "labels": [[4, 5, -100, -100]], "attention_mask": [[1, 1, 1, 1]]},
    {"input_ids": [[4, 5, 6, 7]], "labels": [[5, 6, -100, -100]], "attention_mask": [[1, 1, 1, 1]]},
]


def _identity():
    return {
        "protocol_id": "resume-test-v1", "source_revision": "fixture",
        "data_manifest_hash": hashlib.sha256(b"resume fixture").hexdigest(),
        "tokenizer_revision": "fixture-tokenizer-v1", "eval_revision": "fixture-eval-v1",
        "source_kind": "synthetic_fixture",
    }


def _session(path):
    return TrainingSession(MoETransformer(CONFIG), CONFIG, TRAINING, copy.deepcopy(BATCHES), _identity(), path)


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assert_nested_equal(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_nested_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def test_continuous_equals_stop_and_resume(tmp_path):
    _seed_all(94)
    continuous = _session(tmp_path / "continuous")
    result_full = continuous.fit()
    full_python_rng = random.getstate()
    full_numpy_rng = np.random.get_state()
    full_torch_rng = torch.get_rng_state().clone()

    _seed_all(94)
    interrupted = _session(tmp_path / "resumed")
    result_partial = interrupted.fit(max_optimizer_steps=2)
    assert result_partial["status"] == "incomplete"
    resumed = _session(tmp_path / "resumed")
    resumed.restore_latest()
    result_resumed = resumed.fit()

    assert result_full["trained_target_tokens"] == result_resumed["trained_target_tokens"] == 8
    assert result_full["optimizer_step"] == result_resumed["optimizer_step"] == 4
    for key, tensor in continuous.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[key]), key
    _assert_nested_equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict(), "optimizer")
    assert continuous.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert (continuous.epoch, continuous.cursor) == (resumed.epoch, resumed.cursor)
    assert random.getstate() == full_python_rng
    actual_numpy_rng = np.random.get_state()
    assert actual_numpy_rng[0] == full_numpy_rng[0]
    assert np.array_equal(actual_numpy_rng[1], full_numpy_rng[1])
    assert actual_numpy_rng[2:] == full_numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), full_torch_rng)


def test_resume_rejects_identity_and_missing_state(tmp_path):
    session = _session(tmp_path / "run")
    session.fit(max_optimizer_steps=1)
    pointer = json.loads((tmp_path / "run/checkpoints/latest.json").read_text())
    path = tmp_path / "run/checkpoints" / pointer["filename"]
    payload = torch.load(path, map_location="cpu", weights_only=False)

    bad_identity = _session(tmp_path / "other")
    bad_identity.identity["data_manifest_hash"] = "f" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        bad_identity._validate_checkpoint_payload(payload)

    for missing in ("optimizer_state_dict", "rng_state", "cursor"):
        broken = dict(payload)
        broken.pop(missing)
        with pytest.raises(ValueError, match="missing required state"):
            _session(tmp_path / f"missing-{missing}")._validate_checkpoint_payload(broken)

    altered = dict(payload)
    altered["cursor"] = len(BATCHES) + 1
    with pytest.raises(ValueError, match="cursor"):
        _session(tmp_path / "bad-cursor")._validate_checkpoint_payload(altered)

    # A valid-range edit is also caught by the atomic pointer's content digest.
    payload["cursor"] = min(len(BATCHES), payload["cursor"] + 1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="integrity hash"):
        _session(tmp_path / "run").restore_latest()
