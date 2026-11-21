"""Exact state comparison for trusted, SHA-verified seed401 trainer checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import re

FIELDS = (
    "schema_version", "execution_device", "checkpoint_id", "latest_verified_checkpoint_id",
    "last_finite_checkpoint_id", "run_id", "config", "canonical_config_json", "config_hash",
    "training_config", "training_config_hash", "source_revision", "data_manifest_hash",
    "tokenizer_revision", "eval_revision", "protocol_id", "source_kind", "purpose",
    "model_state_dict", "optimizer_state_dict", "last_step_diagnostics", "scheduler_state_dict",
    "scaler_state", "rng_state", "data_order", "data_order_hash", "epoch", "cursor",
    "trained_target_tokens", "optimizer_step",
)
UUID_FIELDS = frozenset(("checkpoint_id", "latest_verified_checkpoint_id", "last_finite_checkpoint_id"))
HEX = re.compile(r"[0-9a-f]{64}")
SAFE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


class StateMismatch(ValueError):
    """Contains a field path only; never tensor/scalar/token contents."""


def mismatch(path):
    raise StateMismatch("checkpoint state mismatch at " + path)


def key_path(path, key, index):
    if isinstance(key, str) and SAFE_KEY.fullmatch(key):
        return path + "." + key
    if type(key) is int:
        return path + "[integer-key]"
    return path + "[key-index-" + str(index) + "]"


def equal_exact(left, right, path="state"):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            mismatch(path)
        if (left.dtype != right.dtype or left.layout != right.layout or left.shape != right.shape
                or not torch.equal(left.cpu(), right.cpu())):
            mismatch(path)
        return
    if isinstance(left, np.ndarray):
        if (not isinstance(right, np.ndarray) or left.dtype != right.dtype or left.shape != right.shape
                or not np.array_equal(left, right)):
            mismatch(path)
        return
    if type(left) is not type(right):
        mismatch(path)
    if isinstance(left, dict):
        if left.keys() != right.keys():
            mismatch(path + ".keys")
        for i, key in enumerate(left):
            equal_exact(left[key], right[key], key_path(path, key, i))
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            mismatch(path + ".length")
        for i, (a, b) in enumerate(zip(left, right)):
            equal_exact(a, b, path + "[" + str(i) + "]")
    elif isinstance(left, (str, bytes, int, float, bool, type(None), np.generic)):
        if left != right:
            mismatch(path)
    else:
        mismatch(path + ".unsupported_type")


def validate(payload, expected, path):
    import torch
    if not isinstance(payload, dict) or set(payload) != set(FIELDS):
        mismatch(path + ".required_fields")
    if (expected.get("seed") != 401 or expected.get("cell") not in ("HL", "HS", "LS", "LL")
            or not isinstance(expected.get("run_id"), str) or not expected["run_id"]):
        mismatch("expected.identity")
    if not isinstance(payload["config"], dict):
        mismatch(path + ".config")
    for key in ("seed", "cell"):
        if payload["config"].get(key) != expected[key]:
            mismatch(path + ".config." + key)
    for key in ("run_id", "source_revision", "data_manifest_hash", "eval_revision"):
        if key != "run_id" and not HEX.fullmatch(str(expected.get(key, ""))):
            mismatch("expected." + key)
        if payload[key] != expected.get(key):
            mismatch(path + "." + key)
    if payload["schema_version"] != 1 or payload["execution_device"] not in ("cpu", "cuda"):
        mismatch(path + ".schema_or_device")
    if (not isinstance(payload["checkpoint_id"], str) or not payload["checkpoint_id"]
            or any(payload[k] != payload["checkpoint_id"] for k in UUID_FIELDS)):
        mismatch(path + ".checkpoint_id")
    if payload["scaler_state"] is not None:
        mismatch(path + ".scaler_state")
    for field in ("epoch", "cursor", "optimizer_step", "trained_target_tokens"):
        if type(payload[field]) is not int or payload[field] < 0:
            mismatch(path + "." + field)
    if (payload["epoch"] != 0 or payload["cursor"] != payload["optimizer_step"] * 8
            or payload["trained_target_tokens"] != payload["optimizer_step"] * 4096):
        mismatch(path + ".counters")
    scheduler = payload["scheduler_state_dict"]
    if not isinstance(scheduler, dict) or scheduler.get("tokens_seen") != payload["trained_target_tokens"]:
        mismatch(path + ".scheduler_state_dict")
    optimizer = payload["optimizer_state_dict"]
    if not isinstance(optimizer, dict) or not {"state", "param_groups"}.issubset(optimizer):
        mismatch(path + ".optimizer_state_dict.required_fields")
    for state in optimizer["state"].values():
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            mismatch(path + ".optimizer_state_dict.state.required_fields")
    rng = payload["rng_state"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "used_device_rng_states"}:
        mismatch(path + ".rng_state.required_fields")
    device_rng = rng["used_device_rng_states"]
    required_rng = {"cpu", "cuda"} if payload["execution_device"] == "cuda" else {"cpu"}
    if not isinstance(device_rng, dict) or set(device_rng) != required_rng:
        mismatch(path + ".rng_state.used_device_rng_states.required_fields")
    if not isinstance(rng["torch_cpu"], torch.Tensor) or any(
            not isinstance(device_rng[k], torch.Tensor) for k in required_rng):
        mismatch(path + ".rng_state.tensor")
    data_order = payload["data_order"]
    if not isinstance(data_order, list):
        mismatch(path + ".data_order")
    actual_order_hash = hashlib.sha256(json.dumps(data_order, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if payload["data_order_hash"] != actual_order_hash:
        mismatch(path + ".data_order_hash")


def verified_load(path, expected_sha256):
    import torch
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not HEX.fullmatch(str(expected_sha256)):
        mismatch("input.file_or_hash")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            mismatch("input.sha256")
        stream.seek(0)
        # Trusted local trainer artifacts only. Hashes do not make untrusted pickle safe.
        return torch.load(stream, map_location="cpu", weights_only=False)


def compare_payloads(left, right, expected, *, mode="restored"):
    if mode not in ("restored", "continuation"):
        mismatch("mode")
    validate(left, expected, "left")
    validate(right, expected, "right")
    excluded = UUID_FIELDS if mode == "continuation" else frozenset()
    for field in FIELDS:
        if field not in excluded:
            equal_exact(left[field], right[field], "state." + field)
    return {"status": "exact-state-verified", "mode": mode,
            "checked_fields": [field for field in FIELDS if field not in excluded],
            "excluded_fields": sorted(excluded), "seed": 401, "cell": expected["cell"],
            "optimizer_step": left["optimizer_step"],
            "scope": "trusted local checkpoint payloads; runtime restoration/remote durability require separate evidence"}


def compare_files(left, left_sha256, right, right_sha256, expected, *, mode="restored"):
    return compare_payloads(verified_load(left, left_sha256), verified_load(right, right_sha256),
                            expected, mode=mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-sha256", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-sha256", required=True)
    parser.add_argument("--expected-identity", type=Path, required=True)
    parser.add_argument("--mode", choices=("restored", "continuation"), default="restored")
    args = parser.parse_args()
    try:
        expected = json.loads(args.expected_identity.read_text())
        result = compare_files(args.left, args.left_sha256, args.right, args.right_sha256,
                               expected, mode=args.mode)
    except StateMismatch as exc:
        parser.exit(1, str(exc) + "\n")
    except Exception:
        # Never echo parsing, pickle or tensor exceptions containing input values.
        parser.exit(1, "checkpoint comparison input invalid; no state contents emitted\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
