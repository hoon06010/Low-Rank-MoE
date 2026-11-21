"""Algebraic parameter accounting for the Low-Rank-MoE pilot grid."""

from __future__ import annotations

import hashlib
import json
from typing import Any


BASE_CONFIG: dict[str, Any] = {
    "layers": 6,
    "d_model": 384,
    "heads": 6,
    "routed_experts": 8,
    "top_k": 2,
    "routed_width": 768,
    "vocab_size": 50257,
    "context_length": 512,
    "seed": 17,
}

CELL_SHAPES = {
    "HS": (512, 64),
    "HL": (1280, 64),
    "LS": (512, 32),
    "LL": (1280, 32),
    "M": (896, 48),
}


def canonical_config_json(config: dict[str, Any]) -> str:
    """Serialize config deterministically, excluding its own hash field."""
    identity = {key: value for key, value in config.items() if key != "config_hash"}
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    """Reject malformed architecture dimensions before doing any arithmetic."""
    required = (
        "family", "cell", "layers", "d_model", "heads", "routed_experts",
        "top_k", "routed_width", "rank", "shared_width", "vocab_size",
        "context_length", "seed",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    if config["family"] not in {"LR", "DW", "SHARED_ONLY"}:
        raise ValueError("family must be LR, DW, or SHARED_ONLY")
    if type(config.get("shared_experts", 1)) is not int or config.get("shared_experts", 1) <= 0:
        raise ValueError("shared_experts must be a positive integer")

    for key in ("layers", "d_model", "heads", "vocab_size", "context_length"):
        value = config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["d_model"] % config["heads"]:
        raise ValueError("d_model must be divisible by heads")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")

    family = config["family"]
    route_fields = ("routed_experts", "top_k", "routed_width", "rank")
    if family == "SHARED_ONLY":
        if any(type(config[key]) is not int or config[key] != 0 for key in route_fields):
            raise ValueError("SHARED_ONLY requires routed_experts, top_k, routed_width, and rank to be zero")
        if type(config["shared_width"]) is not int or config["shared_width"] <= 0:
            raise ValueError("shared_width must be positive for SHARED_ONLY")
        return

    for key in route_fields + ("shared_width",):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer for routed conditions")
    if config["top_k"] != 2:
        raise ValueError("the pilot routed grid requires top_k=2")
    if config["top_k"] > config["routed_experts"]:
        raise ValueError("top_k cannot exceed routed_experts")
    if family == "LR":
        if config["rank"] > min(config["d_model"], config["routed_width"]):
            raise ValueError("LR rank exceeds a factorized matrix dimension")
    elif config["routed_width"] != 3 * config["rank"]:
        raise ValueError("DW routed_width must equal 3 * rank budget label")


def parameter_counts(config: dict[str, Any]) -> dict[str, Any]:
    """Return algebraic whole-model and per-layer total/active counts.

    Active expert parameters include one shared SwiGLU and the top-k selected
    routed experts. Shared parameters and all non-FFN parameters are active.
    Tied input/output embeddings are counted once.
    """
    validate_config(config)
    layers = config["layers"]
    d_model = config["d_model"]
    shared = 3 * d_model * config["shared_width"] * config.get("shared_experts", 1)
    embedding = config["vocab_size"] * d_model
    attention_per_layer = 4 * d_model * d_model
    norms_per_layer = 2 * d_model
    final_norm = d_model

    if config["family"] == "SHARED_ONLY":
        routed_total_per_layer = routed_active_per_layer = router_per_layer = 0
    elif config["family"] == "LR":
        # Three factorized SwiGLU matrices: gate, up, and down.
        routed_per_expert = 3 * config["rank"] * (d_model + config["routed_width"])
        routed_total_per_layer = config["routed_experts"] * routed_per_expert
        routed_active_per_layer = config["top_k"] * routed_per_expert
        router_per_layer = d_model * config["routed_experts"]
    else:
        # Bias-free dense SwiGLU: gate, up, and down projections.
        routed_per_expert = 3 * d_model * config["routed_width"]
        routed_total_per_layer = config["routed_experts"] * routed_per_expert
        routed_active_per_layer = config["top_k"] * routed_per_expert
        router_per_layer = d_model * config["routed_experts"]

    expert_total_per_layer = shared + routed_total_per_layer
    expert_active_per_layer = shared + routed_active_per_layer
    non_expert_per_layer = attention_per_layer + norms_per_layer + router_per_layer
    total_per_layer = non_expert_per_layer + expert_total_per_layer
    active_per_layer = non_expert_per_layer + expert_active_per_layer
    base_total = embedding + layers * (attention_per_layer + norms_per_layer + router_per_layer) + final_norm
    whole_total = base_total + layers * expert_total_per_layer
    whole_active = base_total + layers * expert_active_per_layer

    return {
        "whole_model_total": whole_total,
        "whole_model_active": whole_active,
        "per_layer_total": total_per_layer,
        "per_layer_active": active_per_layer,
        "components": {
            "tied_embedding": embedding,
            "attention_per_layer": attention_per_layer,
            "rmsnorm_per_layer": norms_per_layer,
            "final_rmsnorm": final_norm,
            "router_per_layer": router_per_layer,
            "shared_swiglu_per_layer": shared,
            "routed_experts_total_per_layer": routed_total_per_layer,
            "routed_experts_active_per_layer": routed_active_per_layer,
            "expert_total_per_layer": expert_total_per_layer,
            "expert_active_per_layer": expert_active_per_layer,
        },
    }


def _condition(family: str, cell: str, shared_width: int, rank: int) -> dict[str, Any]:
    config = dict(BASE_CONFIG)
    config.update(
        config_id=f"{family}-{cell}", family=family, cell=cell,
        rank=rank, shared_width=shared_width,
    )
    if family == "DW":
        config["routed_width"] = 3 * rank
    config["config_hash"] = config_hash(config)
    validate_config(config)
    return config


def build_grid() -> list[dict[str, Any]]:
    """Build LR/DW HS/HL/LS/LL/M and one shared-only condition."""
    conditions = [
        _condition(family, cell, *CELL_SHAPES[cell])
        for cell in ("HS", "HL", "LS", "LL", "M")
        for family in ("LR", "DW")
    ]
    shared_only = dict(BASE_CONFIG)
    shared_only.update(
        config_id="SHARED_ONLY", family="SHARED_ONLY", cell="SHARED_ONLY",
        routed_experts=0, top_k=0, routed_width=0, rank=0, shared_width=2048,
    )
    shared_only["config_hash"] = config_hash(shared_only)
    validate_config(shared_only)
    conditions.append(shared_only)
    return conditions


def build_document() -> dict[str, Any]:
    proposals = []
    for rank in (80, 16):
        proposal = _condition("LR", f"proposal-r{rank}", 2048 - 24 * rank, rank)
        proposals.append(proposal)
    return {
        "schema_version": 1,
        "identity": "SHA-256 of canonical JSON with sorted keys and compact separators, excluding config_hash",
        "provenance": "algebraic; not measured",
        "conditions": build_grid(),
        "proposals": proposals,
    }


def build_ledger_rows() -> list[dict[str, Any]]:
    conditions = build_grid()
    baseline = next(parameter_counts(c) for c in conditions if c["config_id"] == "LR-HS")
    by_id = {c["config_id"]: c for c in conditions}
    rows = []
    for config in conditions:
        counts = parameter_counts(config)
        family, cell = config["family"], config["cell"]
        paired_total_equal = paired_active_equal = ""
        if family in ("LR", "DW"):
            other = "DW" if family == "LR" else "LR"
            paired = parameter_counts(by_id[f"{other}-{cell}"])
            paired_total_equal = counts["whole_model_total"] == paired["whole_model_total"]
            paired_active_equal = counts["whole_model_active"] == paired["whole_model_active"]
        rows.append({
            "config_id": config["config_id"], "config_hash": config["config_hash"],
            "family": family, "cell": cell, "rank": config["rank"],
            "shared_width": config["shared_width"], "routed_width": config["routed_width"],
            "routed_experts": config["routed_experts"], "top_k": config["top_k"],
            "tied_embedding": counts["components"]["tied_embedding"],
            "attention_per_layer": counts["components"]["attention_per_layer"],
            "rmsnorm_per_layer": counts["components"]["rmsnorm_per_layer"],
            "final_rmsnorm": counts["components"]["final_rmsnorm"],
            "router_per_layer": counts["components"]["router_per_layer"],
            "shared_swiglu_per_layer": counts["components"]["shared_swiglu_per_layer"],
            "routed_experts_total_per_layer": counts["components"]["routed_experts_total_per_layer"],
            "routed_experts_active_per_layer": counts["components"]["routed_experts_active_per_layer"],
            "per_layer_total": counts["per_layer_total"],
            "per_layer_active": counts["per_layer_active"],
            "whole_model_total": counts["whole_model_total"],
            "whole_model_active": counts["whole_model_active"],
            "total_equal_to_HS": counts["whole_model_total"] == baseline["whole_model_total"],
            "active_equal_to_HS": counts["whole_model_active"] == baseline["whole_model_active"],
            "paired_LR_DW_total_equal": paired_total_equal,
            "paired_LR_DW_active_equal": paired_active_equal,
            "provenance": "algebraic",
            "formula": "V*d + L*(4*d^2 + 2*d + router + shared + routed_total) + d; active substitutes routed_active",
        })
    return rows
