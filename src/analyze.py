"""Seed-paired analysis for the prespecified Low-Rank-MoE contrasts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRIC_KEY = ("run_id", "split", "checkpoint_id")
IDENTITY_FIELDS = (
    "protocol_id", "data_manifest_hash", "tokenizer_revision", "eval_revision",
    "split", "trained_target_tokens", "eval_target_tokens",
)
CELLS = ("HS", "HL", "LS", "LL")
CONTRASTS = {
    "I_LR": (("LR", "LS", 1), ("LR", "LL", -1), ("LR", "HS", -1), ("LR", "HL", 1)),
    "I_DW": (("DW", "LS", 1), ("DW", "LL", -1), ("DW", "HS", -1), ("DW", "HL", 1)),
    "J": (("LR", "LS", 1), ("LR", "LL", -1), ("LR", "HS", -1), ("LR", "HL", 1),
          ("DW", "LS", -1), ("DW", "LL", 1), ("DW", "HS", 1), ("DW", "HL", -1)),
    "G_low": (("LR", "LS", 1), ("LR", "LL", -1)),
    "delta_budget": (("LR", "LL", 1), ("LR", "HS", -1)),
    "delta": (("LR", "LL", 1), ("DW", "LL", -1)),
}


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta using a stable continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    def fraction(aa: float, bb: float, xx: float) -> float:
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        c, d = 1.0, 1.0 - qab * xx / qap
        if abs(d) < 1e-300:
            d = 1e-300
        d = 1.0 / d
        h = d
        for m in range(1, 201):
            m2 = 2 * m
            aa_cf = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + aa_cf * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa_cf / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            h *= d * c
            aa_cf = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + aa_cf * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa_cf / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-14:
                break
        return h

    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * fraction(a, b, x) / a
    return 1.0 - front * fraction(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    tail = 0.5 * _regularized_beta(x, df / 2.0, 0.5)
    return 1.0 - tail if t >= 0 else tail


def _t_critical_975(df: int) -> float:
    low, high = 0.0, 1.0
    while _t_cdf(high, df) < 0.975:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if _t_cdf(mid, df) < 0.975:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _summary(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n if n else None
    if n < 2:
        return {"n": n, "mean": mean, "sample_sd": None, "paired_t_95_ci": None}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    sd = math.sqrt(variance)
    margin = _t_critical_975(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sample_sd": sd, "paired_t_95_ci": [mean - margin, mean + margin]}


def analyze(manifests: Iterable[dict], metrics: Iterable[dict], *, split: str = "dev", allow_synthetic: bool = False,
            endpoint_tokens: int | None = None, epsilon: float = 0.01) -> dict[str, Any]:
    """Join attempts and metrics, then compute six prespecified contrasts.

    The manifest's checkpoint_id must match the metric checkpoint, thereby
    selecting only the final verified checkpoint of a completed attempt.
    ``endpoint_tokens`` can freeze a protocol endpoint explicitly; when omitted,
    all cells in each paired contrast must still share the same endpoint.
    """
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if endpoint_tokens is not None and (type(endpoint_tokens) is not int or endpoint_tokens <= 0):
        raise ValueError("endpoint_tokens must be a positive integer")
    epsilon = _number(epsilon, "epsilon")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    manifest_list = list(manifests)
    metric_list = list(metrics)
    manifests_by_id: dict[str, dict] = {}
    duplicate_runs: set[str] = set()
    for row in manifest_list:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        if run_id in manifests_by_id:
            duplicate_runs.add(run_id)
        else:
            manifests_by_id[run_id] = row

    metric_keys: set[tuple[str, str, str]] = set()
    metrics_by_run: dict[str, list[dict]] = defaultdict(list)
    duplicate_metric_runs: set[str] = set()
    for row in metric_list:
        key = tuple(str(row.get(field, "")) for field in METRIC_KEY)
        if key in metric_keys:
            duplicate_metric_runs.add(key[0])
        metric_keys.add(key)
        if key[1] == split:
            metrics_by_run[key[0]].append(row)

    joined_candidates: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    invalid: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    seeds: set[int] = set()
    attempts_per_condition: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for row in manifest_list:
        try:
            condition_key = (str(row["family"]), str(row["cell"]), int(row["seed"]))
            attempts_per_condition[condition_key].append(str(row.get("run_id", "")))
            seeds.add(condition_key[2])
        except (KeyError, TypeError, ValueError):
            continue
    for run_id, manifest in manifests_by_id.items():
        try:
            seed = int(manifest["seed"])
            family, cell = str(manifest["family"]), str(manifest["cell"])
        except (KeyError, TypeError, ValueError):
            continue
        seeds.add(seed)
        key = (family, cell, seed)
        if len(attempts_per_condition.get(key, [])) > 1:
            invalid[key].append("duplicate_condition_attempts_for_seed")
            continue
        if manifest.get("purpose") == "pipeline_smoke_only":
            invalid[key].append("pipeline_smoke_only_excluded_from_research")
            continue
        if manifest.get("purpose") == "mini_exploration_only":
            invalid[key].append("mini_exploration_only_excluded_from_research")
            continue
        if manifest.get("source_kind") not in {"measured", "synthetic_fixture"}:
            invalid[key].append("missing_or_invalid_source_kind")
            continue
        if run_id in duplicate_runs:
            invalid[key].append("duplicate_run_id")
            continue
        if run_id in duplicate_metric_runs:
            invalid[key].append("duplicate_metric_key")
            continue
        if manifest.get("source_kind") == "synthetic_fixture" and not allow_synthetic:
            invalid[key].append("synthetic_fixture_requires_opt_in")
            continue
        if manifest.get("status") != "completed":
            invalid[key].append(f"run_status_{manifest.get('status', 'missing')}")
            continue
        missing_manifest_fields = [field for field in
                                   ("protocol_id", "config_id", "config_hash", "recipe_id", "data_manifest_hash",
                                    "tokenizer_revision", "eval_revision", "source_commit", "reason")
                                   if manifest.get(field) in (None, "")]
        if missing_manifest_fields:
            invalid[key].append("missing_manifest_identity:" + ",".join(missing_manifest_fields))
            continue
        if manifest.get("source_kind") == "measured" and not re.fullmatch(r"[0-9a-f]{64}", str(manifest["eval_revision"])):
            invalid[key].append("measured_eval_revision_must_be_sha256")
            continue
        manifest_tokens = _int_field(manifest, "trained_target_tokens")
        if manifest_tokens is None or manifest_tokens <= 0:
            invalid[key].append("invalid_trained_target_tokens")
            continue
        if endpoint_tokens is not None and manifest_tokens != endpoint_tokens:
            invalid[key].append("not_at_fixed_target_endpoint")
            continue
        if _int_field(manifest, "checkpoint_step") is None or _int_field(manifest, "checkpoint_step") <= 0:
            invalid[key].append("missing_final_checkpoint_step")
            continue
        if manifest.get("checkpoint_id") in (None, ""):
            invalid[key].append("missing_final_checkpoint_id")
            continue
        candidates = [row for row in metrics_by_run.get(run_id, [])
                      if str(row.get("checkpoint_id", "")) == str(manifest.get("checkpoint_id", ""))]
        if len(candidates) != 1:
            invalid[key].append("missing_or_duplicate_final_checkpoint_metric")
            continue
        metric = candidates[0]
        repeated_fields = ("protocol_id", "config_id", "family", "cell", "seed", "trained_target_tokens",
                           "checkpoint_id", "status", "source_kind")
        mismatch = []
        for field in repeated_fields:
            if field not in metric or str(metric[field]) != str(manifest.get(field, "")):
                mismatch.append(field)
        # M0 allows eval_revision as an optional metrics column, but when supplied
        # it must agree. The manifest's eval_revision is the authoritative target identity.
        for field in ("data_manifest_hash", "tokenizer_revision", "eval_revision"):
            if field in metric and str(metric[field]) != str(manifest.get(field, "")):
                mismatch.append(field)
        if mismatch:
            invalid[key].append("manifest_metric_identity_mismatch:" + ",".join(mismatch))
            continue
        if str(metric.get("checkpoint_id", "")) != str(manifest.get("checkpoint_id", "")):
            invalid[key].append("metric_not_final_manifest_checkpoint")
            continue
        if str(metric.get("status", "")) != "completed":
            invalid[key].append("metric_not_completed")
            continue
        if metric.get("source_kind", manifest.get("source_kind")) != manifest.get("source_kind"):
            invalid[key].append("source_kind_mismatch")
            continue
        if _int_field(metric, "trained_target_tokens") != manifest_tokens:
            invalid[key].append("trained_target_token_mismatch")
            continue
        eval_targets = _int_field(metric, "eval_target_tokens")
        if eval_targets is None or eval_targets <= 0:
            invalid[key].append("invalid_eval_target_count")
            continue
        try:
            loss = _number(metric.get("loss_nats"), "loss_nats")
        except ValueError:
            invalid[key].append("invalid_loss")
            continue
        if loss < 0:
            invalid[key].append("invalid_loss")
            continue
        joined_candidates[key].append({
            "family": family, "cell": cell, "seed": seed, "loss_nats": loss,
            "run_id": run_id, "recipe_id": manifest.get("recipe_id"),
            "checkpoint_id": manifest.get("checkpoint_id"),
            "protocol_id": manifest.get("protocol_id"),
            "data_manifest_hash": manifest.get("data_manifest_hash"),
            "tokenizer_revision": manifest.get("tokenizer_revision"),
            "eval_revision": manifest.get("eval_revision"), "split": split,
            "trained_target_tokens": manifest_tokens, "eval_target_tokens": eval_targets,
        })

    joined: dict[tuple[str, str, int], dict] = {}
    for key, rows in joined_candidates.items():
        if len(rows) != 1:
            invalid[key].append("duplicate_completed_condition_attempt")
        else:
            joined[key] = rows[0]

    contrast_results = {}
    for name, formula in CONTRASTS.items():
        required_cells = sorted({(family, cell) for family, cell, _ in formula})
        per_seed, exclusions = [], []
        for seed in sorted(seeds):
            rows, reasons = [], []
            for family, cell in required_cells:
                key = (family, cell, seed)
                row = joined.get(key)
                if row is None:
                    why = invalid.get(key, ["missing_cell"])
                    reasons.extend(f"{family}-{cell}:{reason}" for reason in why)
                else:
                    rows.append(row)
            if reasons:
                exclusions.append({"seed": seed, "reasons": sorted(set(reasons))})
                continue
            identities = {field: {row[field] for row in rows} for field in IDENTITY_FIELDS}
            mismatches = [field for field, values in identities.items() if len(values) != 1]
            recipe_families = {family: {str(row["recipe_id"]) for row in rows if row["family"] == family}
                               for family in {row["family"] for row in rows}}
            recipe_mismatches = sorted(family for family, recipes in recipe_families.items() if len(recipes) != 1)
            if recipe_mismatches:
                mismatches.append("recipe_id_within_family:" + ",".join(recipe_mismatches))
            if mismatches:
                exclusions.append({"seed": seed, "reasons": ["paired_identity_mismatch:" + ",".join(mismatches)]})
                continue
            lookup = {(row["family"], row["cell"]): row for row in rows}
            value = sum(sign * lookup[(family, cell)]["loss_nats"] for family, cell, sign in formula)
            per_seed.append({
                "seed": seed, "value": value,
                "conditions": {f"{row['family']}-{row['cell']}": row["loss_nats"] for row in rows},
                "recipe_ids_by_family": {
                    family: sorted({str(row["recipe_id"]) for row in rows if row["family"] == family})
                    for family in sorted({row["family"] for row in rows})
                },
                "trained_target_tokens": rows[0]["trained_target_tokens"],
                "eval_target_tokens": rows[0]["eval_target_tokens"],
            })
        values = [row["value"] for row in per_seed]
        summary = _summary(values)
        decision = _epsilon_decision(name, summary, epsilon)
        contrast_results[name] = {"per_seed": per_seed, "summary": summary, "exclusions": exclusions,
                                  "epsilon_proposal": epsilon, "epsilon_decision": decision}
    return {"split": split, "source_kinds_allowed": ["measured", "synthetic_fixture"] if allow_synthetic else ["measured"],
            "contrasts": contrast_results}


def _int_field(row: dict, field: str) -> int | None:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _epsilon_decision(name: str, summary: dict, epsilon: float) -> dict[str, Any]:
    ci = summary["paired_t_95_ci"]
    if ci is None:
        status = "not_estimable_n_lt_2"
    elif name in {"I_LR", "I_DW", "J", "G_low"}:
        status = "positive_lower_bound" if ci[0] > 0 else "not_supported"
    elif name == "delta_budget":
        status = ("improvement_supported" if ci[1] < 0 else
                  "noninferiority_proposal_met" if ci[1] < epsilon else "not_supported")
    else:
        status = ("LR_quality_better_than_DW" if ci[1] < 0 else
                  "noninferiority_proposal_met" if ci[1] < epsilon else "not_supported")
    return {"status": status, "threshold": epsilon, "kind": "provisional_epsilon_proposal"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL attempt manifest")
    parser.add_argument("--metrics", required=True, help="metrics CSV")
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--endpoint-tokens", type=int)
    parser.add_argument("--epsilon", type=float, default=.01)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output", help="optional JSON result path")
    args = parser.parse_args(argv)
    with Path(args.metrics).open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    result = analyze(_read_jsonl(Path(args.manifest)), metrics, split=args.split,
                     allow_synthetic=args.allow_synthetic, endpoint_tokens=args.endpoint_tokens,
                     epsilon=args.epsilon)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
