import copy

import pytest

from src.analyze import analyze


def _inputs(seeds=(101, 202)):
    manifests, metrics = [], []
    bases = {
        "LR": {"HS": 3.00, "HL": 3.05, "LS": 3.10, "LL": 3.12},
        "DW": {"HS": 2.99, "HL": 3.02, "LS": 3.08, "LL": 3.10},
    }
    for seed in seeds:
        for family in ("LR", "DW"):
            for cell in ("HS", "HL", "LS", "LL"):
                run_id = f"{family}-{cell}-{seed}"
                ckpt = f"ckpt-{run_id}"
                loss = bases[family][cell] + (seed == 202) * .01 + (family == "LR" and cell == "LS" and seed == 202) * .02
                manifests.append({
                    "run_id": run_id, "protocol_id": "p1", "config_id": run_id,
                    "config_hash": f"hash-{run_id}",
                    "family": family, "cell": cell, "seed": seed,
                    "recipe_id": f"{family}-recipe", "data_manifest_hash": "d" * 64,
                    "tokenizer_revision": "tok1", "eval_revision": "eval1",
                    "source_commit": "fixture-commit", "reason": "target token budget reached",
                    "checkpoint_id": ckpt, "checkpoint_step": 100,
                    "trained_target_tokens": 1_000_000, "status": "completed",
                    "source_kind": "synthetic_fixture",
                })
                metrics.append({
                    "run_id": run_id, "protocol_id": "p1", "config_id": run_id,
                    "family": family, "cell": cell, "seed": str(seed), "split": "dev",
                    "checkpoint_id": ckpt, "trained_target_tokens": "1000000",
                    "eval_target_tokens": "500", "loss_nats": str(loss),
                    "status": "completed", "source_kind": "synthetic_fixture",
                    "data_manifest_hash": "d" * 64, "tokenizer_revision": "tok1", "eval_revision": "eval1",
                })
    return manifests, metrics


def test_prespecified_counterexample_formulas_recipes_and_paired_ci():
    manifests, metrics = _inputs()
    default = analyze(manifests, metrics)
    assert "synthetic_fixture_requires_opt_in" in default["contrasts"]["I_LR"]["exclusions"][0]["reasons"][0]
    result = analyze(manifests, metrics, allow_synthetic=True, endpoint_tokens=1_000_000)
    contrasts = result["contrasts"]
    assert contrasts["I_LR"]["per_seed"][0]["value"] == pytest.approx(.03)
    assert contrasts["G_low"]["per_seed"][0]["value"] == pytest.approx(-.02)
    assert contrasts["I_LR"]["summary"]["n"] == 2
    assert contrasts["I_LR"]["per_seed"][1]["value"] == pytest.approx(.05)
    ci = contrasts["I_LR"]["summary"]["paired_t_95_ci"]
    assert ci == pytest.approx([.04 - 12.706204736 * .01, .04 + 12.706204736 * .01])
    assert contrasts["J"]["per_seed"][0]["recipe_ids_by_family"] == {"DW": ["DW-recipe"], "LR": ["LR-recipe"]}
    assert contrasts["delta_budget"]["epsilon_decision"]["kind"] == "provisional_epsilon_proposal"


def test_ci_not_reported_for_one_seed_and_synthetic_default_excluded():
    manifests, metrics = _inputs(seeds=(101,))
    result = analyze(manifests, metrics, allow_synthetic=True)
    assert result["contrasts"]["I_LR"]["summary"]["paired_t_95_ci"] is None
    rejected = analyze(manifests, metrics)
    row = rejected["contrasts"]["I_LR"]["exclusions"][0]
    assert "synthetic_fixture_requires_opt_in" in row["reasons"][0]


def test_failed_duplicate_identity_and_nonfinal_rows_are_not_seeds():
    manifests, metrics = _inputs(seeds=(101, 202))
    # Fail one required cell: contrasts requiring it are incomplete for seed 101.
    manifests[0]["status"] = "failed"
    # Add a duplicate run attempt for another cell/seed.
    manifests.append(copy.deepcopy(manifests[5]))
    # Mismatch a paired family identity, which excludes J but leaves I_LR calculable.
    dw_hs = next(row for row in manifests if row["run_id"] == "DW-HS-101")
    dw_hs["data_manifest_hash"] = "e" * 64
    # Add an extra non-final checkpoint metric; only final manifest checkpoint is eligible.
    extra = copy.deepcopy(metrics[1])
    extra["checkpoint_id"] = "intermediate-checkpoint"
    metrics.append(extra)
    result = analyze(manifests, metrics, allow_synthetic=True)
    assert result["contrasts"]["I_LR"]["summary"]["n"] == 1
    assert result["contrasts"]["I_LR"]["exclusions"]
    assert result["contrasts"]["J"]["exclusions"]
    assert result["contrasts"]["I_LR"]["summary"]["paired_t_95_ci"] is None


def test_duplicate_metric_primary_key_excludes_run():
    manifests, metrics = _inputs(seeds=(101,))
    metrics.append(copy.deepcopy(metrics[0]))
    result = analyze(manifests, metrics, allow_synthetic=True)
    reasons = result["contrasts"]["I_LR"]["exclusions"][0]["reasons"]
    assert any("duplicate_metric_key" in reason for reason in reasons)


def test_missing_repeated_metric_identity_is_rejected():
    manifests, metrics = _inputs(seeds=(101,))
    metrics[0].pop("family")
    result = analyze(manifests, metrics, allow_synthetic=True)
    reasons = result["contrasts"]["I_LR"]["exclusions"][0]["reasons"]
    assert any("manifest_metric_identity_mismatch:family" in reason for reason in reasons)


@pytest.mark.parametrize("missing", ["config_hash", "source_commit", "reason"])
def test_completed_manifest_requires_m0_audit_fields(missing):
    manifests, metrics = _inputs(seeds=(101,))
    manifests[0].pop(missing)
    result = analyze(manifests, metrics, allow_synthetic=True)
    reasons = result["contrasts"]["I_LR"]["exclusions"][0]["reasons"]
    assert any(f"missing_manifest_identity:{missing}" in reason for reason in reasons)


def test_family_recipe_mismatch_excludes_only_affected_contrasts():
    manifests, metrics = _inputs()
    row = next(row for row in manifests if row["run_id"] == "LR-LS-101")
    row["recipe_id"] = "LR-other-recipe"
    result = analyze(manifests, metrics, allow_synthetic=True)
    contrasts = result["contrasts"]
    for name in ("I_LR", "J", "G_low"):
        assert any("recipe_id_within_family:LR" in reason for exclusion in contrasts[name]["exclusions"]
                   for reason in exclusion["reasons"])
    assert contrasts["I_DW"]["summary"]["n"] == 2


def test_measured_eval_revision_must_be_hash_and_pair_identity_must_match():
    manifests, metrics = _inputs()
    for row in manifests:
        row["source_kind"] = "measured"
        row["eval_revision"] = "a" * 64
    for row in metrics:
        row["source_kind"] = "measured"
        row["eval_revision"] = "a" * 64

    accepted = analyze(manifests, metrics)
    assert accepted["contrasts"]["I_LR"]["summary"]["n"] == 2

    invalid_manifests, invalid_metrics = _inputs(seeds=(101,))
    for row in invalid_manifests:
        row["source_kind"] = "measured"
        row["eval_revision"] = "not-a-sha"
    for row in invalid_metrics:
        row["source_kind"] = "measured"
        row["eval_revision"] = "not-a-sha"
    invalid = analyze(invalid_manifests, invalid_metrics)
    assert any("measured_eval_revision_must_be_sha256" in reason
               for exclusion in invalid["contrasts"]["I_LR"]["exclusions"] for reason in exclusion["reasons"])

    mismatch_manifests, mismatch_metrics = _inputs(seeds=(101,))
    for row in mismatch_manifests:
        row["source_kind"] = "measured"
        row["eval_revision"] = "a" * 64
    for row in mismatch_metrics:
        row["source_kind"] = "measured"
        row["eval_revision"] = "a" * 64
    changed_manifest = next(row for row in mismatch_manifests if row["run_id"] == "LR-LS-101")
    changed_metric = next(row for row in mismatch_metrics if row["run_id"] == "LR-LS-101")
    changed_manifest["eval_revision"] = "b" * 64
    changed_metric["eval_revision"] = "b" * 64
    mismatch = analyze(mismatch_manifests, mismatch_metrics)
    assert any("paired_identity_mismatch:eval_revision" in reason
               for exclusion in mismatch["contrasts"]["I_LR"]["exclusions"] for reason in exclusion["reasons"])


def test_real_setup_runs_are_excluded_from_research_contrasts():
    manifests, metrics = _inputs()
    for row in manifests:
        row.update(source_kind='measured',eval_revision='a'*64,purpose='pipeline_smoke_only')
    for row in metrics:
        row.update(source_kind='measured',eval_revision='a'*64)
    result=analyze(manifests,metrics)
    assert result['contrasts']['I_LR']['summary']['n']==0
    assert any('pipeline_smoke_only' in reason for reason in result['contrasts']['I_LR']['exclusions'][0]['reasons'])


def test_mini_exploration_runs_are_excluded_from_research_contrasts():
    manifests, metrics = _inputs()
    for row in manifests:
        row.update(source_kind='measured', eval_revision='a'*64, purpose='mini_exploration_only')
    for row in metrics:
        row.update(source_kind='measured', eval_revision='a'*64)
    result = analyze(manifests, metrics)
    assert result['contrasts']['I_LR']['summary']['n'] == 0
    assert any('mini_exploration_only_excluded_from_research' in reason
               for reason in result['contrasts']['I_LR']['exclusions'][0]['reasons'])
