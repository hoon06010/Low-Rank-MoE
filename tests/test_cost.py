import pytest

from src.cost import estimate_cost


def _condition(**overrides):
    row = {
        "condition_id": "LR-HS", "target_tokens": 3600, "tokens_per_second": 10,
        "repetitions": 2, "eval_seconds_per_run": 30,
        "save_seconds_per_run": 5, "other_overhead_seconds_per_run": 10,
        "extra_cpu_ram_charge_per_run": 3,
        "storage_transfer_charge_per_run": 2,
        "other_direct_charge_per_run": 1,
        "direct_charge_evidence": {
            "extra_cpu_ram_charge": "quoted_separate_charge",
            "storage_transfer_charge": "measured_separate_charge",
            "other_direct_charge": "quoted_separate_charge",
        },
    }
    row.update(overrides)
    return row


def test_cost_components_tax_reserve_and_repetitions_are_separate():
    report = estimate_cost([_condition()], pretax_price_per_hour=36, tax_rate=.1, reserve_fraction=.2)
    row = report["conditions"][0]
    assert row["train_seconds"] == 720  # 3600 / 10, repeated twice
    assert row["evaluation_seconds"] == 60
    assert row["save_seconds"] == 10
    assert row["other_overhead_seconds"] == 20
    assert row["total_seconds"] == 810
    assert row["pretax_price_per_hour"] == 36
    assert row["tax_inclusive_price_per_hour"] == pytest.approx(39.6)
    assert row["base_time_cost"] == pytest.approx(8.1)
    assert row["time_tax_cost"] == pytest.approx(.81)
    assert row["tax_inclusive_time_cost"] == pytest.approx(8.91)
    assert row["extra_cpu_ram_charge"] == 6
    assert row["storage_transfer_charge"] == 4
    assert row["other_direct_charge"] == 2
    assert row["direct_charges_total"] == 12
    assert row["pre_reserve_total_cost"] == pytest.approx(20.91)
    assert row["reserve_cost"] == pytest.approx(4.182)
    assert row["reserved_total_cost"] == pytest.approx(25.092)
    assert report["totals"]["train_seconds"] == 720


@pytest.mark.parametrize("field", [
    "target_tokens", "tokens_per_second", "repetitions", "eval_seconds_per_run",
    "save_seconds_per_run", "other_overhead_seconds_per_run", "extra_cpu_ram_charge_per_run",
    "storage_transfer_charge_per_run", "other_direct_charge_per_run",
])
def test_missing_measured_inputs_are_unknown_not_zero(field):
    row = _condition()
    row[field] = None
    with pytest.raises(ValueError, match="required"):
        estimate_cost([row], pretax_price_per_hour=10, tax_rate=.1, reserve_fraction=.1)


def test_invalid_rates_and_duplicate_conditions_rejected():
    with pytest.raises(ValueError):
        estimate_cost([_condition(tokens_per_second=0)], pretax_price_per_hour=10, tax_rate=.1, reserve_fraction=.1)
    with pytest.raises(ValueError, match="unique"):
        estimate_cost([_condition(), _condition(condition_id="LR-HS")], pretax_price_per_hour=10, tax_rate=.1, reserve_fraction=.1)


def test_pre_tax_rate_and_explicit_zero_charge_evidence_are_required():
    with pytest.raises(TypeError):
        estimate_cost([_condition()], price_per_hour=36, tax_rate=.1, reserve_fraction=.1)
    zero = _condition(extra_cpu_ram_charge_per_run=0)
    with pytest.raises(ValueError, match="explicit applicable charge evidence"):
        estimate_cost([zero], pretax_price_per_hour=36, tax_rate=.1, reserve_fraction=.1)
    zero["direct_charge_evidence"]["extra_cpu_ram_charge"] = "verified_no_charge"
    result = estimate_cost([zero], pretax_price_per_hour=36, tax_rate=.1, reserve_fraction=.1)
    assert result["conditions"][0]["extra_cpu_ram_charge"] == 0


def test_experiment_plan_tax_inclusive_price_conversion_equivalence():
    plan_p = 39.6  # experiment_plan.md §9 defines p as tax-inclusive
    tax_rate = .1
    converted = estimate_cost(
        [_condition(extra_cpu_ram_charge_per_run=0, storage_transfer_charge_per_run=0,
                    other_direct_charge_per_run=0,
                    direct_charge_evidence={
                        "extra_cpu_ram_charge": "verified_no_charge",
                        "storage_transfer_charge": "included_in_hourly_rate",
                        "other_direct_charge": "verified_no_charge",
                    })],
        pretax_price_per_hour=plan_p / (1 + tax_rate), tax_rate=tax_rate, reserve_fraction=0,
    )
    row = converted["conditions"][0]
    assert row["tax_inclusive_price_per_hour"] == pytest.approx(plan_p)
    assert row["tax_inclusive_time_cost"] == pytest.approx(row["total_seconds"] / 3600 * plan_p)


def test_quoted_direct_charge_provenance_is_preserved_without_measured_claim():
    condition = _condition(direct_charge_evidence={
        "extra_cpu_ram_charge": "quoted_separate_charge",
        "storage_transfer_charge": "quoted_separate_charge",
        "other_direct_charge": "quoted_separate_charge",
    })
    report = estimate_cost([condition], pretax_price_per_hour=10, tax_rate=.1, reserve_fraction=.1)
    output = report["conditions"][0]
    assert output["direct_charge_evidence"] == condition["direct_charge_evidence"]
    assert "quoted_separate_charge" in output["direct_charge_evidence"].values()
    assert "measured inputs" not in report["provenance"]
    assert "calculated from supplied inputs" in report["provenance"]
