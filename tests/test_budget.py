import copy
import csv
import json
from pathlib import Path

import pytest

from src.budget import (
    build_document,
    build_grid,
    build_ledger_rows,
    config_hash,
    parameter_counts,
    validate_config,
)


EXPECTED_TOTALS = {
    "LR-HS": 37_016_832,
    "DW-HS": 37_016_832,
    "LR-LL": 37_016_832,
    "DW-LL": 37_016_832,
    "LR-M": 37_016_832,
    "DW-M": 37_016_832,
    "LR-LS": 31_708_416,
    "DW-LS": 31_708_416,
    "LR-HL": 42_325_248,
    "DW-HL": 42_325_248,
    "SHARED_ONLY": 36_998_400,
}


def test_grid_has_eleven_conditions_and_separate_proposals():
    document = build_document()
    grid = document["conditions"]
    assert len(grid) == 11
    assert {c["config_id"] for c in grid} == set(EXPECTED_TOTALS)
    assert [p["rank"] for p in document["proposals"]] == [80, 16]
    assert [p["shared_width"] for p in document["proposals"]] == [128, 1664]
    assert not ({p["config_id"] for p in document["proposals"]} & set(EXPECTED_TOTALS))


def test_config_hash_is_canonical_and_excludes_hash_field():
    config = build_grid()[0]
    assert config["config_hash"] == config_hash(config)
    reordered = dict(reversed(list(config.items())))
    assert config_hash(reordered) == config["config_hash"]
    changed = copy.deepcopy(config)
    changed["shared_width"] += 1
    assert config_hash(changed) != config["config_hash"]


def test_fixed_expected_whole_model_totals_and_components():
    grid = {c["config_id"]: c for c in build_grid()}
    for config_id, expected in EXPECTED_TOTALS.items():
        assert parameter_counts(grid[config_id])["whole_model_total"] == expected
    lr = parameter_counts(grid["LR-HS"])
    dw = parameter_counts(grid["DW-HS"])
    assert lr["components"]["shared_swiglu_per_layer"] == 1152 * 512
    assert lr["components"]["routed_experts_total_per_layer"] == 8 * 3456 * 64
    assert lr["components"]["routed_experts_active_per_layer"] == 2 * 3456 * 64
    assert grid["DW-HS"]["routed_width"] == 3 * grid["DW-HS"]["rank"]
    assert lr["whole_model_total"] == dw["whole_model_total"]
    assert lr["whole_model_active"] == dw["whole_model_active"]
    assert parameter_counts(grid["SHARED_ONLY"])["whole_model_active"] == 36_998_400


def test_fixed_width_shared_expert_count_matches_legacy_aggregate_budget():
    grid = {c["config_id"]: c for c in build_grid()}
    for cell, count in (("HS", 2), ("HL", 5), ("LS", 2), ("LL", 5)):
        legacy = grid[f"LR-{cell}"]
        mini = dict(legacy, shared_width=256, shared_experts=count)
        assert parameter_counts(mini) == parameter_counts(legacy)
    invalid = dict(grid["LR-HS"], shared_experts=0)
    with pytest.raises(ValueError, match="shared_experts"):
        validate_config(invalid)


def test_ledger_marks_algebraic_and_total_active_equalities_separately():
    rows = {row["config_id"]: row for row in build_ledger_rows()}
    assert len(rows) == 11
    for row in rows.values():
        assert row["provenance"] == "algebraic"
        assert row["paired_LR_DW_total_equal"] in (True, "")
        assert row["paired_LR_DW_active_equal"] in (True, "")
    assert rows["LR-HS"]["total_equal_to_HS"] is True
    assert rows["LR-HL"]["total_equal_to_HS"] is False
    assert rows["LR-LS"]["active_equal_to_HS"] is False
    assert rows["SHARED_ONLY"]["paired_LR_DW_total_equal"] == ""


def test_rejects_invalid_rank_shared_width_and_shape():
    lr = next(c for c in build_grid() if c["config_id"] == "LR-HS")
    bad = copy.deepcopy(lr)
    bad["rank"] = bad["d_model"] + 1
    with pytest.raises(ValueError, match="rank"):
        validate_config(bad)
    bad = copy.deepcopy(lr)
    bad["shared_width"] = 0
    with pytest.raises(ValueError, match="shared_width"):
        validate_config(bad)
    bad = copy.deepcopy(lr)
    bad["d_model"] += 1
    with pytest.raises(ValueError, match="divisible"):
        validate_config(bad)
    bad = copy.deepcopy(lr)
    bad["routed_width"] = 1
    with pytest.raises(ValueError, match="rank"):
        validate_config(bad)
    dw = next(c for c in build_grid() if c["config_id"] == "DW-HS")
    bad = copy.deepcopy(dw)
    bad["routed_width"] += 1
    with pytest.raises(ValueError, match=r"3 \* rank"):
        validate_config(bad)
    shared_only = next(c for c in build_grid() if c["family"] == "SHARED_ONLY")
    bad = copy.deepcopy(shared_only)
    bad["top_k"] = 1
    with pytest.raises(ValueError, match="zero"):
        validate_config(bad)


def test_written_config_and_csv_match_generated_algebra():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/architecture_grid.json").open() as stream:
        document = json.load(stream)
    assert document == build_document()
    with (root / "results/preflight/parameter_ledger.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)

    expected_rows = build_ledger_rows()
    assert fieldnames == list(expected_rows[0])
    serialized_expected = [
        {key: "" if value == "" else str(value) for key, value in row.items()}
        for row in expected_rows
    ]
    assert rows == serialized_expected
