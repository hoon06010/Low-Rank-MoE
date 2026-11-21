"""Run-cost accounting from measured and quoted inputs; unknowns are not zero."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} is required; unknown inputs cannot be treated as zero")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return number


def estimate_cost(conditions: Iterable[dict], *, pretax_price_per_hour: float, tax_rate: float,
                  reserve_fraction: float) -> dict[str, Any]:
    """Price each condition from measured and quoted inputs, keeping time and direct charges separate.

    Each condition requires ``condition_id``, actual ``target_tokens``, measured
    ``tokens_per_second``, ``repetitions``, ``eval_seconds_per_run``,
    ``save_seconds_per_run``, ``other_overhead_seconds_per_run``, and explicit
    per-run extra CPU/RAM, storage/transfer, and other direct charge amounts.
    The server rate is a PRETAX wall-clock hourly rate. In experiment_plan.md
    section 9, ``p`` is tax-inclusive, so callers must pass
    ``pretax_price_per_hour=p/(1+tax_rate)``; the reported
    ``tax_inclusive_price_per_hour`` is then exactly ``p``. Direct charge amounts
    must already be tax-inclusive. A direct charge of zero requires evidence
    ``included_in_hourly_rate`` or ``verified_no_charge``; a positive amount
    requires ``quoted_separate_charge`` or ``measured_separate_charge``.
    Reserve is applied to the full time-plus-direct charge total.
    """
    pretax_price = _finite_number(pretax_price_per_hour, "pretax_price_per_hour")
    tax = _finite_number(tax_rate, "tax_rate")
    reserve = _finite_number(reserve_fraction, "reserve_fraction")
    if tax >= 1:
        raise ValueError("tax_rate must be expressed as a fraction below 1")
    if reserve >= 1:
        raise ValueError("reserve_fraction must be expressed as a fraction below 1")
    tax_inclusive_price = pretax_price * (1.0 + tax)
    rows = list(conditions)
    if not rows:
        raise ValueError("at least one condition is required")
    ids, output = set(), []
    totals = {key: 0.0 for key in (
        "train_seconds", "evaluation_seconds", "save_seconds", "other_overhead_seconds", "total_seconds",
        "base_time_cost", "time_tax_cost", "tax_inclusive_time_cost", "extra_cpu_ram_charge",
        "storage_transfer_charge", "other_direct_charge", "direct_charges_total", "pre_reserve_total_cost",
        "reserve_cost", "reserved_total_cost",
    )}
    for index, condition in enumerate(rows):
        label = str(condition.get("condition_id", ""))
        if not label or label in ids:
            raise ValueError(f"condition_id must be present and unique (row {index})")
        ids.add(label)
        tokens = _finite_number(condition.get("target_tokens"), f"{label}.target_tokens", minimum=1.0)
        if not tokens.is_integer():
            raise ValueError(f"{label}.target_tokens must be an integer actual token count")
        throughput = _finite_number(condition.get("tokens_per_second"), f"{label}.tokens_per_second", minimum=1e-300)
        repetitions = _finite_number(condition.get("repetitions"), f"{label}.repetitions", minimum=1.0)
        if not repetitions.is_integer():
            raise ValueError(f"{label}.repetitions must be an integer")
        repetitions = int(repetitions)
        evaluation = _finite_number(condition.get("eval_seconds_per_run"), f"{label}.eval_seconds_per_run") * repetitions
        saving = _finite_number(condition.get("save_seconds_per_run"), f"{label}.save_seconds_per_run") * repetitions
        overhead = _finite_number(condition.get("other_overhead_seconds_per_run"), f"{label}.other_overhead_seconds_per_run") * repetitions
        training = tokens / throughput * repetitions
        total_seconds = training + evaluation + saving + overhead
        base_time_cost = total_seconds / 3600.0 * pretax_price
        time_tax_cost = base_time_cost * tax
        tax_inclusive_time_cost = base_time_cost + time_tax_cost
        direct_fields = {
            "extra_cpu_ram_charge": "extra_cpu_ram_charge_per_run",
            "storage_transfer_charge": "storage_transfer_charge_per_run",
            "other_direct_charge": "other_direct_charge_per_run",
        }
        evidence = condition.get("direct_charge_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"{label}.direct_charge_evidence is required")
        direct_charges = {}
        for charge_name, input_name in direct_fields.items():
            amount = _finite_number(condition.get(input_name), f"{label}.{input_name}") * repetitions
            basis = evidence.get(charge_name)
            allowed = {"included_in_hourly_rate", "verified_no_charge"} if amount == 0 else {
                "quoted_separate_charge", "measured_separate_charge"
            }
            if basis not in allowed:
                raise ValueError(f"{label}.{charge_name} requires explicit applicable charge evidence")
            direct_charges[charge_name] = amount
        direct_total = sum(direct_charges.values())
        pre_reserve_total = tax_inclusive_time_cost + direct_total
        reserve_cost = pre_reserve_total * reserve
        values = {
            "condition_id": label, "target_tokens_per_run": int(tokens), "repetitions": repetitions,
            "measured_tokens_per_second": throughput,
            "train_seconds": training, "evaluation_seconds": evaluation,
            "save_seconds": saving, "other_overhead_seconds": overhead,
            "total_seconds": total_seconds, "pretax_price_per_hour": pretax_price,
            "tax_inclusive_price_per_hour": tax_inclusive_price,
            "base_time_cost": base_time_cost, "time_tax_cost": time_tax_cost,
            "tax_inclusive_time_cost": tax_inclusive_time_cost,
            **direct_charges, "direct_charge_evidence": dict(evidence),
            "direct_charges_total": direct_total,
            "pre_reserve_total_cost": pre_reserve_total, "reserve_fraction": reserve,
            "reserve_cost": reserve_cost, "reserved_total_cost": pre_reserve_total + reserve_cost,
        }
        output.append(values)
        for key in totals:
            totals[key] += values[key]
    return {
        "currency": "unspecified currency units",
        "pretax_price_per_hour": pretax_price,
        "tax_inclusive_price_per_hour": tax_inclusive_price,
        "tax_rate": tax, "reserve_fraction": reserve,
        "conditions": output, "totals": totals,
        "provenance": "calculated from supplied inputs; direct-charge evidence basis is retained per condition; no missing field is imputed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="JSON with conditions, pretax_price_per_hour, tax_rate and reserve_fraction; for experiment_plan.md p, pass pretax_price_per_hour=p/(1+tax_rate)",
    )
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args(argv)
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = estimate_cost(request["conditions"], pretax_price_per_hour=request["pretax_price_per_hour"],
                           tax_rate=request["tax_rate"], reserve_fraction=request["reserve_fraction"])
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
