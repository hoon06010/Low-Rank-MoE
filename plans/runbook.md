# Preparation runbook

Status: local M0–M6 focused checks, full integration, and the M5 toy CPU preflight have run. Commands in the first section completed on the selected project interpreter. Later real-data/GPU steps are conditional instructions, not evidence of completed training.

## Verified local setup and focused checks

Working directory: `/path/to/Low-Rank-MoE`. Interpreter: `.venv/bin/python` (Python 3.13); installed project-local packages recorded in `plans/environment.md`. The `.venv` was created and torch/NumPy/pytest imports plus CPU backward were verified there. No GPU was rented or used.

```sh
.venv/bin/python -m pytest tests/test_budget.py -q
.venv/bin/python -m pytest tests/test_factorization.py tests/test_routing.py tests/test_model.py -q
.venv/bin/python -m pytest tests/test_data.py -q
.venv/bin/python -m pytest tests/test_resume.py tests/test_training_smoke.py -q
.venv/bin/python -m pytest tests/test_contrasts.py tests/test_cost.py -q
.venv/bin/python -m pytest -q
.venv/bin/python -m src.preflight --help
.venv/bin/python -m src.preflight --config configs/toy_cpu.json --output results/preflight/toy_cpu
.venv/bin/python -m src.cost --help
python3 -m json.tool data/manifests/schema.json >/dev/null
```

Last root runs returned exit 0: respectively 6, 12, 11, 30, 24, and 83 tests passed; help and toy preflight also exited 0, and the schema parsed. The final full-suite log is `reports/test_full.log`; exact hashes and limits are in `reports/verification.md`. `configs/architecture_grid.json` is the 11-condition pilot grid; `results/preflight/parameter_ledger.csv` is algebraic, not measured. `data/manifests/README.md` records source metadata candidates and the missing actual-data audit.

The toy output must carry `source_kind=synthetic_fixture`, and its token counts, loss, CPU time and checkpoint status are only an implementation smoke check. It cannot populate actual experiment Results.

## Conditional real-data and GPU stage, not authorized by this bundle

Before any real training, choose an immutable FineWeb-Edu shard set and GPT-2 tokenizer artifact revisions, review applicable rights/terms, download only within the user's authorized scope, compute local checksums, run document/duplicate/split/tokenizer audits, retain a frozen ordered evaluation target/label manifest and its SHA-256 `eval_revision`, and validate a `status=ready` data manifest. The current `TrainingSession` deliberately rejects `source_kind=measured`; M9 must implement and test this provenance gate before enabling it. The current in-memory document preparation is capped and is not a full 10BT streaming pipeline.

Freeze run protocol, optimizer/tuning opportunity, independent seed list, valid-target endpoint, failure handling and cost model, then measure representative throughput/peak memory with a bounded pilot. Record actual hardware, precision, tokens/s, overhead, taxes and reserve before considering the larger training package. `experiment_plan.md` §9 defines hourly `p` as tax-inclusive; `src.cost` takes `pretax_price_per_hour=p/(1+tax_rate)` and separately evidenced direct charges. The existing 500,000 KRW ceiling does not authorize a purchase. Never infer GPU throughput or full-run completion time from the algebraic ledger or CPU toy output. `plans/extension_protocol.md` contains separately costed follow-up comparisons.
