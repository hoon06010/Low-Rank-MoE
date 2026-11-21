# Preparation-bundle verification log

This file records commands actually run by the root controller. It is not an experiment-results report. M1–M6 artifacts and the full local integration suite have been checked.

## Environment and scope

- Project: `/path/to/Low-Rank-MoE`; selected interpreter: `.venv/bin/python` (Python 3.13), with torch 2.9.1, pytest 8.4.2, NumPy 2.3.3. Installation/import and CPU backward evidence is in `plans/environment.md`.
- GPU rental, payment, and large-scale model training: none. Apple M1 host; CUDA unavailable. The CPU toy and algebraic ledger are preparation evidence only.
- Interface source: `plans/interfaces.md`; model conventions: `plans/model_details.md`; plan source: `plans/plan.md` §4 and §6.

## Verified task checks so far

| Task | Actual command or check | Result | Evidence boundary |
|---|---|---|---|
| M0 | Import torch/NumPy/pytest, CPU backward, pytest executable (see `plans/environment.md`); SHA-256 of environment/interfaces and source plans | Pass | Local environment only; no GPU availability beyond host check |
| M1 | Root opened primary StructMoE, PMLR, DeepSeekMoE and relevant arXiv sources; parsed `literature/evidence_ledger.csv` as eight nonempty rows; checked comparison matrix and hashes | Pass for bounded literature scope | Not an exhaustive novelty search; 2024 official code and 2026 under-review code not audited |
| M2 | `.venv/bin/python -m pytest tests/test_budget.py -q` | Exit 0, 6 passed | Algebraic 11-condition counts and checked-in CSV/JSON synchronization; no measured FLOPs or throughput |
| M3 | `.venv/bin/python -m pytest tests/test_factorization.py tests/test_routing.py tests/test_model.py -q` | Exit 0, 12 passed | CPU factor/routing references and all 11 actual parameter counts; deterministic FP32 gate-sum check, no 37M model training or GPU throughput conclusion |
| M4 | `.venv/bin/python -m pytest tests/test_data.py -q`; `python3 -m json.tool data/manifests/schema.json` | Exit 0, 11 passed; schema parses | Synthetic document fixtures and manifest readiness gate; actual dataset/tokenizer bytes were not downloaded/audited |
| M5 | `.venv/bin/python -m pytest tests/test_resume.py tests/test_training_smoke.py -q`; `.venv/bin/python -m src.preflight --help`; `.venv/bin/python -m src.preflight --config configs/toy_cpu.json --output results/preflight/toy_cpu` | Exit 0; 30 passed; help and toy CPU run completed | Synthetic run trained 8 valid targets in one step and evaluated 5 targets. It reported CE 3.558094692230225 nats/target; GPU memory is null/unmeasured. The producer explicitly rejects measured runs until audited evaluation provenance exists. Independent spec and quality reviews passed. |
| M6 | `.venv/bin/python -m pytest tests/test_contrasts.py tests/test_cost.py -q`; `.venv/bin/python -m src.cost --help` | Exit 0; 24 passed; help displays the §9 tax-inclusive-rate conversion | Synthetic contrasts and cost formula only. No actual throughput, price, or effect is measured. Independent spec and quality reviews passed. |

M2 and M4 quality issues were fixed and the affected tests rerun. M3 numeric RoPE and pre-RMSNorm order tests were added and all focused tests rerun. The clarified M0 auxiliary objective invalidated downstream hashes; 29 M2–M4 focused tests passed after that change. The later measured-evaluation-revision rule invalidated them again; unchanged output hashes and the 11 M4 data tests were checked before restoring M0/M2/M3/M4 verified state. Independent Luna reviews and transitions are in `plans/events.jsonl`. The root checked that the latest M5 toy output `results/preflight/toy_cpu/preflight.json` joined to exactly one manifest record and metric row by run/checkpoint identity, with `source_kind=synthetic_fixture` throughout.

## Final local integration check

The first full-suite run found one test-only failure: an unseeded FP32 gate-sum assertion used `1e-10` tolerance. A deterministic seed 3 reproduced a `5.9604644775e-08` difference, below FP32 epsilon `1.1920928955e-07`. `tests/test_routing.py` now fixes the local random input and applies a two-epsilon gate-only tolerance; strict FP64 forward and gradient comparisons remain. Independent M3 spec/quality delta reviews passed.

On the final code, `.venv/bin/python -m pytest -q > reports/test_full.log 2>&1` exited **0** and the log says **83 passed in 8.96s**. Log SHA-256: `a07cf38e8e38246c632e65bc31ccbaf259acd43db940a0ca27ed304716a26426`. Interpreter/package versions: Python 3.13.5, torch 2.9.1, pytest 8.4.2, NumPy 2.3.3. The root reran the exact toy preflight command above after the final M5 changes; it exited 0. Latest run ID `b7632778-92e3-48ea-bb8d-a1232722f6ba` matched one manifest and metric row at checkpoint `8619255578d446e4820de2598989b310`, with 8 trained and 5 evaluation targets. The synthetic `preflight.json` SHA-256 is `f72786d9c376db732e621856116a66733f299409558ff021b6b4487617f2dd3b`; cumulative manifest/metrics hashes are `27e97a0c815b73485bc3f6d8cdac6d99304d21ce05b0d44a9f6b0a74f9314f9b` / `1cf6705c0b00e9525e83aa354b691766e27eacc77e802cd80e6e00a920086e3d`. All are synthetic preparation artifacts.

Cross-module checks: the algebraic 11-condition parameter ledger agrees with actual model parameter counts; routing sparse/reference outputs and gradients agree within their stated dtypes; data split/duplicate/cursor fixtures pass; checkpoint resume reproduces model, optimizer, scheduler, cursor and RNG; the preflight manifest joins metrics on run/checkpoint; six contrast formulas, missing-block handling and cost arithmetic pass. These are correctness and readiness checks, not measured research effects.

## Known implementation limit

`src/model.py` currently converts routing diagnostics to host Python values during forward, which can synchronize GPU execution. Any future measured throughput must profile the actual instrumented path and must not be inferred from algebraic parameter counts. `src/data.py` caps in-memory document preparation at 100,000 records by default; full FineWeb-Edu preparation requires a separately designed streaming or external-sort path and a real source audit.
