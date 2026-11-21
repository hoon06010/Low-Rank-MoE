# Seed repeat exploratory collector

Use `scripts/analyze_seed_repeat_c.py`; the frozen `src/analyze.py` is intentionally unchanged.
This collector preserves `purpose=mini_exploration_only`. It does not promote these runs to confirmatory evidence.

Prepare an input JSON list with one record per run directory:

```json
[
  {"experiment_id":"stage-c-init-seed-repeat-v1","phase":"main","path":"/absolute/path/to/main/LR-HL"},
  {"experiment_id":"seed401-spectron-fixed-recipe-v1","phase":"main","path":"/absolute/path/to/spectron/LR-HS"}
]
```

Run from the repository root:

```sh
.venv/bin/python -B scripts/analyze_seed_repeat_c.py --inputs /absolute/path/to/inputs.json
```

The command prints JSON and does not write experiment files. The same interface is available as
`analyze(specs)` from `scripts.analyze_seed_repeat_c`.

The input inventory supplies experiment/phase scope for historical AdamW directories that do not record
these fields. Explicit fields in reports, manifests, or `experiment_spec.json` must agree. Supply only the
retained main outputs; label pilots/recovery correctly. Other experiments, pilots, seed17, and unfinished
runs appear under `excluded`. Conflicting or duplicate completed evidence raises an error.

For completed runs, checks cover model/config/recipe/optimizer, report identity, fixed endpoint2048 /
8388608targets, eight measured full-dev evaluations (65536 targets each), data/evaluation pairing,
source/training/config hashes, final checkpoint pointer and file SHA256. Run-supplied source maps must
match the independently retained AdamW or Spectron package inventory; bundle and dev identities must
match the frozen experiment policy and verified local dev file. Checkpoint payloads are **not
unpickled**; this is a file-integrity and metadata audit, not a replacement for independent tensor/state
restore verification or GPU validation.

AdamW reports per-seed Delta_budget, G_low, G_high, I_LR, mean and sample SD. Descriptive paired t95% CI
is emitted only when all seeds401/402/403 for that contrast are complete. Partial observations have no
3-seed CI. There is no practical-superiority threshold or automatic hypothesis-test verdict.
Spectron seed401 HS/LL remains separate: cell-wise Spectron−AdamW and Spectron Delta_budget only.
Missing anchor cells remain visibly absent; no imputation or optimizer pooling is performed.

Full dev curves and selected raw resource fields accompany each run. Resource times are the final
invocation's fields, **not total execution cost**. The preserved session ledger and all invocation reports
must supply cumulative training/evaluation/storage/idle/failure time for the final research report.
CSV/report presentation and independent saved-output download verification remain separate finalization steps.
