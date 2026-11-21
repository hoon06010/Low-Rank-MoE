# seed401 storage and recovery validation plan

Scope: new seed401 pilot/recovery only. Main remains unapproved. Existing seed17, Stage C and other seeds are never retention targets.

## Local retention boundary

Before generating a run, register its absent output:
`python reports/seed_repeat_c/seed-401/checkpoint_retention.py --output reports/seed_repeat_c/preflight/pilot/seed-401/LR-HL --register-new`

The external enrollment must survive runtime recovery and be downloaded independently alongside evidence. It is not an experiment input. A restored checkpoint with preserved archive mtimes is eligible only when its original enrollment is present; do not reenroll an existing output.

Run the helper at a stopped verified save boundary. Pass an immutable Drive receipt and independent publish readback evidence. The receipt identity must include experiment_id, seed401, cell, phase, output, run_id, source_revision, data_manifest_hash, eval_revision and config_file_sha256. All must agree with the local single run manifest. A receipt merely fabricated on disk is not remote validation. Use actual Drive media-download evidence.

Inspect first with `--receipt RECEIPT.json --evidence publish-evidence.json`; default is dry-run. Explicit `--apply` removes only enrolled seed401 checkpoint files. Latest pointer and one newest additional checkpoint are retained. No remote deletion is performed by this helper. Remote generations remain immutable; choose two verified committed generations operationally without purging remote files until independent archive/checkpoint hashes and fresh-runtime recovery have passed.

The helper reads trusted enrolled local trainer checkpoint payloads using torch.load to confirm seed/cell/run/source/data/eval identity for every file; do not use it on untrusted downloaded pickle files.

The helper rejects symlinks, hardlinks, foreign paths, missing enrollment, unknown checkpoint contents, old timestamps, identity/hash/byte mismatches and a running execution state. Stop the training process while applying retention. Timestamp ordering can be imperfect, so the latest pointer always overrides recency. Whole-file hash verification detects checkpoint changes; enrollment and trusted evidence are operational prerequisites, not cryptographic attestation of when a file originated.

## Space

Historical largest mature checkpoint: HL 508,461,500 bytes. Conservative four-cell estimate with two retained local and remote generations each:

- local retained checkpoints: 4,067,692,000 bytes
- local peak: 7,222,603,620 bytes (one new checkpoint, tar, staged copy, independent download plus 1GiB reserve and 47,323,796-byte package)
- remote peak: 5,649,895,324 bytes (two per cell plus incoming generation and 1GiB reserve)

Add measured data, reports, Python/library installation and runtime baseline separately. Remote quota must be measured in provider bytes, not virtual mount capacity. Storage fixtures can be tiny for API correctness, but transfer time and free-space gates require one mature-sized bounded transfer before GPU pilot. Never allocate a synthetic torch checkpoint and call that model recovery.

Without retention each pilot produces around nine checkpoints (0,5,32,37,64,69,96,101,105), so four pilot files alone approach 14.8GB. This helper is not automatically integrated into the frozen trainer; use a seed401 adapter at verified publish boundaries or stop between approved invocations. An uninterrupted pilot may create several files before its next hook; reserve peak accordingly.

## Exact state restore

Compare immediately after TrainingSession.restore_latest in a new process using recursive exact equality:
- model_state_dict tensors
- optimizer_state_dict (AdamW steps, exp_avg, exp_avg_sq, parameter groups)
- scheduler_state_dict and tokens_seen
- Python random, NumPy RNG arrays, torch_cpu and used_device_rng_states (including used CUDA device)
- data_order and data_order_hash
- epoch, cursor, optimizer_step and trained_target_tokens
- run_id, canonical config/config hash/training hash, source revision, data/tokenizer/eval/protocol identities, purpose and source_kind
- execution_device and scaler_state=None

CPU checkpoint restore stays CPU; the trainer rejects cross-device restoration. A CPU storage fixture proves transport only. GPU model recovery requires GPU same-device-class checkpoints and matching measured libraries.

Use current seed401 real audited data/config for later GPU evidence. Save step0, publish and independently download/hash archive/checkpoint. New process resumes step0 to step1: cursor8, targets4096, epoch0. Terminate runtime. New runtime restores the same verified step1 checkpoint at a fresh output and compares exact state before training. Advance to step2: cursor16, targets8192, epoch0.

For actual continuation equivalence, branch the same verified step1 checkpoint into two fresh processes and compare step2 payload states, excluding freshly generated checkpoint UUID metadata. This catches restoration omissions that a successful status or incremented counter would miss. CUDA exact numerical equivalence may depend on deterministic operations; record any nondeterminism explicitly instead of silently relaxing to tolerance.

## Time

Previous seed17 T4 pilots total training+evaluation 625.62 seconds; setup, network and recovery are excluded. This is an estimate for planning, not seed401 evidence. Remaining approval is 115 minutes, session90 minutes, additional spend0. Measure CPU mature archive transfer and runtime bootstrap, charge setup/idle/failure/reserve, then budget four independent fresh105-step pilots and a fresh-runtime restore. Stop on identity mismatch, storage failure, NaN/OOM or a resource deadline; no automatic retry. Main approval is not implied by passing these gates.

## Executable comparison

Use trusted local trainer artifacts only. SHA verification protects file integrity; it does not make an unknown pickle safe. The comparator requires all 30 keys actually emitted by current TrainingSession._checkpoint_payload; the tests derive the key set from src/train.py AST to catch schema drift.

Create expected-identity.json with seed401, cell, run_id, source_revision, data_manifest_hash and eval_revision from the verified run identity. Supply externally verified hashes for both local files:

```sh
python reports/seed_repeat_c/seed-401/checkpoint_state_compare.py --left SAVED.pt --left-sha256 VERIFIED_SAVED_SHA256 --right RESTORED_STATE.pt --right-sha256 VERIFIED_RESTORED_SHA256 --expected-identity expected-identity.json --mode restored
```

RESTORED_STATE.pt must be captured immediately after actual TrainingSession.restore_latest by torch.save(session._checkpoint_payload(saved_checkpoint_id), file), before taking another training step. Comparing a downloaded file with itself proves byte/state equality only, not runtime restoration. The comparator reports that limitation explicitly.

For continuation equivalence, compare two same-step checkpoints from independently resumed branches using --mode continuation. Only checkpoint_id, latest_verified_checkpoint_id and last_finite_checkpoint_id differ legitimately; every other field, including last_step_diagnostics, must match exactly. Step1 versus step2 is deliberately rejected.

Failure prints the first field path only and no scalar/tensor/token contents. Missing required top-level keys, RNG entries, AdamW moment entries, corrupted data-order hash, foreign identity, wrong input SHA, tensor dtype/shape/value or optimizer/RNG differences fail closed. CPU snapshots and CUDA snapshots must have the same execution_device for equality.

## 2026-10-07 approved execution amendment

The earlier 115-minute remaining allowance and main-unapproved statements are historical. Current human approval permits seed401 recovery/pilots within remaining6,300seconds and conditionally permits main within28,800seconds only after actual preflight evidence passes. Per-session cap5,400seconds, cost0KRW, combined output cap30GiB. Frozen baseline is preserved; a separately hashed direct-Drive adapter/sidecar deployment implements verified boundaries and resource accounting. Do not infer CUDA recovery from archive extraction. Actual step0/step1 runtime-state comparisons and independent step2 continuation comparisons remain mandatory.
