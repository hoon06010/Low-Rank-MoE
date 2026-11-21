# Conditional manuscript outline

Status: preparation outline only. There are no measured Low-Rank-MoE losses, throughput figures, or completed large training runs in this bundle. Claims and figures below are contingent on future runs and the gates in `experiment_plan.md` and `paper_development_plan.md`.

## Main thread

1. **Allocation curve:** At a fixed whole-model parameter budget, compare how native routed SwiGLU rank and separate dense shared SwiGLU width are allocated. Start with LR/DW HS, HL, LS, LL, M and SHARED_ONLY; only HS/LL/M are fixed-total points. Pair LR and DW at each cell for total and active parameter control. Report measured loss with independent seed uncertainty rather than inferring effects from parameter algebra.
2. **Alternative explanations:** Distinguish relative complementarity (`I_LR`, `J`) from actual low-rank improvement (`G_low`) and fixed-budget quality (`Δ_budget`). Test active computation, optimization/recipe stability, routing dependence, and structural controls only with their own schedules and budgets. A positive interaction alone does not demonstrate better loss or a shared/common-knowledge mechanism.
3. **Conditional selection rule:** If replicated allocation and compute curves support it, specify a small-search allocation rule using held-out scale/budget/data settings, separate selection and final-evaluation seeds, and report regret and total search cost. If transfer fails, retain the conditional empirical curve and do not claim a general rule.

## Sections and evidence gates

| Section | Content | Evidence needed before writing findings |
|---|---|---|
| Introduction | Fixed-budget allocation question and distinction from routed low-rank deltas, gradient projection rank, shared spectral subspaces | `literature/evidence_ledger.csv`, `literature/comparison_matrix.md`, exact primary source versions |
| Method | LR/DW/shared-only parameterization, top-2 router, shared output, total/active accounting, data split, token-weighted CE, run identity | Verified M2–M5 implementation and run manifests; measured FLOPs only if actually profiled |
| Primary comparisons | Original losses for all eight core cells, `I_LR`, `I_DW`, `J`, `G_low`, `Δ_budget`, `Δ` per seed with paired intervals and 0.01 margins kept distinct | Complete measured seed blocks with matching protocol/data/tokenizer/eval/training endpoint; no synthetic fixtures |
| Efficiency and explanations | Same-token and separately trained same-compute results, measured tokens/s and peak memory, stable common recipe eight-cell rerun, routing perturbation and balanced-routing retraining, dense and StructMoE-style controls | Protocols and additional authorized runs; current preparation has none |
| Rank response and transfer | Fixed-shared rank grid, prespecified high-rank reference/epsilon, uncertainty and independent validation; candidate rule and held-out scale/data | Additional authorized rank/transfer experiments; multiplicity/selection accounted for |
| Discussion and limitations | Tested range, confounding, three-seed precision, failed conditions, architecture and source-version boundaries | All attempted runs including failures and null results |

## Six proposed figures and required data

1. **Fixed-total allocation curve:** `protocol_id`, `config_id`, `family`, `cell`, `rank`, `shared_width`, `whole_model_total`, `whole_model_active`, `seed`, `loss_nats`, `trained_target_tokens`, `eval_target_tokens`, `source_kind`; measured repeated middle and boundary points are required for a curve claim. SHARED_ONLY is a separate control.
2. **Compute/cost versus loss:** all figure-1 keys plus actually measured training FLOPs or an explicitly calibrated compute model, tokens/s, optimizer/evaluation/storage time, GPU SKU/rate/tax/reserve; same-compute runs must start from scratch with endpoint-specific schedules.
3. **Eight-cell original losses and contrasts:** eight original loss columns per seed plus `I_LR`, `I_DW`, `J`, `G_low`, `Δ_budget`, `Δ`, paired mean/SD/CI and missing-block reasons; all points use the same protocol and endpoint. Synthetic counterexamples stay under `tests/fixtures/` or `results/preflight/`.
4. **Shared by rank response surface:** shared width, routed rank, whole/active budget, recipe, trained/eval tokens, loss and CI, high-rank reference ID, prespecified epsilon, new validation seeds or multiplicity rule, compute increase; fixed-budget diagonal alone is insufficient.
5. **Routing/optimizer/transfer controls:** matched gate weights, load, active compute, perturbation or balanced-routing retraining flag, recipe, architecture/source version, scale/data condition, loss and throughput. Plot distinct interventions separately.
6. **Held-out allocation selection:** development settings, held-out scale/budget/data, candidate set, rule-selected configuration, fixed-share/equal/random baselines, selection seed, evaluation seed, observed candidate-best loss, regret, profiling/tuning/training/transfer cost. Omit if the rule is unsupported.

No Results or Abstract performance sentence should be written until the corresponding measured run manifest and test-set release conditions are checked.
