# M1-B evidence: native low-rank rank and allocation

Checked: 2026-09-25. Scope is bounded to the core StructMoE line and three adjacent follow-ups (Spectron, MoARa, SD-MoE). This is a targeted prior-work check, not an exhaustive novelty search. Source documents `plans/plan.md` and `experiment_plan.md` matched the hashes supplied in the task.

## Classification that matters for this project

| Rank or sharing term | What is restricted or shared | Does it change learned model capacity? | Relevance here |
|---|---|---|---|
| Native weight-factorization rank | A learned model matrix is parameterized as a product such as `A B`; its representable matrix rank is capped by the factor width `r` from initialization | Yes | This is the Low-Rank-MoE routed expert rank |
| Gradient-projection rank (GaLore and MoARa) | The optimizer projects a full-size matrix gradient into a low-rank subspace to reduce optimizer-state memory; the model weight itself stays full-size | No architectural weight-rank cap | Relevant to memory-efficient optimization, not evidence that a routed expert has rank `r` |
| Post-training compression / low-rank residualization (e.g. LightMoE) | A trained expert is replaced or approximated after training using a common/base component and low-rank residuals | Changes the deployed compressed representation after training | Related to compression and shared parameterization; not native low-rank pretraining |
| Shared FFN | A separate dense nonlinear FFN is evaluated for every token alongside routed outputs | Yes; it adds always-active model parameters and computation | This is the project's independent full-rank shared SwiGLU; none of the checked StructMoE variants has this same path |
| Shared spectral component (SD-MoE) | Expert matrices are decomposed into common low-rank spectral directions and expert-specific orthogonal complements; gradient updates are split consistently | It changes expert parameterization/training structure, but its common subspace is not a separate always-on FFN | A competing explanation for allocating common versus expert-specific directions; not equivalent to the proposed shared FFN/routed factor-rank budget |

## Core: StructMoE has two distinct paper versions

### A. `Low Rank Experts Enable Specialization in Dense Transformers` (StructMoE)

- **Version/source:** anonymous ICLR 2026 under-review PDF, OpenReview file `7eda692ef4f9549b74c2de9283097ae444af2fd6.pdf`; source: https://openreview.net/pdf/7eda692ef4f9549b74c2de9283097ae444af2fd6.pdf . No public revision identifier is printed in the PDF. Do not cite it as an accepted ICLR paper.
- **Exact relevant locations:** §3.1–3.5 for formulation and accounting; §4.1–4.5 for runs; Table 1, Table 2, Table 3; Appendix B.1, Figure 5 for rank/count allocation. PDF pages 3–6 and 12–13.
- **Architecture / rank meaning:** a dense MLP remains the base path. A per-token router chooses top-k low-rank matrix contributions `A_l B_l`, summed into the dense up-projection *before* its nonlinearity; the result then uses the base down-projection. Thus the low-rank path alters the weight action before activation. It is not a bank of independent nonlinear SwiGLU outputs added after activation, and it is not a separate always-active shared FFN.
- **Training:** from-scratch decoder-only Transformer training is reported, with 0.9B and 1.6B models and 345B / 400B tokens respectively (Table 1). Main setting uses `L=16` LoREs, factor rank `r=16`, top-k `k=1`; training settings say identical data and optimization settings for parameter-matched dense comparisons (§4.3–4.4). The paper reports 32 A100 40GB GPUs, ZeRO-1 and data parallelism (§4.3).
- **Controls and allocation:** Eq. 7–8 / Table 1 shrink dense FFN width from 7168 to 6618 to pay for router and LoRE parameters, matching total parameter count. Table 3 gives modeled/achieved compute and wall time: fewer estimated/achieved TFLOPs but about 5.8–6.5% slower step time due to routing, gather/scatter and grouped-GEMM overhead. Appendix B.1 holds `L*r=256` and total parameters constant across `(r,L)=(1,256),(8,32),(16,16),(32,8)`; Figure 5 favors `(16,16)` in its small-scale sweep. This is direct evidence that native routed low-rank **rank/count allocation** and parameter-matched native training already exist.
- **What it does not test:** no independent dense shared FFN whose width trades off against routed factor rank; the dense MLP is a shared base that receives routed pre-activation low-rank contributions. It does not establish the effect of the project's different nonlinear-path decomposition.
- **Official code status / divergence:** §8 Reproducibility, PDF p. 10, says code will be released upon acceptance. No official repository link is included in this under-review PDF, and the claimed implementation could not be checked. Therefore no code-paper divergence was verifiable for this version.
- **Evidence confidence:** paper formulation, parameter-matching arithmetic, rank/count ablation, training and runtime reporting checked in the full PDF. Kernel behavior/results are author-reported; no independent run or code audit.

### B. `StructMoE: Structured Mixture of Experts Using Low Rank Experts` (Sarwar et al., 2024)

- **Version/source:** PMLR 262, pp. 182–193, NeurIPS Efficient Natural Language and Speech Processing Workshop (2024): https://proceedings.mlr.press/v262/sarwar24a.html . Full paper: https://raw.githubusercontent.com/mlresearch/v262/main/assets/sarwar24a/sarwar24a.pdf .
- **Exact relevant locations:** §3 method and experimental setup/results; Appendix and paper footnote for code. See PMLR page and publisher-hosted PDF.
- **Distinction from A:** this is not the anonymous 2026 dense-Transformer StructMoE manuscript. The 2024 formulation augments each ordinary MoE expert with a secondary router that selects low-rank experts per token/per expert, entangled inside the expert's up-projection before activation. It is nested inside an existing sparse MoE, rather than a routed low-rank branch inside a single dense MLP.
- **Training/controls:** the PMLR abstract describes held-out validation loss against a parameter-matched MoE baseline. Detailed quantitative controls were not re-extracted in this bounded pass, so do not transfer the 2026 manuscript's setup or results to this paper.
- **Official code:** the publisher-hosted paper footnote links https://github.com/zainsarwar865/StructuredMoE . Repository contents/commit were not inspected in this pass; code-paper divergence is unknown.
- **Evidence confidence:** bibliographic identity, abstract, and code link checked against publisher-hosted sources; full experimental details and code behavior remain unchecked.

## Follow-ups (three)

### 1. Spectron: `Stabilizing Native Low-Rank LLM Pretraining`

- **Version/source:** arXiv:2602.12429v2, 2026-07-15; current HTML: https://arxiv.org/html/2602.12429 ; official authors' code: https://github.com/Pauljanson002/spectron .
- **Exact relevant locations:** §3–5, §6 compute-optimal allocation, Appendix B.2 rank-ratio ablation, Appendix B.4 FFN-only factorization; Algorithm 1. The repository README documents low-rank factorization and run options.
- **What rank means / training:** native weight factorization from scratch for non-embedding matrices. Spectron changes factor updates using orthogonalization and spectral renormalization; it is not a gradient projection rank. Paper compares native low-rank optimizer variants (including naive AdamW and self-guided), dense training, and compute-oriented configurations on FineWeb. It identifies weight-update spectral instability as an optimization failure mode, so low-rank performance cannot automatically be attributed to representational capacity.
- **Controls/limits:** model/compute and rank-ratio sweeps are in §5–6 and Appendix B.2; Appendix B.4 explicitly tests fully connected layers only. This is dense Transformer pretraining, not sparse routing or shared-FFN allocation. No LoRE or shared-vs-routed allocation is tested.
- **Code-paper check:** official repository exists and README describes dense, low-rank and self-guided modes. Exact reproduction of each paper table/config from repository commits was not audited. Paper version v2 and current `main` can drift; no divergence claim is made.

### 2. MoARa: `Module-Aware Rank Allocation and Structure-Preserving Decomposition for Low-Rank LLM Pre-training`

- **Version/source:** arXiv:2609.15037v1, 2026-09-14: https://arxiv.org/html/2609.15037 . The paper identifies reference optimizer implementations but a new MoARa code repository was not located in the paper's checked text; official implementation status is unverified.
- **Exact relevant locations:** §3.2 (projection-rank reallocation), §4.1–4.3, §5, Appendix C (allocation procedure) and D (config/training); Table 2 for cross-architecture experiments.
- **What rank means / training:** low-rank **gradient projection** rank allocation across modules, preserving full-size model weights. A short profiling run measures gradient-subspace alignment; a static, budget-preserving rank reallocation shifts projection dimensions among transformer modules. This is optimizer memory/convergence allocation, not reducing native FFN weight rank or reallocating model parameters from routed experts to a shared FFN.
- **Controls/results:** tests GaLore and five other low-rank optimizer families, Llama/Qwen/DeepSeek architectures from 300M to 7B, with a three-seed 1B benchmark table; reports steps/wall-clock to a target quality, not only equal-step quality. Allocation rule transfer across architectures is reported without target-architecture profiling (§4.2). This is a useful rank-*allocation* precedent whose manipulated resource and estimand differ from this study.
- **Code-paper divergence / caveat:** paper lists upstream reference implementations for baseline optimizers, not a confirmed MoARa repository. The paper is a new v1 preprint; results are author-reported and not independently reproduced.

### 3. SD-MoE: `Spectral Decomposition for Effective Expert Specialization`

- **Version/source:** arXiv:2602.12556, paper HTML: https://arxiv.org/html/2602.12556 . No official code link was identified in the checked paper text; implementation status unknown.
- **Exact relevant locations:** §2 parameter/gradient spectral analysis, §3.1 decomposition and update, §4 results, Table 2 / Appendix F common-rank sensitivity.
- **What rank means / training:** a common **weight spectral subspace** (e.g. tested common rank `k∈{2,4,8,16,32}`) is separated from expert-specific orthogonal complements, with gradients decomposed and routed to corresponding shared/unique components. This is a parameter-sharing/decomposition rule inside existing MoE experts, not a native factorized rank cap applied to each independent expert and not an independent always-on dense FFN.
- **Controls/results:** analyzes Qwen/DeepSeek MoEs and reports roughly 5% training throughput overhead; common-rank sensitivity is a downstream task comparison in Table 2, with the paper reporting a peak at k=8 and <0.6% range over tested k. Sections 2–3 include both pretrained-model analyses and trained-from-scratch Qwen baseline/SD-MoE experiments (Appendix figure captions). Thus separate those two evidence sources when citing.
- **Code-paper divergence / caveat:** no implementation was available to compare. The authors' spectral interpretation and reported gains need independent evaluation; common spectral rank is not algebraically interchangeable with factorized expert rank or total shared-FFN parameters.

## Implication for the proposed study

The direct novelty boundary should be phrased narrowly: StructMoE already combines native training, routed low-rank weight contributions, dense shared computation and fixed-total-parameter controls, and its 2026 under-review draft directly sweeps rank against LoRE count under a fixed LoRE/parameter budget. The potentially distinct question in `experiment_plan.md` is whether a **separate nonlinear, full-rank shared SwiGLU width** conditions the quality response to **native factorization rank of routed SwiGLUs** under a fixed total parameter budget, with an equal-budget dense-width control. The checked StructMoE source does not test that same interaction. This supports a bounded distinction, not a first-ever claim: literature search is not exhaustive, and the 2026 StructMoE result remains an unreviewed preprint without audited code.

Do not describe MoARa as native low-rank expert-weight training: its rank is gradient projection. Do not use Spectron's successful native low-rank optimizer results as evidence that routed low-rank factors will be stable under sparse, uneven expert updates. Do not convert SD-MoE common-subspace rank or LightMoE's compressed residual rank into the project's routed-weight rank or shared-FFN parameter fraction.

## Local materials consulted

- `plans/plan.md` §5 (M1 scope and preliminary literature state; SHA-256 matched task hash).
- `experiment_plan.md` §§3–5 (target distinction, rank definition, model structure; SHA-256 matched task hash).
- `paper_development_plan.md` §5 and §§3–4 (paper claim boundaries, further comparisons).
- `papers/05-lightmoe.md` (post-training compression distinction); `papers/08-loro.md` and `papers/09-spectron.md` (native weight factorization/optimization); `papers/03-comprehensive-scaling-law.md` and `papers/04-slicing-and-dicing.md` (shared capacity and comparator context).

## Unresolved

- The official StructuredMoE repository for the 2024 version was linked but not inspected, so implementation divergence remains open.
- The under-review 2026 StructMoE code is promised after acceptance, not available for this check. Its appendices/results could change in a future version.
- SD-MoE's official implementation location was not established.
- No exhaustive search, citation chasing, or novelty certification was performed; the comparison is limited to the core StructMoE line and three adjacent papers.
