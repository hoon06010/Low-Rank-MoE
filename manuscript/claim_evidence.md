# Claim-to-evidence gates

> **2026-10-03 현재 상태:** Stage C는 2026-10-01 Kaggle에서 네 조건(HL→HS→LS→LL), seed 17, 조건당 2,048 steps / 8,388,608 targets로 완료했다. 최종 Successful #13 아카이브와 네 체크포인트는 독립 다운로드 후 SHA-256 일치를 검증했다. 보수적 사용 시간은 384.00/480분이다. 독립 seed·DW 대조·동일 FLOPs 확증은 미완료다. C는 단일 seed 탐색이며, 아래 날짜별 준비·게이트 기록은 당시 판단이다. 최신 근거: [C 완료 보고서](../reports/c_execution_2026-10-01.md). 이번 갱신은 로컬 완료 기록을 기준으로 하며 외부 서비스는 재조회하지 않았다.

Current status: proposed claims, not findings. `results/preflight/` and synthetic fixtures cannot substantiate performance or causal conclusions.

| Candidate claim | Minimum evidence | Current status |
|---|---|---|
| Shared expansion relatively compensates low routed rank | Complete same-seed 4-cell LR blocks; `I_LR` paired CI lower bound > 0; original losses shown | Unmeasured |
| Compensation differs from ordinary width reduction | Complete 8-cell LR/DW blocks; both `I_LR` and `J` paired CI lower bounds > 0; family recipes shown; common-recipe 8-cell rerun for robustness under that prespecified recipe, without generalizing across optimizers | Unmeasured |
| Low-rank loss actually improves at larger shared width | `G_low = L_LR,LS - L_LR,LL` CI lower bound > 0, separate from `I_LR` | Unmeasured |
| Fixed-total HS→LL reallocation maintains quality | `Δ_budget = L_LR,LL - L_LR,HS` paired CI upper bound < prespecified `ε_budget=0.01` nats/target | Unmeasured |
| LR-LL is noninferior to DW-LL | `Δ = L_LR,LL - L_DW,LL` paired CI upper bound < separate `ε_DW=0.01`; does not replace the HS→LL test | Unmeasured |
| Input-dependent routing helps beyond a path substitute | Controlled routing perturbation plus HS/LL input-independent balanced-routing retraining with gate weight, load and compute checks | Unmeasured; perturbation alone is insufficient |
| A minimum tested routed rank decreases with shared width | Fixed-shared rank grids, prespecified higher-rank reference and ε, uncertainty, new seeds or multiplicity control, compute cost | Unmeasured; fixed-total diagonal cannot establish it |
| An allocation rule transfers to new settings | Held-out setting-level scale/budget/data tests, separated selection/evaluation seeds, fixed/equal/random baselines, regret and full exploration cost | Unmeasured; conditional future claim |

Established preparation facts: the 11-condition ledger algebra and actual model parameter counts agree in the tested implementation; synthetic correctness tests cover routing, split isolation, resume, paired contrasts and cost arithmetic. The final local suite passed 83 tests (`reports/test_full.log`). These facts validate preparation tools, not any loss advantage. Bounded literature evidence shows the 2026 under-review StructMoE already includes native routed low-rank components and a fixed-budget rank/count sweep; the manuscript must not claim that broad idea as first. See `literature/comparison_matrix.md` for structure and budget distinctions.
