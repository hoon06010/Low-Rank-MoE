# M1-A: StructMoE와 shared capacity 선행연구 증거

- 작성일: 2026-09-25
- 범위: `plans/plan.md` M1 중 (a) StructMoE 버전/구현, (b) shared 용량·계산통제. 본 메모는 전수 novelty 검토가 아니다.
- 입력 해시 확인: `plans/plan.md` = `d10fcb382480d6aecfcb465c9e013bf2c4e02b307dfa7dc9ca6f5e1b2abfc775`; `experiment_plan.md` = `c51e4d958406d952386604fe9056a263033d3c675603626da87da8b7ef2955dd`.
- 비교 기준: 현재 설계는 무작위 초기화부터 native pretraining을 하며, routed expert weight를 직접 rank-factorized하고 별도 shared FFN 출력을 합산한다. 따라서 native weight rank, gradient/update rank, 내부 low-rank branch, 별도 shared FFN은 구별한다.

## 요약 비교

| 연구·확인 버전 | 구조와 rank 종류 | 비교 통제·학습 | 현재 질문과의 관계 | 확인 경계 |
|---|---|---|---|---|
| StructMoE, 익명 ICLR 2026 원고, OpenReview PDF hash ID `7eda692ef4f9549b74c2de9283097ae444af2fd6` | Dense MLP up-projection 안에 token-routed LoRE 은행을 비선형성 전에 더한다. LoRE 각 행렬곱의 rank는 `r`; 이는 routed FFN weight 전체의 native rank 제약이 아니라 dense base에 추가하는 내부 low-rank 경로다. 별도 shared FFN은 없다. | §3.3 식 (7)–(9)은 router/LoRE 파라미터를 위해 base FFN expansion을 줄여 dense 기준과 파라미터를 일치시킨다. FLOPs는 논문 산식상 감소 가능. 표 1은 0.9B/1.6B dense 대조, 같은 모델별 학습 token 양을 제시한다. from-scratch pretraining. | 구조적으로 가장 가까운 routed low-rank 경로 대조지만, 고정 총 예산에서 native routed rank 절약분을 별도 shared FFN에 옮기는 실험은 아니다. | PDF 표지는 “Under review as a conference paper at ICLR 2026”; 정확한 공개 제출/개정 날짜와 저자는 확인되지 않는다. 공식 공개 코드 링크·코드와 이 PDF 버전의 일치도 확인되지 않았다. 원문에서 확인한 학습/구조를 구현 완료로 해석하지 않는다. |
| StructMoE, Sarwar et al., ENLSP-IV 2024 / PMLR 262, 182–193, 2024-12-14 | 메인 expert별 secondary router가 token별 LoRE를 고르고 up-projection 안에서 activation 전 결합한다. native routed expert weight rank가 아니며, 논문 초록에 별도 always-on shared FFN은 없다. | 공식 PMLR 서지 및 PDF 초록 확인. 요약은 “parameter matched MoE baseline”, validation loss. 전체 본문 표/조건과 후속 ICLR 원고 간 실험 변경을 이 세션에서 대조하지 못함. | 이름이 같은 확정 출판 버전이다. ICLR 2026 익명 원고와 같은 실험 버전이라고 가정하면 안 된다. | PMLR PDF가 명시한 코드 URL은 `https://github.com/zainsarwar865/StructuredMoE`. 저장소 접근이 도구에서 실패하여 commit/date, README, 실제 코드, 논문과의 구현 차이는 미확인. |
| DeepSeekMoE, arXiv v1 2024-01-11; ACL 2024 최종본 | FFN intermediate width를 분할하는 fine-grained expert 구성은 expert 내부 작은 폭이지 factorized weight의 native rank가 아니다. 별도 always-active shared FFN 전문가를 갖는다. | §§3.1–3.2: granularity를 높일 때 선택 개수를 함께 늘려 expert 활성 용량/계산을 유지한다. shared 수만큼 routed 활성 선택 수를 줄여 계산을 유지한다. §4.3 Fig. 3: 총 expert 및 활성 expert 수를 일정하게 하고 shared 1/2/4개를 비교, Pile loss 1.808/1.806/1.811. Native pretraining. | 가장 직접적인 shared-expert 격리와 계산 통제의 선행이다. shared count를 시험하지만 shared 폭과 native routed rank 간 재배분은 시험하지 않는다. | 내부 pretraining corpus. 공식 코드 링크 존재(저장소의 이번 시점 commit/실험 재현 여부는 별도 미확인). |
| Towards a Comprehensive Scaling Law of MoE, arXiv:2509.23678 v1 2025-09-28 | 일반 MoE의 total/active parameter, 활성 전문가 수 `G`, 활성 전문가 내 shared count 비율 `S`를 다룬다. weight rank factorization은 없음. 별도 shared FFN/expert를 구조 변수로 다룸. | §§3–5 및 Appendix L: 446 controlled experiments. 실험 범위 `N=133M–3.4B`, `D=10–50B tokens`; validation은 최대 9B/100B. 실험 그룹은 주요 변수 한 축의 marginal effect를 볼 때 다른 요인을 고정한다. Native pretraining. | shared 개수와 활성 규모의 상호 조건성을 다루는 넓은 비교군이다. `S`는 expert 수 비율이지 총 파라미터 중 shared 비율이나 shared 폭이 아니다. native low-rank rank와 shared 용량의 결합은 없다. | 이 논문에서 독립 공개된 공식 실험 코드 링크는 로컬 노트 및 확인한 본문에서 찾지 못함. 노트는 공개 프레임워크 사용과 실험 코드 공개를 구분한다. |
| Slicing and Dicing: Configuring Optimal Mixtures of Experts, arXiv:2605.11689 v1 2026-05-12 | Dense/generalist(shared) FFN과 routed FFN의 크기·선택량을 다루나 native low-rank weight는 없음. generalist는 routed MoE와 병렬의 별도 always-active 경로다. | §§2.1, 3.2: dense 대비 active FFN parameter/FLOPs를 `g_gen + Σ k_i g_i = 1`로 맞추고 generalist 크기 `1/2, 1/4, 1/8` 및 여러 expert 구성을 평가. 2,000회 이상 pretraining, 10M–6.6B total scale. 공식 repo 링크는 arXiv 본문이 제공한다. | 공통 경로의 이득이 routed 용량 감소를 보상하지 않을 수 있다는 반대 결과 및 active-budget 대조다. 저자들은 generalist 추가가 전반적으로 no-generalist와 비슷하거나 약간 나쁘다고 보고한다. 특정 구조·예산 결과를 native low-rank 조건에 일반화할 수 없다. | 공식 arXiv HTML §§2.1, 3.2를 직접 확인. repo commit 및 코드-논문 차이는 확인하지 않았다. |

## StructMoE 버전 판별과 근거

### 익명 ICLR 2026 원고 (`7eda...`)

- 공식 PDF: https://openreview.net/pdf/7eda692ef4f9549b74c2de9283097ae444af2fd6.pdf
- PDF 표지: “Under review as a conference paper at ICLR 2026”; 본문 제목은 “LOW RANK EXPERTS ENABLE SPECIALIZATION IN DENSE TRANSFORMERS”. 공개 날짜/버전 문자열은 PDF에서 확인되지 않아 날짜를 붙이지 않는다.
- §3.1–3.2 및 Fig. 1: token router가 LoRE 은행에서 고정 top-k를 골라 `x W_up + Σ s_l(x) x A_l B_l`을 activation 전에 구성하고 표준 down-projection에 넣는다. 각 `A_l B_l`의 rank는 `r`. 이는 dense base up-projection의 token-conditioned update다.
- §3.3 식 (7)–(9): `P_StructMoE = 2 H D'_ff + Lr(H + D'_ff) + HL`; dense와 parameter count를 맞추기 위해 `D'_ff`를 낮춘다. FLOPs 식은 dense보다 작을 수 있음을 보인다. 따라서 parameter matching을 compute matching으로 바꿔 부르면 안 된다.
- Table 1: dense 및 StructMoE의 0.9B와 1.6B 설정을 제시하고 각각 학습 tokens를 345B, 400B로 보고한다. Table 3은 parameter-matched dense/StructMoE의 runtime와 throughput을 제시한다. §5 계열 및 본문은 from-scratch LM pretraining 설정이다.
- §3.5는 full MoE와 동등하지 않고, dense expansion 일부를 routed low-rank 조정으로 바꾼 구조임을 명시한다. 별도 shared FFN과 native routed expert weight rank의 비교는 아니다.
- 구현: 이 PDF에서 확인 가능한 저장소/commit 링크를 확보하지 못했다. 2024 PMLR PDF가 연결하는 공개 repo를 이 2026 원고의 구현이라고 동일시하지 않는다.

### 2024 출판 버전과 제출 기록

- 공식 PMLR 서지: https://proceedings.mlr.press/v262/sarwar24a.html — Sarwar et al., *StructMoE: Structured Mixture of Experts Using Low Rank Experts*, NeurIPS Efficient Natural Language and Speech Processing Workshop, PMLR 262:182–193, 2024-12-14. PMLR PDF 파일명/본문 표제는 “StructMoE: Augmenting MoEs with Hierarchically Routed Low Rank Experts”로 다르다. 이는 서지 표제 차이를 기록한 것이며 별도 실험 버전이 확정됐다는 뜻은 아니다.
- PMLR 페이지는 논문 PDF와 `https://github.com/zainsarwar865/StructuredMoE`를 코드 URL로 명시한다. GitHub 본문과 파일/commit을 열지 못했으므로 실제 코드의 date, scope, paper-code mismatch는 미검증이다.
- OpenReview 검색 결과에는 같은 “Augmenting MoEs…” 제목의 anonymized ICLR 2025 submission (submitted 2024-09-25, modified 2024-10-16, withdrawn)과 anonymized 2024 NeurIPS submission PDF가 따로 나타난다. 2024 workshop 최종 PDF는 실명 저자와 확대된 저자 목록을 싣는다. 이 제출/출판 기록을 ICLR 2026의 `7eda...` 원고와 혼합하지 않는다.
- 별도의 OpenReview PDF `https://openreview.net/pdf?id=QSoc7HGc6Q`는 검색 결과상 “LOW RANK EXPERTS ENABLE SPECIALIZATION IN DENSE TRANSFORMERS”, 익명 ICLR 2026 원고로 표시되고, `7eda...` PDF와 구조·산식이 겹친다. 다만 OpenReview 페이지가 브라우저 challenge를 반환했고 제출 메타데이터/개정 날짜/두 PDF의 byte-level·실험 차이는 확인되지 않았다. 현재는 동일 원고의 다른 revision인지 별도 record인지 미결로 남긴다.

## 세부 해석

### 총 파라미터, 활성 파라미터, FLOPs

DeepSeekMoE와 Slicing and Dicing은 shared 경로를 늘릴 때 routed activation을 줄이는 계산량 통제를 명시한다. Comprehensive Scaling Law는 활성 expert 수 및 shared-count ratio를 통제 변수로 연구한다. StructMoE ICLR 원고는 dense base width를 줄여 total parameter를 맞추되, 그 식의 FLOPs는 오히려 작아질 수 있다. 서로 다른 통제축이므로 표에서 “같은 total”, “같은 active”, “같은 FLOPs”, “같은 학습 tokens”를 한 항목으로 합치면 안 된다.

### 현재 설계가 추가하는 비교

현재 연구의 별도 shared FFN 출력 합산과 native factorized routed weights 조합은 위 연구들의 한 실험 안에 확인되지 않는다. 이는 조사한 범위의 비교표 결론이지 최초성 확정이 아니다. 본 연구의 `I_LR`, `G_low`, `Δ_budget`, `Δ`는 shared expansion에 따른 상대 격차, 실제 loss 변화, 고정예산 재배분 품질 및 dense-width 비교를 서로 나눠 판정해야 한다. StructMoE를 후속 구조 대조로 삼을 때는 version, total parameter, active computation/FLOPs, nonlinear path 위치 및 rank/branch 예산을 별도 명세해야 한다.

## 출처 및 확인 위치

1. Sarwar et al., PMLR 2024 proceedings page and paper PDF: https://proceedings.mlr.press/v262/sarwar24a.html ; https://raw.githubusercontent.com/mlresearch/v262/main/assets/sarwar24a/sarwar24a.pdf (bibliography, abstract, paper-linked code URL; source title differs from proceedings metadata title).
2. Anonymous ICLR 2026 paper, OpenReview PDF ID `7eda692ef4f9549b74c2de9283097ae444af2fd6`: https://openreview.net/pdf/7eda692ef4f9549b74c2de9283097ae444af2fd6.pdf (cover; §§3.1–3.5; Fig. 1; Eq. 7–9; Tables 1, 3; Appendix/reproducibility statements). The separate PDF ID `QSoc7HGc6Q` was search-readable but OpenReview open returned a browser challenge; no equivalence claim is made.
3. Dai et al., DeepSeekMoE, arXiv v1 and ACL 2024: https://arxiv.org/abs/2401.06066 ; https://aclanthology.org/2024.acl-long.70.pdf (arXiv §§3.1–3.2, §4.3 / Fig. 3; ACL §§3.1–3.2). Official repository named in local bibliographic note: https://github.com/deepseek-ai/DeepSeek-MoE.
4. Zhao et al., *Towards a Comprehensive Scaling Law of Mixture-of-Experts*, arXiv v1: https://arxiv.org/abs/2509.23678 ; HTML: https://arxiv.org/html/2509.23678 (§§1, 3–5; Appendix L).
5. Li et al., *Slicing and Dicing: Configuring Optimal Mixtures of Experts*, arXiv:2605.11689v1 (2026-05-12): https://arxiv.org/abs/2605.11689 ; HTML: https://arxiv.org/html/2605.11689v1 (§§2.1, 3.1–3.2; the paper states that code and data are released at https://github.com/hadasah/slicing_and_dicing).
6. Local context inspected: `plans/plan.md` M1; `experiment_plan.md` §§3–5; `paper_development_plan.md` §5; `papers/01-deepseekmoe.md`, `papers/03-comprehensive-scaling-law.md`, `papers/04-slicing-and-dicing.md`.

## 미해결 확인

- 2024 PMLR StructMoE official repository could not be inspected from this environment. The publication itself supplies the URL, but its commit, code-date, training commands/configs, and code-paper differences remain unknown.
- `7eda...` and `QSoc...` OpenReview records may be successive versions of the same anonymous ICLR 2026 manuscript. The former PDF was fully inspected; the latter was only returned via search snippets and its page hit an access challenge. Compare their OpenReview revision metadata and downloaded PDFs before citing an exact revision chronology.
- Exact submission/version date for the inspected `7eda...` manuscript was not available in the PDF. Only the ICLR 2026 review label is verified.
- The `Slicing and Dicing` official repository snapshot/commit and correspondence to the paper's code/data were not inspected. Its paper-grounded §2.1/§3.2 statements were verified directly in arXiv HTML.
- This search was bounded to the assigned core and three close follow-ups. It is not a full novelty or priority search.
