# 연구질문 타당성·미해결성 검토

> **2026-10-03 현재 상태:** Stage C는 2026-10-01 Kaggle에서 네 조건(HL→HS→LS→LL), seed 17, 조건당 2,048 steps / 8,388,608 targets로 완료했다. 최종 Successful #13 아카이브와 네 체크포인트는 독립 다운로드 후 SHA-256 일치를 검증했다. 보수적 사용 시간은 384.00/480분이다. 독립 seed·DW 대조·동일 FLOPs 확증은 미완료다. C는 단일 seed 탐색이며, 아래 날짜별 준비·게이트 기록은 당시 판단이다. 최신 근거: [C 완료 보고서](../reports/c_execution_2026-10-01.md). 이번 갱신은 로컬 완료 기록을 기준으로 하며 외부 서비스는 재조회하지 않았다.

검토일: 2026-09-29. 사용자 질문에 대한 문헌 기반 평가이며 가설의 통계적 검증이나 연구계획 변경이 아니다.

## 판단

현재 질문은 타당하고 반증 가능하다. 다만 broad architecture novelty는 약하고, 좁게 정의한 상호작용·대조 실험에는 미해결 가능성이 남는다. “아무도 검증하지 않았다”거나 최초성을 확정할 수 없다.

평가 대상은 research_plan.md의 고정 총예산 배분 질문과 그 후속 experiment_plan.md의 H1–H4다. 핵심 질문은 다음과 같다.

> 처음부터 학습하는 언어 MoE에서 별도의 비선형 shared FFN 용량 확대는 native low-rank routed FFN의 rank 축소 손실을 상대적으로 완화하는가? 그 효과는 예산을 맞춘 dense-width 축소에서도 나타나는 일반적인 용량 효과와 다른가?

고정 총예산에서 품질을 유지하는 배분이 있는지, 그 배분에서 입력별 routing이 유용한지, 계산 비용까지 유리한지는 각각 별도 질문이다.

## 검토한 주요 문헌

| 선행연구 | 이미 다룬 내용 | 현재 질문과의 경계 |
|---|---|---|
| [DeepSeekMoE, 2024](https://arxiv.org/html/2401.06066v1) | Shared expert 격리와 fine-grained routed experts | Shared/routed 분리 자체는 새롭지 않다. 작은 expert 폭은 native weight factorization과 다르다. |
| [StructMoE, PMLR 2024](https://proceedings.mlr.press/v262/sarwar24a.html) | Expert 안에 secondary routing으로 low-rank 성분을 activation 전에 결합 | Low-rank experts와 routing 결합의 선행. 별도 shared FFN 출력의 합산과 구별된다. |
| [Low Rank Experts Enable Specialization in Dense Transformers, ICLR 2026 표기 원고](https://openreview.net/pdf/7eda692ef4f9549b74c2de9283097ae444af2fd6.pdf) | Dense up-projection에 token-routed low-rank 성분을 더하고 base expansion을 줄여 파라미터 예산 일치 | Native routed low-rank와 dense 용량 재배분의 넓은 최초성 주장을 배제해야 한다. 오늘은 검색에 색인된 원문 초록을 확인했으며 직접 PDF 열기는 인증 화면으로 막혔다. 기존 evidence_native_rank.md의 부록 B.1 rank/count sweep 기록은 이번에 직접 재검증하지 못했다. 게재 상태도 새로 확정하지 않았다. |
| [Towards Greater Leverage, 2025](https://arxiv.org/html/2507.17702) | §4.1.3에서 모델 크기·활성 비율·활성 expert 총수를 고정해 shared 비율 변경 | 최적 shared 비율을 묻는 넓은 질문에는 선행이 있다. Native routed weight rank는 해당 축이 아니다. |
| [Towards a Comprehensive Scaling Law of MoE, 2025](https://arxiv.org/html/2509.23678) | 총/활성 파라미터, 활성 expert 수, shared 비율의 scaling 분석 | Shared 비율의 실험적 선행. 활성 expert 수 비율을 우리 총 파라미터 배분 비율과 동일시하지 않는다. |
| [Slicing and Dicing, 2026](https://arxiv.org/html/2605.11689v1) | 활성 FFN 예산을 고려한 generalist 추가가 조사 조건에서 동등하거나 더 나쁨 | Shared 확대가 자동으로 유리하다는 가정을 반박한다. Native low-rank 제약 아래의 상대적 보완은 별도다. |
| [MLoRE, 2024](https://arxiv.org/html/2403.17749v1) | 시각 다중작업의 일반 convolution 공통 경로와 low-rank expert; 부록 B.2의 공통 경로 rank 비교 | Full-capacity common path와 low-rank expert 결합의 선행. 언어 native pretraining의 고정예산 rank/shared 상호작용 검증과 다르다. |
| [LightMoE, 2026](https://arxiv.org/html/2603.12645v1) | 학습된 expert를 shared base와 개별 low-rank adapter로 압축 | 행렬 내부의 base 공유이며 별도 always-active nonlinear shared FFN과 다르다. 사후 압축 결과를 native training 근거로 옮길 수 없다. |
| [Spectron, 2026, 확인된 HTML v2](https://arxiv.org/html/2602.12429) | Native low-rank 사전학습에서 weight-update spectral norm과 안정성, 계산 배분 | 낮은 rank의 실패가 표현력 한계인지 최적화 문제인지 분리해야 한다는 근거. MoE shared/routed 상호작용의 직접 증거는 아니다. |
| [SD-MoE, 2026](https://arxiv.org/html/2602.12556) | 공통 low-rank spectral subspace와 expert-specific complement 분해 | 공통/전문 성분의 rank 배치에 관한 경쟁 설명. 별도 shared FFN과 expert 전체의 native rank 제한과는 다르다. |
| [LatentMoE, 2026](https://arxiv.org/html/2601.18089v1) | Routed 경로는 공통 projection을 거친 저차원 공간에서, shared experts는 원래 차원에서 계산. 절약분으로 N과 K를 확대 | 기존 비교표에 없던 중요한 인접 연구. “좁은 routed + 원래 차원 shared” 결합 자체의 차별성을 약화한다. 각 routed gate/up/down의 독립 factor rank를 변경하고 N/K를 고정해 절약분을 shared 폭으로 옮기는 본 실험과는 다르다. |

## 왜 질문이 타당한가

- Shared와 routed에 한정된 용량을 어떻게 나눌지는 품질·활성 계산·모델 저장량에 영향을 미치는 실질적인 설계 문제다.
- 공통 경로가 저랭크 경로의 부담을 덜 수 있다는 가설은 개연성이 있지만 필연적이지 않다. Shared 확대가 routed를 대체하거나, rank를 줄여 잃는 표현력을 복구하지 못하거나, 단지 계산량 증가 때문에 좋아질 수 있다.
- H1/H2에는 부호가 반대이거나 차이가 없는 결과도 존재하므로 반증 가능하다. Positive result를 전제하지 않는 현재 설계가 적절하다.
- 그러나 조건을 많이 붙여 기존 논문과 다른 조합을 만들었다는 사실만으로 학술적 가치가 보장되지는 않는다. 폭 축소 대조, routing 대조, 계산 비교로 경쟁 설명을 실제로 구분해야 한다.

## 아직 직접 답을 확인하지 못한 부분

확인한 주요 문헌에서는 별도 nonlinear shared FFN 폭 × routed SwiGLU의 native factor rank 상호작용을, 같은 셀별 예산의 narrow dense routed 대조와 함께 검증한 직접 실험을 찾지 못했다. 이는 제한된 검색의 결과이며 부재의 증명이 아니다.

H1만 양수이면 shared가 작은 경로를 돕는 일반적인 현상일 수 있다. H2의 LR-vs-DW 상호작용 차이가 핵심 차별성이다. 입력별 routing의 역할이 사라진다면 conditional specialization을 보존했다는 설명도 약해진다.

“필요한 최소 rank가 줄었다”는 표현은 두 rank의 상호작용만으로 뒷받침되지 않는다. 고정 shared별 rank grid, 기준 모델, 사전에 정한 품질 허용오차와 불확실성 평가가 추가로 필요하다. 이 또한 모델/데이터/학습법에 조건부인 실험적 최소 rank이며 보편적 필요 rank는 아니다.

## 현재 프로젝트 증거

reports/colab_free_mini/experiment_summary.md에는 단일 seed 17, 조건당 3,145,728 학습 targets의 4-cell 탐색이 기록되어 있다. I_LR=0.009041 nats, 고정 총파라미터 HS→LL의 loss 차이=-0.036808 nats다. 방향상 후속 연구 동기는 있으나 신뢰구간, DW 대조, 반복 seed, 동일 학습 FLOPs 비교가 없어 H1/H2 확정이나 계산 효율 우위 근거가 아니다.

reports/longer_training/experiment_summary.md 기준 당시 A 파일럿만 완료됐고 B/C는 자원 게이트 때문에 미실행이었다. 이후 B와 C 탐색 학습을 완료했다. 이전 계획서 머리말의 “모델 구현/실험 전” 문구와 manuscript/claim_evidence.md의 Unmeasured는 탐색 결과를 모두 반영하지 않은 부분이 있다. 그럼에도 확증적 주장이 아직 성립하지 않는다는 판단은 유지된다.

## 권고

현재 H1/H2를 중심 질문으로 유지하되 기여를 “새로운 결합의 발명”보다 “shared 용량과 routed rank의 상호작용 및 적용 경계에 대한 통제된 실증”으로 표현한다. 새로운 방향 선택이나 실험 실행은 이번 평가에서 하지 않았다.

다음 증거는 반복 seed의 LR/DW 8-cell 비교, 실제 저rank loss 개선과 고정예산 품질 유지의 분리, shared-only 및 routing 대조, 안정성 점검, 별도 동일 학습 FLOPs 비교다. 기존 실행 계획과 예산을 자동 변경하는 제안은 아니다.

## 검색 범위와 한계

2026-09-29에 arXiv, PMLR, OpenReview의 위 원문 및 원문 색인을 확인하고, low-rank/shared/MoE/native pretraining/rank allocation 조합으로 추가 검색했다. 새 인접 후보로 MEOX(https://arxiv.org/abs/2609.05351)의 초록도 확인했으나 지구관측 multimodal 사전학습의 shared projections/private low-rank residual 연구이므로 상세 구조 비교는 보류했다. 압축·양자화 보상·LoRA fine-tuning 검색 결과는 native LM 가설의 직접 검증과 구별했다. 전수 인용추적, 코드 재현, 모든 2026 후속 논문의 검토를 수행한 것은 아니다.
