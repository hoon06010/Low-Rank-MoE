# Low-rank MoE의 조합 routing: 선행방법 지도

조사일: 2026-11-13 (KST). 대상: DeepSeekMoE형 shared FFN + native low-rank routed FFN에서 전문가 조합을 고르는 새 방법.
상태: 1차 출처에 근거한 제한된 문헌 검토. 게재 확인, 방법 확인, 우리 해석을 구분한다. 최초성 인증이나 실험 재현 결과가 아니다.

## 판단

단순한 “서로 다른 전문가를 같이 고르기”는 새 기여가 아니다. 가장 직접적인 선행은 [Kang et al. — MP-MoE, ICML 2026](https://proceedings.mlr.press/v306/kang26q.html)다. 후보를 현재 선택 집합에 추가할 때의 대리 이득까지 계산한다.

남는 후보는 **native low-rank 출력 구조가 입력별·조합별 실제 손실을 예측하는 데 추가 정보를 주는지 확인하고, 그 정보를 비용이 제한된 집합 선택에 사용하는 것**이다. 기존 covariance를 basis overlap으로 바꾸기만 하면 유사도 추정기의 교체로 평가될 위험이 크다.

[연구 질문과 판정 실험](../../reports/complementarity_routing_question_2026-11-13.md)을 함께 읽는다.

## 1. 직접 비교해야 할 방법

아래 '차이'는 원문들의 비교를 바탕으로 한 이번 조사자의 해석이다. 논문이 하지 않았다는 사실만으로 가치 있는 공백이라고 판단하지 않는다.

| 논문 / 확인된 게재 | 실제 사용하는 정보와 방법 | 우리 후보에 주는 제약 |
|---|---|---|
| [Kang et al. — Breaking the Echo Chamber / MP-MoE](https://proceedings.mlr.press/v306/kang26q.html), ICML 2026 본학회 | 개별 routing 점수와 expert 공동 선택 covariance로 Mahalanobis 집합 목적함수 최적화. prefix 조건부 greedy 선택 | 집합 선택·중복 억제·조건부 증가분 자체는 선행 |
| [Park & Park — STAR](https://proceedings.mlr.press/v306/park26u.html), ICML 2026 본학회 | GHA로 입력 principal subspace를 추적해 일반 gate와 혼합 | 'subspace-aware routing'은 선행. 출력 부분공간과는 구별 |
| [Guo et al. — Advancing Expert Specialization for Better MoE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html), NeurIPS 2025 본학회 | 같은 토큰에 선택된 expert 출력의 projection을 줄이는 loss와 routing variance loss | 출력 다양화, DeepSeek에 적용하는 것 자체도 선행 |
| [PASs-MoE](https://aclanthology.org/2026.acl-long.1474/), ACL 2026 main long | LoRA 입력 factor의 activation energy로 routing하고 rank 방향 변화 규제 | low-rank factor에서 router 신호를 얻는 발상은 선행 |
| [Hu et al. — Awakening Dormant Experts / CoR](https://aclanthology.org/2026.acl-long.2187.pdf), ACL 2026 main long | expert 제거의 loss 영향으로 calibration expert 중요도를 만들고 routing에 반영 | loss 기반 expert 기여도는 선행. 입력별 조합 상호작용과 구분 |
| [Zhao et al. — ProbMoE](https://proceedings.mlr.press/v306/zhao26ap.html), ICML 2026 본학회 | cardinality 제약 아래 subset 확률 모델링·샘플링·미분 가능한 학습 | 'subset을 학습' 자체는 선행. 별도 pair interaction 여부가 중요 |
| [DirMoE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html), ICLR 2026 본학회 | Bernoulli 선택과 Dirichlet mixture contribution 분리 | 선택과 합산 가중치를 분리하는 것 자체는 선행 |
| [FedFMX — Fisher-Routed Mixture of Experts for Federated Class-Incremental Learning](https://arxiv.org/abs/2606.28835), ECCV 2026 | sample별 coalition/Shapley 중요도를 lightweight gate에 증류 | 기여도 증류는 선행. 합산된 LM hidden output의 loss와는 다른 coalition value |

### 가장 가까운 MP-MoE의 경계

[공식 PDF §3–4, pp.4–9](https://raw.githubusercontent.com/mlresearch/v306/main/assets/kang26q/kang26q.pdf)에서 집합 점수는 다음 형태다.

\[
f_x(S)=\mu_S^\top\Sigma_{SS}^{-1}\mu_S,\qquad
\Delta f_x(j\mid S)=
\frac{(\mu_j-\Sigma_{jS}\Sigma_{SS}^{-1}\mu_S)^2}
{\Sigma_{jj}-\Sigma_{jS}\Sigma_{SS}^{-1}\Sigma_{Sj}}.
\]

여기서 \(\Sigma\)는 공동 선택 이력에서 얻는다. 저자들은 출력 유사도와의 경험적 관계도 조사했다. 그러므로 '실제 expert와 무관한 점수'라고 비판하면 부정확하다. 다만 이 증가분은 **언어모델 loss 감소량 자체가 아니다**.

OLMoE 사전학습과 작은 모델 비교에서 개선을 보고한다. 논문 p.7의 추론 overhead 0은 **추론 때 표준 top-k로 복귀**하는 설정이다. 제안법이 추론에도 조합 선택을 실행한다면 비용을 별도로 비교해야 한다.

[저자 demo 코드](https://github.com/kxlkxl1999/MP-MoE/blob/main/megatron-lm/megatron/core/transformer/moe/router.py#L253)에서 `routing_type="otep"`를 유지한 해당 분기는 eval에서도 OTEP 선택을 실행하며, 입력 점수도 논문의 logits 표현과 다른 softmax probability다. 외부 eval 설정의 전환 여부는 미확인이다. demo를 그대로 써서 공식 결과 재현이라고 부르지 않는다.

### 부분공간·다양성 방법의 서로 다른 대상

[STAR 방법 §3, 실험 §5](https://arxiv.org/html/2606.08814v1): 입력 covariance의 principal directions를 일반 gate에 결합한다. 언어모델 사전학습의 주 설정은 top-1이므로 여러 expert의 공동 기여를 검증한 실험이 아니다. 출력 factor의 span을 쓰지 않는다. 게재는 공식 proceedings, 방법은 arXiv v1에서 확인했다.

[Advancing Expert Specialization 최종 PDF §3–4](https://proceedings.neurips.cc/paper_files/paper/2025/file/4598de7d243d528e38eb0c5d8155fb52-Paper-Conference.pdf): DeepSeek 계열을 포함한 post-training에서 출력 projection과 routing variance를 규제한다. 집합 선택 알고리즘을 교체하지 않는다. arXiv v1과 최종 Eq.6의 표시가 달라 최종 PDF를 기준으로 삼아야 하며, 부록 표기와 코드의 대조는 후속 재현 항목이다.

[PASs-MoE PDF §3.3–4.3](https://aclanthology.org/2026.acl-long.1474.pdf): \(s_e(h)=\|A_eh\|^2/r\)처럼 LoRA 입력측 경로의 활성 정도를 쓴다. native routed SwiGLU의 출력공간 \( \operatorname{col}(B_{\rm down}) \)이나 두 expert를 합친 뒤의 loss를 직접 평가하는 방법은 아니다. 입력 factor activation과 출력 span을 같은 'low-rank 기하'로 뭉뚱그리지 않는다.

### 실제 기여를 사용하는 방법의 경계

[CoR PDF §3.2.3–3.3](https://aclanthology.org/2026.acl-long.2187.pdf): 선택 expert를 제거하고 남은 gate를 재정규화해 loss 변화를 측정한다. hard calibration tokens에 대해 평균한 expert별 CEI를 contextual router score와 합치고 층별 활성 수를 재배분한다. 우리 설계의 같은 K 교체 실험은 제거·재정규화·활성 수 변화의 영향을 분리하기 위한 것이다.

[FedFMX §3.3–3.5](https://arxiv.org/html/2606.28835v1): coalition loss는 expert별 loss의 평균으로 정의하고 Fisher stability/plasticity 항을 함께 쓴다. 합산된 hidden output을 후속 LM에 통과시킨 손실과 다르다. Shapley 점수를 gate에 증류하므로 'label 없이 기여도 routing 배포'도 새 기여가 아니다. ECCV 게재는 [공식 accepted 목록](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)과 [Springer proceedings, pp.327–347](https://link.springer.com/book/10.1007/978-3-032-37461-5)에서 확인했다.

[ProbMoE §4–7](https://arxiv.org/html/2606.01509v1): 독립 Bernoulli 요인의 곱을 exact-k 조건으로 정규화한다. exact-k MAP은 개별 logit top-k로 귀결된다. cardinality 때문에 subset 변수들이 의존하지만, 학습한 pairwise interaction potential을 둔 것과 같지는 않다.

[DirMoE §4](https://arxiv.org/html/2602.09001v1): 선택과 contribution을 분리하는 변분 접근이다. 새 연구에서는 membership 개선과 weight 재조정의 개선을 따로 측정해야 한다.

## 2. 기반 방법과 저널 선행

| 논문 / 게재 | 조사한 방법과 관련성 | 이번 확인 범위 |
|---|---|---|
| [Chi et al. — On the Representation Collapse of Sparse Mixture of Experts / XMoE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/df4f371f1f89ec8ba5014b3310578048-Abstract.html), NeurIPS 2022 | 낮은 차원의 hypersphere에서 token–expert 점수를 계산. 저차원 router와 저랭크 FFN은 다름 | 공식 페이지·본문 도입 |
| [Hazimeh et al. — DSelect-k](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html), NeurIPS 2021 | binary encoding을 smooth하게 풀어 at-most-k expert 선택을 미분 가능하게 만듦 | 공식 초록·방법 설명. pair utility 직접 실험은 아님 |
| [Hendawy et al. — Multi-Task Reinforcement Learning with Mixture of Orthogonal Experts / MOORE](https://proceedings.iclr.cc/paper_files/paper/2024/hash/19c145aaad40927c51f4d10eaa339c20-Abstract-Conference.html), ICLR 2024 | Gram–Schmidt expert representation과 task별 조합. 직교화 자체의 선행 | 공식 초록. RL과 sparse LM 차이 유지 |
| [Do et al. — On the Role of Discrete Representation in Sparse Mixture of Experts / VQMoE](https://arxiv.org/abs/2411.19402), TMLR 2025 | codebook으로 expert를 지정하는 discrete 경로와 continuous MoE를 사전학습에서 결합 | [TMLR accepted 목록](https://www.jmlr.org/tmlr/papers/), [본문 v2 §3–5](https://arxiv.org/html/2411.19402v2) |
| [Shapley-MoE — Discovering Important Experts for Mixture-of-Experts Models Pruning Through a Theoretical Perspective](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c66a9db149261435664284a20b6f1d42-Abstract-Conference.html), NeurIPS 2025 | coalition 기여를 이용한 static expert pruning | 본문 방법 확인. 토큰별 online dispatch와는 다름 |
| [Do et al. — SimSMoE](https://aclanthology.org/2025.findings-naacl.107/), Findings NAACL 2025 | 출력 CKA로 collapse를 찾아 다양성을 유도 | 공식 페이지·본문 도입. 본학회 main과 구분 |

[VQMoE §3–5](https://arxiv.org/html/2411.19402v2)는 discrete/continuous 경로와 학습 방식까지 바꾼다. 같은 backbone에서 router만 바꾸는 비교와 동일하지 않다. 이론의 cluster 가정을 일반적인 token별 최적 조합 보장으로 확대 해석하지 않는다. 사전학습·NLP adaptation·vision 실험을 보고하지만 native output-rank와 조합 효과의 상호작용은 본 연구의 별도 확인 대상이다.

이번 검색에서 JMLR·TPAMI·TNNLS의 더 직접적인 native-low-rank joint-output selector는 확인하지 못했다. 이는 부재의 증명이 아니다. 저널 항목을 채우기 위해 영상 분류·시계열·다중모달의 '보완성'을 동일 문제의 해결책처럼 추가하지 않았다.

## 3. 게재 확정 목록과 분리해 읽을 최근 원고

아래 원고는 방법 충돌이나 반증 위험 때문에 포함한다. 이번 조사에서 공식 본학회/저널 게재를 확인하지 못했으며, 검색 사이트의 venue 배지만으로 accepted로 분류하지 않았다.

| 원문 | 확인 내용 | 설계에 반영할 점 |
|---|---|---|
| [Tian et al. — Beyond Geometric Complementarity](https://arxiv.org/html/2607.28308v1), §2–5, S2–S7 | routed token **입력 분포의 PCA 공간**과 frozen-route NLL을 구분. 공간 겹침과 기능적 무용성을 동일시할 수 없다는 관측 | 우리의 weight-factor 출력 span에 대한 직접 반증은 아님. 다만 기하→utility 연결을 검증해야 한다는 강한 경고 |
| [Yoon et al. — When Are Experts Misrouted?](https://arxiv.org/html/2605.07260v1), §3–5 | 같은 K 대안 route를 실제 CE로 평가하고 마지막 router만 preference update | counterfactual route 학습은 이미 선행. label 없는 배포와 평가 split을 엄격히 구분 |
| [Li — Beyond Routing: Decoupling Expert Dispatch and Aggregation](https://arxiv.org/html/2608.08853v1), §3.4–4 | expert 집합을 고정하고 weight를 학습. 여러 utility predictor의 일반화 실패도 보고 | oracle headroom이 있다고 utility가 예측 가능한 것은 아님. membership/aggregation 대조 필수 |
| [GatePro](https://arxiv.org/html/2510.13079v1), §3.2 | router weight 유사도를 사용한 localized competition | 단순 cosine/overlap penalty는 약한 novelty |
| [Nguyen et al. — CompeteSMoE](https://arxiv.org/html/2505.13380v1), §2.2–2.4 | 일부 학습 시점에 모든 expert 반응을 계산하고 cheap router로 증류 | '비싼 expert 신호를 저렴한 router에 증류'는 선행. 원래 [2024 코드](https://github.com/giangdip2410/CompeteSMoE)는 저자가 재현 버그를 공지한 archive |
| [RotMoLE](https://arxiv.org/html/2605.25565v1), §2–3 | low-rank 경로 내부의 입력별 회전 \(B_iR_i(x)A_i\) | 조합 선택보다 expert 변환 자체를 바꿈. NeurIPS 2026 게재는 미확인 |

위 첫 원고를 앞선 대화에서는 '출력 방향 겹침'의 반례 후보로 넓게 표현했다. 본문 확인 후 **측정한 부분공간은 router-input PCA**라고 바로잡는다. 기능적 개입은 유용하지만 native weight-output subspace와 직접 같은 실험은 아니다.

## 4. 인용망과 조사 경계

실제로 확인한 연결:
- [MP-MoE](https://proceedings.mlr.press/v306/kang26q.html) → [GatePro](https://arxiv.org/abs/2510.13079), [CompeteSMoE](https://arxiv.org/abs/2402.02526), [SimSMoE](https://aclanthology.org/2025.findings-naacl.107/).
- [VQMoE](https://arxiv.org/html/2411.19402v2) → [XMoE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/df4f371f1f89ec8ba5014b3310578048-Abstract.html), [SimSMoE](https://aclanthology.org/2025.findings-naacl.107/), [CompeteSMoE](https://arxiv.org/abs/2402.02526).
- [Advancing Expert Specialization](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html) → [MOORE](https://proceedings.iclr.cc/paper_files/paper/2024/hash/19c145aaad40927c51f4d10eaa339c20-Abstract-Conference.html).
- [Beyond Geometric Complementarity](https://arxiv.org/html/2607.28308v1) → [STAR](https://proceedings.mlr.press/v306/park26u.html), [When Are Experts Misrouted?](https://arxiv.org/abs/2605.07260).

대표 검색축: mixture of experts + complementary/diversity/pairwise/subset/marginal contribution/counterfactual/low-rank subspace/DPP; ICML·NeurIPS·ICLR·ACL proceedings 및 TMLR·JMLR·IEEE 저널 출처 확인. 검색 결과의 제3자 해설은 후보 발견에만 사용했다. 인용 수는 검증하지 않아 순위 근거로 쓰지 않았다.

일부 OpenReview는 browser verification, 일부 PDF는 content-type 문제로 직접 열리지 않았다. 해당 항목은 공식 게재 페이지와 읽은 원고 버전을 분리했다. 논문 수치와 주장 전체를 독립 재현한 것은 아니다.

## 5. 이 지도에서 남는 질문

**현재까지 확인된 사실:** 다양화, 기하학적 routing, 공동 선택, 반사실적 utility, utility 증류가 각각 존재한다.

**아직 이번 자료로 답할 수 없는 것:** low-rank FFN 출력 factor의 구조가 입력·선택 이력만으로는 예측하기 어려운 조합의 loss를 추가로 설명하는가; 그 신호가 새 문서·도메인·seed에서도 유효한가; 그 정보를 계산하는 비용까지 포함해 이득인가.

후자의 세 가지를 만족시키는 증거가 없으면, 지금 단계에서 '새로운 구조 논문으로 전환 확정'이라고 말할 수 없다.
