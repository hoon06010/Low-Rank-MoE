# 전문가가 다르면 좋은가, 함께 예측을 잘해야 좋은가?

작성: 2026-11-13 (KST)
상태: 문헌 조사에 따른 **연구 질문 개정 후보**. 모델 구현·학습·checkpoint 개입은 실행하지 않았다. 기존 배분 연구의 결과·recipe·실험 승인 상태는 바꾸지 않는다.
연결 문서: [방법별 선행연구 지도](../literature/2026-11-13-complementarity-routing/field_map.md).

## 1. 추천 연구 질문

> **DeepSeekMoE형 shared FFN + native low-rank routed FFN에서, 출력 부분공간 정보가 입력과 기존 routing 점수만으로는 설명하지 못하는 expert 조합의 실제 예측 기여를 알려 주는가? 그 정보를 이용한 입력별 조합 선택이 기존 다양성·공동 선택 기반 routing보다, 추가 학습비와 선택비를 포함한 비용 대비 품질을 개선하는가?**

쉬운 말로 하면, “A가 좋아 보이고 B도 좋아 보인다”를 넘어 **“이 입력에는 A+B와 A+C 중 어느 쪽이 실제로 더 도움이 되는가?”**를 학습한다. A와 C가 다르다는 이유만으로 A+C를 보상하지 않는다.

세 단계가 모두 필요하다.

1. **현상이 있는가:** 개별 점수가 비슷한 후보라도 함께 선택하는 expert에 따라 실제 좋은 후보가 달라지는가?
2. **low-rank 정보가 필요한가:** 그 차이를 output factor 구조가 추가로 설명·예측하는가?
3. **방법으로 쓸 수 있는가:** 정답·미선택 expert의 실제 출력을 추론 때 보지 않고, 추가 비용보다 큰 이득을 얻는가?

첫 단계만 통과하면 routing 진단이다. 두 번째까지 통과해야 low-rank의 역할이 생긴다. 세 번째까지 통과해야 효율적인 새 방법이라는 주장을 검토할 수 있다.

## 2. 왜 처음 아이디어를 그대로 쓰면 약한가

가장 가까운 [MP-MoE, ICML 2026](https://proceedings.mlr.press/v306/kang26q.html)는 이미 공동 선택 이력을 이용한 집합 선택을 한다. [Advancing Expert Specialization, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html)는 선택 expert 출력의 중복을 규제한다. [CoR, ACL 2026](https://aclanthology.org/2026.acl-long.2187.pdf)와 [FedFMX, ECCV 2026](https://arxiv.org/abs/2606.28835)은 기여도와 routing의 연결을 이미 다룬다.

따라서 기여 후보는 '다양성', '조건부 증가분', 'counterfactual', 'distillation'이라는 단어가 아니다. **rank 제약이 만드는 조합 실패를 확인하고, 현재 토큰에서 실제 합산된 expert 출력의 비가산적 효과를 저렴하게 포착하는 것**이어야 한다. 이 조합의 최초성은 이번 조사만으로 확정하지 않는다.

## 3. low-rank의 정확한 역할

현재 로컬 [모델](../src/model.py)은 routed SwiGLU의 gate/up/down을 factorize하고, 별도 shared 출력과 합산한다. 이 문서의 핵심은 down projection이다.

\[
E_i(h)=B_i z_i(h),\qquad
B_i\in\mathbb R^{d\times r},\quad z_i(h)\in\mathbb R^r.
\]

따라서 단일 expert 출력은 \(\operatorname{col}(B_i)\)에 들어간다. 이는 **출력 가능한 방향의 상한**이다. 모든 입력에서 그 방향을 다 사용한다거나, 다른 방향이 예측에 유용하다는 보장은 없다.

동일 rank이며 full-column-rank인 \(B_i\)의 orthonormal basis를 \(Q_i\)라 두면,

\[
\rho_{ij}=\frac{\|Q_i^\top Q_j\|_F^2}{r}
\]

는 기저 선택에 불변인 overlap 지표다. 값이 클수록 공간이 겹친다. rank 결손은 수치 rank와 정규화 규칙을 별도로 고정한다.

Raw \(B_i^\top B_j\)를 그대로 쓰면 안 된다. \(B_iA_i=(B_iR)(R^{-1}A_i)\)처럼 같은 함수의 factor 좌표를 바꾸어도 값이 달라질 수 있기 때문이다. 가능하면 실제 down 행렬의 effective output span과 상한인 \(\operatorname{col}(B_i)\)도 구분한다.

**주의할 점**
- 좁은 dense FFN도 폭에 따른 출력 rank 상한이 있다. low-rank만 부분공간을 갖는 것은 아니다.
- full-rank shared weight의 전체 span을 제거하려고 하면 전체 공간을 제거할 수 있다. 'shared가 이미 처리한 방향'은 실제 토큰별 출력이나 별도 기능 측정으로 정의해야 한다.
- 실제 \(z_i(h)\)를 모든 expert에 대해 구하는 것은 일반적으로 공짜가 아니다. 그것을 사용한 router는 이미 expert 계산비를 썼다.
- QR/SVD와 pairwise overlap의 갱신비·메모리도 비용이다. 기저가 주어진다는 이유로 overhead 0이라고 주장하지 않는다.

### 다양성이 유용성을 보장하지 않는 직접적인 예

설명용 2차원 예로 \(E_A=(1,0), E_B=(1,0), E_C=(0,1)\), 두 expert의 가중치는 각각 1/2라고 하자. 세 출력은 rank-1 경로로 표현할 수 있다.

| 목표 출력 | A+B의 제곱오차 | A+C의 제곱오차 | 좋은 조합 |
|---|---:|---:|---|
| \((1,0)\) | 0 | 0.5 | 공간이 완전히 겹치는 A+B |
| \((0.5,0.5)\) | 0.5 | 0 | 서로 다른 공간의 A+C |

이것은 LM에서 측정한 결과가 아니라 수학적 반례다. 같은 기하학에서도 목표에 따라 좋은 조합이 바뀐다. 그래서 overlap에 일률적으로 음의 계수를 붙이는 대신, task loss와 연결을 검증한다.

최근 [Beyond Geometric Complementarity](https://arxiv.org/html/2607.28308v1)도 기하학과 기능을 구분해야 할 근거를 준다. 다만 그 논문이 측정한 입력 PCA 공간과 위의 native output-factor 공간은 서로 다르다.

## 4. 검증할 가설과 반증 조건

| 가설 | 필요한 증거 | 반증·중단 조건 |
|---|---|---|
| H1: 조합 조건이 유용한 정보를 준다 | 동반 expert를 바꿀 때 후보의 실제 교체 이득 순위가 달라지고, 독립 점수 모델보다 집합 조건부 모델의 held-out regret가 낮음 | 차이가 sampling noise이거나 독립 점수만으로 동등하게 예측 |
| H2: output geometry가 추가 정보를 준다 | 같은 크기의 interaction router에서 실제 기하 feature가 no-geometry·shuffled geometry보다 좋음 | 정보 제거·shuffle 후 결과가 같음 |
| H3: rank 제약과 관련된 효과다 | rank별 router 개선량과 matched dense-width 대조에서 효과 차이가 재현됨 | dense에서도 같은 개선이면서 LR 특유 메커니즘 증거 없음 |
| H4: 비용을 지불할 가치가 있다 | teacher/probe 학습비, cache 갱신비, 추론 selector비까지 포함한 품질–비용 비교에서 우위 | 활성 expert 수만 같고 실제 시간·FLOPs를 맞추면 이득 소멸 |

H1이 없으면 조합 router 개발을 멈춘다. H2가 없으면 low-rank 기하학을 기여에서 뺀다. H3가 없으면 '일반 interaction routing' 문제로 재분류하고 선행검색을 다시 한다. H4가 없으면 효율 주장을 하지 않는다.

## 5. 먼저 할 진단: 같은 K에서 후보만 교체

새 대규모 학습 전에 고정 checkpoint의 마지막 MoE layer에서 시작한다. 마지막 층은 뒤쪽 attention과 다른 router가 함께 변하는 문제를 줄인다. 기존 checkpoint가 충분히 학습됐다는 가정은 하지 않고 학습량·seed를 명시한다.

입력 문맥 \(x\), 다음 token 정답 \(y\), shared 출력과 나머지 모델을 고정한다. 선택된 동반 expert \(a\)에 대해 후보 \(b\)와 \(c\)를 교체한다.

\[
G_x(b\succ c\mid a)
=\ell_x(a,c)-\ell_x(a,b).
\]

양수면 a와 함께 쓸 때 b가 더 좋다. 두 route 모두 K=2다. 추가·제거 비교를 주 지표로 삼으면 활성 수와 출력 크기가 함께 바뀐다.

**기제 진단에서는 gate의 slot별 가중치를 고정한다.** 예컨대 원래 top-2의 \((w_1,w_2)\)를 유지하며 \(w_1E_a+w_2E_b\)와 \(w_1E_a+w_2E_c\)를 비교한다. 후보가 바뀌었다는 이유로 가중치를 다시 정규화하지 않는다.

별도의 **배포 평가**에서는 후보 집합 내 원래 router logits를 softmax해 실제 선택 규칙을 적용한다. 이 결과를 slot 고정 진단과 섞지 않는다.

동반 expert를 \(a'\)로 바꾼 2×2 비교도 수행한다.

\[
I_x =
[\ell_x(a,c)-\ell_x(a,b)]-
[\ell_x(a',c)-\ell_x(a',b)].
\]

이는 동반 expert에 따라 b와 c의 상대 이득이 얼마나 달라지는지 나타낸다. 양수만 '좋은' 값으로 보지 않는다. 부호와 크기를 예측하는 것이 목적이다. 출력 loss의 비선형성과 합산 가중치 규칙에도 의존하므로 '전문 지식 분리'의 직접 증거로 해석하지 않는다.

수집할 값:
- native router logits, score gap, load, 공동 선택 통계
- \(\rho_{ab},\rho_{ac}\) 등 output-space overlap
- 같은 K 교체 loss와 2×2 interaction
- actual-output cosine/norm은 **비배포 진단 변수**로 분리
- token별 변화량을 문서/문맥 단위로 집계한 불확실성

훈련·개발·최종평가 split은 원본 문서 단위로 나눈다. 동일 문서의 인접 토큰이 train/test에 섞이지 않게 한다. 모델·feature·threshold는 개발 split에서 정하고 최종 split을 보고 다시 선택하지 않는다.

## 6. 진단이 통과할 때 시험할 최소 방법 후보

임시 설명 이름: **loss-conditioned pair routing with output-subspace features**. 확정 모델명이나 최초성 주장이 아니다.

현재 작은 N=8, K=2 실험 설정을 출발점으로 하면 후보 쌍은 28개다. 이것은 28개 expert 쌍을 모두 실행한다는 뜻이 아니라 **실행 전 pair score 28개를 비교**한다는 뜻이다.

\[
F_\theta(h,\{i,j\})
=s_i(h)+s_j(h)
+f_\theta\big(h,\operatorname{sym}(s_i,s_j,e_i,e_j),\rho_{ij}\big).
\]

- \(s_i\): 일반 router의 개별 logit.
- \(e_i\): expert identity embedding 등 보조 입력. 이것만으로 pair ID를 암기할 수 있으므로 관련 대조군이 필수다.
- sym: 순서를 바꿔도 같은 결과를 내는 조합. 실제 구현에서 permutation invariance를 확인한다.
- \(f_\theta\): 양수·음수 모두 가능한 상호작용 보정. overlap이 크다고 반드시 벌점을 주지 않는다.
- 보정이 0이면 sum-of-logits pair 선택은 기존 top-2와 같다.

선택 후 두 expert만 실행한다. membership 효과를 분리하려고 합산 가중치는 기존 개별 logits의 선택집합 softmax를 유지한다. 별도 learned aggregator는 첫 방법에 넣지 않는다.

출력 구조를 쓰지 않는 동일 용량 pair predictor가 핵심 대조다. 고정 checkpoint의 기하 feature는 pair ID에 따라 정해지는 값이므로, 정보이론적으로 새로운 정보가 아니라 제한된 데이터·모델에서 도움이 될 수 있는 구조적 사전정보다. expert ID를 포함한 대조와 checkpoint/seed 간 전이를 통해 pair 암기와 구분한다. 저랭크 feature는 frozen checkpoint에서는 캐시할 수 있지만, joint training에서는 갱신 주기와 stale geometry 영향까지 실험해야 한다. 모든 후보의 실제 \(E_i(h)\), 미래 token, loss, loss gradient는 배포 router 입력에서 제외한다.

### 학습 신호와 비용

먼저 backbone을 고정하고 일부 훈련 문맥에서 같은 K 대안 pair를 계산한다. 실제 pair loss를 이용한 pair ranking 또는 soft target 학습을 시험한다. label을 학습 타깃으로 쓰는 것은 가능하지만 추론 입력으로 넣지 않는다.

비교하는 학습 predictor들은 같은 counterfactual labels·문맥·튜닝 예산을 제공받는다. geometry를 쓰는 모델에만 더 많은 정답 신호를 주지 않는다.

전체 28쌍을 사용하는 작은 진단과, 제한된 후보만 표본추출하는 학습을 구분한다. 후보 생성은 native 상위 후보뿐 아니라 일정한 탐색 표본도 포함해 초기 router 오류를 영구 고정하지 않게 한다. 어떤 표본을 얼마나 썼는지와 손실 평가비를 공개한다.

이 학습 형태 자체도 선행하는 아이디어다. 특히 [When Are Experts Misrouted?](https://arxiv.org/html/2605.07260v1)의 router preference update와 비교해야 한다. 우리의 추가 기여가 되려면 geometry와 non-additive context가 더 주는 정보가 있어야 한다.

정답을 본 oracle의 이득은 **주어진 후보·개입 규칙에서의 진단 상한**이다. 새 입력에서 예측 가능한 성능 상한은 아니다. [Beyond Routing §3.4](https://arxiv.org/html/2608.08853v1)가 보여 주는 utility predictor 실패 가능성을 먼저 검증한다.

Frozen-backbone 성공 후에만 공동 학습을 검토한다. 공동 학습을 처음부터 시작하면 expert가 달라진 효과와 selector 개선이 섞여 원인을 찾기 어렵다.

## 7. 공정한 비교를 위한 실험 순서

### A. 저비용 판정

동일 checkpoint·동일 후보·동일 문맥에서 다음을 비교한다.

| 비교 | 묻는 질문 |
|---|---|
| 개별 router 점수 | 원래 판단으로 충분한가 |
| 공동 선택 covariance 점수 | 이력 기반 관계로 충분한가 |
| static output-overlap penalty | 단순 기하학 보상으로 충분한가 |
| token-conditioned 독립 expert predictor | 일반적인 router 표현력 증가인가 |
| geometry 없는 동일 용량 pair predictor | 조합 모델 자체의 효과인가 |
| geometry를 추가한 pair predictor | low-rank 구조가 추가 정보를 주는가 |
| shuffled geometry / token / pair identity 대조 | 구조 신호·입력 조건 대신 암기나 정적 편향인가 |

'MP-MoE 점수 적용'이라는 사후 진단은 원 논문의 training-only 방법 재현과 다르다. 구분해서 기록한다.

### B. 작은 방법 비교

A 통과 후 동일 LR backbone에서 standard top-k, 원래 설정의 MP-MoE, output-diversity loss, CoR식 static CEI 대조, 제안법을 우선 비교한다. STAR와 PASs-style score는 기하학적 독립 선택의 추가 대조다. 다른 구조에 옮기는 경우 '원 논문 재현' 대신 '동일 backbone에 이식한 변형'이라고 명시한다.

CoR 원래 방법은 층별 K도 바꾸므로, fixed-K CEI-only는 별도 ablation이다. PASs-MoE의 LoRA 입력 factor를 native SwiGLU의 gate/up factor에 대응시키는 규칙도 새 변형이며 원래 방법과 동일하다고 부르지 않는다.

### C. low-rank 역할과 일반화

각 rank 내부에서는 backbone을 같게 두고 router만 비교한다. 이후 낮은/높은 rank, matched narrow-dense, down-only factorization으로 효과 차이를 본다. 처음부터 shared 폭·N·K·rank·router를 모두 동시에 바꾸지 않는다.

현재 모든 gate/up/down을 factorize하는 모델만으로는 output bottleneck과 input bottleneck 효과가 섞인다. 출력공간을 기제로 주장하려면 down-only 대조가 중요하다.

마지막 층 성공 후 중간 층과 전체 layer 적용을 구분한다. 작은 모델·짧은 학습의 결과를 대형 DeepSeek checkpoint나 장기 사전학습으로 일반화하지 않는다.

## 8. 무엇을 성과로 보고할 것인가

주 지표:
- 고정 후보 개입에서의 held-out **route regret**: 선택 pair loss − oracle 최소 pair loss
- 전체 배포 규칙에서 next-token CE, 변경 전보다 나빠진 문맥 비율
- 동일 checkpoint에서 geometry를 추가한 순수 개선량
- 여러 훈련 seed에서 짝지은 평균 차이와 불확실성
- 실제 wall time·throughput·peak memory·학습 총 FLOPs·추론 FLOPs

N=8,K=2는 검증하기 쉬운 시작점이다. N이 커지면 pair 점수는 O(N²)이므로 top-M shortlist나 순차 선택이 필요할 수 있다. shortlist 자체가 oracle 좋은 쌍을 제외하는 손실과 첫 선택 오류를 따로 측정한다. K>2는 pairwise 근사로 충분한지도 새 문제다.

같은 K는 같은 총 FLOPs를 뜻하지 않는다. parameter count에는 predictor를, training cost에는 counterfactual label 생성을, inference cost에는 cache·pair scoring을 포함한다. 같은 토큰 수 비교와 같은 총 학습 FLOPs 비교는 별개로 보고한다.

실용 최소 효과·허용 overhead는 **실험 결과를 보기 전에** 자원과 활용 목적에 맞춰 정한다. 아직 수치가 정해지지 않았으므로 이 문서는 실행 확정 계획이 아니다. 한 번 유의한 숫자가 나올 때까지 조건을 늘리는 방식은 사용하지 않는다.

## 9. 지금 결정할 수 있는 것

**권고:** 새로운 routing 연구의 후보로 남기되, 처음 제안한 '보완적 부분공간 선택'을 그대로 논문 기여로 확정하지 않는다. 다음 단계의 목적은 학습 성능 최고점을 만드는 것이 아니라 **geometry가 실제 조합 utility에 추가 정보를 주는지 판정하는 것**이다.

현재 확보한 결과는 선행방법 비교, 연구 질문, 방법 후보와 반증 조건이다. 새로운 성능 개선·novelty 확정·실행 가능 비용을 입증한 결과는 아직 없다.

임시 연구 제목:
**When Does Output-Subspace Geometry Help Joint Routing in Low-Rank Mixture-of-Experts?**

간단한 설명:
**저랭크 전문가가 낼 수 있는 출력 방향을 알면, 함께 일할 전문가를 더 잘 고를 수 있는가?**
