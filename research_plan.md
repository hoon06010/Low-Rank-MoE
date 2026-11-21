# 고정 파라미터 예산에서 Low-Rank Routed Experts와 Full-Rank Shared Experts의 용량 배분

> **2026-11-13 연구 질문 개정안:** 같은 총 파라미터·주어진 학습 토큰 예산에서 shared 용량을 늘릴 때, routed FFN을 **넓지만 저랭크로 유지할지, 좁은 dense로 만들지**를 비교한다. [질문·설계 이유·반증 조건](reports/research_question_refinement_2026-11-13.md)에 현재 증거와 후속 대비를 정리했다. 로컬 기록상 Stage C seed17·seed401 AdamW 네 조건은 완료했으며, LR 고유 이득·동일 FLOPs 우위는 미확인이다. 아래는 2026-09-20 초안의 이력이며 당시 미구현·미실행 표현을 현재 상태로 읽지 않는다.

- 작성일: 2026-09-20
- 상태: 연구 계획 초안. 선행연구 검토를 바탕으로 작성했으며, 모델 구현과 실험은 아직 수행하지 않았다.
- 연구 방식: 소규모 언어 모델의 native pretraining 실험
- 문헌 검토 범위: 본 대화에서 확인한 원문과 실험 조건. 전수 검색이나 독창성 확정은 아니다.

## 1. 핵심 연구 질문

> 같은 총 파라미터 예산에서 routed expert의 rank를 낮춰 확보한 파라미터를 full-rank shared expert의 수에 얼마나 재투자하는 것이 유리한가?

이 연구는 routed expert 수와 top-K를 우선 고정하고, routed rank를 낮추면서 항상 활성화되는 full-rank shared 용량을 늘릴 때의 성능 곡선을 찾는다. shared expert 하나의 폭을 고정하면 shared 수가 shared 용량의 단위가 된다.

핵심 결과물은 특정 구조의 단일 성능 점수가 아니라, 같은 총 파라미터 예산에서 가능한 `(routed rank, shared expert 수)` 조합과 각 조합의 검증 손실·계산 비용이다. 중간 지점에 최적값이 존재하거나 shared 확대가 반드시 유리하다고 가정하지 않는다.

## 2. 구조와 용어

한 MoE 층의 expert 출력을 다음과 같이 정의한다. Transformer의 외부 residual connection은 여기서 생략한다.

$$
\begin{aligned}
y(h) &= \sum_{j=1}^{S} E_j^{\mathrm{shared}}(h) \\
     &\quad + \sum_{i \in \operatorname{TopK}(h)} g_i(h) E_i^{\mathrm{routed}}(h).
\end{aligned}
$$

| 기호 | 의미 | 초기 비교에서의 취급 |
|---|---|---|
| $S$ | 항상 활성화되는 shared expert 수 | 변경 |
| $N$ | routed expert 총수 | 고정 |
| $K$ | 토큰당 활성 routed expert 수 | 고정 |
| $r$ | routed expert의 factorized matrix rank | 변경 |
| $p_s$ | shared expert 하나의 파라미터 수 | 폭과 구조를 고정하여 일정하게 유지 |
| $p_r(r)$ | routed expert 하나의 파라미터 수 | rank에 따라 변경 |

shared expert는 factorization을 강제하지 않은 일반적인 dense weight를 학습한다. 여기서 full-rank라는 표현은 **학습된 행렬의 수학적 rank가 항상 최대라는 보장이 아니라, 낮은 rank로 매개변수화를 제한하지 않는다는 뜻**이다.

routed expert는 대상 행렬을 처음부터 다음처럼 매개변수화하고 학습한다.

$$
\begin{aligned}
W_i &= B_i A_i, \\
B_i &\in \mathbb{R}^{d_{\mathrm{out}} \times r}, \qquad
A_i \in \mathbb{R}^{r \times d_{\mathrm{in}}}.
\end{aligned}
$$

이미 학습된 full-rank weight에 adapter만 추가하는 LoRA fine-tuning과 구별한다. shared와 routed 경로 모두 무작위 초기화부터 언어 모델 학습에 참여한다.

실제 expert는 비선형 FFN으로 정의한다. SwiGLU를 사용한다면 gate/up/down 중 어느 행렬을 factorize하는지 사전에 고정해야 한다. 초기 후보는 routed FFN의 세 행렬을 모두 같은 rank로 factorize하는 설정이며, 최종 선택은 구현 전에 확정한다. attention과 embedding은 첫 비교에서 동일하게 유지한다.

### Shared 수와 shared 폭의 관계

같은 입력을 받고 독립적인 normalization이나 입력별 gate 없이 출력을 합산하는 일반적인 FFN/SwiGLU shared experts는 가중치를 이어 붙여 하나의 더 넓은 FFN으로 표현할 수 있다. 따라서 이 설정에서 shared 수의 증가는 주로 **shared FFN 용량의 증가**로 해석한다. 별도 expert로 분리했다는 사실 자체에 독립적인 표현력 이득을 부여하지 않는다. 초기화나 optimizer 설정에 따른 학습 차이는 별개다.

비선형성이 전혀 없는 shared expert들을 사용하는 경우에는

$$
\sum_{j=1}^{S} W_j h = \left(\sum_{j=1}^{S} W_j\right) h
$$

이므로 shared 수를 늘려도 표현 가능한 선형 함수의 범위는 늘지 않는다. 이 점을 피하기 위해 본 연구의 shared expert는 비선형 FFN으로 정의한다.

## 3. 고정 예산과 활성 계산량

동일한 층에서 expert에 배정된 파라미터 예산을 $P_E$라 하면

$$
P_E = S p_s + N p_r(r).
$$

attention, embedding, normalization, router 등 나머지 파라미터는 별도로 집계한다. 처음에는 층 수와 층별 구조를 동일하게 유지해 expert 예산 비교가 모델 전체의 고정 예산 비교로 이어지도록 한다.

행렬 하나를 factorize하면 dense 파라미터 수는 $d_{\mathrm{out}}d_{\mathrm{in}}$, factor 파라미터 수는 $r(d_{\mathrm{out}}+d_{\mathrm{in}})$이다. 따라서 rank 제한만으로 항상 파라미터가 절약되는 것은 아니다. factor 파라미터 수가 dense보다 작은 범위를 사용한다.

예를 들어 bias 없는 SwiGLU에서 입력 차원이 $d$, intermediate width가 $m$이고 세 행렬을 모두 rank $r$로 factorize하면

$$
p_{\mathrm{dense}} = 3 d m, \qquad p_r(r) = 3 r(d + m).
$$

실제 실험 예산은 이 근사식만 사용하지 않고 구현된 모델의 파라미터 수로 확인한다. 정수 rank와 shared 수 때문에 예산을 정확히 맞추지 못하면 실제 차이를 공개하고, 허용 오차를 실행 전에 정한다.

### 같은 총 파라미터가 같은 활성 파라미터를 뜻하지 않는다

토큰 하나에 활성화되는 expert 파라미터는

$$
\begin{aligned}
P_{\mathrm{act},E}
&= S p_s + K p_r(r) \\
&= \frac{K}{N} P_E + \left(1 - \frac{K}{N}\right) S p_s.
\end{aligned}
$$

따라서 $P_E$, $N$, $K$, $p_s$가 고정되고 rank 절약분을 전부 shared에 재투자하면

$$
\Delta P_{\mathrm{act},E}
= \left(1 - \frac{K}{N}\right) \Delta S\, p_s.
$$

**$K<N$인 sparse MoE에서는 shared 확대에 따라 활성 expert 파라미터가 반드시 증가한다.** 최초 실험은 고정된 모델 용량에서 품질과 활성 계산량을 교환하는 곡선이다. 이득이 관찰되어도 계산 효율 개선으로 곧바로 해석하지 않는다.

동일 토큰 수 비교를 먼저 수행하고, 유망한 조합에 대해 동일 총 학습 FLOPs에서도 비교한다. 후자는 모델별 학습 토큰 수가 달라질 수 있으므로 서로 다른 질문에 대한 결과로 보고한다. 동일 총 파라미터·동일 활성 파라미터·고정 N/K를 모두 만족시키면서 S를 자유롭게 바꾸는 것은 위 조건에서 불가능하다.

## 4. 선행연구와 본 연구의 위치

### 4.1 Shared 용량 배분에 직접 관련된 연구

| 연구 | 확인한 실험과 결과 | 본 연구와의 차이 |
|---|---|---|
| **DeepSeekMoE** [1] | §4.4에서 총 expert 64개와 활성 expert 8개를 유지하며 shared를 1·2·4개로 변경했다. Pile loss는 1.808·1.806·1.811이며, 저자들은 차이가 크지 않다고 설명한다. | shared 수에 따라 routed 총수와 활성 수도 바뀐다. routed rank를 낮춰 shared 용량을 확보하는 실험이 아니다. |
| **Towards Greater Leverage** [2] | §4.1.3에서 shared 비율을 바꾸며 loss 곡선을 조사한다. 본문은 모델 크기·활성 비율·활성 expert 총수를 통제한다고 설명한다. 낮은 nonzero shared 비율이 유리하며, 조사한 계산 예산에 따라 16.7% 또는 8.3%가 유리하다고 보고한다. | 같은 크기의 expert에서 shared/routed 활성 구성을 바꾼다. native low-rank routed expert를 다루지 않는다. 수치 재현 시 부록의 실제 모델별 파라미터 집계를 함께 확인해야 한다. |
| **Towards a Comprehensive Scaling Law of MoE** [3] | 총 파라미터·활성 파라미터·활성 expert 수·shared 비율 등을 포함한 446개 실험으로 scaling law를 적합했다. shared 비율 13–31%를 fitted optimum 주변의 손실 차이가 작은 범위로 제시한다. | 비율은 활성 expert 중 shared의 비율이다. 이를 hybrid 모델의 총 파라미터 배분 비율로 사용해서는 안 된다. rank는 실험 축이 아니다. |
| **Slicing and Dicing** [4] | 2026년 프리프린트. 2,000회 이상 사전학습으로 expert 수·크기·shared 용량 등을 조사했다. §3.2에서 활성 FFN 파라미터를 맞춘 조건의 shared/generalist 추가는 대체로 동등하거나 조금 나빴다. | shared 확대의 보편적인 이득을 지지하지 않는다. 다만 native low-rank routed expert가 있는 조건의 보완 효과는 검증하지 않았다. |

이 연구들의 결과는 shared expert의 이득이나 최적 비율에 대한 합의가 없음을 보여 준다. 특정 논문의 최적 shared 비율을 출발 가정으로 고정하지 않는다. 모델 규모, routed 용량, 활성 계산량과 학습 조건을 고려해야 한다.

### 4.2 Full-rank 공통 용량과 low-rank 개별 용량을 결합한 연구

**LightMoE [5]**는 이미 학습된 MoE에서 일부 experts를 여러 shared base와 expert별 low-rank 성분으로 대체한다. §3.2의 구조는 대략 $W_i=W_{\mathrm{base},c(i)}+B_iA_i$이며, §4.3과 부록 A.4에서 압축률별 adaptation rank와 shared base 그룹 설정을 비교한다. 예산 배분 관점에서 가까운 선행연구이므로 우선적으로 읽어야 한다. 그러나 base는 expert들이 공유하는 행렬 성분이며 모든 토큰에 독립적으로 실행되는 shared FFN이 아니다. 학습도 native pretraining이 아닌 압축 후 복구·적응이다. 비선형 FFN 내부의 행렬 공유와 별도 FFN 출력의 합산은 일반적으로 동등하지 않다.

**MLoRE [6]**는 시각 다중 작업에서 일반 convolution 공통 경로와 선택적인 low-rank expert 경로를 결합한다. Table 2에서 공통 경로 추가 시 파라미터가 213M에서 243M으로 늘어난다. 부록 B.2에서는 공통 경로를 low-rank로 제한한 경우보다 일반 convolution을 사용한 경우가 좋았다. 이는 full-rank 공통 경로의 가치에 대한 근거지만, 고정 예산에서 routed rank 절약분을 공통 용량에 재투자한 결과는 아니다. 경로의 선형성과 재매개변수화에 기반한 계산 특성 역시 Transformer FFN에 그대로 적용할 수 없다.

**Soup-of-Experts [7]**는 같은 파라미터 수에서 low-rank expert 수를 늘리는 접근이 dense expert보다 불리한 결과를 보고하고 최적화 문제의 가능성을 제기한다. expert 파라미터를 조합하는 구조여서 본 연구의 토큰별 sparse routing과 다르다. 낮은 rank를 통한 재배분이 자동으로 유리하지 않음을 보여 주는 보조 근거로 사용한다.

### 4.3 Optimization 관련 연구

**LORO [8]**는 native low-rank factor를 독립적으로 최적화할 때의 기하학적 문제를 다루고, low-rank manifold에서의 업데이트를 제안한다. **Stabilizing Native Low-Rank LLM Pretraining / Spectron [9]**는 factorized 학습에서 weight update의 spectral norm 증가와 학습 불안정성을 다룬다.

본 연구에서는 이들을 최적 배분 곡선이 낮은 rank에서의 최적화 실패에 의해 왜곡되지 않았는지 확인하는 비교 대상으로 사용한다. 기존 방법의 적용 자체를 optimization novelty로 주장하지 않는다. factor별 AdamW, LORO, Spectron의 차이를 단순한 learning rate 차이로 환원하지 않고 각 방법의 초기화·스케일링·업데이트 구현을 확인해야 한다.

### 4.4 잠정적인 연구 공백

이미 연구된 것은 shared/routed 역할 분리, shared 비율의 최적화, low-rank expert, shared base와 low-rank 성분의 결합이다.

이번에 확인한 문헌에서는 다음 조건을 함께 만족하는 직접적인 실험을 찾지 못했다.

1. 언어 모델을 처음부터 학습한다.
2. 항상 활성화되는 shared FFN은 factorization으로 제한하지 않는다.
3. routed FFN은 native low-rank로 학습한다.
4. routed 총수와 top-K를 고정한다.
5. rank 절약분을 shared 용량에 재투자하여 총 파라미터를 맞춘다.
6. 그 배분 곡선을 최적화 방법과 계산 비용까지 포함해 평가한다.

이는 제한된 문헌 검토의 결과이며 최초성의 증명은 아니다. 구조 novelty와 예산 배분에 대한 실증적 기여를 구분해 주장한다.

## 5. 검증할 주장과 가능한 반대 결과

| 질문 | 검증할 주장 | 가능한 반대 결과 |
|---|---|---|
| 배분 효과 | 일부 예산 범위에서 routed rank를 낮추고 shared 용량을 늘린 조합이 같은 예산의 비교 모델보다 좋은가? | 재배분 내내 손실이 증가하거나, 대부분을 shared에 쓰는 경계가 최적일 수 있다. |
| Low-rank 방식의 가치 | 같은 예산을 shared에 넘길 때 routed FFN의 폭 축소보다 factorization이 유리한가? | 단순한 폭 축소가 동등하거나 더 좋을 수 있다. |
| Shared 보완 효과 | shared 확대의 이득이 낮은 routed rank에서 더 커지는가? | shared의 이득이 rank와 무관하거나, routed 기능을 대신하면서 conditional 경로의 기여가 사라질 수 있다. |
| 최적화 의존성 | 학습법을 개선하면 낮은 rank 구간의 성능과 최적 배분점이 달라지는가? | 안정화해도 rank 제한으로 인한 손실이 남을 수 있다. |

‘conditional specialization은 low-rank로 충분하다’는 설명은 검증 대상이다. 특정 토큰 집합을 처리한다는 사실만으로 필요한 변환의 rank가 낮다고 결론낼 수 없다.

## 6. 단계별 실험안

아래는 실행 전 구체화할 실험 설계 초안이다. 숫자와 학습 설정이 확정된 계획 또는 완료된 결과를 뜻하지 않는다.

### 단계 A: 소규모 모델과 예산 계산 검증

- 작은 decoder-only Transformer/MoE를 사용한다. 모델 크기와 토큰 수는 확보 가능한 GPU 자원을 확인한 뒤 정한다.
- depth, hidden width, attention, tokenizer, 데이터, MoE 층 위치, routed 수 N, top-K를 고정한다.
- shared expert 하나의 폭과 routed factorization 대상을 정한다.
- 고정 예산에 들어오는 `(S, r)` 조합을 계산하고, 실제 파라미터 수 및 실행 가능한 차원인지 확인한다.
- 짧은 학습으로 forward/backward, 초기 출력 규모, router 동작과 수치 안정성을 확인한다.

### 단계 B: 고정 총 파라미터의 배분 곡선

처음부터 큰 grid를 돌리지 않고 실현 가능한 3–5개 배분점으로 시작한다. 동일 데이터 순서와 동일 학습 토큰 수에서 검증 손실을 비교한다. shared가 작은 쪽과 큰 쪽을 모두 포함해 한쪽 경계가 최적인지도 확인한다.

필수 비교군은 다음과 같다.

| 비교군 | 목적 |
|---|---|
| Dense routed FFN을 쓰는 기준 MoE | 기존의 용량 배분과 비교한다. 동일 예산을 위한 폭 조정은 명시한다. |
| Low-rank routed + full-rank shared | 핵심 rank/shared 배분 곡선을 측정한다. |
| 폭을 줄인 dense routed + 같은 shared 용량 | 같은 routed 예산에서 factorization 고유의 이득을 확인한다. |
| 같은 expert 예산을 shared FFN에 배정한 dense 경계 | conditional 용량을 유지하는 이득이 있는지 확인한다. router 제거에 따른 작은 예산 차이도 집계한다. |

초기 탐색 후 유망한 점과 양쪽 인접점을 복수 seed로 반복한다. 정확한 seed 수와 실질적으로 의미 있다고 볼 손실 차이는 본 실험 전에 정한다. 작은 차이를 단일 seed의 순위로 결론내리지 않는다.

### 단계 C: 보완 효과와 최적화 효과 분리

고정 예산 곡선에서는 S와 r이 함께 변하므로 곡선만으로 ‘shared가 low-rank 손실을 보완했다’는 인과적 설명을 확정할 수 없다. 대표적인 높은/낮은 rank와 작은/큰 shared 용량의 2×2 조합을 추가 비교한다. 이 보조 비교는 고정 예산 밖의 점을 포함하며, 주 예산 비교와 구별한다.

대표 rank와 배분점에서 naive factor optimizer, LORO 방식, Spectron 방식의 적용 가능성과 효과를 비교한다. 각 방법에 합리적인 learning rate·초기화 조정을 허용하되 탐색 예산을 기록한다. 전체 조합의 대규모 교차 실험은 초기 결과를 본 뒤 결정한다.

loss spike가 관찰되면 factor/weight-update norm과 routing 지표를 함께 확인한다. 필요할 경우 짧은 고정 routing 대조 실험으로 원인을 좁힌다. 안정화 방법을 적용했다는 이유만으로 표현력 문제가 제거되었다고 해석하지 않는다.

### 단계 D: 계산 예산과 재현성 확인

동일 토큰 수에서 유망했던 배분점에 대해 동일 총 학습 FLOPs에서도 비교한다. 같은 장비·precision·batch·sequence length에서 처리량과 peak memory를 측정한다. 모델 규모를 늘리는 것은 작은 설정에서 배분 효과가 반복 관찰된 뒤 진행한다.

## 7. 기록할 지표와 해석

| 범주 | 기록할 항목 | 해석상 주의점 |
|---|---|---|
| 품질 | held-out next-token cross-entropy, 필요시 perplexity와 domain별 loss | 동일 tokenizer와 평가 데이터를 사용한다. 전체 평균이 특정 domain의 악화를 가릴 수 있다. |
| 파라미터 | 총수, shared/routed/기타 파라미터, 토큰당 활성 expert 파라미터 | 총수와 활성 수를 혼용하지 않는다. |
| 계산 | forward 및 학습 FLOPs 추정, optimizer 연산 포함 여부 | factorization과 안정화의 추가 비용을 명시한다. |
| 메모리 | 학습 peak GPU memory, weight·gradient·optimizer state·activation | 파라미터 절약률을 그대로 peak memory 절약률로 사용하지 않는다. |
| 속도 | 학습 tokens/s, 필요시 별도 inference throughput | 작은 GEMM, dispatch, 커널 지원으로 FLOPs 감소가 속도 향상과 일치하지 않을 수 있다. |
| Routing | expert별 token 수, 선택 확률의 entropy, 미사용 expert, overflow/drop 비율 | 정상적인 specialization과 routing collapse를 구별한다. |
| Low-rank 안정성 | factor norm, effective weight/update spectral norm의 추정, gradient norm, loss spike | 큰 행렬을 매 step 재구성하는 계측은 피하고 계측 비용을 기록한다. |
| 경로 기여 | shared/routed 출력 규모와 경로 제거 시 loss 변화 | 출력 norm만으로 역할 분담을 증명하지 않는다. 제거 실험도 학습된 모델의 의존성을 보여 줄 뿐 인과적 역할의 완전한 증명은 아니다. |

주 결과 그림은 x축에 shared 수 또는 shared 파라미터 비율, y축에 검증 loss를 놓고 각 점의 routed rank와 활성 파라미터를 표시한다. 같은 총 예산의 점만 한 곡선으로 연결한다. 계산 비용 대비 loss 그림을 별도로 제시한다.

## 8. 기여의 판단 기준

- 특정 배분이 더 좋다면, 검증된 규모·데이터·예산 범위에서의 배분 효과로 보고한다.
- 폭 축소 대조군과 차이가 없다면, 결과를 low-rank 고유의 장점으로 주장하지 않는다.
- 동일 토큰 수에서만 유리하다면, 추가 활성 계산량을 사용한 품질 개선으로 해석하고 compute efficiency 주장을 제한한다.
- Shared 확대가 낮은 rank의 손실을 회복하지 못해도, 어떤 지점부터 conditional 용량 감소가 지배하는지 보고할 수 있다.
- 기존 optimizer의 적용은 실험 통제 또는 적용 연구로 기술한다. 새로운 optimization 기여는 별도 방법과 근거가 있을 때만 주장한다.
- 단일 모델의 최적 S/r을 보편적인 최적 비율이나 scaling law로 일반화하지 않는다.

## 9. 실행 전에 확정할 사항과 다음 작업

1. GPU 종류·수·사용 시간에 따른 작은 모델과 학습 예산을 정한다.
2. 데이터셋, 이용 조건, tokenizer, train/validation 분할 및 평가 프로토콜을 확정한다.
3. FFN 종류, factorize할 행렬, shared 폭, N/K, MoE 층 배치를 정한다.
4. 초기화와 shared 출력 합산의 스케일 규칙을 정해 S 변화가 초기 출력 크기 변화로 혼동되지 않도록 한다.
5. 실제 파라미터 수에 기반한 3–5개 배분점과 예산 허용 오차를 정한다.
6. seed 수, optimizer 탐색 예산, 실패·중단 기준을 본 실험 전에 기록한다.
7. 우선 읽을 원문은 Slicing and Dicing, Towards Greater Leverage, LightMoE이다. 이어 LORO와 Spectron의 실제 업데이트 및 초기화 조건을 확인한다.


## 10. 참고문헌

아래 링크는 본 계획의 근거로 검토한 원문이다. 세부 학회 상태를 확인하지 않은 연구에는 임의의 학회명을 부여하지 않았다.

1. Dai et al. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.** 2024. [원문](https://arxiv.org/html/2401.06066v1) — §3.2, §4.4.
2. Tian et al. **Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models.** 2025. [원문](https://arxiv.org/html/2507.17702) — §4.1.3, 부록의 모델 설정.
3. Zhao et al. **Towards a Comprehensive Scaling Law of Mixture-of-Experts.** 2025. [원문](https://arxiv.org/html/2509.23678) — §4.5, §5.2.
4. Li et al. **Slicing and Dicing: Configuring Optimal Mixtures of Experts.** 2026, arXiv 프리프린트. [원문](https://arxiv.org/html/2605.11689) — §2.1, §3.1–3.2.
5. Hao et al. **LightMoE: Reducing Mixture-of-Experts Redundancy through Expert Replacing.** 2026, arXiv 프리프린트. [원문](https://arxiv.org/html/2603.12645v1) — §3.2, §4.3, 부록 A.4.
6. Yang et al. **Multi-Task Dense Prediction via Mixture of Low-Rank Experts.** CVPR 2024. [원문](https://arxiv.org/html/2403.17749v1) — §3.3, Table 2, 부록 B.2.
7. Ablin et al. **Soup-of-Experts: Pretraining Specialist Models via Parameters Averaging.** 2025. [원문](https://arxiv.org/html/2502.01804v1) — low-rank expert 및 고정 파라미터 비교.
8. Mo et al. **Parameter and Memory Efficient Pretraining via Low-rank Riemannian Optimization.** ICLR 2025. [공식 논문 페이지](https://proceedings.iclr.cc/paper_files/paper/2025/hash/16fafdc5e8ab22780873d607ce4387a8-Abstract-Conference.html).
9. Janson et al. **Stabilizing Native Low-Rank LLM Pretraining.** 2026. [논문](https://arxiv.org/abs/2602.12429) — Spectron과 native low-rank 학습 안정성.
