# Shared 용량은 조건부 계산에 필요한 routed rank를 낮추는가?

> **2026-10-07 seed401 개정:** [seed 반복 계획](docs/public_release.md)에 AdamW [HL → HS → LS → LL] → Spectron [HS → LL] 순서를 확정했다. 추가 두 조건은 동일 seed/학습량·LR 3e-4 고정 recipe 탐색 비교이며, 기존 AdamW 3-seed 반복과 별도로 보고한다. 전용 실행 준비·GPU 검증은 미완료다.

> **2026-10-01 다음 실험 보완:** [C 이후 대조 계획](docs/public_release.md)에 무제약 DENSE_ROUTED-S/L과 Spectron/AdamW anchor 비교를 추가했다. LR/DW 핵심 8셀·확증 seeds {101,202,303}·기존 대비는 유지하며, 추가 조건·tuning·GPU 비용은 별도 산정한다.
## Low-Rank-MoE 실험계획 v0.2

- 작성일: 2026-09-24
- 문서 상태: 실행 전 검토용 초안. 실험 승인·사전등록·결과 보고서가 아니다.
- 확인한 것: 기존 연구계획, 관련 원문, 아래 구성의 대수적 파라미터 수.
- 아직 확인하지 않은 것: 구현의 실제 파라미터/FLOPs, 데이터 파일·revision, 학습 안정성, GPU 처리량·비용, 가설의 성립.
- 목적: shared 확대가 작은 routed 경로를 보완하는지, 입력별 expert 선택의 가치가 남는지, 폭 축소로도 설명되는지를 구분한다.
- 범위: 영어 소규모 decoder-only LM의 native pretraining. 모든 모델 가중치를 무작위 초기화한다.
- 문서 관계: [research_plan.md](research_plan.md)의 후속 실행 설계. 기존 연구계획·논문 메모는 수정하지 않는다.
- 실행 경계: 이번 작업은 계획서 작성이다. 데이터 다운로드, 환경 설치, 모델 학습, GPU 결제는 수행하지 않았다.
- 예산 전제: [gpu_rental_survey.md](gpu_rental_survey.md)에 기록된 총 500,000원 상한을 설계에 적용한다. 과거 GPU 가격·환율은 현재 견적으로 사용하지 않는다.
- 수치의 위상: 아래 모델·토큰·학습률은 실행 가능한 후보를 만들기 위한 제안이다. 측정값이나 사용자 확정 사항이 아니다. 파일럿 후 §11의 실행 명세를 고정한다.

## 1. 쉽게 읽는 실험 목표

공통 담당자(shared)를 키우면, 선택적으로 일하는 전문가(routed)를 작게 만들어도 되는지 확인한다. 성능만 좋아지는 것 외에 다음을 함께 확인한다.

1. 작은 전문가가 큰 전문가보다 shared 확대의 도움을 더 받는가?
2. 전문가를 단순히 좁게 만들어도 같은 결과가 나오는가?
3. 입력에 맞는 전문가 선택이 여전히 유용한가?
4. 계산을 더 사용한 효과를 제외하고도 실제 이득이 있는가?

최적 shared 개수 한 점을 찾는 대신, 품질·routing 의존성·계산 비용이 함께 변하는 구간을 기록한다. 여기서 rank는 factorized weight의 제한이며, 지식량이나 전문화 정도의 직접 측정치가 아니다.

## 2. 질문, 가설, 반증 조건

| ID | 사전 질문 | 지지에 필요한 관측 | 반대 결과의 의미 |
|---|---|---|---|
| H1 | shared 확대가 낮은 rank와 높은 rank의 loss 격차를 줄이는가? | §7의 I_LR가 양수 | 상대적 보완을 지지하지 않음; 실제 개선과 별도 판정 |
| H2 | 그 상대적 보완이 폭 축소 대조보다 큰가? | I_LR와 J가 모두 양수 | LR에 특수한 상대적 보완은 미지지 또는 불확실 |
| H3 | 고정예산 재배분 후 품질이 유지되어도 입력별 expert 선택이 유용한가? | §7.3의 Δ_budget 비열등성 및 routing 교란·추가 학습 대조의 일관된 결과 | 품질 유지 또는 routing 유용성은 미확인; 교란 손해가 작으면 경로 대체·expert 호환성과 양립 |
| H4 | 같은 총예산의 배분이 계산 비용까지 고려해 유리한가? | 반복된 배분 비교 및 별도 동일 학습 FLOPs 비교 | 동일 토큰에서의 품질–계산량 교환으로 주장 제한 |

주 분석은 H1/H2의 상대적 보완과 §7.3의 고정예산 품질 유지(Δ_budget)다. 실제 개선(G_low)과 LL에서의 DW 대비 품질(Δ)은 별도 사전 대비로 구분한다. H3는 기능적 의존성 진단, H4는 실용성 확장이다. H1/H2가 실패해도 결과를 숨기지 않으며, H3/H4만 좋아졌다는 이유로 주가설 성공으로 바꾸지 않는다.

## 3. 선행연구를 반영한 범위

- StructMoE 원고는 dense MLP 안에 routed low-rank 성분을 넣고 폭을 조정하여 파라미터를 맞춘 native pretraining을 수행한다. 우리 계획은 별도 비선형 FFN 출력의 합산이다. 결합 자체나 native 학습 자체의 최초성을 주장하지 않는다. [원문](https://openreview.net/pdf/7eda692ef4f9549b74c2de9283097ae444af2fd6.pdf)
- Slicing and Dicing의 generalist 결과는 shared 확대가 보편적으로 유리하지 않을 수 있음을 보여 준다. 저랭크 제약 아래의 조건부 효과를 검증한다. [원문 §3.2](https://arxiv.org/html/2605.11689)
- SD-MoE의 공통 spectral 성분과 expert별 성분 구분은 rank 배치에 대한 경쟁 설명이다. shared spectral rank와 native factorized parameter budget은 동일하지 않다. [원문 §3](https://arxiv.org/html/2602.12556)
- Spectron은 native low-rank 최적화의 안정성을 다루므로 표현력과 최적화 효과를 구분하는 후속 대조로 사용한다. [원문, FFN-only 부록 포함](https://arxiv.org/html/2602.12429)
- 선행연구의 shared 개수 비율을 우리 shared 파라미터 비율로 옮기지 않는다. 문헌 검색은 전수 검토가 아니며, 모든 후속 대조를 처음부터 수행하지 않는다.

## 4. 제안 모델과 변경 변수

### 4.1 파일럿 시작 구성

| 항목 | 제안 값 |
|---|---|
| 층 수 / hidden dimension | 6 / 384 |
| Attention | full multi-head attention, 6 heads, head dimension 64; Q/K/V/O projection 모두 bias 없음 |
| 위치 / normalization | RoPE / pre-RMSNorm, 층당 2개 및 final RMSNorm |
| FFN 위치 | 6개 층 모두 MoE |
| Expert | bias 없는 SwiGLU |
| Routed 수 N / 토큰당 선택 K | 8 / 2, 전 조건 고정 |
| Low-rank routed intermediate width | 768, 전 조건 고정 |
| Factorization | gate/up/down 3개 행렬 모두 동일 r; factor 사이 추가 activation 없음 |
| Shared | 하나의 dense SwiGLU, 폭 m_s를 조절; 별도 입력별 gate 없음 |
| FFN 합산 | shared 출력 + 선택된 routed 출력의 gate-weighted sum |
| Router | bias 없는 384×8 linear, FP32 softmax, 선택된 top-2 weight 재정규화 |
| Token dropping | 없음; 모든 토큰 처리 |
| Load balancing | 아래 정의의 보조손실 및 계수 0.01, 전 조건 동일 |
| Vocabulary / embedding | GPT-2 tokenizer, 50,257; input/output embedding tying |
| Context length | 512 |
| Dropout | 0 |
| Precision | GPU 지원 시 BF16 matmul, FP32 master/optimizer state; 실제 지원 확인 후 고정 |

GPT-2의 tokenizer만 사용하고 pretrained model weight는 사용하지 않는다. tokenizer revision과 special token 설정을 기록한다. [공개 tokenizer/model card](https://huggingface.co/openai-community/gpt2)

Shared 수 대신 폭을 쓰는 이유: 같은 입력을 받고 출력을 단순 합산하는 독립 SwiGLU들은 하나의 넓은 SwiGLU로 표현할 수 있다. 별도의 gate/normalization을 추가하지 않는다. 본 계획의 “full-rank”는 factorization을 강제하지 않는다는 뜻이며 실제 학습 rank의 최대값을 보장하지 않는다.

Router 보조손실 정의: 유효 token 수 T, 선택 횟수 기반 f_i=count_i/(KT), top-K 이전 softmax 확률 평균 p_i를 사용해 L_bal=N∑_i f_i p_i로 둔다. f_i의 hard assignment에는 gradient를 흘리지 않고 p_i로 router를 학습한다. 균일 사용이면 L_bal=1이며 coefficient 0.01을 곱한다. 논문 구현의 정규화와 다르면 재현이라고 부르지 않는다.

### 4.2 초기화와 학습의 공통 원칙

- 모든 조건에서 attention/embedding/router 등의 공통 shape는 모듈별 고정 RNG stream으로 같은 seed 초기화를 사용한다. 변수 크기가 달라도 뒤쪽 공통 모듈의 RNG 소비가 달라지지 않게 한다.
- Dense linear는 fan-in 분산을 따르고, factorized W=BA는 A의 원소 분산 1/d_in, B의 원소 분산 1/r로 시작한다. 그러면 독립 초기화에서 W 원소 분산은 1/d_in이다. factor 중 하나를 0으로 초기화하지 않는다.
- 각 FFN의 down projection과 attention output에는 동일한 depth scaling 1/sqrt(2L)을 적용한다. 이는 제안 규칙이며 파일럿에서 출력 RMS·gradient 유한성을 확인한다.
- Shared와 routed의 상대 출력 계수는 1로 고정한다. 조건마다 평가 성능에 맞춰 출력 scale을 튜닝하지 않는다. 폭 변화에 따른 초기 출력 RMS를 기록한다.
- 같은 seed에서는 동일한 학습 문서 순서, global batch 및 토큰 budget을 사용한다. shape가 다른 expert weight를 억지로 같은 난수 행렬이라고 취급하지 않는다.

## 5. 파라미터 산술과 비교군

d=384, routed width m=768이면 한 low-rank expert의 파라미터 수는 다음과 같다.

$$
p_r(r)=3r(d+m)=3456r,\qquad Q(m_s)=3dm_s=1152m_s.
$$

따라서 router를 제외한 층당 expert 총예산과 활성 파라미터는 다음과 같다.

$$
P_E=1152m_s+8(3456r),\qquad
P_{\mathrm{act},E}=1152m_s+2(3456r).
$$

고정 총 expert 예산을 2,359,296으로 두면 m_s+24r=2048이다.

| 배분점 | r | shared 폭 m_s | 폭 대조군의 routed 폭 m' | 층당 총 expert | 층당 활성 expert |
|---|---:|---:|---:|---:|---:|
| A: routed 쪽 용량 큼 | 64 | 512 | 192 | 2,359,296 | 1,032,192 |
| M: 중간점 | 48 | 896 | 144 | 2,359,296 | 1,363,968 |
| B: shared 쪽 용량 큼 | 32 | 1280 | 96 | 2,359,296 | 1,695,744 |

폭 축소 dense 대조군(DW)은 3dm'=p_r(r), 즉 m'=3r로 둔다. LR과 DW의 각 대응 셀은 shared 구조, 총/활성 expert 파라미터가 정확히 같다. matrix multiply 항 외 activation·kernel·optimizer 연산은 다르므로 FLOPs 및 시간까지 같다고 단정하지 않는다. DW의 r 표기는 비교용 예산 라벨이며 DW에 rank 제한이 있다는 뜻이 아니다.

### 5.1 핵심 2×2×2 설계: 8조건

| 셀 | rank 예산 라벨 | shared 폭 | 층당 총 expert | LR 모델 / DW 모델 |
|---|---:|---:|---:|---|
| HS | 64 | 512 | 2,359,296 | LR-HS / DW-HS |
| HL | 64 | 1280 | 3,244,032 | LR-HL / DW-HL |
| LS | 32 | 512 | 1,474,560 | LR-LS / DW-LS |
| LL | 32 | 1280 | 2,359,296 | LR-LL / DW-LL |

HS와 LL이 고정 총예산 배분곡선의 양 끝이다. HL/LS는 상호작용을 분리하는 예산 밖 대조군이며 고정예산 성능 순위에 섞지 않는다. HL/LS까지 예산을 같게 만들기 위해 hidden size나 층 수를 바꾸지 않는다.

### 5.2 추가 탐색 3조건

- LR-M, DW-M: 위 표의 중간 배분점.
- SHARED-ONLY: router와 routed experts 없이 shared 폭 2048. 층당 expert 예산은 같지만 router 제거로 층당 3,072개 파라미터가 적다. 숨기거나 사용되지 않는 dummy parameter로 메우지 않는다.

총 11개 architecture 조건이며, 핵심 확증은 8조건 전체를 대상으로 한다. 중간점과 dense 경계는 탐색 후 별도 예산이 있을 때 동일한 반복을 추가한다.

### 5.3 대수 계산과 상한 점검

위 attention/RMSNorm/bias/embedding 가정을 적용하면 A/M/B 모델은 37,016,832개, LS는 31,708,416개, HL은 42,325,248개다. SHARED-ONLY는 36,998,400개로 고정예산 모델보다 18,432개(전체 약 0.0498%) 적다. 이 수치는 구현 실측이 아니라 대수값이다.

- 파라미터 절약 조건: r < dm/(d+m)=256. 후보 32/48/64는 이 범위 안이다.
- Shared 폭 비음수 조건: r≤2048/24. 고정예산곡선은 정수 차원에서만 구성한다.
- r=0은 일반 factorization 조건이 아니라 routed를 제거한 별도 경계다.
- 고정예산에서 P_act,E=(K/N)P_E+(1-K/N)Q. shared를 늘리면 활성 계산이 증가하는 방향이다. 품질 증가가 이를 넘어선 효율 이득인지는 별도 확인한다.
- Cross-entropy 차이에 유용한 보편적 효과 상한은 이 산술만으로 도출할 수 없다. 예상 효과 크기·검정력은 미확인이다.
- I/J는 모델별 관측 손실에서 계산되며 rank나 Q의 대수식을 loss에 대입해 효과를 만들어내지 않는다. 강제로 낮춘 weight rank를 가설의 성공 지표로 사용하지 않는다.

## 6. 데이터와 평가 누수 방지

### 6.1 기본 후보

FineWeb-Edu의 영어 텍스트를 기본 후보로 삼는다. 공개 데이터 카드에서 영어 데이터와 ODC-BY 표기를 확인했다. 세부 config·파일 revision·샘플 품질·실제 토큰 수는 아직 검사하지 않았다. [데이터 카드](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

실행 전 다음 순서로 데이터 manifest를 만든다.

1. config, immutable revision, 사용할 shard 목록과 각 checksum을 고정한다. 파일 순서의 앞부분만 편의대로 사용하는 대신 고정 seed로 shard/문서 표집한다.
2. 본문 결측·빈 문자열·언어·중복 id를 검사하고 제외 수를 기록한다.
3. 같은 canonical URL 또는 정규화 본문 SHA-256을 공유하는 문서들을 한 그룹으로 묶는다. 같은 그룹은 한 split에만 둔다.
4. 그룹 hash로 train 98%, dev 1%, test 1%를 배정한다. 데이터 적재 순서와 무관한 고정 규칙을 사용한다.
5. split 간 exact text overlap은 0임을 확인한다. 근접 중복은 표본 감사하고 잔존율·미해결 범위를 기록한다. exact dedup만으로 완전한 누수 제거를 주장하지 않는다.
6. train/dev/test를 따로 토큰화한 뒤 EOS로 문서를 구분하고 길이 512로 packing한다. split 경계를 넘는 packing은 금지한다. 문서 간 causal context 허용 여부는 “허용, EOS 구분”으로 전 조건 고정한다.
7. dev/test는 각각 약 2M 평가 target tokens를 확보하고 최종 실제 개수를 고정한다. train은 최대 계획 토큰량을 중복 순환 없이 공급하도록 준비한다. 부족하면 config 변경을 기록하고 실행 명세를 다시 고정한다.

Dev는 학습률/파일럿 선택용이다. Test는 확증 설정을 고정한 뒤 최종 체크포인트 평가에만 사용한다. 중간 체크포인트의 test 순위로 모델을 고르지 않는다. 단일 교육 웹 데이터 결과로 도메인별 전문화·한국어·코드·일반 LLM 성능을 주장하지 않는다.

## 7. 측정값과 사전 대비

### 7.1 주 지표

- Token-weighted next-token cross-entropy, 단위 nats/token; padding은 제외한다.
- Perplexity는 exp(loss)로 보조 보고한다.
- Router 보조손실은 주 평가 loss에 포함하지 않고 별도로 기록한다.
- 마지막 고정 token checkpoint를 비교한다. dev-best checkpoint를 주 결과로 대체하지 않는다.

### 7.2 보완 효과와 폭 축소 대조

L_f(r,s)는 family f∈{LR,DW}의 held-out loss다. s_S=512, s_L=1280, r_L=32, r_H=64로 둔다.

$$
I_f=
[L_f(r_L,s_S)-L_f(r_L,s_L)]
-
[L_f(r_H,s_S)-L_f(r_H,s_L)],\qquad
J=I_{\mathrm{LR}}-I_{\mathrm{DW}}.
$$

I_LR>0은 shared 확대에 따라 낮은 rank와 높은 rank의 loss 격차가 줄어드는 상대적 보완을 뜻한다. 양쪽 loss가 모두 악화되어도 낮은 rank의 악화가 더 작으면 성립한다. J>0만으로 상대적 보완이 있는 것도 아니다. I_LR와 I_DW가 모두 음수여도 J는 양수일 수 있다. H2의 지지는 I_LR>0과 J>0을 함께 요구한다.

낮은 rank에서 shared 확대가 실제 loss를 낮추는지는 별도 사전 대비 G_low=L_LR,LS−L_LR,LL로 측정한다. G_low의 95% CI 하한이 0보다 클 때만 실제 개선을 잠정 지지한다. “실제 개선을 동반한 상대적 보완”은 I_LR와 G_low의 CI 하한이 모두 0보다 커야 한다. LR에 특수한 상대적 보완까지 주장하려면 J의 CI 하한도 0보다 커야 한다.

가상 반례: LR의 (HS,HL,LS,LL) loss가 (3.00,3.05,3.10,3.12)이면 I_LR=0.03이지만 G_low=−0.02다. shared 확대가 양쪽 모두 해로우므로 실제 개선이라고 부르지 않는다. 이 수치는 관측 결과가 아니다.

J는 예산을 맞춘 두 매개변수화 방식의 상호작용 차이다. 공통 지식과 전문 지식이 내부적으로 분리되었다는 증명으로 해석하지 않는다. 변화량과 함께 네 셀의 원 loss 및 DW 결과를 반드시 공개한다.

### 7.3 “품질 유지”의 범위

고정예산 품질 유지의 사전 주 비교는 LR family의 HS→LL 재배분이다. Δ_budget=L_LR,LL−L_LR,HS로 정의한다. 두 셀은 총 파라미터가 같고 rank/shared 배분이 다르다. 기존 핵심 24회에서 seed별 paired 차이를 계산하므로 추가 학습은 필요 없다.

LL 셀에서 매개변수화 방식의 품질 차이는 별도 사전 대비 Δ=L_LR,LL−L_DW,LL로 유지한다. Δ의 비열등성이 Δ_budget의 비열등성을 대신하지 않는다. DW의 HS→LL 차이와 나머지 셀의 LR−DW 차이는 보조 결과로 모두 보고하되 좋은 비교를 골라 주 비교로 바꾸지 않는다.

잠정 실용 허용치는 ε_budget=0.01 nats/token, ε_DW=0.01 nats/token으로 각각 제안한다. 같은 tokenizer에서 perplexity 비 약 exp(0.01)=1.010에 해당한다. 이 값들은 문헌의 보편적 기준이나 측정된 노이즈가 아니다. 확증 전 각각 고정하고 결과에 맞춰 넓히지 않는다.

| 판정 대상 | 사전 기준 | 허용되는 해석 |
|---|---|---|
| 고정예산 재배분의 품질 유지 | Δ_budget의 95% CI 상한 < ε_budget | 시험한 HS→LL 재배분에서 비열등성 잠정 지지 |
| 고정예산 재배분의 품질 개선 | Δ_budget의 95% CI 상한 < 0 | 시험한 HS→LL 재배분에서 loss 개선 잠정 지지 |
| LL의 DW 대비 품질 유지 | Δ의 95% CI 상한 < ε_DW | LL의 예산 일치 DW 대비 비열등성 잠정 지지 |

각 판정은 해당 질문에만 적용한다. H1/H2만 통과했다고 고정예산 품질 유지를 선언하지 않으며, Δ_budget만 통과했다고 상대적 보완이나 LR 고유의 이득을 선언하지 않는다. 두 비열등성을 함께 주장할 때는 두 기준 모두 통과해야 한다. 여러 대비 중 하나라도 통과하면 연구 전체가 성공이라는 판정은 사용하지 않는다.

고정예산 품질 유지도 같은 활성 계산량이나 계산 효율을 뜻하지 않는다. LL은 HS보다 활성 expert 파라미터가 많으므로 §8 단계 4의 동일 학습 FLOPs 비교를 별도로 수행한다. DW 대비 비열등성은 같은 큰 intermediate width의 무제약 dense routed와 동등하다는 주장도 아니다.

현재 핵심 반복은 고정예산 양 끝점 비교를 제공한다. 중간점과 SHARED-ONLY의 반복은 §8 단계 4의 선택적 확장이므로 전체 배분곡선이나 최적 배분을 확정하지 않는다. 현재 grid만으로 “필요 최소 rank”를 산출하지 않는다. 경계가 보일 때 인접 rank를 추가하고 “시험한 범위의 최소 rank”로만 보고한다.

### 7.4 Routing의 기능적 진단

- 원래 routing과 교란 routing의 loss 차이 R=L_shuffle−L_normal을 계산한다.
- 한 층씩 개입한다. 고정된 평가 microbatch의 모든 유효 token position 사이에서 top-K expert id와 gate weight 묶음 전체를 permutation한다. Expert별 부하·전체 gate-weight 분포·K는 보존한다.
- Dropout을 끄고, 다른 층은 정상 routing으로 실행한다. 같은 token 집합/배치 순서와 permutation seed를 전 모델에 사용한다.
- dev에서 고정 선택한 약 256K target tokens에 대해 층당 5개 permutation을 수행한다. 원래 배정과 같은 비율, R의 층별 값/seed별 값을 기록한다. 배치 크기에 따른 민감도는 파일럿에서 확인한다.
- 여러 permutation은 평가 Monte Carlo 반복이며 독립 학습 seed 수에 더하지 않는다.
- R가 작다는 결과만으로 전문가가 무의미하다거나 shared가 모든 일을 한다고 확정하지 않는다. expert 호환성·측정 민감도도 가능한 설명이다.
- 후속 확증은 대표 조건에서 입력과 무관한 균형 무작위 routing으로 처음부터 학습한 대조다. K/expected load/데이터/학습량을 맞춘다. 이것이 없으면 “학습된 routing에 대한 사후 의존성” 수준으로 제한한다.

### 7.5 시스템·안정성 지표

총/활성 파라미터, forward/backward 및 optimizer FLOPs 산정 범위, tokens/s, peak allocated/reserved GPU memory, 평가·저장 시간을 기록한다. Router utilization, entropy, gradient norm, loss spike, factor/update norm, shared/routed 출력 RMS를 기록한다. 출력 RMS나 token 부하 균형은 전문화의 증거로 대체하지 않는다.

## 8. 실행 단계와 진행 기준

### 단계 0 — 구현 정확성 및 비용 파일럿

- CPU의 작은 tensor에서 factorized forward/gradient를 명시적 BA와 비교한다.
- 실제 모델 parameter count를 §5와 대조한다. paired LR/DW의 trainable parameter 오차는 0이어야 한다. 차이가 나면 학습 전에 원인을 수정한다.
- top-K weight 합, dropless token accounting, router 보조손실의 K 정규화, tied embedding, packing/target mask를 확인한다.
- checkpoint 저장/복원 후 짧은 continuation을 비교한다. dtype별 허용 오차와 RNG 복원을 기록한다.
- GPU에서 최대 모델 HL 및 A/M/B 양 family를 측정한다. global target batch를 유지하도록 microbatch/gradient accumulation을 정한다. 모든 조건에 공통으로 들어가는 microbatch를 우선 사용한다.
- warmup 이후 100 step 이상 처리량을 측정하고 평가/저장 시간을 포함해 비용을 예상한다. 미래 긴 학습의 안정성을 보장하는 검사는 아니다.
- 유한하지 않은 loss/gradient, parameter mismatch, 복구 실패가 있으면 다음 단계로 가지 않는다.

### 구현 보완 기록 (2026-09-27)

Spectron factor optimizer와 별도 AdamW 대조 recipe를 실행 가능한 코드로 추가했다. 세부 설정·원문 대비 차이·검증 경계는 [optimizer 안정성 구현 기록](docs/optimizer_stability.md)을 따른다. 새 mini stability protocol은 optimizer-step 전체의 router balancing과 idle expert zero-gradient 갱신을 명시한다. global balancing의 추가 forward 및 optimizer·진단 비용을 처리량/FLOPs에 포함한다. 기존 `microbatch` balancing 결과와 혼합하지 않는다. 이는 구현 준비이며 GPU 품질·안정성 검증 완료를 뜻하지 않는다.

### 단계 1 — 공정한 짧은 학습률 탐색

기본 AdamW 후보: beta=(0.9,0.95), eps=1e-8, weight decay=0.1(embedding/norm/bias 제외), gradient clip=1.0, peak LR∈{1e-4,3e-4,1e-3}. LR factor A/B 양쪽에 decay를 적용하는 규칙을 기록한다. Warmup은 target token budget의 5%, 이후 cosine으로 peak의 10%까지 감소시킨다.

Global batch 후보는 32,768 target tokens/optimizer step이다. padding/shift로 실제 유효 token 수가 달라지면 실제 count로 schedule과 예산을 계산한다.

- Family별 HS/LL 두 anchor × 3개 LR × seed 17, run당 약 8M target tokens: 최대 12회.
- family 내부 두 anchor의 마지막 dev loss 평균이 가장 낮은 안정적 LR을 선택한다. 한 anchor라도 발산한 후보는 선택하지 않는다.
- Family마다 하나의 LR을 이후 모든 셀에 고정한다. LR family와 DW family의 최적 LR이 다를 수 있으므로 주 결과는 “동일 tuning budget의 학습 recipe 비교”다.
- SHARED-ONLY는 DW family에서 선택한 LR을 그대로 사용한다. 별도 tuning run은 추가하지 않으며 단계 1은 최대 12회로 유지한다. 이는 독립 최적화한 dense baseline이 아닌 DW recipe를 적용한 경계 비교다. SHARED-ONLY 실패 시 임의의 LR 재탐색으로 교체하지 않고 실패를 기록한다. 별도 tuning이 필요하면 변경 이유와 추가 예산을 실행 전에 기록하고 독립 seed 평가 전에 명세를 다시 고정한다.
- Family별 튜닝 때문에 구조 효과와 optimizer 설정이 완전히 분리되는 것은 아니다. 대표점의 공통 LR/optimizer 대조는 해당 점의 민감도만 진단한다. J 전체에 대한 recipe 의존성을 점검하려면 공통으로 고정한 안정적 recipe 아래 핵심 8셀을 같은 독립 seeds/학습량으로 반복하여 I/J를 재추정해야 한다. 이 추가 실험은 현재 핵심 24회 예산에 포함되지 않는다. 수행하지 않으면 J는 동일 tuning budget의 recipe 비교로 해석한다. 수행해도 시험한 공통 recipe 내 구조 비교이며 optimizer 전반에 독립적인 효과라고 일반화하지 않는다.
- NaN/Inf는 즉시 실패로 기록한다. LR 세 후보가 모두 실패한 family가 있으면 표현력 결론을 내리지 않고 최적화 문제를 먼저 수정한다. 숨겨진 추가 tuning을 하지 않는다.

### 단계 2 — 탐색

11조건 × seed 17 × 약 32M target tokens를 처음부터 학습한다. 단계 1 checkpoint를 이어 쓰지 않는다. 토큰 종료점은 global batch의 정수 배수로 함께 올림하여 사전에 기록한다.

탐색의 목적은 예산·안정성 확인, 곡선의 형태와 효과 방향 파악이다. 단일 seed 결과는 확증에 합치지 않는다. 전체 8셀을 본 뒤 유리한 셀만 남겨 H1/H2를 검정하지 않는다.

- 둘 이상의 family/셀에서 반복적으로 불안정하면 확증 전에 recipe를 개정한다.
- 신호가 작거나 상반되면 효과를 단정하지 않는다. 가장 유리한 점을 선택하는 대신 전체 대비의 불확실성을 기록한다.
- 예산이 부족하면 탐색 결과까지만 보고하고 중단할 수 있다. 이를 부정 결과의 확증으로 부르지 않는다.

### 단계 3 — 독립 seed 재현성 확인 및 잠정 추정: 핵심 8조건

확증 후보는 8조건 전체 × seed {101,202,303}=24회다. 모든 run을 처음부터 학습한다. 독립 단위는 seed이며 같은 seed 내 셀을 paired 비교한다.

학습량 후보 D∈{128M,256M,512M} target tokens 중, 모든 24회와 평가/저장/실패 여유를 포함해 §9 예산에 들어가는 가장 큰 값을 선택한다. 처리량과 비용으로 선택하며 test 성능으로 선택하지 않는다. D는 batch 정수 배수로 공동 올림한다.

- 최소 128M도 감당할 수 없으면 seed 수나 불리한 셀을 몰래 줄이지 않는다. 확증을 보류하고 더 작은 모델의 새로운 프로토콜 또는 추가 자원을 검토한다.
- 128M/256M/512M은 수렴 보장이 아니다. 해당 유한 학습 budget에서의 결과로 기술한다.
- 중간 dev curve는 안정성 점검에 사용하고, 결과가 좋아 보이는 시점에 특정 조건만 조기 종료하지 않는다.
- 3 seed의 검정력은 미확인이다. I_LR, I_DW, J, G_low, Δ_budget, Δ 각각을 seed 내에서 계산한 뒤, 개별 seed 대비, 평균, 표준편차와 df=2의 paired t 기반 95% CI를 보고한다. 정규성 검증이나 높은 검정력을 확보했다고 주장하지 않는다.
- H1은 I_LR의 CI 하한>0, H2는 I_LR와 J의 CI 하한이 모두>0인 경우에만 상대적 보완의 잠정 지지를 표시한다. 실제 개선은 §7.2의 G_low 기준, 고정예산 및 DW 대비 품질 유지는 §7.3의 각 비열등성 기준으로 따로 판정한다. 두 양의 조건을 함께 요구하는 판정이며 어느 하나만 골라 성공으로 바꾸지 않는다. 그 외는 부호와 CI를 그대로 보고하고 불확실/미지지로 구별한다. 3 seed CI는 분포 가정에 민감하여 강한 확증으로 부르지 않는다.
- Token, checkpoint, layer, permutation을 독립 표본으로 늘려 p-value를 작게 만들지 않는다. 보조 비교는 탐색적으로 표시한다.
- 발산은 결과다. 재현 가능한 수치 실패를 좋은 seed로 대체하지 않는다. 하드웨어 장애는 동일 config/seed로 복구하고 비용·사유를 기록한다. 불완전 seed block이 생기면 완전한 paired contrast가 없는 seed는 그 대비를 계산할 수 없다고 표시한다. 나머지 완결 seed만으로 사전 반복 요건을 충족했다고 주장하지 않는다.

### 단계 4 — 배분곡선과 계산 예산의 확장

추가 예산이 있으면 LR-M, DW-M, SHARED-ONLY에 같은 3 seeds/D를 적용한다: 추가 9회, 총 33회. 이것이 없으면 중간점/경계 곡선은 탐색 결과이며 최적 배분점 확정으로 쓰지 않는다.

동일 학습 FLOPs 비교는 별도 run이다. 기본 비교는 LR-HS, LR-LL, DW-LL의 3조건이다. 각 condition의 모델/optimizer 연산을 포함한 비용 모델을 고정하고, LR-HS의 D-token 학습 비용 C를 기준으로 각 condition의 토큰량을 역산한다. 모든 run은 처음부터 해당 종료점에 맞춘 동일 형태의 schedule로 학습한다. 공통 train stream prefix를 사용하며 데이터 길이 차이를 보고한다.

이미 학습한 token-matched run을 중간에서 잘라 최종 scheduler run과 동등하다고 취급하지 않는다. 실행하지 못하면 compute efficiency를 주장하지 않는다. Wall-clock 효율은 같은 GPU/SW/batch에서 별도 보고한다.

## 9. 총비용과 실행 가능성

| 용도 | 제안 상한 |
|---|---:|
| 환경·정확성·처리량·학습률 파일럿 | 50,000원 |
| 11조건 탐색 | 70,000원 |
| 핵심 24회 반복 및 여유가 있는 확장 | 280,000원 |
| 저장·전송·중단·재실행 등 예비비 | 100,000원 |
| 총액 | 500,000원 |

이 배분은 지출 승인이 아니다. 실제 서버 가격·세금·재고·계정 권한·지속 실행 가능성은 대여 직전에 확인한다.

Condition c의 측정 처리량을 v_c, target tokens를 D_c, 반복 수를 n_c, 평가/저장 시간을 h_c, 세금 포함 전체 서버 시간당 금액을 p라 하면 다음으로 계산한다.

$$
H=\sum_c n_c\left(\frac{D_c}{3600v_c}+h_c\right),\qquad
B_{\mathrm{compute}}=pH.
$$

GPU뿐 아니라 CPU/RAM 기본 과금, idle 설치, checkpoint 저장소·전송, optimizer 진단 overhead와 재실행도 포함한다. 핵심 24회만 최소 3.072B target tokens이며, 실제 step 올림 때문에 소폭 증가한다. 이 산술을 단일 GPU 완주 가능성의 증거로 해석하지 않는다.

잔여 예산이 부족하면 우선 §10의 추가 모델/도메인/역배치 실험과 단계 4 확장을 미룬다. 핵심 8셀 전체 반복이 불가능하면 실행 범위를 재설계하고 문서 버전을 올린다. 복수 seed와 필수 대조를 희생해 큰 모델 한 개의 좋은 결과만 남기지 않는다.

## 10. 결과에 따라 수행할 후속 실험

| 관측 | 다음 실험 | 주장 제한 |
|---|---|---|
| I_LR>0, J도 양의 방향 | 대표점의 공통 LR/optimizer 민감도 진단; J에 대한 추가 주장은 공통 recipe의 핵심 8셀 반복 후 재추정 | 대표점만 반복하면 J는 recipe 비교로 제한; 8셀 반복도 해당 recipe 범위의 근거 |
| 낮은 rank의 loss spike/발산 | Spectron 적용 가능성 검증, 동일 tuning budget 대조 | 안정화 전 표현력 부족으로 결론내리지 않음 |
| Δ_budget 비열등성 지지, routing 교란 손해 작음 | 입력 독립 routing으로 처음부터 학습 | 경로 대체인지 전문가 호환성인지 구분 |
| DW와 동등 | 일반적인 용량 배분 연구로 재정의 | low-rank 고유 장점 주장 철회 |
| 안정적 효과 재현 | StructMoE식 결합 대조 | 기존 방식에서도 가능한 효과인지 검증 |
| rank 배치의 근거가 필요 | low-rank shared/dense routed 역배치 | SD-MoE 자체의 재현이라고 부르지 않음 |
| 한 데이터·규모에서 유망 | 별도 데이터 또는 두 번째 규모 | 보편성/scaling law는 그 이후 |

역배치·StructMoE 대조는 branch별 예산과 nonlinearity 위치 차이를 명시한 별도 프로토콜이 필요하다. 현재 계획의 24회 비용에 포함되어 있다고 오해하지 않는다. 처음부터 모두 교차 실험하지 않는다.

## 11. 실행 명세를 고정하는 체크리스트

측정이 필요한 아래 항목은 현재 “미확인”이다. 파일럿 뒤 값을 채운 실행 명세를 추가하며 현재 문서를 승인된 것으로 자동 변경하지 않는다.

- [ ] model config, 실제 total/active parameter count, source commit, dependency lock, precision
- [ ] dataset/tokenizer immutable revisions, shard/checksum manifest, split/duplicate 감사 결과
- [ ] checkpoint/seed 규칙, global batch, 유효 token count 및 종료 step
- [ ] family별 LR 선택 근거, 초기화·decay·router loss 정의, tuning 실패 전체 목록
- [ ] D, 조건 목록, seed 목록, test 공개 시점, ε_budget/ε_DW 및 I/J/G_low/Δ_budget/Δ 사전 대비
- [ ] GPU SKU/VRAM, SW stack, condition별 tokens/s 및 평가 시간, 최종 비용표
- [ ] 모든 핵심 조건의 비용이 상한에 들어가는지, 부족할 때의 보류 결정
- [ ] deviation log와 결과 공개 범위; 본 실험을 시작할 연구 판단

ε_budget/ε_DW 등의 변경이 필요하면 확증 결과를 보기 전에 사유·새 값·시점을 남긴다. 파일럿 loss를 본 뒤 만든 가설은 탐색에서 나온 가설로 표시한다.

## 12. 산출물과 작업 순서

아래 경로는 후속 구현에서 만들 산출물 계획이며 현재 존재한다고 주장하지 않는다.

| 예정 경로 | 목적 |
|---|---|
| configs/experiment_manifest.yaml | 모델·데이터·optimizer·seed·token budget 고정 |
| data/manifests/ | source revision, shard, split, checksum과 중복 감사 |
| src/model.py, src/train.py, src/evaluate.py | 구조·학습·평가 구현 |
| tests/test_budget.py, tests/test_factorization.py, tests/test_routing.py | 산술·forward/gradient·routing·복구 정확성 |
| results/run_manifest.csv | 모든 성공/실패 run의 config·비용·종료 사유 |
| results/metrics.csv | seed/cell별 원 loss와 시스템 지표 |
| results/contrasts.csv | I_LR, I_DW, J, G_low, Δ_budget, Δ, R와 계산 코드 버전 |
| results/figures/ | 2×2 interaction, 고정예산 곡선, routing, loss–compute 그래프 |
| experiment_decisions.md | 실행 선택·변경·중단 사유의 append-only 기록 |

- [ ] 먼저 parameter ledger와 11조건 config를 만든다.
- [ ] 단일 MoE 층을 구현하고 dense/explicit-factor 참조와 비교한다.
- [ ] 전체 LM, checkpoint 복구, 데이터 manifest를 준비한다.
- [ ] 파일럿으로 정확성·학습 안정성·비용을 검증한다.
- [ ] 실행 명세를 고정한 뒤 탐색/독립 반복을 순서대로 수행한다.
- [ ] 전체 셀·실패·비용을 포함해 사전 대비를 계산한다.
- [ ] 확인된 범위에 맞춰 주장하고 필요한 후속 실험만 선택한다.

실제 구현 명령과 코드는 코드베이스 생성 때 별도 구현계획으로 작성한다. 현재 저장소에는 본 실험의 학습 코드나 측정 결과가 확인되지 않았으므로 존재하지 않는 명령을 실행 가능한 것으로 제시하지 않는다.

## 13. 이 계획이 실패하는 지점

1. 가장 큰 HL 조건을 포함한 정확한 구현과 핵심 반복이 예산에 들어가지 않으면 현재 규모에서 실행 불가다.
2. 저랭크 family가 합리적인 tuning에서 불안정하면 보완 가설 검정보다 최적화 진단이 먼저다.
3. 3 seed의 CI가 넓으면 부호 하나로 결론내리지 않는다. 무효과와 증거 부족을 구분한다.
4. I_LR가 양수여도 J가 뚜렷하지 않으면 LR에 특수한 상대적 보완은 미확인이다. I/J가 양수여도 G_low의 개선과 Δ_budget의 비열등성이 지지되지 않으면 실제 개선이나 고정예산 품질 유지를 주장하지 않는다.
5. Token-matched 이득만 있으면 계산 효율이나 속도 개선을 주장할 수 없다.
6. Routing 교란 손해만으로 인간이 이해하는 도메인 전문화를 증명할 수 없다.
7. 작은 모델·짧은 학습의 반대 결과를 모든 규모의 불가능성으로 일반화하지 않는다.
8. 같은 총 파라미터라도 shared가 큰 조건의 더 많은 활성 계산을 숨기면 질문을 잘못 답하게 된다.

## 14. 검토·개정 기록

- 2026-09-24 v0.1: 기존 연구계획에서 후속 실험계획을 분리. 핵심 질문을 shared×rank 보완 및 matched-width 대비로 명시.
- 대수 검토: 2개 독립 검토 갈래 중 예산 검토에서 11조건 수·고정예산점·dense-width 대응을 확인하고 주 작성자가 계산을 재실행했다. 실제 모델 count와 처리량은 미검증이다.
- 식별성 검토: J>0만으로 보완 효과가 성립하지 않는 문제를 반영하여 I_LR>0도 함께 요구했다. 작은 seed 수, recipe별 tuning, routing 교란의 해석 제한을 명시했다.
- 확장 제외 사유: 역배치·StructMoE·다중 규모/도메인 전수 교차는 비용을 통제하고 주 대비의 반복을 우선하기 위해 핵심 실행 목록에서 제외했다.
- 기존 research_plan.md, gpu_rental_survey.md, papers/는 이번 작성에서 수정하지 않았다.
- 본 문서는 계산 및 설계 검토를 거친 초안이다. 실험의 성공·충분한 검정력·최초성·50만 원 내 완주는 확인되지 않았다.

### 2026-09-24 v0.2 — 검토 의견 반영

- §2/7/8/10/11/12/13: 상대적 보완(I/J), 낮은 rank에서의 실제 개선(G_low), 고정예산 HS→LL 품질 유지(Δ_budget), LL의 DW 대비 품질(Δ)을 구분했다. G_low와 Δ_budget은 기존 핵심 24회에서 산출한다. 가상 반례를 추가하고 판정·산출물·실패 조건을 연결했다.
- §4: attention Q/K/V/O projection의 bias 없음을 명시했다. §5의 파라미터 산술과 일치하며 총계는 변경하지 않았다.
- §8/10: 대표점의 공통 LR 대조는 해당 점의 진단으로 제한했다. J의 공통 recipe 재검증에는 핵심 8셀 전체 반복이 필요하고, 추가 비용은 별도라고 명시했다.
- §8: SHARED-ONLY에는 DW의 선택 LR을 적용한다. 독립 tuning은 수행하지 않는 경계 비교로 명시하여 12회 tuning/11조건 탐색/24회 핵심 반복을 유지했다.
- §3/5/6/9의 문헌·산술·데이터·예산 본문은 이번 개정에서 수정하지 않았다. research_plan.md, gpu_rental_survey.md, papers/도 수정하지 않았다.
- 이번 개정은 실행 전 설계 수정이다. 효과·검정력·처리량·완주 비용은 여전히 미확인이다. 이전 v0.1 기록은 이력으로 보존하며 실행 판정은 v0.2 본문을 따른다.

| 철회·대체한 설명 | 사유 |
|---|---|
| I_LR>0을 실제 loss 감소로 읽을 수 있는 설명 | 두 rank에서 모두 악화되어도 양수가 될 수 있어 상대적 보완과 실제 개선을 분리 |
| LL의 DW 대비만을 품질 유지의 주 비교로 사용 | 고정예산 재배분 질문에 직접 답하도록 Δ_budget을 주 비교로 추가하고 Δ는 별도 비교로 유지 |
| 대표점 대조만으로 J의 구조적 해석 강화 | J는 8셀 대비이므로 일부 점만 반복하면 전체 recipe 의존성을 점검할 수 없음 |
