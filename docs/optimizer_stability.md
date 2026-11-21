# 선행연구 기반 optimizer 안정성 보완

2026-09-27. 적용 코드는 `src/optim.py`, `src/model.py`, `src/train.py` 및 mini runner다.
이 문서는 구현과 CPU 검증 기록이다. GPU 장기 안정성이나 품질 개선의 실험 결과가 아니다.

## 근거와 구현 범위

- [Spectron 논문](https://arxiv.org/html/2602.12429), Algorithm 1–3: factor momentum을 Newton–Schulz로 직교화하고, 업데이트 전 두 factor의 spectral norm 추정값을 합해 공통 step scale을 계산한다.
- [저자 공개 코드](https://github.com/Pauljanson002/spectron): `muon_local.py`, `power_iter.py`를 확인했다. 공식 코드에는 Nesterov, shape별 LR scaling, spectral weight decay 등의 선택지가 있다. 이번 구현은 논문 Algorithm 1 기반의 FFN factor 적용이며 공식 학습 recipe 전체 재현이 아니다.
- 프로젝트 `experiment_plan.md` §4·§8의 `N sum(f_i p_i)`와 factor decay 정의를 점검했다. shared/dropless 구조와 초기화 분산은 기존 정의를 유지한다. 다른 논문의 구조를 임의로 섞어 새로운 모델로 바꾸지 않았다.

## Spectron과 AdamW 분리

`optimizer="spectron"`일 때 `FactorizedLinear`의 A/B만 Spectron으로 갱신한다. 이 프로젝트는 W=B@A이므로 factor의 이름과 방향을 원문의 W=ABᵀ와 혼동하지 않는다. attention, shared FFN, router, embedding, norm은 기존 AdamW 규칙을 사용한다. embedding/norm/bias에는 decay가 없다.

- momentum 0.95, Newton–Schulz 5회, warm-start power iteration 1회가 기본이다.
- 두 factor 모두 갱신 전 norm으로 `rho=lr/(sigma_A+sigma_B+1)`를 계산한다.
- FP32 master/state를 사용한다. 테스트용 FP64는 보존한다. BF16 kernel, Nesterov, shape LR scaling, spectral weight decay는 구현하지 않았다.
- Newton–Schulz의 작은 Gram 행렬 계산은 원문 다항식과 대수적으로 같으며, tall/wide 모두 SVD의 singular-value 다항식과 독립 대조했다.
- power vector, momentum, step 및 진단 scalar를 checkpoint에 보존한다. model 및 두 optimizer를 CPU에서 bitwise 동일하게 재개하는 테스트가 있다.
- 0 행렬과 null-space power 초기값을 처리한다. trainable factor 한쪽만 존재하면 오류로 종료한다. optimizer 이름 오타를 AdamW로 조용히 대체하지 않는다.

유한 NS 반복은 정확한 polar factor가 아니며 power iteration은 실제 spectral norm을 과소추정할 수 있다. 따라서 실용적 근사 제어이며 `||ΔW||₂≤lr`을 엄밀히 보장한다고 주장하지 않는다. 이론 bound는 정확한 norm/직교화 및 rho 조건을 전제로 한다.

Algorithm 1에 맞춰 `spectron_weight_decay=0`이 기본이다. 0보다 큰 factor decay를 명시적으로 줄 수 있으나 추가 decay 변화는 해당 bound 밖이다. AdamW baseline은 factor decay 0.1이므로 제공된 비교는 optimizer recipe 비교다. 직교화만의 인과 효과라고 해석하지 않는다. 필요하면 factor decay와 LR tuning budget을 맞춘 별도 통제를 사전 고정해야 한다.

## 발견한 두 학습 문제

### microbatch별 balancing

`N sum(f_i p_i)`는 batch 통계에 비선형이므로 microbatch 손실의 평균은 전체 batch 손실과 다르다. 실제 작은 예에서 전체 batch 값 1.0000, 나눠 계산한 평균 1.9051이었다.

새 설정의 `router_balance_scope="optimizer_step"`은 gradient 없이 전체 group의 hard assignment를 먼저 집계하고, gradient pass에서 고정한 global f와 각 microbatch의 확률 평균을 결합한다. CE는 유효 target 수, balancing은 유효 input token 수로 가중한다. label이 모두 -100인 microbatch도 유효 input이 있으면 router balancing에 참여한다. group 전체에 target이 없으면 optimizer step은 진행하지 않는다.

현재 모델에는 dropout/forward 상태 변화가 없다. 이를 추가하면 prepass와 gradient pass의 routing을 일치시키는 처리를 추가해야 한다. 추가 forward 1회의 시간과 FLOPs를 포함해야 하며 기존 AdamW 처리량을 새 설정의 비용으로 재사용하면 안 된다. 기본 `microbatch` 모드는 호환성을 위해 남아 있으므로 아래 새 설정을 사용한다.

### idle expert 갱신

기존 sparse dispatch는 idle expert의 grad를 None으로 남겨 AdamW의 decay와 momentum, step counter까지 생략했다. 기본 `idle_expert_policy="zero_grad"`에서는 backward 후 이 gradient를 0으로 만들어 dense zero-gradient optimizer 의미를 유지한다. 이는 idle expert에 토큰을 보내거나 expert forward를 실행하는 처리가 아니다. Spectron에서도 잔여 momentum과 선택한 decay 규칙이 이어진다. 이전 동작은 `idle_expert_policy="skip"`으로 명시할 수 있다.

## 실행 가능한 대조 설정

- `configs/colab_stability_adamw/{LR-HS,LR-HL,LR-LS,LR-LL}.json`
- `configs/colab_stability_spectron/{LR-HS,LR-HL,LR-LS,LR-LL}.json`

두 집합은 seed, 토큰 수, 모델, global balancing과 idle 정책을 공유하며, 새 protocol `colab-free-mini-stability-v1` 및 각각 다른 recipe를 기록한다. 기존 mini의 3e-4 LR을 시작값으로 유지했으며 Spectron에 최적인 LR로 검증한 값은 아니다. 조건별 동등한 LR tuning budget과 여러 seed가 필요하다. 이 mini grid는 기존 계획의 고정예산 핵심 8셀을 대신하지 않는다.

예시(실제 GPU 실행은 별도):

```bash
.venv/bin/python scripts/run_colab_mini.py \
  --config configs/colab_stability_spectron/LR-LL.json \
  --bundle data/prepared/colab_free_mini_v1/bundle.json \
  --output runs/colab_stability_spectron/LR-LL \
  --device cuda --max-steps 32
```

`package_colab_mini.py`의 allowlist에도 새 설정을 포함했다. 기존 tar.gz와 원격 실행 환경은 자동 갱신하지 않았다. 원격에서 사용하려면 새 소스로 payload를 다시 만들어 전달해야 한다.

기존 checkpoint에는 새 optimizer/정책 state가 없으므로 설정 hash 또는 source identity가 다르면 재개를 거부한다. 이전 checkpoint에 Spectron을 덧씌우지 말고 새 output 경로에서 시작한다. 과거 결과와 새 결과를 같은 recipe로 합치지 않는다.

## 진단과 검증 한계

각 finite step은 `diagnostics/<run_id>/step-XXXXXXXX.json`에 CE loss, balancing loss, clipping 전 gradient norm, LR 및 idle parameter 수를 기록한다. Spectron은 factor spectral norm 추정치, factor update Frobenius norm, rho의 최댓값을 추가한다. 이것은 합성 ΔW spectral norm 측정값이 아니다. 각 factor의 개별 추정값은 optimizer checkpoint state에서 확인할 수 있다.

마지막 step 진단은 checkpoint에도 저장·복구한다. 진단 파일은 checkpoint보다 먼저 기록될 수 있으므로 중단/rollback 시 `latest.json`의 검증된 checkpoint step보다 뒤의 파일은 확정 학습 기록에서 제외한다. step별 파일 저장 시간은 실제 처리량 측정에 포함된다.

회귀 검증: scalar 수식, tall/wide NS 대조, null/zero 처리, optimizer partition, 중단·재개, idle expert momentum/decay, 불균등 input mask 및 zero-target microbatch의 분할/비분할 gradient, runner 설정 검증. 검증은 CPU synthetic fixture이며 CUDA/실데이터 장기 학습과 wall-clock 개선은 미검증이다.
