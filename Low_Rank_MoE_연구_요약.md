# Low-Rank MoE 연구 요약

작성일: 2026-11-15 (KST)  
범위: 프로젝트에 보존된 계획서, 실행 보고서, 원시 결과, 2026-11-13 연구 질문 정교화 문서. 원격 GPU 상태나 문헌의 최신 개정·게재 상태는 이번 요약에서 재조회하지 않았다.

## 1. 연구의 핵심

**제한된 파라미터 예산을 항상 실행되는 shared FFN과 선택적으로 실행되는 routed FFN에 어떻게 나눌 것인가**를 연구한다. Routed FFN을 줄이는 방법으로 넓은 비선형 폭을 유지하는 저랭크 분해와, 폭 자체를 줄이는 dense FFN을 비교하려는 프로젝트다.

현재 가장 구체적인 질문은 다음과 같다.

> 같은 총 파라미터 수와 학습 토큰 수에서, shared 용량에 따라 넓은 저랭크 routed FFN과 좁은 dense routed FFN 중 유리한 방식이 달라지는가?

현재까지 두 초기화 seed에서 **routed rank를 낮추고 shared 용량을 늘린 조합이 같은 파라미터 수의 기준 조합보다 낮은 dev loss**를 보였다. 그러나 좁은 dense 대조가 아직 없고 활성 계산량도 증가했으므로, 저랭크 고유의 이득이나 계산 효율 개선은 입증되지 않았다.

## 2. 출발 가설과 연구 질문의 변화

초기 가설은 shared가 공통 기능을 담당하면 routed에는 상대적으로 작은 rank로도 충분한 조건부 기능만 남을 수 있다는 것이었다. 이에 routed rank를 줄여 절약한 파라미터를 dense shared에 재배분하는 설계를 검토했다.

이 설명은 아직 검증된 메커니즘이 아니다. 실제 모델에서 shared와 routed는 **같은 정규화 입력을 병렬로 받아 출력을 합산**한다. Shared가 먼저 처리한 잔차를 routed가 입력받는 구조가 아니며, 공통 지식과 전문 지식이 실제로 분리되었다는 증거도 없다.

따라서 2026-11-13 정교화에서는 “저랭크로도 전문성이 유지된다”는 주장을 앞세우기보다, **rank 제한과 dense 폭 축소라는 두 용량 절약 방법의 차이를 shared 용량별로 측정**하는 질문으로 좁혔다. 저랭크가 더 좋지 않다는 결과도 설계 선택에 유용한 결론이다.

## 3. 모델과 예산의 정의

사전학습된 모델에 LoRA를 부착하는 방식이 아니다. 모든 가중치를 무작위 초기화해 학습하는 작은 decoder-only 언어 모델이다.

| 항목 | 기존 주요 실험 설정 |
|---|---|
| Transformer | 6층, hidden dimension 384, attention head 6 |
| Context length | 512 |
| Routed experts | 층당 8개, 토큰별 top-2 선택 |
| Routed FFN | SwiGLU, intermediate width 768 |
| 저랭크 적용 | gate/up/down 세 행렬 모두 같은 rank로 분해 |
| Routed rank | 64 또는 32 |
| Shared FFN | dense SwiGLU, 개당 intermediate width 256 |
| Shared 수 | 층당 2개 또는 5개 |
| 데이터·토크나이저 | FineWeb-Edu 첫 10,000문서 기반 준비 데이터, GPT-2 tokenizer |
| 주요 학습 설정 | AdamW, FP32, 초기 learning rate 3e-4, global batch 4,096 유효 정답 토큰 |

Routed 행렬은 W=BA로 매개변수화한다. 입력 차원 d, 중간 폭 m, rank r인 bias 없는 SwiGLU의 파라미터 수는 다음과 같다.

- Dense: p_dense = 3dm
- Low-rank: p_LR = 3r(d+m)
- 같은 예산의 좁은 dense 폭: m′ = r(d+m)/d

현재 d=384, m=768에서는 m′=3r이다. 따라서 rank 64인 LR과 폭 192인 dense, rank 32인 LR과 폭 96인 dense가 대응 비교군이다. 저랭크는 중간 폭을 보존하는 대신 변환 방향을 제한하고, 좁은 dense는 중간 특징 수를 줄인다. 이 차이만으로 어느 쪽이 우수하다고 결정할 수는 없다.

단일 저랭크 expert의 출력은 down factor B의 열공간에 놓여 최대 r차원으로 제한된다. 전체 모델에는 router·shared·attention·residual이 있으므로 전체 모델의 표현을 rank r로 제한한다고 해석하면 안 된다.

Shared 여러 개를 같은 입력에 적용해 단순 합산하는 현재 구조는 하나의 더 넓은 SwiGLU로 표현할 수 있다. 따라서 shared 수 증가는 주로 **항상 활성화되는 FFN 용량 증가**로 해석한다.

### 저장 예산과 활성 계산량

층당 shared 파라미터를 P_s, routed 하나의 파라미터를 p_r라 하면:

- 총 expert 파라미터: P_E = P_s + N p_r
- 토큰당 활성 expert 파라미터: P_act,E = P_s + K p_r
- 따라서 P_act,E = (K/N)P_E + (1−K/N)P_s

N=8, K=2를 고정하고 총 예산을 유지해도, shared에 더 배분하면 활성 파라미터가 증가한다. **같은 총 파라미터 수는 같은 FLOPs·실행 시간·GPU peak memory를 의미하지 않는다.**

## 4. 수행한 작업과 실험 상태

| 작업 | 보존된 기록상 상태 | 해석 범위 |
|---|---|---|
| 모델·예산·routing·학습·재개 구현 | 구현 및 로컬 검증 기록 있음 | 성능 가설 입증과 별개 |
| 소량 실데이터 CPU 검증 | 데이터 준비 및 중단·재개 일치 검증 | 파이프라인 검증 |
| Colab 미니 탐색 | seed17, LR 4조건, 조건당 768 steps / 3,145,728 targets 완료 | 초기 탐색 |
| Stage C | seed17, LR 4조건, 조건당 2,048 steps / 8,388,608 targets 완료 | 단일 seed 탐색 |
| AdamW seed401 반복 | 같은 LR 4조건, 조건당 2,048 steps / 8,388,608 targets 완료 | 추가 초기화에서 관측 반복 |
| seed402 | 준비 감사 기록 있음, 운영 adapter 보완 대기, 학습 미시작 | 결과 없음 |
| seed403 | 상태 문서상 미시작, 완료 결과 미확인 | 과거 준비 차단 사유를 현재 사실로 단정하지 않음 |
| Spectron seed401 | 학습 전 저장 자원 게이트에서 중단, optimizer steps=0 | optimizer 성능 실험 결과가 아님 |
| LR 대 좁은 dense(DW) | 핵심 비교 미완료 | 저랭크 고유 이득 미검증 |
| 동일 FLOPs·routing 대조 | 미완료 | 계산 효율·조건부 선택 가치 미검증 |
| 출력 부분공간 기반 조합 routing | 연구 질문과 방법 후보 정리 | 구현·학습·성능 검증 없음 |

Spectron은 보존된 AdamW checkpoint 크기보다 작은 checkpoint 상한 설정으로 자원 검사에서 중단되었다. 이를 Spectron 학습의 발산이나 저랭크 최적화 실패로 해석할 수 없다.

## 5. 핵심 실험 결과

H/L은 높은/낮은 rank(64/32), 뒤의 S/L은 작은/큰 shared 용량(2/5개)을 뜻한다. 네 조건 전체가 같은 파라미터 수인 것은 아니다. **HS와 LL만 동일 총 파라미터 비교**이며, HL과 LS는 2×2 분해를 위한 보조 조건이다.

조건당 8,388,608 학습 targets, 2,048 optimizer steps, dev 65,536 targets에서의 최종 CE다. 낮을수록 좋으며 단위는 nats/target이다.

| 조건 | Rank | Shared 수 | 총 파라미터 | seed17 CE | seed401 CE |
|---|---:|---:|---:|---:|---:|
| HL | 64 | 5 | 42,325,248 | 6.05776107 | 6.05213559 |
| HS | 64 | 2 | 37,016,832 | 6.11435759 | 6.10328890 |
| LS | 32 | 2 | 31,708,416 | 6.13254436 | 6.12923248 |
| LL | 32 | 5 | 37,016,832 | 6.07361295 | 6.06640364 |

| 대비 | 정의 | seed17 | seed401 |
|---|---|---:|---:|
| 낮은 rank에서 shared 확대 이득 | G_low = LS−LL | +0.05893141 | +0.06282885 |
| 높은 rank에서 shared 확대 이득 | G_high = HS−HL | +0.05659652 | +0.05115331 |
| Shared 확대 이득의 rank별 차이 | I_LR = G_low−G_high | +0.00233489 | +0.01167554 |
| 동일 예산 재배분 차이 | Δ_budget = LL−HS | −0.04074464 | −0.03688526 |

초기 Colab 미니 탐색에서도 HS→LL의 차이는 −0.036808 nats였다. 다만 동일 seed17의 짧은 탐색이므로 독립 반복으로 추가 집계하지 않는다.

### 관찰에서 말할 수 있는 것

1. 같은 shared 용량에서 rank를 낮추면 두 seed 모두 CE가 높아졌다. 낮은 rank 자체의 우위는 보이지 않았다.
2. HS→LL에서는 shared 확대 이득이 rank 축소 손해보다 컸다. 시험한 학습량에서 동일 저장 예산의 재배분이 유리했다.
3. Shared 확대는 높은 rank에서도 상당한 이득을 보였다. 저랭크만 특별히 보완했다는 설명은 아직 약하다.
4. 최종 I_LR는 두 seed 모두 양수지만 크기가 다르다. Seed17의 중간 평가에서는 음수였던 구간도 있어 학습 시점 의존성이 있다.

Seed17은 탐색 seed이며 seed401과 실행 환경도 다르다(Python 3.12.13/PyTorch 2.10.0 대 Python 3.13.15/PyTorch 2.11.0). 이 두 결과를 완결된 사전 확증 반복이나 통계적 유의성의 증거로 부르지 않는다.

## 6. 아직 결론낼 수 없는 이유

**계산 증가와 구조 이득이 분리되지 않았다.** HS→LL에서 총 파라미터는 37,016,832로 같지만, 대수적 전체 활성 파라미터는 29,054,208→33,035,520으로 약 13.70% 증가한다. 층당 활성 expert 부분은 1,032,192→1,695,744로 약 64.29% 증가한다. 이 비율은 실제 FLOPs나 실행 시간 증가율이 아니다.

**좁은 dense 대조가 없다.** 동일한 절약분을 dense 폭 축소로 확보해도 같은 결과가 나올 수 있다. LR/DW 비교 없이 재배분의 성공을 저랭크 고유 장점으로 바꿔 말할 수 없다.

**초기 출력 크기가 shared 수와 함께 변한다.** 현재 구현은 shared 합 출력에 개수별 정규화를 적용하지 않는다. 11월 13일 합성 입력 초기화 진단에서는 shared 2→5개일 때 RMS 비율이 약 1.57–1.59였다. 이는 학습 없는 진단이며, CE 차이의 원인이나 효과 크기를 입증하지 않는다.

**짧은 학습의 결과다.** Shared는 모든 토큰으로 학습되므로 초기에 유리할 수 있다. 더 긴 학습·새 데이터에서도 같은 순위인지, 수렴한 표현력 차이인지 확인하지 않았다.

**상호작용은 역할 분리의 증명이 아니다.** I_LR>0은 큰 shared에서 rank 축소 손해가 작아졌다는 관측이다. 일반적인 용량 증가의 한계효용 체감으로도 발생할 수 있다. 공통/전문 지식 분리나 전문화 보존은 별도 증거가 필요하다.

**자원 지표를 분리해야 한다.** 파라미터 수, FLOPs, 처리량, peak memory, 통신량, 전체 운영 시간, 금전 비용은 다른 지표다. 기존 기록의 마지막 실행 구간 시간만으로 전체 실험 비용을 계산하면 안 된다. Stage C 보고서상 유료 compute 사용은 0원이지만, 무료 GPU의 시간·저장·복구 비용은 존재했다. 통신과 동일 FLOPs 효율은 입증되지 않았다.

## 7. 선행연구 검토에서 얻은 연구의 위치

아래는 기존 로컬 문헌 검토의 요지이며, 이번 요약에서 새 문헌 검색을 수행한 결과는 아니다.

| 검토 축 | 프로젝트에서 검토한 연구 | 연구 설계에 남긴 판단 |
|---|---|---|
| Shared와 routed의 구성·배분 | DeepSeekMoE, Towards Greater Leverage, Slicing and Dicing | Shared 자체와 배분 연구는 선행하며, 보편적 최적 비율을 가정하지 않음 |
| 저랭크·공유 구조 | StructMoE, LightMoE, MLoRE, LatentMoE | Low-rank expert 또는 공유 성분 결합 자체를 새 기여로 주장하지 않음 |
| Rank와 폭의 구분 | GS-MoE 등 | 학습 후 압축과 처음부터의 학습을 구별하고, rank와 width를 별도 축으로 비교 |
| 저랭크 최적화 | LORO, Spectron | 낮은 rank의 손해가 표현 제약인지 최적화 조건인지 분리할 필요 |

잠정 기여 후보는 **native 학습에서 별도의 dense shared 용량에 따라 예산이 같은 LR/DW routed의 상대 성능이 어떻게 달라지는지 실증하는 것**이다. 최초성이나 보편적 효율 우위는 확정되지 않았다. 상세 원문 링크와 검토 범위는 [연구 질문 정교화 문서](reports/research_question_refinement_2026-11-13.md)에 있다.

## 8. 별도 후속 후보: 출력 부분공간과 조합 routing

2026-11-13에는 다음 질문도 정리했다.

> 저랭크 expert가 낼 수 있는 출력 방향을 알면, 이 입력에서 함께 선택할 expert 조합을 더 잘 예측할 수 있는가?

개별 expert 점수가 높다고 둘을 합친 결과가 반드시 좋은 것은 아니다. 다만 서로 다른 부분공간을 선택하는 것만으로도 예측이 개선되지는 않는다. 핵심은 **기하학적 다양성 자체가 아니라 실제 task loss에 도움이 되는 조합을 예측하는지**다.

제안된 검증 순서는 다음과 같다.

1. 고정 checkpoint에서 K=2를 유지하며 후보 expert를 교체해, 동반 expert에 따른 실제 loss 차이가 있는지 측정한다.
2. 기존 router 점수와 geometry 없는 동일 용량 pair predictor를 넘어 출력 부분공간 정보가 추가 예측력을 주는지 확인한다.
3. Geometry shuffle, matched narrow-dense, 필요시 down-only 대조로 암기·일반 router 개선·저랭크 구조 효과를 구분한다.
4. Counterfactual label 생성, QR/SVD·cache 갱신, pair scoring까지 포함한 비용 대비 품질을 평가한다.

N=8, K=2에서는 후보 쌍이 28개다. 추론 전에 28개 pair 점수를 비교하는 것과 28개 조합의 expert를 모두 실행하는 것은 다르다. 정답이나 미선택 expert의 실제 출력을 배포 router 입력으로 사용하면 안 된다.

이 방향은 **연구 질문 개정 후보**이며, 기존 배분 연구를 대체하기로 확정한 계획이나 성능 개선 결과가 아니다. 구현·학습·checkpoint 개입은 수행되지 않았다.

## 9. 다음 검증의 우선순위

기존 실행 순서나 승인 범위를 변경하는 지시가 아니라, 연구 질문에 답하기 위한 후속 설계의 우선순위다.

1. **LR와 DW의 동일 예산 비교:** 대응 셀의 shared 용량·총 파라미터·데이터·토큰 수를 맞추고, LR-HS/LL과 DW-HS/LL의 상대 CE를 비교한다. 두 가족의 2×2 상호작용 차이 J를 주장하려면 총 8개 셀이 필요하다.
2. **재현성과 초기화 통제:** 복수 seed와 공정한 tuning 기회를 확보하고, shared 출력 scale 민감도를 별도 실험으로 구분한다. 완료된 frozen recipe는 소급 변경하지 않는다.
3. **주장별 추가 대조:** 계산 효율에는 동일 학습 FLOPs, 조건부 선택 가치에는 입력 독립 균형 routing 및 shared-only 경계, 장기 유효성에는 긴 학습이 필요하다.
4. **메커니즘을 주장할 때만 추가 분해:** 출력 차원 제한을 설명하려면 down-only와 rank grid로 입력·출력 병목을 분리한다.
5. **조합 routing은 먼저 진단:** 부분공간 정보의 추가 가치가 없으면 geometry 기반 방법 개발을 확대하지 않는다.

LR/DW 차이가 없으면 용량 재배분 현상으로, 동일 FLOPs 이득이 없으면 추가 활성 계산을 동반한 품질 개선으로 범위를 제한한다. 현재 연구의 핵심 미해결점은 **“재배분이 유리한가”에서 나아가 “왜 저랭크 방식으로 재배분해야 하는가”에 답하는 것**이다.

## 10. 근거 파일

- [프로젝트 README](README.md): 전체 구조와 초기 탐색 결과.
- [연구계획](research_plan.md), [실험계획](experiment_plan.md): 가설·예산·비교군·해석 기준.
- [2026-11-13 연구 질문 정교화](reports/research_question_refinement_2026-11-13.md): 현재 질문, 두 seed 비교, 초기화 진단, LR/DW 대조.
- [Stage C 완료 보고서](reports/c_execution_2026-10-01.md): seed17 결과와 저장 감사.
- [Seed401 AdamW 완료 보고서](reports/seed_repeat_c/seed-401/adamw-completion-2026-10-09.md): 반복 결과와 당시 검증 범위.
- [Seed17 원자료](artifacts/kaggle/stage-c-final/C/), [seed401 원자료](reports/seed401_spectron/execution-2026-10-09/adamw-anchor/main/): 조건별 mini_report.json 등.
- [Seed402 상태](docs/public_release.md), [seed403 상태](docs/public_release.md): 준비·미실행 기록.
- [Spectron 실행 보고서](reports/seed401_spectron/execution-2026-10-09/execution-report.md): 학습 전 자원 게이트 중단.
- [조합 routing 질문](reports/complementarity_routing_question_2026-11-13.md), [관련 문헌 지도](literature/2026-11-13-complementarity-routing/field_map.md): 별도 후속 후보와 반증 조건.

문서의 결과표는 보존된 조건별 JSON과 대조했다. 이번 작업에서는 새 학습, checkpoint 복구 검증, 원격 상태 확인을 수행하지 않았다.
