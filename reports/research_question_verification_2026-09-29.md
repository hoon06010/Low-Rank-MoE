# 연구 질문 확인 계획 실행 판정 (2026-09-29)

> **2026-10-03 현재 상태:** Stage C는 2026-10-01 Kaggle에서 네 조건(HL→HS→LS→LL), seed 17, 조건당 2,048 steps / 8,388,608 targets로 완료했다. 최종 Successful #13 아카이브와 네 체크포인트는 독립 다운로드 후 SHA-256 일치를 검증했다. 보수적 사용 시간은 384.00/480분이다. 독립 seed·DW 대조·동일 FLOPs 확증은 미완료다. C는 단일 seed 탐색이며, 아래 날짜별 준비·게이트 기록은 당시 판단이다. 최신 근거: [C 완료 보고서](c_execution_2026-10-01.md). 이번 갱신은 로컬 완료 기록을 기준으로 하며 외부 서비스는 재조회하지 않았다.

> 기존 미니 탐색 수치 I_LR=0.009041, Delta_budget=-0.036808은 새 AdamW B 완료 수치가 아니다. 연구질문 계획의 단계 B(명세 검토)와 야간 탐색 B(학습)는 다른 구분이다.

기준: `plans/2026-09-29-research-question-verification-plan.md`. 이 기록은 문헌·실험 준비의 확인 결과이며 새 GPU 학습 결과가 아니다.

## 현재 결론

- 연구 질문은 별도 비선형 shared FFN 폭과 native routed SwiGLU factor rank의 **통제된 상호작용**, dense-width 대조, 동일 토큰과 동일 학습 FLOPs의 비용 경계로 좁힌다. HS↔LL은 총 파라미터를 맞춘 재배분이고 LR↔DW는 대응 셀별 총·활성 파라미터를 맞춘 대조다. 2×2 전체가 동일 총량이나 동일 FLOPs인 것은 아니다. 표적 문헌에서는 이 전체 대조를 직접 수행한 실험을 확인하지 못했으나, 이는 최초성이나 보편적 우위의 증거가 아니다.
- 기존 seed 17의 4셀 결과는 탐색이다. `I_LR=0.009041`, `Δ_budget=-0.036808` nats/target은 독립 반복, DW 대조, 동일 FLOPs 대조가 없어 확증 근거가 아니다.
- 당시 B/C 탐색 학습과 핵심 8셀×독립 3 seed 확증 학습은 진입 게이트 미충족으로 보류했다. 이후 B/C 탐색 학습은 완료했으며 핵심 확증 학습은 미완료다. B 네 셀의 기존 견적은 예비 시간을 포함해 139.65분으로 당시 120분 상한을 넘었다. 보존 checkpoint 예상 44.47GB도 당시 로컬 여유 17.315GB를 넘었다. 무료 Colab 야간안의 controller·Drive 왕복·복구 리허설은 미완료다.

## 단계별 증거와 판정

| 단계 | 판정 | 세부 근거 |
|---|---|---|
| A 문헌 | 좁힌 질문과 반대 결과를 기록 | `literature/research_question_verification_2026-09-29.md`, `literature/comparison_matrix.md`, `literature/evidence_ledger.csv` |
| B 실행 명세 | 자원 게이트 보류 | `plans/2026-09-29-stage-b-execution-gate.md`, `reports/longer_training/resource_gate.json` |
| C 탐색·확증·동일 FLOPs | C 탐색 완료; 확증·동일 FLOPs 미완료 | C 네 셀 seed 17, 조건당 2,048 steps 완료 및 저장본 검증; 독립 seed·DW·동일 FLOPs 비교는 후속 작업 |

## 반대 결과의 해석

- `Δ_budget`만 독립 반복에서 유지되고 `I_LR`·`J`가 뒷받침되지 않으면 시험한 예산 재배분의 품질 차이만 보고한다.
- LR와 DW의 차이가 불명확하면 low-rank 고유 우위는 미확인으로 남긴다. 사전 양방향 기준과 좁은 CI 없이 동등하다고 표현하지 않는다.
- 동일 토큰의 이득이 동일 학습 FLOPs에서 사라지면 효율 우위를 주장하지 않는다.
- 긴 학습에서 차이가 사라지거나 seed/recipe에 민감하면 조건부 또는 불확실한 결과로 보존한다.

## 재진입에 필요한 작업

1. LR/DW 8셀 전체와 독립 seed의 데이터 revision·tokenizer·train/dev target manifest, 중복/누수 감사, 설정·recipe·종료점·허용치를 결과 확인 전에 고정한다.
2. DW 처리량, 전체 24회 및 동일 FLOPs 세 조건의 평가·저장·전송·복구를 포함한 시간·공간·금액을 실측 기반으로 견적한다.
3. B/C용 무료 Colab 야간안을 선택한다면 별도 계획의 notebook controller, 보존 정책, Drive 왕복 및 새 세션 복구 리허설을 통과시킨다. 다른 자원을 선택하면 범위와 비용을 별도로 고정한다.

어느 경로에서도 유리해 보이는 일부 셀만 남겨 확증으로 부르지 않는다. 기존 `experiment_plan.md`의 대비 방향과 잠정 허용치를 사후 변경하지 않는다.

## 2·3단계 재진입 점검 (2026-09-29)

- 단계 B의 핵심 8셀 config 해시와 대수적 총·활성 파라미터 수를 다시 확인했다. 대응 LR/DW 쌍은 일치한다. 확증용 데이터 manifest, seed {101,202,303} 설정, DW 실측 처리량, 24회 전체 비용은 아직 없다. 현재 실행기는 LR 네 셀과 seed 17에 한정된다.
- 야간 탐색용 `payload.tar.gz`는 47,327,359바이트이고 SHA-256은 `f1a4bfb1f266d57c0b6bdca3a0cbeaddf506ed7a04ad9746112af3bb81d35372`이다. Manifest의 49개 로컬 파일 해시와 일치한다. 관련 로컬 테스트 13개가 통과했다.
- GPU의 Drive 왕복, 새 런타임에서 512→513 step 복구, 실제 quota·전송 시간, 단계별 시간 한도는 확인되지 않았다. 따라서 B/C 탐색과 8셀 확증, 동일 FLOPs 학습은 시작하지 않았다. 로컬 테스트 통과를 GPU 리허설 통과로 간주하지 않는다.
