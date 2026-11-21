# Low-Rank-MoE

**고정 파라미터 예산에서 low-rank routed expert와 dense shared expert의 용량 배분을 연구합니다.**

작은 decoder-only 언어 모델을 무작위 초기화부터 학습하며, routed FFN의 rank를 줄여 확보한 파라미터를 항상 활성화되는 shared FFN에 배분합니다. 중심 질문은 **같은 총 파라미터 수와 학습 토큰 수에서 shared 용량에 따라 넓은 저랭크 routed FFN과 좁은 dense routed FFN 중 유리한 방식이 달라지는가**입니다.

> **연구 단계: 탐색.** 두 초기화 seed에서 동일 파라미터 예산의 재배분 조합이 낮은 dev loss를 보였습니다. 좁은 dense 대조와 동일 FLOPs 비교는 미완료이므로, 저랭크 고유의 이득이나 계산 효율 향상을 입증한 결과는 아닙니다.

## 모델과 비교 조건

- Transformer 6층, hidden dimension 384, attention head 6, context length 512.
- 층당 routed expert 8개 중 토큰별 top-2를 선택합니다.
- Routed SwiGLU의 gate/up/down 행렬을 각각 `W = BA`로 매개변수화합니다. Intermediate width는 768, rank는 64 또는 32입니다.
- Shared expert는 개당 width 256인 dense SwiGLU이며, 층당 2개 또는 5개를 사용합니다.
- Shared와 routed는 같은 정규화 입력을 병렬로 받아 출력을 합산합니다. Shared가 공통 지식, routed가 전문 지식을 분리해 학습한다는 메커니즘은 아직 검증되지 않았습니다.
- 사전학습 모델의 LoRA 미세조정이 아니라 전체 모델을 처음부터 학습합니다.

H/L은 높은/낮은 routed rank, 뒤의 S/L은 작은/큰 shared 용량을 뜻합니다. **HS와 LL만 동일 총 파라미터 비교**이며, 네 조건 전체가 같은 예산은 아닙니다.

## 보존된 실험 결과

FineWeb-Edu 첫 10,000문서 기반 준비 데이터와 GPT-2 tokenizer를 사용했습니다. 아래 결과는 AdamW·FP32, 조건당 **2,048 optimizer steps / 8,388,608 학습 targets**, dev 65,536 targets의 최종 CE입니다. 단위는 nats/target이며 낮을수록 좋습니다.

| 조건 | Routed rank | Shared 수 | 총 파라미터 | seed17 dev CE ↓ | seed401 dev CE ↓ |
|---|---:|---:|---:|---:|---:|
| HL | 64 | 5 | 42,325,248 | 6.05776107 | 6.05213559 |
| HS | 64 | 2 | 37,016,832 | 6.11435759 | 6.10328890 |
| LS | 32 | 2 | 31,708,416 | 6.13254436 | 6.12923248 |
| LL | 32 | 5 | 37,016,832 | 6.07361295 | 6.06640364 |

동일 예산의 `LL − HS`는 seed17에서 **−0.04074464**, seed401에서 **−0.03688526**입니다. 같은 shared 용량에서 rank만 낮추면 두 seed 모두 CE가 높아졌습니다. 관측한 이득은 rank 축소와 shared 확대를 함께 적용한 조합의 결과입니다.

원자료: [seed17 조건별 JSON](artifacts/kaggle/stage-c-final/C/) · [seed401 조건별 JSON](reports/seed401_spectron/execution-2026-10-09/adamw-anchor/main/).
실행 기록: [Stage C](reports/c_execution_2026-10-01.md) · [seed401 완료 보고서](reports/seed_repeat_c/seed-401/adamw-completion-2026-10-09.md).

### 해석의 한계와 다음 실험

- **같은 파라미터 수는 같은 계산량이 아닙니다.** HS→LL에서 shared 용량이 늘어 활성 파라미터와 계산 비용도 달라집니다. 실제 속도·FLOPs·메모리 효율 우위는 별도 측정해야 합니다.
- **좁은 dense(DW) 대조가 필요합니다.** 현재 차이를 저랭크 고유 효과로 해석할 수 없습니다.
- Seed17은 탐색 seed이며 seed401과 실행 환경도 다릅니다. 두 결과만으로 통계적 유의성이나 완결된 확증 실험을 주장하지 않습니다.
- Shared 출력 scale 통제, 복수 seed, 더 긴 학습, 입력 독립 routing 대조가 남아 있습니다.
- seed402·403의 완료 결과는 이 자료에 없습니다. Spectron seed401은 학습 전 자원 게이트에서 중단되어 성능 비교 결과가 아닙니다.
- 출력 부분공간을 활용한 조합 routing은 [후속 연구 질문](reports/complementarity_routing_question_2026-11-13.md) 단계입니다.

상세한 연구 상태와 해석은 [연구 요약](Low_Rank_MoE_연구_요약.md), [연구 질문 정교화](reports/research_question_refinement_2026-11-13.md)를 참고하세요.

## 빠른 시작

모든 명령은 저장소 루트에서 실행합니다. 아래 의존성은 보존된 Python 3.13 CPU 환경 기록을 바탕으로 정리했습니다. CUDA 환경은 해당 장치에 맞는 PyTorch 설치가 별도로 필요합니다.

```sh
git clone https://github.com/hoon06010/Low-Rank-MoE.git
cd Low-Rank-MoE
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

디스크 여유를 확인한 뒤 설치하세요. PyTorch와 가상환경 설치 용량은 이 저장소의 작은 소스 파일 용량과 별개입니다.

### 합성 데이터 CPU smoke

외부 데이터 다운로드 없이 작은 모델의 학습·평가 경로를 실행합니다. 출력 경로는 기존 실험과 겹치지 않게 지정합니다.

```sh
.venv/bin/python -m src.preflight \
  --config configs/toy_cpu.json \
  --output results/preflight/readme_toy_cpu
```

이 실행은 `synthetic_fixture`이며 연구 성능 결과가 아닙니다.

### 핵심 테스트

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_budget.py tests/test_model.py tests/test_factorization.py \
  tests/test_routing.py tests/test_cost.py tests/test_contrasts.py
```

과거 복구·운영 테스트 일부는 Drive에 보관된 archive·ledger·notebook 등 추가 fixture를 요구합니다. `tests/` 전체를 실행하려면 해당 자료를 별도로 준비해야 합니다.

### 실데이터 실행

실데이터는 저장소에 포함하지 않습니다. [데이터 이용 조건](plans/m9/data_terms.md), [manifest 계약](data/manifests/README.md), [준비 및 실행 안내](plans/m9/runbook.md)를 먼저 확인하세요. 데이터 준비 스크립트에는 `pyarrow`, `tokenizers`, `jsonschema` 등의 의존성이 필요하며, [당시 환경 기록](reports/m9/environment.freeze.txt)은 CUDA용 lockfile이 아닙니다.

준비된 bundle과 그 참조 파일이 모두 있는 경우:

```sh
OMP_NUM_THREADS=2 .venv/bin/python -m src.real_preflight \
  --config configs/real_cpu_smoke.json \
  --bundle data/prepared/fineweb_10k/bundle.json \
  --output results/preflight/readme_real_cpu \
  --device cpu
```

이 명령은 데이터를 자동 다운로드하지 않습니다. 재개는 동일 설정·데이터·소스 해시를 요구합니다. GPU 실행은 [Colab 실행 안내](plans/colab_free_mini_runbook.md), [Stage C 계획](plans/2026-10-01-stage-c-kaggle-execution-plan.md)을 참고하세요.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| [src/](src/) | 모델, 학습·재개, 데이터 처리, 평가, 파라미터·비용 계산 |
| [configs/](configs/) | CPU smoke, LR/DW, Colab, seed 반복 설정 |
| [scripts/](scripts/) | 데이터 준비, 실행, 패키징, 복구 도구 |
| [tests/](tests/) | 모델·예산·routing·학습·복구 테스트 |
| [reports/](reports/) | 선택한 실행 보고서와 작은 결과 파일 |
| [artifacts/](artifacts/) | seed17 최종 조건별 결과 JSON |
| [plans/](plans/) | 모델·인터페이스 명세와 당시 실행 계획 |
| [literature/](literature/), [concept/](concept/) | 연구 관련 문헌 메모와 SVD·PCA 설명 |
| [manuscript/](manuscript/) | 논문 개요와 주장–근거 대응 |
| [data/manifests/](data/manifests/) | 데이터 manifest 스키마와 안내 |
| [results/preflight/parameter_ledger.csv](results/preflight/parameter_ledger.csv) | 코드로 계산한 파라미터 예산표; 측정 성능 아님 |

## 이 업로드의 범위와 검증

2026-11-21에 Google Drive의 프로젝트 스냅샷에서 작은 코드·설정·문서·결과 파일을 선별했습니다. 문서 속 과거 날짜와 실험 상태는 당시 기록입니다. 개인 경로와 외부 계정 식별자는 공개본에서 제거했습니다.

- 원본 데이터, packed tokens, tokenizer 파일, checkpoint, 압축 백업, 가상환경, 논문 PDF·번역 산출물과 별도의 광범위한 LLM 문헌 모음은 포함하지 않았습니다.
- 따라서 이 저장소는 **코드와 연구 근거를 공유하는 경량 사본**이며 checkpoint 복구용 전체 백업은 아닙니다. 과거 문서의 일부 링크는 제외된 자료를 가리킬 수 있습니다.
- [공개 파일 manifest](IMPORT_MANIFEST.json)는 공개본의 경로·크기·SHA-256만 기록합니다. 개인정보와 운영 기록의 제외 범위는 [공개 범위 안내](docs/public_release.md)에 정리했습니다.
- 이번 업로드에서 Python 문법·JSON 파싱, 결과 수치와 파라미터 예산을 확인했습니다. 현재 환경에 PyTorch·pytest가 없어 학습·GPU 실행·전체 테스트는 재실행하지 않았습니다.
- [업로드 검증 기록](docs/github_import_verification.md)에 검사 범위를 남겼습니다.

기존 연구계획: [research_plan.md](research_plan.md) · [experiment_plan.md](experiment_plan.md) · [paper_development_plan.md](paper_development_plan.md).
