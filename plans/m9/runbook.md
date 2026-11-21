# 실제 데이터 준비·학습 경로 실행 안내

작업 디렉터리: `/path/to/Low-Rank-MoE`.

## 현재 확인된 범위

FineWeb-Edu sample-10BT 고정 revision의 첫 shard 2,152,819,114 bytes를 다운로드하고 공식 SHA-256과 대조했다. 첫 10,000문서를 사용했으며 전체 데이터의 무작위 표본은 아니다. 정확한 중복 1건을 제외한 9,999문서를 train 9,810 / dev 95 / test 94로 분리했다. 준비된 토큰은 train 65,536 / dev 4,096 / test 4,096개 유효 next-token target이다. 토큰 순서·정답·문서 출처는 `data/prepared/fineweb_10k/`에 보존했다. test split은 준비·누수 감사만 했고 모델 평가는 하지 않았다.

`data/prepared/fineweb_10k/bundle.json`의 ready는 이 준비용 표본에만 적용된다. 유사중복 검토는 최대 128문서/split, 문서당 앞 4,096단어, 5-word shingle Jaccard >= 0.8 기준의 표본 비교다. 전체 데이터의 누수 부재나 연구 성능을 증명하지 않는다.

## 로컬 재실행

다운로드된 데이터로 새 CPU 실행:

```sh
cd /path/to/Low-Rank-MoE
OMP_NUM_THREADS=2 .venv/bin/python -m src.real_preflight --config configs/real_cpu_smoke.json --bundle data/prepared/fineweb_10k/bundle.json --output results/preflight/real_cpu_manual --device cpu
```

중간 체크포인트가 있는 동일 실행 재개:

```sh
OMP_NUM_THREADS=2 .venv/bin/python -m src.real_preflight --config configs/real_cpu_smoke.json --bundle data/prepared/fineweb_10k/bundle.json --output results/preflight/real_cpu_manual --device cpu --resume
```

새 실행은 기존 체크포인트 디렉터리를 덮어쓰지 않는다. 재개는 모델·훈련 설정·데이터·토크나이저·평가 토큰·소스 해시가 같아야 한다. 이미 완료된 실행의 재개는 추가 학습을 하지 않으며 새 처리량 측정으로 해석하지 않는다. CPU와 CUDA 간 체크포인트 이동은 이 정확 재개 경로에서 지원하지 않는다.

모든 검증:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```

데이터 준비 스크립트 `scripts/prepare_fineweb_10k.py`는 이미 받은 원본에서 준비한다. 기존 bundle이 있으면 보존을 위해 중단한다. 원본/토크나이저/packed split 중 어느 파일이 바뀌어도 `load_audited_bundle`의 재해시 검증을 통과해야 한다. `reports/m9/environment.freeze.txt`는 이번 macOS 환경 기록이며 CUDA 설치 명령이 아니다.

## 후속 GPU 파일럿 — 실행·결제 미수행

CUDA가 동작하는 PyTorch 환경과 해당 파일럿의 시간·지출 범위가 마련된 뒤 실행한다. 코드, configs, 그리고 bundle이 참조하는 data/raw 및 data/prepared 디렉터리를 같은 상대 구조로 준비해야 한다. 이 준비 작업은 원격 전송이나 GPU 대여를 수행하지 않았다.

대표 LR-HL 조건의 명령:

```sh
python -m src.real_preflight --config configs/gpu_pilot/LR-HL.json --bundle data/prepared/fineweb_10k/bundle.json --output results/preflight/gpu_LR-HL --device cuda
```

나머지 준비 설정: LR-HS, LR-LL, DW-HL, DW-HS, DW-LL. 각 config는 기존 architecture_grid의 모델을 사용하며, 2 warmup optimizer steps와 최대 10 measurement steps, 총 6,144 유효 학습 target을 설정했다. FP32·batch size 1 기준의 최초 기능/메모리 파일럿이다. 본실험 recipe, mixed precision 또는 대규모 배치 처리량을 대신하지 않는다.

보고서는 warmup 이후의 처리량(체크포인트 포함/제외), 체크포인트 시간, 평가 시간, 학습 측정 구간 peak allocated/reserved GPU memory를 분리한다. CPU에는 GPU 메모리값을 넣지 않는다. 측정 경로에는 routing의 CPU 진단 변환과 finite 검사가 포함된다. 평가 메모리는 학습 peak 수치에 포함되지 않는다. 실제 CUDA 검증 테스트는 CUDA 장비에서 다시 실행해야 한다.

모든 이 경로의 실행에는 `purpose=pipeline_smoke_only`가 기록된다. `src.analyze`는 이 실행을 연구 대비에서 제외한다. 본실험은 파일럿의 실제 비용·메모리·속도를 본 뒤 별도 프로토콜로 고정한다.
