# 실데이터 준비 및 실행 경로 검증 — 2026-09-26

요청한 소량 실데이터 준비와 CPU/CUDA 학습 경로 구현을 완료했다. 실제 실행 검증은 CPU에서 했으며 CUDA 장비 검증은 남아 있다. GPU 대여·결제·본실험은 하지 않았다.

## 데이터

- FineWeb-Edu `sample-10BT`, revision `84e8104e779e409e2267ac60609138e3dda2cbd2`의 첫 shard를 받았다. 원본 SHA-256이 공식 LFS 값 `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871`과 일치했다.
- 처음 10,000문서를 고정 사용했다. 정확한 중복 1건 제거, 9,999문서 보존. 학습 9,810 / 개발 95 / 평가 94문서. 문서 본문·URL·ID·연결 그룹의 분할 간 겹침 0건.
- 유사 문서 표본 33,122쌍을 비교했으며 Jaccard >= 0.8인 쌍 0건. 표본 방식·문서 길이 제한이 있으므로 전수 유사중복 검사로 해석하지 않는다.
- GPT-2 tokenizer revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, 어휘 50,257개, EOS 50,256. 파일 해시·빈 문자열·영문 왕복·EOS 동작을 확인했다.
- 문서 순서를 고정하고 train 65,536 / dev 4,096 / test 4,096 유효 정답 토큰을 보존했다. test split으로 모델 성능을 평가하지 않았다.
- 원본 약 2.0 GiB, 준비 결과 약 52 MiB. 전체 14 shard를 받거나 전체 100억 토큰을 준비한 것은 아니다.

## 구현과 검증

`src.data`에 제한된 토큰 패킹, 유사중복 표본 검사, 원본·토크나이저·정답 파일 해시 재검증을 추가했다. `TrainingSession`은 감사된 묶음과 일치하는 입력·토크나이저·평가 출처가 있을 때만 measured 실행을 허용한다. CPU/CUDA FP32 실행과 사용한 장치 RNG 저장·복구를 연결했다. `src.real_preflight`는 warmup 뒤 처리량, 저장 시간, 평가 시간, CUDA 학습 구간 최대 메모리를 기록한다.

작은 모델로 실제 영어 데이터 2,048 유효 학습 토큰을 4 optimizer steps 학습하고 dev 4,096 토큰을 평가했다. 별도 실행은 512토큰에서 중단 후 재개했다. 연속 실행과 재개 실행의 모델 파라미터, optimizer, scheduler, Python/NumPy/PyTorch RNG, 데이터 순서/cursor/epoch, 학습 토큰 수, step 수, dev loss가 모두 정확히 일치했다. 이는 학습 경로 검증이며 rank/shared 가설의 성능 결과가 아니다.

최종 전체 테스트: **93 passed, 1 skipped**. 건너뛴 1개는 실제 CUDA 장비가 필요한 fit/evaluate/RNG 복구 테스트다. CUDA 실행 요청은 현재 환경에서 명확히 unavailable로 종료됐다. GPU 처리량·메모리·실험 비용은 미측정이다.

준비용 실데이터 실행에는 `purpose=pipeline_smoke_only`가 붙고 연구 대비 계산에서는 제외된다. 원본 준비 단계 보고서들은 당시 검증 기록으로 보존했다.

## 근거와 다음 실행

- 데이터 묶음: `data/prepared/fineweb_10k/bundle.json`
- 문서·토큰 감사: `data/prepared/fineweb_10k/audit.json`
- 유사중복 범위: `data/prepared/fineweb_10k/near_duplicate_audit.json`
- 실제 연속/재개 비교: `reports/m9/real_cpu_verification.json`와 `.log`
- 최신 실제 실행: `results/preflight/real_cpu_continuous_v2/`, `results/preflight/real_cpu_resumed_v2/`
- 전체 테스트 로그: `reports/m9/test_full.log`
- 설치 환경: `reports/m9/environment.freeze.txt`
- 코드·설정·검증 결과 해시: `reports/m9/artifact_hashes.json`
- 재실행 명령·GPU 경계: `plans/m9/runbook.md`

다음 단계는 제한된 GPU 파일럿이다. 대표 6조건 설정을 `configs/gpu_pilot/`에 준비했다. 장비·시간·지출 상한을 정하고 실제 CUDA 검증과 측정을 수행한 후 본실험 비용을 판단한다. 현재의 CPU 수치로 GPU 비용을 추정하지 않는다.
