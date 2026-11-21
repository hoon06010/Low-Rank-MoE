# seed401 직접 Drive 전송 CPU 사전검사

이 도구는 CPU synthetic fixture의 원격 저장과 빈 cache 복구를 검사한다. 학습 가능한 torch checkpoint가 아니며 CUDA step0/중간 checkpoint 복구나 stock GPU runner의 영속 저장 승인을 대신하지 않는다. 공통 source/package는 변경하지 않는다.

## 인증

Kaggle runtime의 비공개 환경변수 `GOOGLE_DRIVE_ACCESS_TOKEN`이 필요하다. Drive 파일 생성·다운로드 권한이 있는 유효한 OAuth token을 사용한다. token을 notebook 소스, 출력, 파일, Git, 채팅에 쓰지 않는다. 도구는 token을 환경에서 요청마다 읽고 저장하지 않는다. 만료·HTTP 오류 시 자동 학습 진행이나 재승인은 하지 않는다.

`--folder-id`는 사용자가 소유하고 쓰기 가능한 전용 private Drive 폴더 ID를 지정한다. 이 도구는 sharing permission을 만들거나 바꾸지 않는다. notebook Internet도 필요하다. runtime이 실제로 이 API를 사용할 수 있는지는 원격 검사 결과로만 확인한다.

Drive API가 활성화된 사용자 소유 Google Cloud project와 quota 프로젝트 권한이 필요한 환경에서는 `GOOGLE_DRIVE_QUOTA_PROJECT`에 그 project ID를 지정한다. `X-Goog-User-Project` 요청 헤더로 전달된다. API가 비활성화되어 있으면 인증만 존재해도 실행할 수 없다. API 활성화·권한은 실제 HTTP 검사로 확인한다. 이 도구는 프로젝트나 API 설정을 변경하지 않는다.

TLS 검증은 Python urllib 기본 검증을 유지한다. 실행 Python의 CA 설정이 없으면 `SSL_CERT_FILE`에 설치된 certifi의 CA bundle 경로를 지정할 수 있다. HTTPS 검증을 끄거나 unverified context를 쓰지 않는다. 예: 해당 Python에서 `python -m certifi`로 CA 경로를 확인한 뒤 같은 runtime의 `SSL_CERT_FILE` 환경변수에 지정한다.

## 파일 배치와 신원

프로젝트가 원래 경로 구조로 Kaggle runtime에 존재해야 한다. 기존 package에 이 도구가 없으므로 동일 상대 경로에 전용 파일을 별도 제공한다. `scripts/overnight_store.py`는 원본 package에서 import한다. 별도 작업 폴더에 아래 JSON identity를 준비하고 publish와 fresh restore에 동일 파일을 사용한다.

```json
{"seed":401,"phase":"cpu-storage-fixture","source_revision":"1d67b6451e97f946a065550e2f9a51a8240eb116dbacef9208656cf68132a646"}
```

이 hash 값은 기존 기록이다. 실행 전에 실제 고정 source hash를 재검증해야 하며 fixture가 이를 자동 검증하지 않는다.

## CPU runtime 1: 업로드와 독립 다운로드

```sh
python reports/seed_repeat_c/seed-401/drive_transport.py quota
python reports/seed_repeat_c/seed-401/drive_transport.py fixture-publish \
  --folder-id PRIVATE_FOLDER_ID \
  --identity-file /kaggle/working/cpu-storage-identity.json \
  --work-dir /kaggle/working/drive-fixture-publish-UNIQUE \
  --timeout-seconds 300
```

원격 quota가 확인되고 archive bytes+1MiB 이상의 여유가 있어야 업로드한다. resumable upload 뒤 file ID로 새 파일에 다운로드하여 bytes/SHA256을 확인한다. receipt도 독립 다운로드 검증하며 immutable `commit-generation-...json`을 마지막에 생성·다운로드 검증한다. 성공 evidence에는 archive/receipt/pointer IDs와 hash가 있다. token은 없다. `pointer_id`를 다음 runtime에 전달한다.

이 commit은 공유 mutable `latest.json`을 원격에 덮어쓰지 않는다. 특정 commit ID를 사용하므로 기존 증거를 보존하고 동시 overwrite를 피한다. 최신 commit 선택, remote lock, failure marker와 resource ledger 동기화는 GPU 운영 adapter에서 별도 구현해야 한다.

## CPU runtime 2: 새 runtime에서 복구

runtime 1을 종료하고 새 CPU runtime을 시작한다. 같은 원본 package, 전용 transport 파일, 동일 identity JSON과 private token을 준비한다. 이전 runtime cache는 복사하지 않는다.

```sh
python reports/seed_repeat_c/seed-401/drive_transport.py fresh-restore \
  --pointer-id VERIFIED_POINTER_ID \
  --identity-file /kaggle/working/cpu-storage-identity.json \
  --work-dir /kaggle/working/drive-fixture-restore-UNIQUE \
  --timeout-seconds 300
```

commit→receipt→archive를 ID로 내려받고 hash/bytes/identity를 검증한 뒤 기존 안전 tar 복구 및 checkpoint ID/hash 검증을 사용한다. `restore-evidence.json`은 빈 cache 복구를 증명한다. 실제 새 runtime였다는 확인은 runtime ID·시간·package/source 재검증과 함께 외부 검사 기록에 남긴다. 도구 자체는 프로세스가 새 Kaggle runtime인지 판정하지 않는다.

## 보존과 범위

원격 파일이나 입력 checkpoint를 삭제하지 않는다. 성공/실패 cache도 남긴다. 다운로드 중 생긴 자기 소유의 불완전 파일만 오류 시 제거한다. 따라서 재시도는 항상 새로운 `--work-dir`를 사용한다. local 다운로드 전 blob 크기+1GiB reserve를 검사하며 기존 snapshot 제작·복구의 공간 검사를 유지한다.

이 사전검사는 GPU를 할당하지 않으며 main 실험을 실행하지 않는다. 기존 고정 runner의 mounted persistent_root, receipt/lock/ledger/failure 계약은 변경되지 않았다. API transport 검증 후 GPU 운영 연결과 실제 CUDA step0/중간 복구 검증이 추가로 필요하다.

공식 근거: [resumable upload](https://developers.google.com/workspace/drive/api/guides/manage-uploads), [blob download](https://developers.google.com/workspace/drive/api/guides/manage-downloads).
