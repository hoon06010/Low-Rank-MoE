"""Build an additive, default-OFF recovery notebook without changing frozen inputs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/seed_repeat_c/seed-401/local'
SOURCES = ('scripts/prepare_seed401_local.py', 'scripts/seed401_local.py',
           'scripts/seed401_recovery.py')


def build(root=ROOT, destination=OUT):
    root, destination = Path(root), Path(destination)
    embedded = {name: (root / name).read_text() for name in SOURCES}
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    cells = []

    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(True),
                'id': f'recovery-ready-{len(cells)}'}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Seed401 AdamW — 검증된 저장본에서 재개

기본 복구 대상은 `low-rank-moe-seed401-recovery` **Version 2**, HL128입니다.
GPU 시작 전에는 정확한 Successful Notebook Output만 연결하세요.
Start session으로 GPU가 Running이 된 뒤 비공개 `seed401-local-input`을 추가하세요.
두 입력을 동시에 연결한 채 새 GPU 세션을 시작하면 Adding data/status 70 오류가 관측됐습니다.
데이터셋 연결 완료 후에만 준비·복구 셀을 실행하세요. 다음 세션도 이 순서를 따릅니다.
그다음 세션부터는 직전 저장 때 별도로 보관한 `recovery-pin.json`을 사용합니다.

CPU에서 준비·복구를 먼저 검증할 수 있습니다. CPU→GPU 전환으로 런타임이 바뀌면
준비와 Input 복구를 다시 수행하세요. 학습은 T4 GPU0 하나·FP32입니다.
GPU 할당 시각은 실제 시각으로 입력하며 부팅/설치 시간을 빼지 않습니다.
사용자 종료 시각 제한은 2026-10-07 해제되었습니다. 세션90분·저장여유17분은 유지합니다.
기존 실패 차감과 장부를 보존합니다. 이 노트북은 AdamW 네 조건만 실행합니다.
''')
    add('code', '''START_TRAINING = False
QUOTA_HOURS = 0.0  # 지금 Kaggle에 표시된 남은 GPU 시간
ALLOCATED_AT_ISO = ""  # 실제 GPU 할당 시작, 예: YYYY-MM-DDTHH:MM:SS+09:00
RESTORE_FROM = ""  # /kaggle/input/.../seed401_local_state, Version2 또는 직전 Successful
PIN_PATH = ""  # 첫 Version2는 빈 문자열; 이후는 별도 보관한 recovery-pin.json 경로
''')
    add('code', '''# 준비만 수행합니다. GPU 학습을 시작하지 않습니다.
from pathlib import Path
import hashlib, json, os, sys
ROOT = Path('/kaggle/working/seed401_local')
''' + 'EMBEDDED = ' + repr(embedded) + '\nSOURCE_HASHES = ' + repr(hashes) + '''
for name, source in EMBEDDED.items():
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError('embedded operator checksum mismatch')
namespace = {}
exec(EMBEDDED['scripts/prepare_seed401_local.py'], namespace)
namespace['prepare'](Path('/kaggle/input'), ROOT)
for name, source in EMBEDDED.items():
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from scripts import seed401_local as local
from scripts import seed401_recovery as recovery
local.validate_payload()
print('Frozen payload and additive operator verified; training OFF.')
''')
    add('code', '''# 복구는 빈 작업 폴더에 한 번만 수행합니다. 기존 상태를 지우지 않습니다.
if not RESTORE_FROM:
    raise ValueError('Successful Output의 복구 경로를 입력하세요. 새 pilot은 시작하지 않습니다.')
source = Path(RESTORE_FROM).resolve()
if not source.is_relative_to(Path('/kaggle/input')):
    raise ValueError('복구 대상은 연결한 Notebook Output이어야 합니다.')
pin = json.loads(Path(PIN_PATH).read_text()) if PIN_PATH else None
recovery.restore_recovery(source, local.STATE, pin=pin)
print('Pinned snapshot, identities, ledger, and free disk verified.')
''')
    add('code', '''# 실제 GPU 할당 시각과 현재 quota를 확인한 뒤 명시적으로 실행합니다.
if START_TRAINING:
    from datetime import datetime
    import subprocess, time
    allocated = datetime.fromisoformat(ALLOCATED_AT_ISO)
    if allocated.tzinfo is None:
        raise ValueError('할당 시각에 +09:00 또는 +00:00 timezone을 포함하세요.')
    allocated_at = allocated.timestamp()
    stop_at = allocated_at + local.SESSION_SECONDS
    recovery.session_budget(allocated_at, stop_at, time.time(), QUOTA_HOURS)
    invocation = (
        'import sys; from scripts.seed401_recovery import run_recovery; '
        'run_recovery(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]))'
    )
    log_path = local.STATE / ('console-recovery-' + str(time.time_ns()) + '.log')
    command = [sys.executable, '-u', '-c', invocation,
               str(QUOTA_HOURS), str(allocated_at), str(stop_at)]
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                returncode = process.wait()
            except BaseException:
                process.terminate(); process.wait()
                raise
            finally:
                process.stdout.close()
    finally:
        # Interrupted sessions remain unresolved; preserve the closed console too.
        snapshot_hash = local.seal(local.STATE)
        print('FINAL SNAPSHOT:', snapshot_hash)
    if returncode:
        raise RuntimeError('실패 로그와 Output을 보존하세요. 자동 재시도하지 않습니다.')
    receipt = recovery.make_pin(local.STATE)
    receipt_path = Path('/kaggle/working/recovery-pin.json')
    local.write(receipt_path, receipt)
    print('EXTERNAL RECOVERY PIN:', json.dumps(receipt, sort_keys=True))
    print('Save output → Successful 확인 → Stop Session. Pin을 별도 보관하세요.')
else:
    print('Training OFF. 실제 할당 시각·quota 입력 후 START_TRAINING=True로 실행합니다.')
''')
    add('markdown', '''## 저장과 다음 세션

Save Version → Quick Save → Save output → **Successful** 확인 → Stop Session.
`recovery-pin.json`과 최종 snapshot 해시를 별도로 보관하세요. pin은 snapshot 폴더 밖에 있습니다.
다음 세션은 그 정확한 저장 버전과 pin을 연결합니다. 로컬 체크포인트만으로는 영속 보존되지 않습니다.
실패 세션을 closed로 바꾸거나 장부를 초기화하지 마세요. 새 실패는 원인을 검토한 뒤 처리합니다.
완료 조건: AdamW HL→HS→LS→LL 각각2048steps/8388608targets, dev8회, identity/hash 일치.
Spectron은 별도 package와 GPU pilot·복구 검증을 사용합니다.
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': cells}
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'seed401-recovery-ready.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    manifest = destination / 'seed401-recovery-ready.manifest.json'
    manifest.write_text(json.dumps({'schema': 'seed401-recovery-notebook-v1',
        'embedded_sources': hashes, 'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + '\n')
    return path, manifest


if __name__ == '__main__':
    for path in build():
        print(path)
