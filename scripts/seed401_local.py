"""Seed401: single T4, local checkpoints, explicit Kaggle Output save/restore.

No authentication, remote storage, automatic retry, or GPU execution on import.
The frozen training code and experiment configs are imported without modification.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('HL', 'HS', 'LS', 'LL')
STATE = Path('/kaggle/working/seed401_local_state')
PRIOR_PREFLIGHT_SECONDS = 4214.246299505234
LIMITS = {'pilot': 9900, 'main': 86400}
SESSION_SECONDS = 5400
SAVE_RESERVE = 1020


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def new_state():
    return {'schema': 'seed401-local-v1', 'seed': 401,
            'prior_preflight_seconds': PRIOR_PREFLIGHT_SECONDS,
            'limits': LIMITS, 'sessions': [],
            'failed_pilot_preserved': 'be7ee742-8a0b-4cde-9730-fb62473d5b05',
            'approval': '2026-10-07: approved restart and user requested Drive-free seed17-style setup'}


def remaining(state, phase):
    if state['schema'] != 'seed401-local-v1' or state['limits'] != LIMITS or state['prior_preflight_seconds'] != PRIOR_PREFLIGHT_SECONDS:
        raise ValueError('ledger identity mismatch')
    return LIMITS[phase] - (PRIOR_PREFLIGHT_SECONDS if phase == 'pilot' else 0) - sum(
        r['reserved_seconds'] for r in state['sessions'] if r['phase'] == phase)


def reserve(state, phase, seconds, runtime):
    if phase not in LIMITS or not math.isfinite(seconds) or not 0 < seconds <= SESSION_SECONDS:
        raise ValueError('invalid session reservation')
    if any(r['status'] in ('running', 'failed') for r in state['sessions']):
        raise ValueError('unresolved session: preserve outputs and inspect; no automatic retry')
    if any(r['runtime'] == runtime for r in state['sessions']):
        raise ValueError('use a new Kaggle runtime after saving this session')
    if remaining(state, phase) < seconds:
        raise ValueError('resource budget exhausted')
    row = {'id': uuid.uuid4().hex, 'runtime':runtime, 'phase':phase,
           'reserved_seconds':seconds, 'status':'running', 'started_at':time.time()}
    state['sessions'].append(row)
    return row


def pointer(out):
    out = Path(out)
    p = json.loads((out/'checkpoints/latest.json').read_text())
    name = p['filename']
    if Path(name).name != name or not name.endswith('.pt'):
        raise ValueError('unsafe checkpoint filename')
    cp = out/'checkpoints'/name
    if cp.is_symlink() or sha(cp) != p['sha256']:
        raise ValueError('checkpoint checksum mismatch')
    return p, cp


def retain_two(out):
    _, latest = pointer(out)
    checkpoints = sorted((Path(out)/'checkpoints').glob('checkpoint-*.pt'), key=lambda p:p.stat().st_mtime_ns)
    keep = {latest, *[p for p in checkpoints if p != latest][-1:]}
    for p in checkpoints:
        if p not in keep:
            p.unlink()


def seal(state_dir):
    state_dir = Path(state_dir)
    records = {}
    for p in sorted(state_dir.rglob('*')):
        if p.is_symlink():
            raise ValueError('symlink in output')
        if p.is_file() and p.name != 'snapshot.json':
            records[str(p.relative_to(state_dir))] = {'sha256':sha(p), 'bytes':p.stat().st_size}
    write(state_dir/'snapshot.json', {'schema':'seed401-local-v1', 'files':records})
    return sha(state_dir/'snapshot.json')


def restore(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('restore requires empty destination')
    spec = json.loads((source/'snapshot.json').read_text())
    if spec['schema'] != 'seed401-local-v1' or 'ledger.json' not in spec['files']:
        raise ValueError('invalid snapshot')
    for name, record in spec['files'].items():
        p = source/name
        if not p.resolve().is_relative_to(source) or p.is_symlink() or p.stat().st_size != record['bytes'] or sha(p) != record['sha256']:
            raise ValueError('snapshot checksum mismatch')
    state = json.loads((source/'ledger.json').read_text())
    remaining(state, 'pilot'); remaining(state, 'main')
    if any(r['status'] != 'closed' for r in state['sessions']):
        raise ValueError('unresolved saved session; inspect before restoring')
    destination.mkdir(parents=True, exist_ok=True)
    for name in [*spec['files'], 'snapshot.json']:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    for p in destination.glob('*/LR-*/checkpoints/latest.json'):
        pointer(p.parent.parent)


def pilot_estimate(reports):
    total = 0
    run_ids = set()
    for cell in ORDER:
        r = reports[cell]
        if (r.get('pilot_usable') is not True or r.get('pilot_measurement_complete') is not True
            or r.get('pilot_measured_steps') != 100 or r.get('optimizer_step') != 105
            or r.get('trained_target_tokens') != 430080 or r.get('status') == 'failed'
            or r['model']['seed'] != 401 or r['model']['cell'] != cell or r['run_id'] in run_ids):
            raise ValueError('pilot gate requires four distinct complete seed401 pilots')
        run_ids.add(r['run_id'])
        train, evaluation = r['elapsed_training_seconds_including_checkpoint'], r['evaluation_seconds_this_invocation']
        if not math.isfinite(train) or train <= 0 or not math.isfinite(evaluation) or evaluation < 0:
            raise ValueError('invalid pilot timing')
        total += train * 2048 / 100 + evaluation * 8
    # Includes a conservative 30% margin and one hour of setup/save/restart reserve.
    return total * 1.3 + 3600


def verify_final(out, report, cell):
    if report['status'] != 'completed':
        return
    from scripts.run_seed_repeat_c import verify_final as frozen_verify
    frozen_verify(out, report, {'seed':401, 'cell':cell})
    pointer(out)


def validate_payload():
    manifest_path = ROOT/'seed401_payload_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['payload_sha256'] != '877e874871dfc74fa203a1243dfd1f7644685fdb6b892940f8cfed176f5cf268':
        raise ValueError('unexpected frozen payload')
    for rec in [*manifest['files'], *manifest['remote_downloads']]:
        if sha(ROOT/rec['path']) != rec['sha256']:
            raise ValueError('frozen payload checksum mismatch: ' + rec['path'])
    return manifest


def run(phase, quota_hours, allocated_at, state_dir=STATE):
    import fcntl
    import torch
    from scripts.run_colab_mini import run_colab_mini
    state_dir = Path(state_dir)
    if phase not in LIMITS or not math.isfinite(quota_hours) or quota_hours <= 0:
        raise ValueError('enter the current remaining Kaggle GPU quota')
    elapsed = time.time() - allocated_at
    if not math.isfinite(elapsed) or elapsed < 0 or elapsed >= SESSION_SECONDS - SAVE_RESERVE:
        raise ValueError('allocation start invalid or session save deadline reached')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise ValueError('single Tesla T4 required; no training started')
    torch.cuda.set_device(0)
    manifest = validate_payload()
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir/'operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = state_dir/'ledger.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else new_state()
        if phase == 'main':
            reports = {c:json.loads((state_dir/'pilot'/f'LR-{c}'/'mini_report.json').read_text()) for c in ORDER}
            estimate = pilot_estimate(reports)
            # First main launch must fit all four conditions. Later sessions keep this initial estimate as evidence.
            if not any(r['phase'] == 'main' for r in ledger['sessions']):
                if estimate > min(remaining(ledger,'main'), quota_hours*3600):
                    raise ValueError('pilot projection exceeds remaining time/quota')
                ledger['main_estimate_seconds'] = estimate
        if quota_hours*3600 < SESSION_SECONDS:
            raise ValueError('at least 90 minutes remaining quota required')
        runtime = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        row = reserve(ledger, phase, SESSION_SECONDS, runtime)
        row.update(allocated_at=allocated_at, quota_hours_observed=quota_hours,
                   payload_sha256=manifest['payload_sha256'], torch_version=torch.__version__, cuda_version=torch.version.cuda)
        write(ledger_path, ledger)
        deadline = allocated_at + SESSION_SECONDS - SAVE_RESERVE
        def timeout(*_):
            raise TimeoutError('session training deadline; preserve checkpoint and save output')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1, deadline-time.time()))
        try:
            def boundary(session, stage):
                if shutil.disk_usage(state_dir).free < 3*1024**3:
                    raise ValueError('less than 3 GiB free; stop and save')
                if stage in ('initial', 'after_chunk', 'after_report'):
                    if not (session.epoch == 0 and session.cursor == session.optimizer_step*8
                            and session.trained_target_tokens == session.optimizer_step*4096):
                        raise ValueError('training counter mismatch')
                    retain_two(session.output_dir)
                used = sum(p.stat().st_size for p in state_dir.rglob('*') if p.is_file())
                if used >= 30*1024**3:
                    raise ValueError('30 GiB output limit reached')
                return time.time() < deadline-180 if stage == 'before_chunk' else True
            for cell in ORDER:
                out = state_dir/phase/f'LR-{cell}'
                report_path = out/'mini_report.json'
                if report_path.exists():
                    prior = json.loads(report_path.read_text())
                    if phase == 'pilot':
                        if prior.get('pilot_usable') is not True:
                            raise ValueError('failed/incomplete pilot; no automatic retry')
                        pointer(out)
                        continue
                    if prior['status'] == 'completed':
                        verify_final(out, prior, cell)
                        continue
                if time.time() >= deadline - (1000 if phase == 'pilot' else 300):
                    break
                resume = (out/'checkpoints/latest.json').exists()
                if phase == 'pilot' and resume:
                    raise ValueError('pilot must be fresh')
                row.update(cell=cell)
                write(ledger_path,ledger)
                report = run_colab_mini(ROOT/f'configs/seed_repeat_c/seed-401/LR-{cell}.json',
                    ROOT/'data/prepared/colab_free_mini_v1/bundle.json', out, device='cuda',
                    resume=resume, max_steps=2048, max_seconds=min(900,deadline-time.time()-180) if phase=='pilot' else deadline-time.time()-180,
                    pilot=phase=='pilot', boundary_hook=boundary)
                if report['status'] == 'failed' or phase == 'pilot' and report.get('pilot_usable') is not True:
                    raise RuntimeError('training/pilot failed; see original logs and run manifest')
                pointer(out)
                if phase == 'main':
                    verify_final(out, report, cell)
                gc.collect(); torch.cuda.empty_cache()
                if phase == 'main' and report['status'] != 'completed':
                    break
            row.update(status='closed', elapsed_allocation_seconds=time.time()-allocated_at,
                       note='Full 90-minute reservation retained, including setup and output saving.')
        except BaseException:
            row['status'] = 'failed'
            (state_dir/f'failure-{row["id"]}.log').write_text(traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,old_handler)
            write(ledger_path,ledger)
            snapshot_hash = seal(state_dir)
            print('LOCAL SNAPSHOT:', snapshot_hash, flush=True)
            print('Quick Save + Save output; confirm Successful before stopping GPU. Local files alone are not persistent.', flush=True)
        if phase == 'pilot':
            reports = {c:json.loads((state_dir/'pilot'/f'LR-{c}'/'mini_report.json').read_text()) for c in ORDER}
            print('Main estimate seconds:', pilot_estimate(reports), flush=True)
        else:
            complete = all((state_dir/'main'/f'LR-{c}'/'mini_report.json').exists() and
                json.loads((state_dir/'main'/f'LR-{c}'/'mini_report.json').read_text())['status']=='completed' for c in ORDER)
            print('ALL FOUR COMPLETE' if complete else 'SESSION COMPLETE: save output and continue in a new runtime', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['pilot','main'])
    p.add_argument('--quota-hours', type=float, required=True)
    p.add_argument('--allocated-at', type=float, required=True)
    p.add_argument('--state-dir', type=Path, default=STATE)
    a=p.parse_args()
    run(a.phase,a.quota_hours,a.allocated_at,a.state_dir)


if __name__ == '__main__':
    main()
