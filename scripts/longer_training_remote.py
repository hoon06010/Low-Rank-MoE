"""One externally gated Colab A/B/C invocation; never advances to another cell/stage."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid

ROOT = Path('/content/low_rank_moe_longer')
BUNDLE = 'data/prepared/colab_free_mini_v1/bundle.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pointer_state(out):
    pointer = json.loads((out / 'checkpoints/latest.json').read_text())
    name = pointer.get('filename', '')
    if not name or Path(name).name != name or not name.endswith('.pt'):
        raise ValueError('unsafe checkpoint pointer')
    path = out / 'checkpoints' / name
    if path.is_symlink() or digest(path) != pointer.get('sha256'):
        raise ValueError('checkpoint pointer hash mismatch')
    return pointer, path


def inspect(out):
    files = sorted(p for p in out.rglob('*') if p.is_file()) if out.exists() else []
    result = {'output': str(out), 'exists': out.exists(), 'bytes': sum(p.stat().st_size for p in files),
              'checkpoint_count': sum(p.suffix == '.pt' for p in files),
              'disk_free_bytes': shutil.disk_usage(out if out.exists() else ROOT).free}
    if (out / 'mini_report.json').exists():
        result['report'] = json.loads((out / 'mini_report.json').read_text())
    if (out / 'checkpoints/latest.json').exists():
        result['verified_pointer'], path = pointer_state(out)
        result['latest_checkpoint_bytes'] = path.stat().st_size
    return result


def equal_state(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left == right


def recovery_worker(config, out):
    """Restore through the real runner and compare state immediately before its one step."""
    import torch
    from src.train import TrainingSession
    from scripts.run_colab_mini import run_colab_mini
    original = TrainingSession.restore_latest
    verified = []
    def checked_restore(session):
        pointer, path = pointer_state(out)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['optimizer_step'], saved['trained_target_tokens']) != (105, 430080):
            raise ValueError('recovery source must be completed A HL pilot at 105 steps')
        original(session)
        current = session._checkpoint_payload(saved['checkpoint_id'])
        fields = ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'rng_state',
                  'data_order', 'data_order_hash', 'epoch', 'cursor', 'optimizer_step', 'trained_target_tokens')
        checks = {key: bool(equal_state(saved[key], current[key])) for key in fields}
        if not all(checks.values()):
            raise ValueError('restored state differs from checkpoint: ' + str(checks))
        verified.append({'pointer': pointer, 'checks': checks, 'before_steps': 105, 'before_targets': 430080})
    TrainingSession.restore_latest = checked_restore
    try:
        report = run_colab_mini(config, ROOT / BUNDLE, out, device='cuda', resume=True, max_steps=1, pilot=False)
    finally:
        TrainingSession.restore_latest = original
    if not verified or report['status'] == 'failed' or (report['optimizer_step'], report['trained_target_tokens']) != (106, 434176):
        raise ValueError('recovery did not advance 105->106 / 430080->434176')
    result = {**verified[0], 'after_steps': 106, 'after_targets': 434176, 'status': 'verified',
              'claim_boundary': 'saved/retrieved resume only; no uninterrupted CUDA bitwise equivalence claim'}
    (out / 'recovery_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def extract_recovery(archive, sha256, out):
    if digest(archive) != sha256:
        raise ValueError('recovery archive checksum mismatch')
    if out.exists():
        raise ValueError('recovery destination must be new')
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError('duplicate recovery members')
        for member in members:
            if not member.isfile() or not (out / member.name).resolve().is_relative_to(out.resolve()):
                raise ValueError('unsafe recovery archive member')
        if shutil.disk_usage(ROOT).free < sum(m.size for m in members) + 1024**3:
            raise ValueError('insufficient recovery extraction space')
        out.mkdir(parents=True)
        tar.extractall(out, filter='data')
    pointer_state(out)
    report = json.loads((out / 'mini_report.json').read_text())
    if report.get('model', {}).get('cell') != 'HL' or not report.get('pilot_usable') or report.get('optimizer_step') != 105:
        raise ValueError('recovery archive is not a usable HL A pilot')


def validate_budget(timeout, remaining, reserve):
    if not all(math.isfinite(x) for x in (timeout, remaining, reserve)) or timeout <= 0 or reserve < 0:
        raise ValueError('invalid invocation/session time budget')
    if timeout + reserve > remaining:
        raise ValueError('not enough session time for invocation and recovery reserve')


def timed_child(command, timeout, log):
    started = time.monotonic()
    with log.open('w') as stream:
        environment = {k: v for k, v in os.environ.items() if k != "LONGER_ARGS"}
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
        try:
            code = process.wait(timeout=timeout)
            status = 'returned'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code, status = -9, 'hard_timeout'
    return {'returncode': code, 'status': status, 'wall_seconds': time.monotonic()-started, 'log': str(log)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['A','B','C'], required=True)
    parser.add_argument('--cell', choices=['HL','HS','LS','LL'], required=True)
    parser.add_argument('--mode', choices=['run','inspect','export','recover'], default='inspect')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recovery-output', action='store_true')
    parser.add_argument('--max-steps', type=int, default=32)
    parser.add_argument('--timeout-seconds', type=float)
    parser.add_argument('--session-remaining-seconds', type=float)
    parser.add_argument('--reserve-seconds', type=float, default=600)
    parser.add_argument('--recovery-tar', type=Path)
    parser.add_argument('--recovery-sha256')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    if argv is None and 'LONGER_ARGS' in os.environ:
        argv = json.loads(os.environ['LONGER_ARGS'])
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError('LONGER_ARGS must be a JSON string list')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if args.recovery_output and (args.mode not in ('inspect', 'export') or args.stage != 'A' or args.cell != 'HL'):
        raise ValueError('recovery-output requires inspect/export A HL')
    stage = 'recovery' if args.mode == 'recover' or args.recovery_output else args.stage
    out = ROOT / f'runs/longer_training/{stage}/adamw/17/{args.cell}'
    config = ROOT / f'configs/longer_training_{"b" if args.stage == "B" else "c"}/LR-{args.cell}.json'
    if args.worker:
        if args.mode != 'recover' or args.stage != 'A' or args.cell != 'HL':
            raise ValueError('invalid recovery worker')
        recovery_worker(config, out)
        return 0
    if args.mode == 'inspect':
        print(json.dumps(inspect(out), indent=2)); return 0
    if args.timeout_seconds is None or args.session_remaining_seconds is None:
        raise ValueError('run/recover/export require explicit timeout and remaining session budget')
    validate_budget(args.timeout_seconds, args.session_remaining_seconds, args.reserve_seconds)
    if args.mode == 'export':
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('export hard timeout')))
        signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        pointer_state(out)
        files = sorted(p for p in out.rglob('*') if p.is_file())
        if any(p.is_symlink() for p in out.rglob('*')):
            raise ValueError('symlinks forbidden in run exports')
        size = sum(p.stat().st_size for p in files)
        if shutil.disk_usage(ROOT).free < size + 1024**3:
            raise ValueError('insufficient space to retain run and full uncompressed export')
        archive = ROOT / f'longer_{stage}_{args.cell}_{uuid.uuid4().hex}.tar'
        started = time.monotonic()
        with tarfile.open(archive, 'w:') as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(out)), recursive=False)
        print(json.dumps({'archive': str(archive), 'sha256': digest(archive), 'bytes': archive.stat().st_size,
                          'export_seconds': time.monotonic()-started, 'files': len(files)}))
        signal.setitimer(signal.ITIMER_REAL, 0)
        return 0
    if args.timeout_seconds is None or args.session_remaining_seconds is None:
        raise ValueError('run/recover require explicit timeout and remaining session budget')
    validate_budget(args.timeout_seconds, args.session_remaining_seconds, args.reserve_seconds)
    if not 1 <= args.max_steps <= 32:
        raise ValueError('at most 32 steps per B/C invocation')
    if args.stage == 'A' and args.resume:
        raise ValueError('A pilot cannot resume; use separate recover mode')
    started = time.monotonic()
    logs = ROOT / 'reports/longer_training/invocations'
    logs.mkdir(parents=True, exist_ok=True)
    invocation = uuid.uuid4().hex
    if args.mode == 'recover':
        if args.stage != 'A' or args.cell != 'HL' or not args.recovery_tar or not args.recovery_sha256:
            raise ValueError('recovery requires stage A cell HL and retrieved full archive + SHA256')
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('recovery setup hard timeout')))
        signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        try:
            extract_recovery(args.recovery_tar, args.recovery_sha256, out)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        command = [sys.executable, str(ROOT / 'scripts/longer_training_remote.py'), '--stage','A','--cell','HL','--mode','recover','--worker']
    else:
        command = [sys.executable, '-m', 'scripts.run_colab_mini', '--config', str(config), '--bundle', str(ROOT/BUNDLE),
                   '--output', str(out), '--device', 'cuda']
        command += ['--pilot'] if args.stage == 'A' else ['--max-steps', str(args.max_steps)]
        if args.resume:
            command.append('--resume')
    # The controller owns measured cost, disk, free-tier and stage gates. No progression here.
    elapsed_setup = time.monotonic()-started
    timeout = args.timeout_seconds-elapsed_setup
    if timeout <= 0:
        raise ValueError('invocation budget exhausted during setup')
    result = timed_child(command, timeout, logs / f'{invocation}.log')
    result.update({'stage': args.stage, 'cell': args.cell, 'mode': args.mode, 'command': command,
                   'setup_seconds': elapsed_setup, 'total_seconds': time.monotonic()-started,
                   'timeout_seconds': args.timeout_seconds, 'session_remaining_at_start': args.session_remaining_seconds,
                   'reserve_seconds': args.reserve_seconds, 'disk_free_bytes': shutil.disk_usage(ROOT).free})
    (logs / f'{invocation}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['returncode'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
