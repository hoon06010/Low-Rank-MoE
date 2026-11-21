"""One explicitly gated seed/cell invocation with verified mounted-store boundaries."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import seed_repeat_policy as policy
from scripts.overnight_store import (deadline, publish_snapshot, restore_snapshot,
                                    _checkpoint, _atomic_json)

ROOT = policy.ROOT


def build_spec(seed, cell, phase='main'):
    if type(seed) is not int or seed not in policy.SEEDS or cell not in policy.CELLS or phase not in ('main', 'pilot', 'recovery'):
        raise ValueError('unsupported seed, cell, or phase')
    config_path = ROOT / f'configs/seed_repeat_c/seed-{seed}/LR-{cell}.json'
    config = json.loads(config_path.read_text())
    policy.validate_config(config_path, config)
    manifest = json.loads((ROOT / policy.POLICY).read_text())
    if policy.digest(ROOT / policy.BUNDLE) != manifest['bundle_sha256']:
        raise ValueError('frozen bundle hash mismatch')
    bundle = json.loads((ROOT / policy.BUNDLE).read_text())
    dev_path = (ROOT / policy.BUNDLE).parent / bundle['splits']['dev']['path']
    if policy.digest(dev_path) != bundle['splits']['dev']['sha256']:
        raise ValueError('dev file hash mismatch')
    base = 'reports/seed_repeat_c' + (f'/preflight/{phase}' if phase != 'main' else '')
    sources = policy.source_hashes()
    return {'experiment_id': policy.EXPERIMENT_ID, 'train_order_policy': policy.ORDER_POLICY,
            'seed': seed, 'cell': cell, 'phase': phase,
            'config': str(config_path.relative_to(ROOT)), 'config_file_sha256': policy.digest(config_path),
            'config_json_hash': policy.json_hash(config), 'policy_sha256': policy.digest(ROOT / policy.POLICY),
            'source_file_hashes': sources, 'source_revision': policy.json_hash(sources),
            'data_manifest_hash': manifest['bundle_sha256'],
            'train_file_sha256': bundle['splits']['train']['sha256'],
            'eval_revision': policy.json_hash(json.loads(dev_path.read_text())),
            'output': f'{base}/seed-{seed}/LR-{cell}'}


def validate_gate(gate, spec):
    if gate.get('approved') is not True or not gate.get('approval_reference'):
        raise ValueError('explicit resource approval is required')
    if gate.get('experiment_id') != policy.EXPERIMENT_ID or spec['seed'] not in gate.get('seeds', []):
        raise ValueError('resource approval scope mismatch')
    if spec['phase'] not in gate.get('phases', []):
        raise ValueError('phase is not approved')
    for key in ('provider', 'device_name', 'torch_version', 'cuda_version', 'python_version',
                'quota_checked_at', 'persistent_root', 'persistent_storage_evidence', 'session_id'):
        if not isinstance(gate.get(key), str) or not gate[key].strip():
            raise ValueError(f'missing resource gate: {key}')
    if gate.get('precision') != 'float32' or gate.get('source_revision') != spec['source_revision']:
        raise ValueError('precision or frozen source gate mismatch')
    for key in ('quota_seconds', 'total_limit_seconds', 'session_limit_seconds', 'session_started_at',
                'invocation_seconds', 'reserve_seconds', 'min_free_bytes', 'max_output_bytes'):
        value = gate.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid resource limit: {key}')
    cost = gate.get('cost_limit')
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0 or not gate.get('cost_currency'):
        raise ValueError('explicit cost limit is required, including zero for free resources')
    for key in ('cost_per_hour', 'prior_charged_seconds'):
        value = gate.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid accounting gate: {key}')
    if gate['total_limit_seconds'] * gate['cost_per_hour'] / 3600 > gate['cost_limit']:
        raise ValueError('approved time exceeds cost limit')
    if gate['total_limit_seconds'] > gate['quota_seconds']:
        raise ValueError('total time exceeds verified quota')
    elapsed = time.time() - gate['session_started_at']
    if elapsed < 0 or elapsed + gate['invocation_seconds'] + gate['reserve_seconds'] > gate['session_limit_seconds']:
        raise ValueError('insufficient session time including save/recovery reserve')
    if spec['phase'] == 'main':
        for key in ('four_cell_pilot_evidence', 'step0_recovery_evidence', 'bounded_recovery_evidence'):
            record = gate.get(key, {})
            path = Path(record.get('path', ''))
            if not path.is_file() or policy.digest(path) != record.get('sha256'):
                raise ValueError(f'missing verified preflight evidence: {key}')
    return gate


def validate_counters(session):
    step = session.optimizer_step
    if not (0 <= step <= 2048 and session.trained_target_tokens == step * 4096
            and session.cursor == step * 8 and session.epoch == 0):
        raise ValueError('seed-repeat counter/cycling mismatch')


def verify_receipt(out, store, spec, receipt=None):
    pointer, _ = _checkpoint(out)
    generation = receipt['generation'] if receipt else json.loads((store / 'latest.json').read_text())['generation']
    if Path(generation).name != generation:
        raise ValueError('unsafe generation')
    receipt = json.loads((store / f'{generation}.json').read_text())
    archive = store / receipt['archive']
    if (Path(receipt['archive']).name != receipt['archive'] or receipt['identity'] != spec
            or receipt['checkpoint_sha256'] != pointer['sha256']
            or receipt['checkpoint_id'] != pointer['checkpoint_id']
            or policy.digest(archive) != receipt['sha256']):
        raise ValueError('persistent receipt identity/hash mismatch')
    return receipt


def verify_final(out, report, spec):
    if report['status'] != 'completed':
        return
    if (report['optimizer_step'], report['trained_target_tokens']) != (2048, 8388608):
        raise ValueError('final counter mismatch')
    rows = list(csv.DictReader((out / 'metrics.csv').open()))
    expected_tokens = [1048576 * i for i in range(1, 9)]
    if [int(r['trained_target_tokens']) for r in rows] != expected_tokens:
        raise ValueError('final acceptance requires exactly eight dev evaluations')
    for row in rows:
        if (row['seed'] != str(spec['seed']) or row['config_id'] != f"LR-{spec['cell']}"
                or row['run_id'] != report['run_id'] or row['split'] != 'dev'
                or not math.isfinite(float(row['loss_nats']))):
            raise ValueError('final metric identity/finite mismatch')
    if rows[-1]['checkpoint_id'] != report['checkpoint_id'] or float(rows[-1]['loss_nats']) != report['loss_nats']:
        raise ValueError('final metric/report mismatch')



def validate_output_state(out, store, mode, resume):
    if (store / 'failure.json').exists():
        raise ValueError('persistent failure marker requires explicit operator review before continuation')
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError('fresh output directory must be empty')
    if not resume and mode != 'restore' and store.exists() and any(store.iterdir()):
        raise ValueError('persistent history exists; restore the same run instead of fresh initialization')


def record_failure(out, store, state, exc):
    """Attempt both failure records independently; do not hide the triggering error."""
    state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    for path in (store / 'failure.json', out / 'execution_state.json'):
        try:
            _atomic_json(path, state)
        except OSError:
            # An unavailable destination cannot be made durable by this process.
            # Its other record and the nonzero CLI exit still stop this invocation.
            pass

def execute(spec, gate, *, mode='run', resume=False, max_steps=32):
    """No progression to another cell. Reservation is retained if the process dies."""
    import torch
    from scripts.run_colab_mini import run_colab_mini
    if mode not in ('run', 'init', 'restore') or type(max_steps) is not int or not 1 <= max_steps <= 32:
        raise ValueError('one invocation must use 1..32 steps')
    if mode in ('restore', 'init') and resume or spec['phase'] == 'pilot' and (resume or mode != 'run'):
        raise ValueError('invalid fresh/resume/pilot mode')
    validate_gate(gate, spec)
    if (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != gate['device_name']
            or torch.__version__ != gate['torch_version'] or torch.version.cuda != gate['cuda_version']
            or platform.python_version() != gate['python_version']):
        raise ValueError('actual CUDA hardware/library gate mismatch')
    out = ROOT / spec['output']
    persistent = Path(gate['persistent_root']).resolve()
    if not Path(gate['persistent_root']).is_absolute() or not persistent.is_dir() or persistent.is_relative_to(ROOT):
        raise ValueError('persistent_root must be an existing external mounted directory')
    store = persistent / spec['phase'] / f"seed-{spec['seed']}" / f"LR-{spec['cell']}"
    validate_output_state(out, store, mode, resume)
    if resume:
        state = json.loads((out / 'execution_state.json').read_text())
        if state['spec'] != spec or state['status'] != 'verified':
            raise ValueError('resume requires matching verified execution state')
        verify_receipt(out, store, spec, state.get('last_verified_snapshot'))
    control = ROOT / 'reports/seed_repeat_c/control'
    control.mkdir(parents=True, exist_ok=True)
    with (control / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (persistent / 'seed-repeat.lock').open('a') as storage_lock:
            fcntl.flock(storage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _execute_locked(spec, gate, out, store, persistent, mode, resume, max_steps, run_colab_mini)


def _execute_locked(spec, gate, out, store, persistent, mode, resume, max_steps, runner):
    ledger_path = persistent / 'resource_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'sessions': {}, 'invocations': [], 'prior_charged_seconds': 0}
    session_id = gate['session_id']
    previous = ledger['sessions'].get(session_id, {'started_at': gate['session_started_at'], 'charged_seconds': 0})
    if previous['started_at'] != gate['session_started_at']:
        raise ValueError('session accounting start changed')
    elapsed = time.time() - gate['session_started_at']
    charge = max(previous['charged_seconds'], elapsed + gate['invocation_seconds'] + gate['reserve_seconds'])
    ledger['sessions'][session_id] = {'started_at': previous['started_at'], 'charged_seconds': charge}
    ledger['prior_charged_seconds'] = max(ledger['prior_charged_seconds'], gate['prior_charged_seconds'])
    ledger['charged_seconds'] = ledger['prior_charged_seconds'] + sum(s['charged_seconds'] for s in ledger['sessions'].values())
    if ledger['charged_seconds'] > gate['total_limit_seconds']:
        raise ValueError('total resource budget exhausted')
    invocation = uuid.uuid4().hex
    ledger['invocations'].append({'id': invocation, 'reserved_seconds': gate['invocation_seconds'],
                                  'seed': spec['seed'], 'cell': spec['cell'], 'mode': mode,
                                  'approval_reference': gate['approval_reference'], 'gate': gate})
    _atomic_json(ledger_path, ledger)
    started = time.monotonic()
    state = {'invocation_id': invocation, 'spec': spec, 'status': 'running', 'last_verified_snapshot': None}

    def storage_check(extra_bytes=0):
        for path in (ROOT, persistent, Path('/tmp')):
            if shutil.disk_usage(path).free < gate['min_free_bytes'] + extra_bytes:
                raise ValueError('storage free-space gate reached')
        used = sum(p.stat().st_size for p in (ROOT / 'reports/seed_repeat_c').rglob('*') if p.is_file())
        used += sum(p.stat().st_size for p in persistent.rglob('*') if p.is_file())
        if used + extra_bytes >= gate['max_output_bytes']:
            raise ValueError('storage usage gate reached')

    def publish(step):
        _, checkpoint = _checkpoint(out)
        selected = [p for p in out.rglob('*') if p.is_file() and (p.suffix != '.pt' or p == checkpoint)]
        archive_bound = sum(p.stat().st_size + 1024 for p in selected) + 10240
        storage_check(2 * archive_bound)  # local tar plus persistent staged copy
        receipt = publish_snapshot(out, store, spec, step=step, timeout_seconds=gate['reserve_seconds'])
        storage_check()
        return receipt

    def boundary(session, phase):
        validate_counters(session)
        storage_check()
        if phase == 'before_chunk':
            _, checkpoint = _checkpoint(out)
            storage_check(10 * checkpoint.stat().st_size)  # first AdamW state + checkpoint/copy overlap
            if mode == 'init':
                return False
            return True
        if phase in ('initial', 'after_chunk', 'after_report'):
            rows = [json.loads(line) for line in (out / 'run_manifest.jsonl').read_text().splitlines()]
            for row in rows:
                row.update(experiment_id=spec['experiment_id'], train_order_policy=spec['train_order_policy'],
                           config_file_sha256=spec['config_file_sha256'], execution_source_revision=spec['source_revision'])
            from src.train import _atomic_bytes
            _atomic_bytes(out / 'run_manifest.jsonl', lambda stream: stream.write((''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows)).encode()))
            receipt = publish(session.optimizer_step)
            state['last_verified_snapshot'] = receipt
            _atomic_json(out / 'execution_state.json', state)
        return True

    try:
        with deadline(gate['invocation_seconds']):
            storage_check()
            if mode == 'restore':
                receipt = restore_snapshot(store, out, spec, timeout_seconds=gate['reserve_seconds'])
                storage_check()
                state.update(status='verified', last_verified_snapshot=receipt)
                _atomic_json(out / 'execution_state.json', state)
                return state
            out.mkdir(parents=True, exist_ok=True)
            # Do not write to an empty fresh output before the underlying runner's guard.
            report = runner(ROOT / spec['config'], ROOT / policy.BUNDLE, out, device='cuda',
                            resume=resume, max_steps=max_steps, pilot=spec['phase'] == 'pilot', boundary_hook=boundary)
            if report['status'] == 'failed':
                raise ValueError('runner failed; automatic continuation is prohibited')
            verify_final(out, report, spec)
            report.update(config_hash=policy.json_hash(report['model']), seed=spec['seed'], config_id=f"LR-{spec['cell']}",
                          experiment_id=spec['experiment_id'], train_order_policy=spec['train_order_policy'],
                          config_file_sha256=spec['config_file_sha256'], execution_source_revision=spec['source_revision'])
            _atomic_json(out / 'mini_report.json', report)
            receipt = publish(report['optimizer_step'])
            verify_receipt(out, store, spec)
            state.update(status='verified', run_id=report['run_id'], optimizer_step=report['optimizer_step'],
                         trained_target_tokens=report['trained_target_tokens'], last_verified_snapshot=receipt)
            _atomic_json(out / 'execution_state.json', state)
            return state
    except BaseException as exc:
        record_failure(out, store, state, exc)
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        # Keep full invocation reservation: includes failed/idle uncertainty; never undercharge a crash.
        ledger['invocations'][-1].update(status=state['status'], measured_seconds=state['elapsed_seconds'])
        original_failure = sys.exc_info()[0] is not None
        for path, record in ((ledger_path, ledger), (out / 'execution_state.json', state)):
            try:
                _atomic_json(path, record)
            except OSError as exc:
                record_failure(out, store, state, exc)
                if not original_failure:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=policy.SEEDS, required=True)
    parser.add_argument('--cell', choices=policy.CELLS, required=True)
    parser.add_argument('--phase', choices=('main', 'pilot', 'recovery'), default='main')
    parser.add_argument('--mode', choices=('inspect', 'run', 'init', 'restore'), default='inspect')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int, default=32)
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args(argv)
    spec = build_spec(args.seed, args.cell, args.phase)
    if args.mode == 'inspect':
        print(json.dumps(spec, indent=2)); return 0
    if args.gate is None:
        parser.error('--gate is required before any GPU initialization or execution')
    state = execute(spec, json.loads(args.gate.read_text()), mode=args.mode, resume=args.resume, max_steps=args.max_steps)
    print(json.dumps({k: state.get(k) for k in ('invocation_id', 'status', 'run_id', 'optimizer_step', 'trained_target_tokens', 'elapsed_seconds')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
