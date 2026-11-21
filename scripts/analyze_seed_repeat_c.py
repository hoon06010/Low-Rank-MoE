"""Read-only, exploratory Stage C collector; does not alter frozen training sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ADAMW = 'stage-c-init-seed-repeat-v1'
SPECTRON = 'seed401-spectron-fixed-recipe-v1'
SEEDS = (401, 402, 403)
CELLS = ('HS', 'HL', 'LS', 'LL')
FORMULAS = {'delta_budget': {'LL': 1, 'HS': -1}, 'G_low': {'LS': 1, 'LL': -1},
            'G_high': {'HS': 1, 'HL': -1}, 'I_LR': {'LS': 1, 'LL': -1, 'HS': -1, 'HL': 1}}


def _json(path):
    return json.loads(Path(path).read_text())


def _equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label}: expected {expected!r}, got {actual!r}')


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def _finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite loss')
    return result


def expected_identity(experiment):
    """Read the independently retained package inventory, not run-supplied identities."""
    package = ROOT / ('reports/seed_repeat_c/package/payload_manifest.json' if experiment == ADAMW else 'reports/seed401_spectron/package/manifest.json')
    inventory = _json(package)['files']
    files = ({name: row['sha256'] for name, row in inventory.items()} if isinstance(inventory, dict)
             else {row['path']: row['sha256'] for row in inventory})
    scripts = ('run_colab_mini.py', 'run_seed_repeat_c.py', 'seed_repeat_policy.py', 'overnight_store.py', 'package_colab_mini.py') if experiment == ADAMW else ('seed401_spectron.py', 'run_colab_mini.py', 'seed401_local.py')
    names = {name for name in files if name.startswith('src/') and name.endswith('.py')}
    names.update('scripts/' + name for name in scripts)
    sources = {name: files[name] for name in sorted(names)}
    policy = _json(ROOT/'configs/seed_repeat_c/experiment_manifest.json')
    bundle_path = ROOT/'data/prepared/colab_free_mini_v1/bundle.json'
    _equal(hashlib.sha256(bundle_path.read_bytes()).hexdigest(), policy['bundle_sha256'], 'expected frozen bundle file')
    bundle = _json(bundle_path)
    dev_path = bundle_path.parent/bundle['splits']['dev']['path']
    _equal(hashlib.sha256(dev_path.read_bytes()).hexdigest(), bundle['splits']['dev']['sha256'], 'expected frozen dev file')
    return {'source_file_hashes': sources, 'data_manifest_hash': policy['bundle_sha256'],
            'eval_revision': _hash_json(_json(dev_path)),
            'tokenizer_revision': bundle['manifest']['tokenizer']['revision']['value']}


def collect(spec):
    """Validate one explicitly scoped main directory, without unpickling checkpoints."""
    out = Path(spec['path'])
    manifests = [json.loads(line) for line in (out/'run_manifest.jsonl').read_text().splitlines() if line.strip()]
    if len(manifests) != 1:
        raise ValueError('duplicate or missing run manifest')
    m = manifests[0]
    if m.get('seed') not in SEEDS:
        return None, 'seed outside 401/402/403'
    experiment = spec['experiment_id']
    optimizer = 'adamw' if experiment == ADAMW else 'spectron'
    seed, cell = m['seed'], m['cell']
    if cell not in CELLS or (optimizer == 'spectron' and (seed != 401 or cell not in ('HS', 'LL'))):
        raise ValueError('condition outside experiment scope')
    for key, value in {'family':'LR', 'optimizer':optimizer, 'source_kind':'measured',
                       'purpose':'mini_exploration_only'}.items():
        _equal(m.get(key), value, key)
    if m.get('status') != 'completed':
        return None, f"unfinished: {m.get('status')}"
    config_path = ROOT / (f'configs/seed_repeat_c/seed-{seed}/LR-{cell}.json' if optimizer == 'adamw' else f'configs/seed401_spectron/LR-{cell}.json')
    config = _json(config_path)
    _equal(m.get('recipe_id'), config['training']['recipe_id'], 'recipe')
    _equal(m.get('config_hash'), _hash_json(config['model']), 'model config hash')
    _equal(m.get('checkpoint_step'), 2048, 'final step')
    _equal(m.get('trained_target_tokens'), 8388608, 'final targets')
    r = _json(out/'mini_report.json')
    for field in ('run_id', 'checkpoint_id', 'status', 'trained_target_tokens', 'data_manifest_hash', 'eval_revision', 'source_kind', 'purpose'):
        _equal(r.get(field), m.get(field), f'report {field}')
    _equal(r.get('model'), config['model'], 'report model')
    for key, value in config['training'].items():
        _equal(r.get('training', {}).get(key), value, f'report training {key}')
    _equal(m.get('training_config_hash'), _hash_json(r['training']), 'training config hash')
    _equal(m.get('source_commit'), _hash_json(r.get('source_file_hashes')), 'source hash')
    expected = expected_identity(experiment)
    _equal(r.get('source_file_hashes'), expected['source_file_hashes'], 'expected frozen source')
    for key in ('data_manifest_hash', 'eval_revision', 'tokenizer_revision'):
        _equal(m.get(key), expected[key], 'expected frozen ' + key)
    _equal(r.get('optimizer_step'), 2048, 'report step')
    _equal(r.get('precision'), 'float32', 'precision')
    _equal(r.get('pilot'), False, 'pilot cannot be main')
    for document in (m, r):
        for key in ('experiment_id', 'phase'):
            if key in document:
                _equal(document[key], spec[key], key)
    if (out/'experiment_spec.json').exists():
        identity = _json(out/'experiment_spec.json')
        for key in ('experiment_id', 'phase'):
            _equal(identity.get(key), spec[key], f'spec {key}')
        for key, value in (('seed',seed), ('cell',cell), ('recipe_id',m['recipe_id'])):
            _equal(identity.get(key), value, f'spec {key}')
    p = _json(out/'checkpoints/latest.json')
    _equal(p.get('checkpoint_id'), m['checkpoint_id'], 'checkpoint identity')
    name = p.get('filename', '')
    if not name or Path(name).name != name or not name.endswith('.pt'):
        raise ValueError('unsafe checkpoint filename')
    checkpoint = out/'checkpoints'/name
    if checkpoint.is_symlink():
        raise ValueError('checkpoint symlink')
    with checkpoint.open('rb') as stream:
        _equal(hashlib.file_digest(stream,'sha256').hexdigest(), p.get('sha256'), 'checkpoint hash')
    with (out/'metrics.csv').open(newline='') as stream:
        metrics = list(csv.DictReader(stream))
    if len(metrics) != 8:
        raise ValueError('exactly eight dev evaluations required')
    tokens, checkpoints, curve = set(), set(), []
    for row in metrics:
        for key in ('run_id', 'protocol_id', 'config_id', 'family', 'cell'):
            _equal(row.get(key), str(m[key]), f'metric {key}')
        _equal(int(row['seed']), seed, 'metric seed')
        _equal(row.get('split'), 'dev', 'metric split')
        _equal(row.get('source_kind'), 'measured', 'metric source')
        _equal(int(row['eval_target_tokens']), 65536, 'dev targets')
        token = int(row['trained_target_tokens'])
        if token in tokens or row['checkpoint_id'] in checkpoints:
            raise ValueError('duplicate evaluation')
        tokens.add(token); checkpoints.add(row['checkpoint_id'])
        curve.append({'trained_target_tokens':token, 'loss_nats':_finite(row['loss_nats'])})
        if token == 8388608:
            _equal(row['checkpoint_id'], m['checkpoint_id'], 'final metric checkpoint')
            _equal(row['status'], 'completed', 'final metric status')
    _equal(tokens, set(range(1048576,8388609,1048576)), 'evaluation cadence')
    curve.sort(key=lambda row:row['trained_target_tokens'])
    if r.get('loss_nats') is not None:
        _equal(_finite(r['loss_nats']),curve[-1]['loss_nats'],'report loss')
    return dict(experiment_id=experiment, seed=seed, cell=cell, optimizer=optimizer,
        recipe_id=m['recipe_id'], run_id=m['run_id'], loss_nats=curve[-1]['loss_nats'], curve=curve,
        identity={key:m.get(key) for key in ('protocol_id','data_manifest_hash','eval_revision','tokenizer_revision')},
        checkpoint_sha256=p['sha256'], source_commit=m.get('source_commit'), path=str(out),
        resource_report={key:r.get(key) for key in ('parameter_count','gpu_memory_allocated_peak_bytes',
        'gpu_memory_reserved_peak_bytes','tokens_per_second_including_checkpoint','elapsed_training_seconds_including_checkpoint')}), None


def analyze(specs):
    rows, excluded, seen = [], [], set()
    for spec in specs:
        if spec.get('experiment_id') not in (ADAMW,SPECTRON) or spec.get('phase') != 'main':
            excluded.append({'spec':spec,'reason':'outside main experiment scope'});continue
        row, reason = collect(spec)
        if row is None:
            excluded.append({'spec':spec,'reason':reason});continue
        key=(row['optimizer'],row['seed'],row['cell'])
        if key in seen:
            raise ValueError(f'duplicate condition {key}')
        seen.add(key);rows.append(row)
    if len({json.dumps(row['identity'],sort_keys=True) for row in rows}) > 1:
        raise ValueError('paired data/evaluation identity mismatch')
    adamw=[row for row in rows if row['optimizer']=='adamw']
    lookup={(row['seed'],row['cell']):row['loss_nats'] for row in adamw}
    contrasts={}
    for name, formula in FORMULAS.items():
        values=[{'seed':seed,'value':sum(sign*lookup[(seed,cell)] for cell,sign in formula.items())}
                for seed in SEEDS if all((seed,cell) in lookup for cell in formula)]
        numbers=[row['value'] for row in values];n=len(numbers)
        mean=statistics.mean(numbers) if n else None
        sd=statistics.stdev(numbers) if n>1 else None
        margin=4.30265273*sd/math.sqrt(3) if n==3 else None
        contrasts[name]=dict(per_seed=values,n=n,mean=mean,sample_sd=sd,
            paired_t_95_ci=[mean-margin,mean+margin] if margin is not None else None,
            status='complete_3_seed_descriptive' if n==3 else 'incomplete_no_3_seed_ci')
    spectron=[row for row in rows if row['optimizer']=='spectron']
    comparison=[{'cell':row['cell'],'spectron_minus_adamw':row['loss_nats']-lookup[(401,row['cell'])]}
                for row in spectron if (401,row['cell']) in lookup]
    sp={row['cell']:row['loss_nats'] for row in spectron}
    return dict(purpose='mini_exploration_only',interpretation='Descriptive initialization repeats; no practical superiority or stability claim.',
        verification='Checkpoint bytes and pointer hash checked; tensor/optimizer restore and GPU correctness are separate execution gates.',
        adamw={'runs':adamw,'contrasts':contrasts},
        spectron={'runs':spectron,'cell_differences':comparison,'n':1 if spectron else 0,
                  'delta_budget':sp['LL']-sp['HS'] if set(sp)=={'HS','LL'} else None},excluded=excluded)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,help='JSON list of experiment_id, phase, path records')
    args=parser.parse_args(argv)
    print(json.dumps(analyze(_json(args.inputs)),indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
