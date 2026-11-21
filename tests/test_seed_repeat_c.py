"""Seed-repeat policy and execution regression tests; no production GPU runs."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_colab_mini as mini

ROOT = Path(__file__).resolve().parents[1]


def test_twelve_configs_change_only_model_seed():
    for seed in (401, 402, 403):
        for cell in ('HL', 'HS', 'LS', 'LL'):
            path = ROOT / f'configs/seed_repeat_c/seed-{seed}/LR-{cell}.json'
            assert path.is_file(), f'missing seed-repeat config: {path}'
            config = mini._config(path)
            mini._validate_grid(path, config)
            assert config['model']['seed'] == seed
            config['model']['seed'] = 17
            assert config == json.loads((ROOT / f'configs/longer_training_c/LR-{cell}.json').read_text())


def test_eval_metric_records_actual_initialization_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(mini, 'evaluate', lambda *a, **k: {'loss_nats': 2., 'eval_target_tokens': 512})
    config = {'protocol_id': 'fixture', 'model': {'config_id': 'LR-HL', 'cell': 'HL', 'seed': 401}}
    result = {'run_id': 'fixture', 'checkpoint_id': 'ckpt', 'trained_target_tokens': 4096, 'status': 'incomplete'}
    mini._eval_metric(SimpleNamespace(model=None), [], config, result, tmp_path, 'cpu')
    assert next(csv.DictReader((tmp_path / 'metrics.csv').open()))['seed'] == '401'


def test_legacy_seed_restriction_stays_closed(tmp_path):
    original = ROOT / 'configs/longer_training_c/LR-HL.json'
    assert mini._config(original)['model']['seed'] == 17
    changed = json.loads(original.read_text())
    changed['model']['seed'] = 401
    path = tmp_path / 'LR-HL.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='seed 17'):
        mini._config(path)


def test_wrapper_exposes_exact_separate_paths():
    from scripts import run_seed_repeat_c as repeat
    for seed in (401, 402, 403):
        for cell in ('HL', 'HS', 'LS', 'LL'):
            spec = repeat.build_spec(seed, cell)
            assert spec['output'] == f'reports/seed_repeat_c/seed-{seed}/LR-{cell}'
            assert spec['seed'] == seed and spec['cell'] == cell
            assert spec['experiment_id'] == 'stage-c-init-seed-repeat-v1'
            assert spec['train_order_policy'] == 'fixed_audited_bundle_order'
            assert repeat.build_spec(seed, cell, 'pilot')['output'].startswith('reports/seed_repeat_c/preflight/')


def test_wrapper_requires_resource_approval():
    from scripts import run_seed_repeat_c as repeat
    with pytest.raises(ValueError, match='approval'):
        repeat.validate_gate({}, repeat.build_spec(401, 'HL'))


def test_counter_gate_forbids_cycle_and_partial_batch():
    from scripts import run_seed_repeat_c as repeat
    session = SimpleNamespace(optimizer_step=32, trained_target_tokens=131072, cursor=256, epoch=0)
    repeat.validate_counters(session)
    session.epoch = 1
    with pytest.raises(ValueError, match='counter'):
        repeat.validate_counters(session)


def test_package_has_seed_repeat_mode():
    from scripts import package_colab_mini as package
    assert 'seed-repeat' in package.MODES


def test_policy_rejects_wrong_seed_and_mixed_grid(tmp_path, monkeypatch):
    import shutil
    from scripts import seed_repeat_policy as policy
    shutil.copytree(ROOT / 'configs/seed_repeat_c', tmp_path / 'configs/seed_repeat_c')
    monkeypatch.setattr(policy, 'ROOT', tmp_path)
    monkeypatch.setattr(mini, 'ROOT', tmp_path)
    path = tmp_path / 'configs/seed_repeat_c/seed-401/LR-HL.json'
    original = json.loads(path.read_text())
    for seed in (17, 404):
        altered = copy.deepcopy(original); altered['model']['seed'] = seed
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match='seed-repeat seed'):
            mini._config(path)
    path.write_text(json.dumps(original))
    # Reformatting fails the frozen byte hash too.
    with pytest.raises(ValueError, match='hash'):
        mini._config(path)
    selected = copy.deepcopy(original)
    selected['model']['seed'] = 402
    path.write_bytes((ROOT / 'configs/seed_repeat_c/seed-401/LR-HL.json').read_bytes())
    with pytest.raises(ValueError, match='model mismatch: seed'):
        mini._validate_grid(path, selected)


def test_initialization_repeat_and_common_shape_pairing():
    import torch
    from src.model import MoETransformer
    from test_resume import CONFIG
    first = MoETransformer(dict(CONFIG, seed=401, shared_experts=2)).state_dict()
    same = MoETransformer(dict(CONFIG, seed=401, shared_experts=2)).state_dict()
    paired = MoETransformer(dict(CONFIG, seed=401, shared_experts=5)).state_dict()
    other = MoETransformer(dict(CONFIG, seed=402, shared_experts=2)).state_dict()
    assert all(torch.equal(first[k], same[k]) for k in first)
    common = [k for k in first if k in paired and first[k].shape == paired[k].shape]
    assert common and all(torch.equal(first[k], paired[k]) for k in common)
    assert any(not torch.equal(first[k], other[k]) for k in first)


def toy_runner(tmp_path, monkeypatch, runner=mini, seed=401):
    """Real trainer/evaluation with a tiny audited synthetic artifact, never C evidence."""
    from test_real_data import bundle
    from test_resume import CONFIG, TRAINING
    from scripts.seed_repeat_policy import digest, json_hash
    path, spec, batches = bundle(tmp_path)
    rows = [copy.deepcopy(batches['train'][0]) for _ in range(4)]
    (tmp_path / 'train.json').write_text(json.dumps(rows))
    spec['purpose'] = 'mini_exploration_only'
    spec['splits']['train'].update(sha256=digest(tmp_path / 'train.json'), content_hash=json_hash(rows), valid_target_tokens=16)
    path.write_text(json.dumps(spec))
    config = {'protocol_id': 'fixture-only', 'model': dict(CONFIG, seed=seed, config_id='LR-HL', cell='HL'),
              'training': dict(TRAINING, total_target_tokens=16)}
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, '_config', lambda _: json.loads(config_path.read_text()))
    monkeypatch.setattr(runner, '_validate_grid', lambda *_: None)
    for name, value in (('ROW_TARGETS', 4), ('GLOBAL_BATCH', 4), ('CHUNK_STEPS', 1), ('EVAL_INTERVAL', 4)):
        monkeypatch.setattr(runner, name, value)
    return config_path, path, config


def test_runner_seed_records_fresh_guard_and_resume(tmp_path, monkeypatch):
    import torch
    from scripts.overnight_store import _checkpoint
    from scripts.longer_training_remote import equal_state
    config_path, bundle_path, config = toy_runner(tmp_path, monkeypatch)
    out = tmp_path / 'run'
    first = mini.run_colab_mini(config_path, bundle_path, out, device='cpu', max_steps=1)
    pointer, checkpoint = _checkpoint(out)
    saved = torch.load(checkpoint, weights_only=False)
    manifest = json.loads((out / 'run_manifest.jsonl').read_text())
    metric = next(csv.DictReader((out / 'metrics.csv').open()))
    assert first['model']['seed'] == saved['config']['seed'] == manifest['seed'] == int(metric['seed']) == 401
    assert saved['config_hash'] == manifest['config_hash']
    assert saved['source_revision'] == manifest['source_commit']
    assert saved['data_manifest_hash'] == first['data_manifest_hash'] == manifest['data_manifest_hash']
    assert saved['eval_revision'] == first['eval_revision'] == manifest['eval_revision']
    with pytest.raises(ValueError, match='not empty'):
        mini.run_colab_mini(config_path, bundle_path, out, device='cpu', max_steps=1)
    changed = copy.deepcopy(config); changed['model']['seed'] = 402
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='identity mismatch'):
        mini.run_colab_mini(config_path, bundle_path, out, device='cpu', max_steps=1, resume=True)
    config_path.write_text(json.dumps(config))
    restored = []
    original = mini.TrainingSession.restore_latest
    def checked_restore(session):
        original(session)
        current = session._checkpoint_payload(saved['checkpoint_id'])
        fields = ('rng_state', 'scheduler_state_dict', 'optimizer_state_dict', 'model_state_dict',
                  'optimizer_step', 'trained_target_tokens', 'cursor', 'epoch', 'data_order_hash')
        assert all(equal_state(saved[k], current[k]) for k in fields)
        restored.append(True)
    monkeypatch.setattr(mini.TrainingSession, 'restore_latest', checked_restore)
    final = mini.run_colab_mini(config_path, bundle_path, out, device='cpu', max_steps=3, resume=True)
    assert restored and final['optimizer_step'] == 4 and final['epoch'] == 0 and final['cursor'] == 4
    assert all(r['seed'] == '401' for r in csv.DictReader((out / 'metrics.csv').open()))


def test_original_c_source_seed17_bounded_equivalence(tmp_path, monkeypatch):
    import tarfile
    import types
    import torch
    from scripts.overnight_store import _checkpoint
    from scripts.longer_training_remote import equal_state
    original = types.ModuleType('original_c_runner')
    original.__file__ = str(ROOT / 'scripts/run_colab_mini.py')
    snapshot = ROOT / 'reports/seed_repeat_c/preparation_audit/source_before.tar'
    with tarfile.open(snapshot) as archive:
        exec(compile(archive.extractfile('scripts/run_colab_mini.py').read(), original.__file__, 'exec'), original.__dict__)
    config_path, bundle_path, _ = toy_runner(tmp_path, monkeypatch, seed=17)
    toy_runner_config = mini._config
    monkeypatch.setattr(original, '_config', toy_runner_config)
    monkeypatch.setattr(original, '_validate_grid', lambda *_: None)
    for name in ('ROW_TARGETS', 'GLOBAL_BATCH', 'CHUNK_STEPS', 'EVAL_INTERVAL'):
        monkeypatch.setattr(original, name, getattr(mini, name))
    initial = []
    def hook(session, phase):
        if phase == 'initial':
            initial.append(session._checkpoint_payload('fixed'))
        return True
    old = original.run_colab_mini(config_path, bundle_path, tmp_path / 'old', device='cpu', max_steps=2, boundary_hook=hook)
    new = mini.run_colab_mini(config_path, bundle_path, tmp_path / 'new', device='cpu', max_steps=2, boundary_hook=hook)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'rng_state', 'data_order'):
        assert equal_state(initial[0][key], initial[1][key]), key
    old_payload = torch.load(_checkpoint(tmp_path / 'old')[1], weights_only=False)
    new_payload = torch.load(_checkpoint(tmp_path / 'new')[1], weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'rng_state',
                'cursor', 'epoch', 'optimizer_step', 'trained_target_tokens'):
        assert equal_state(old_payload[key], new_payload[key]), key
    assert old['loss_nats'] == new['loss_nats']


def test_package_seed_repeat_allowlist(tmp_path, monkeypatch):
    import tarfile
    import shutil
    from scripts import package_colab_mini as package
    from scripts import seed_repeat_policy as policy
    # Small package fixture avoids copying or compressing the production bundle in unit tests.
    for folder in ('configs/seed_repeat_c', 'src', 'scripts'):
        source = ROOT / folder
        if folder.startswith('configs'):
            shutil.copytree(source, tmp_path / folder)
        else:
            (tmp_path / folder).mkdir()
            for path in source.glob('*.py'):
                shutil.copyfile(path, tmp_path / folder / path.name)
    data = tmp_path / 'data/prepared/colab_free_mini_v1'; data.mkdir(parents=True)
    bundle = data / 'bundle.json'
    bundle.write_text(json.dumps({'audit_artifacts': [], 'splits': {}, 'manifest': {
        'tokenizer': {'artifact_checksums': []}, 'dataset': {'revision': {'value': 'fixed'}, 'shards': []}}}))
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    monkeypatch.setattr(package, 'BUNDLE', bundle)
    monkeypatch.setattr(package, 'LONGER_BUNDLE_SHA256', package.digest(bundle))
    monkeypatch.setattr(policy, 'ROOT', tmp_path)
    package.main(['--mode', 'seed-repeat'])
    output = tmp_path / 'reports/seed_repeat_c/package'
    manifest = json.loads((output / 'payload_manifest.json').read_text())
    with tarfile.open(output / 'payload.tar.gz') as archive:
        names = archive.getnames()
    configs = [n for n in names if n.startswith('configs/seed_repeat_c/seed-')]
    assert len(configs) == 12
    assert 'scripts/run_seed_repeat_c.py' in names
    assert all(n.startswith(('configs/seed_repeat_c/', 'scripts/', 'src/', 'data/')) for n in names)
    assert not any('reports/' in n or '.env' in n or 'credential' in n for n in names)
    assert package.digest(output / 'payload.tar.gz') == manifest['payload_sha256']


def approved_fixture(spec, tmp_path):
    import time
    return dict(approved=True, approval_reference='unit-test-only', experiment_id=spec['experiment_id'],
                seeds=[401], phases=['recovery'], provider='fixture', device_name='fixture',
                torch_version='fixture', cuda_version='fixture', python_version='fixture',
                quota_checked_at='fixture', persistent_root=str(tmp_path), persistent_storage_evidence='fixture',
                precision='float32', source_revision=spec['source_revision'], quota_seconds=1000,
                total_limit_seconds=1000, session_limit_seconds=1000, session_started_at=time.time(),
                invocation_seconds=60, reserve_seconds=20, min_free_bytes=1, max_output_bytes=10**9,
                cost_limit=0, cost_currency='USD', cost_per_hour=0, prior_charged_seconds=0, session_id='test-session')


def test_gate_rejects_missing_limits_scope_and_unverified_main(tmp_path):
    import time
    from scripts import run_seed_repeat_c as repeat
    spec = repeat.build_spec(401, 'HL', 'recovery')
    gate = approved_fixture(spec, tmp_path)
    repeat.validate_gate(gate, spec)
    for field, value in (('reserve_seconds', 0), ('quota_seconds', float('nan')),
                         ('session_started_at', time.time()-990), ('source_revision', 'changed'),
                         ('seeds', [402]), ('precision', 'bf16'), ('cost_limit', -1)):
        with pytest.raises(ValueError):
            repeat.validate_gate(dict(gate, **{field: value}), spec)
    main = dict(spec, phase='main')
    with pytest.raises(ValueError, match='preflight evidence'):
        repeat.validate_gate(dict(gate, phases=['main']), main)


def test_init_resume_and_external_snapshot_roundtrip(tmp_path, monkeypatch):
    import torch
    from scripts import run_seed_repeat_c as repeat
    from scripts import seed_repeat_policy as policy
    from scripts.overnight_store import restore_snapshot, _checkpoint
    from test_resume import CONFIG, TRAINING
    config_path, bundle_path, config = toy_runner(tmp_path, monkeypatch)
    # Retain real 512-target rows / accumulation of eight for wrapper counter validation.
    row = {'input_ids': [[1]*512], 'labels': [[1]*512], 'attention_mask': [[1]*512], 'valid_target_count': 512, 'split': 'train'}
    spec_bundle = json.loads(bundle_path.read_text())
    spec_bundle['context_length'] = 512
    for split, count in (('train', 16), ('dev', 1)):
        rows = [dict(row, split=split) for _ in range(count)]
        file = tmp_path / f'{split}.json'; file.write_text(json.dumps(rows))
        spec_bundle['splits'][split].update(sha256=policy.digest(file), content_hash=policy.json_hash(rows), valid_target_tokens=count*512)
    bundle_path.write_text(json.dumps(spec_bundle))
    config['model']['context_length'] = 512
    config['training'].update(total_target_tokens=8192, gradient_accumulation_steps=8)
    config_path.write_text(json.dumps(config))
    for name, value in (('ROW_TARGETS', 512), ('GLOBAL_BATCH', 4096), ('CHUNK_STEPS', 32), ('EVAL_INTERVAL', 1048576)):
        monkeypatch.setattr(mini, name, value)
    monkeypatch.setattr(repeat, 'ROOT', tmp_path)
    monkeypatch.setattr(policy, 'BUNDLE', 'bundle.json')
    spec = dict(experiment_id=policy.EXPERIMENT_ID, train_order_policy=policy.ORDER_POLICY,
                seed=401, cell='HL', phase='recovery', config='config.json', config_file_sha256=policy.digest(config_path),
                source_revision='fixture', output='reports/seed_repeat_c/preflight/recovery/seed-401/LR-HL')
    out = tmp_path / spec['output']; persistent = tmp_path / 'external'; persistent.mkdir()
    store = persistent / 'recovery/seed-401/LR-HL'
    gate = approved_fixture(spec, persistent)
    def cpu_runner(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return mini.run_colab_mini(*args, **kwargs)
    state = repeat._execute_locked(spec, gate, out, store, persistent, 'init', False, 1, cpu_runner)
    assert state['status'] == 'verified' and state['optimizer_step'] == 0
    before = torch.load(_checkpoint(out)[1], weights_only=False)
    restored = tmp_path / 'retrieved'
    restore_snapshot(store, restored, spec)
    assert policy.digest(_checkpoint(restored)[1]) == policy.digest(_checkpoint(out)[1])
    state = repeat._execute_locked(spec, gate, out, store, persistent, 'run', True, 1, cpu_runner)
    assert state['optimizer_step'] == 1 and state['trained_target_tokens'] == 4096
    repeat.verify_receipt(out, store, spec)
    with pytest.raises(ValueError, match='identity/hash'):
        repeat.verify_receipt(out, store, dict(spec, seed=402))
    report = json.loads((out / 'mini_report.json').read_text())
    manifest = json.loads((out / 'run_manifest.jsonl').read_text())
    assert report['experiment_id'] == manifest['experiment_id'] == spec['experiment_id']
    # A persistence exception after training cannot be presented as a successful run.
    monkeypatch.setattr(repeat, 'publish_snapshot', lambda *a, **k: (_ for _ in ()).throw(OSError('simulated storage failure')))
    with pytest.raises(OSError, match='storage failure'):
        repeat._execute_locked(spec, gate, out, store, persistent, 'init', True, 1, cpu_runner)
    assert json.loads((out / 'execution_state.json').read_text())['status'] == 'failed'


def test_final_acceptance_rejects_missing_or_wrong_eval(tmp_path):
    from scripts import run_seed_repeat_c as repeat
    report = dict(status='completed', optimizer_step=2048, trained_target_tokens=8388608,
                  run_id='run', checkpoint_id='last', loss_nats=6.)
    spec = dict(seed=401, cell='HL')
    rows = [dict(seed=401, config_id='LR-HL', run_id='run', checkpoint_id='last' if i==8 else str(i),
                 trained_target_tokens=i*1048576, split='dev', loss_nats=6.) for i in range(1,9)]
    def write(rows):
        with (tmp_path / 'metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(rows); repeat.verify_final(tmp_path, report, spec)
    write(rows[:-1])
    with pytest.raises(ValueError, match='eight dev'):
        repeat.verify_final(tmp_path, report, spec)
    rows[-1]['seed']=402;write(rows)
    with pytest.raises(ValueError, match='identity'):
        repeat.verify_final(tmp_path, report, spec)


def test_new_process_restores_seed_rng_and_schedule(tmp_path):
    import subprocess
    import sys
    from src.model import MoETransformer
    from src.train import TrainingSession
    from test_resume import CONFIG, TRAINING, BATCHES, _identity
    config = dict(CONFIG, seed=401)
    identity = _identity()
    session = TrainingSession(MoETransformer(config), config, TRAINING, BATCHES, identity, tmp_path / 'run')
    session.fit(max_optimizer_steps=1)
    fixture = tmp_path / 'fixture.json'
    fixture.write_text(json.dumps(dict(config=config, training=TRAINING, batches=BATCHES, identity=identity)))
    program = '''
import json,sys,torch
from pathlib import Path
from src.model import MoETransformer
from src.train import TrainingSession
from scripts.overnight_store import _checkpoint
from scripts.longer_training_remote import equal_state
root=Path(sys.argv[1]); f=json.loads((root/'fixture.json').read_text()); out=root/'run'
saved=torch.load(_checkpoint(out)[1],weights_only=False)
s=TrainingSession(MoETransformer(f['config']),f['config'],f['training'],f['batches'],f['identity'],out)
s.restore_latest(); now=s._checkpoint_payload(saved['checkpoint_id'])
fields=('model_state_dict','optimizer_state_dict','rng_state','scheduler_state_dict','cursor','epoch','optimizer_step','trained_target_tokens','data_order_hash')
assert all(equal_state(saved[k],now[k]) for k in fields)
print('fresh process restore verified')
'''
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'fresh process restore verified' in result.stdout


def test_restored_fallback_receipt_and_persistent_failure_gate(tmp_path, monkeypatch):
    from scripts import run_seed_repeat_c as repeat
    from scripts.overnight_store import publish_snapshot, restore_snapshot, _checkpoint
    from src.model import MoETransformer
    from src.train import TrainingSession
    from test_resume import CONFIG, TRAINING, BATCHES, _identity
    spec = repeat.build_spec(401, 'HL', 'recovery')
    out = tmp_path / 'run'; store = tmp_path / 'store'
    session = TrainingSession(MoETransformer(CONFIG), CONFIG, TRAINING, BATCHES, _identity(), out)
    session.fit(max_optimizer_steps=1)
    older = publish_snapshot(out, store, spec, step=1)
    session.fit(max_optimizer_steps=1)
    newer = publish_snapshot(out, store, spec, step=2)
    (store / newer['archive']).write_bytes(b'corrupt')
    restored = tmp_path / 'restored'
    receipt = restore_snapshot(store, restored, spec)
    assert receipt['generation'] == older['generation']
    repeat.verify_receipt(restored, store, spec, receipt)


def test_initial_persistence_failure_never_starts_training(tmp_path, monkeypatch):
    from scripts import run_seed_repeat_c as repeat
    from scripts import seed_repeat_policy as policy
    spec = repeat.build_spec(401, 'HL', 'recovery')
    spec['output']='reports/seed_repeat_c/preflight/recovery/seed-401/LR-HL'
    monkeypatch.setattr(repeat, 'ROOT', tmp_path)
    persistent = tmp_path / 'external'; persistent.mkdir()
    out=tmp_path/spec['output']; store=persistent/'recovery/seed-401/LR-HL'
    gate=approved_fixture(spec,persistent)
    trained=[]
    checkpoint=tmp_path/'tiny.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(repeat,'_checkpoint',lambda _: ({},checkpoint))
    def runner(*args, boundary_hook, **kwargs):
        out.mkdir(parents=True,exist_ok=True)
        (out/'run_manifest.jsonl').write_text('{}\n')
        boundary_hook(SimpleNamespace(optimizer_step=0,trained_target_tokens=0,cursor=0,epoch=0),'initial')
        trained.append(True)
    monkeypatch.setattr(repeat,'publish_snapshot',lambda *a,**k: (_ for _ in ()).throw(OSError('storage failed')))
    with pytest.raises(OSError,match='storage failed'):
        repeat._execute_locked(spec,gate,out,store,persistent,'run',False,1,runner)
    assert not trained
    assert json.loads((store/'failure.json').read_text())['status']=='failed'


def test_fresh_local_directory_cannot_replace_persistent_history(tmp_path):
    from scripts import run_seed_repeat_c as repeat
    store=tmp_path/'store';store.mkdir();(store/'latest.json').write_text('{}')
    with pytest.raises(ValueError,match='persistent history'):
        repeat.validate_output_state(tmp_path/'new-run',store,'run',False)


def test_durable_failure_attempt_survives_local_write_error(tmp_path,monkeypatch):
    from scripts import run_seed_repeat_c as repeat
    writes=[]
    def writer(path,state):
        writes.append(path)
        if path.name=='execution_state.json':
            raise OSError('local disk full')
    monkeypatch.setattr(repeat,'_atomic_json',writer)
    repeat.record_failure(tmp_path/'out',tmp_path/'store',{},OSError('original failure'))
    assert tmp_path/'store/failure.json' in writes
