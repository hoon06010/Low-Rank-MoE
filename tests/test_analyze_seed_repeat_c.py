import csv
import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    assert (ROOT / 'scripts/analyze_seed_repeat_c.py').exists(), 'dedicated collector not implemented'
    return importlib.import_module('scripts.analyze_seed_repeat_c')


@pytest.fixture(autouse=True)
def frozen_identity(monkeypatch):
    module=api()
    monkeypatch.setattr(module, 'expected_identity', lambda experiment: {
        'source_file_hashes':{}, 'data_manifest_hash':'d'*64, 'eval_revision':'e'*64,
        'tokenizer_revision':'tokenizer'})


def fixture(tmp_path, seed=401, cell='HS', loss=3.0):
    config = json.loads((ROOT / f'configs/seed_repeat_c/seed-{seed}/LR-{cell}.json').read_text())
    out = tmp_path / f'{seed}-{cell}'
    out.mkdir()
    rid = f'run-{seed}-{cell}'
    manifest = dict(run_id=rid, family='LR', cell=cell, seed=seed, optimizer='adamw',
        recipe_id=config['training']['recipe_id'], status='completed', source_kind='measured',
        purpose='mini_exploration_only', checkpoint_step=2048, trained_target_tokens=8388608,
        checkpoint_id='cp8', protocol_id=config['protocol_id'], config_id=f'LR-{cell}',
        config_hash=hashlib.sha256(json.dumps(config['model'],sort_keys=True,separators=(',',':')).encode()).hexdigest(), training_config_hash=hashlib.sha256(json.dumps(config['training'],sort_keys=True,separators=(',',':')).encode()).hexdigest(), source_commit=hashlib.sha256(b'{}').hexdigest(),
        data_manifest_hash='d'*64, eval_revision='e'*64, tokenizer_revision='tokenizer')
    (out/'run_manifest.jsonl').write_text(json.dumps(manifest)+'\n')
    report = dict(manifest, model=config['model'], training=config['training'], pilot=False,
        optimizer_step=2048, actual_tokens_total=8388608, precision='float32', loss_nats=loss,
        parameter_count=1, source_file_hashes={})
    (out/'mini_report.json').write_text(json.dumps(report))
    (out/'checkpoints').mkdir()
    (out/'checkpoints/final.pt').write_bytes(b'fixture checkpoint bytes')
    (out/'checkpoints/latest.json').write_text(json.dumps(dict(filename='final.pt',checkpoint_id='cp8',sha256=hashlib.sha256(b'fixture checkpoint bytes').hexdigest())))
    metrics=[]
    for i in range(1,9):
        metrics.append(dict(run_id=rid,protocol_id=config['protocol_id'],config_id=f'LR-{cell}',family='LR',cell=cell,seed=seed,split='dev',checkpoint_id=f'cp{i}',trained_target_tokens=i*1048576,eval_target_tokens=65536,loss_nats=loss,status='completed' if i==8 else 'incomplete',source_kind='measured'))
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=metrics[0]); writer.writeheader();writer.writerows(metrics)
    return dict(experiment_id='stage-c-init-seed-repeat-v1',phase='main',path=str(out))


def test_partial_and_complete_paired_statistics(tmp_path):
    module=api()
    specs=[fixture(tmp_path,seed,cell,3+{'HS':0,'HL':-.1,'LS':.2,'LL':-.05}[cell]+(seed-401)*.01) for seed in (401,402,403) for cell in ('HS','HL','LS','LL')]
    partial=module.analyze(specs[:8])
    assert partial['adamw']['contrasts']['delta_budget']['paired_t_95_ci'] is None
    full=module.analyze(specs)
    assert full['adamw']['contrasts']['delta_budget']['mean']==pytest.approx(-.05)
    assert full['adamw']['contrasts']['G_high']['mean']==pytest.approx(.1)
    assert full['adamw']['contrasts']['I_LR']['mean']==pytest.approx(.15)
    assert full['adamw']['contrasts']['delta_budget']['n']==3
    assert 'epsilon' not in json.dumps(full)


@pytest.mark.parametrize('damage',['duplicate','missing_eval','nonfinite','checkpoint','pilot','identity'])
def test_rejects_invalid_completed_evidence(tmp_path,damage):
    module=api(); spec=fixture(tmp_path);out=Path(spec['path'])
    if damage=='duplicate':
        with pytest.raises(ValueError,match='duplicate'):module.analyze([spec,spec])
        return
    if damage=='missing_eval':
        lines=(out/'metrics.csv').read_text().splitlines();(out/'metrics.csv').write_text('\n'.join(lines[:-1])+'\n')
    elif damage=='nonfinite':
        (out/'metrics.csv').write_text((out/'metrics.csv').read_text().replace('3.0','nan'))
    elif damage=='checkpoint':(out/'checkpoints/final.pt').write_bytes(b'corrupt')
    else:
        report=json.loads((out/'mini_report.json').read_text())
        if damage=='pilot':report['pilot']=True
        else:report['run_id']='different'
        (out/'mini_report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):module.analyze([spec])


def test_excludes_pilot_and_seed17_scope(tmp_path):
    module=api();spec=fixture(tmp_path)
    spec['phase']='pilot'
    result=module.analyze([spec])
    assert result['adamw']['runs']==[]
    assert len(result['excluded'])==1


def test_model_config_hash_is_checked(tmp_path):
    module=api();spec=fixture(tmp_path);out=Path(spec['path'])
    m=json.loads((out/'run_manifest.jsonl').read_text());m['config_hash']='bad'
    (out/'run_manifest.jsonl').write_text(json.dumps(m)+'\n')
    with pytest.raises(ValueError,match='config hash'):module.analyze([spec])


def test_seed17_is_explicitly_excluded(tmp_path):
    module=api();spec=fixture(tmp_path);out=Path(spec['path'])
    m=json.loads((out/'run_manifest.jsonl').read_text());m['seed']=17
    (out/'run_manifest.jsonl').write_text(json.dumps(m)+'\n')
    assert module.analyze([spec])['excluded'][0]['reason']=='seed outside 401/402/403'


def test_cli_emits_partial_json(tmp_path,capsys):
    module=api();path=tmp_path/'inputs.json';path.write_text(json.dumps([fixture(tmp_path)]))
    assert module.main(['--inputs',str(path)])==0
    assert json.loads(capsys.readouterr().out)['adamw']['contrasts']['I_LR']['n']==0


def test_spectron_separate_from_adamw(tmp_path,monkeypatch):
    module=api()
    specs=[fixture(tmp_path,401,cell,loss) for cell,loss in [('HS',3.),('LL',2.9)]]
    config_root=tmp_path/'config-root'
    for cell in ('HS','LL'):
        original=json.loads((ROOT/f'configs/seed_repeat_c/seed-401/LR-{cell}.json').read_text())
        target=config_root/f'configs/seed_repeat_c/seed-401/LR-{cell}.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(original))
        sp_parent=tmp_path/f'spectron-{cell}';sp_parent.mkdir()
        spec=fixture(sp_parent,401,cell,2.8 if cell=='HS' else 2.6)
        spec['experiment_id']=module.SPECTRON
        out=Path(spec['path'])
        original['training'].update(optimizer='spectron',recipe_id=module.SPECTRON)
        target=config_root/f'configs/seed401_spectron/LR-{cell}.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(original))
        m=json.loads((out/'run_manifest.jsonl').read_text())
        m.update(optimizer='spectron',recipe_id=module.SPECTRON,training_config_hash=module._hash_json(original['training']))
        (out/'run_manifest.jsonl').write_text(json.dumps(m)+'\n')
        r=json.loads((out/'mini_report.json').read_text());r['training']=original['training'];r.update(experiment_id=module.SPECTRON,phase='main')
        (out/'mini_report.json').write_text(json.dumps(r));specs.append(spec)
    monkeypatch.setattr(module,'ROOT',config_root)
    result=module.analyze(specs)
    assert len(result['adamw']['runs'])==2
    assert result['adamw']['contrasts']['delta_budget']['n']==1
    assert result['spectron']['delta_budget']==pytest.approx(-.2)
    assert result['spectron']['cell_differences'][0]['spectron_minus_adamw']==pytest.approx(-.2)


@pytest.mark.parametrize('field',['source','data_manifest_hash','eval_revision'])
def test_rejects_self_consistent_wrong_frozen_identity(tmp_path,field):
    module=api();spec=fixture(tmp_path);out=Path(spec['path'])
    m=json.loads((out/'run_manifest.jsonl').read_text());r=json.loads((out/'mini_report.json').read_text())
    if field=='source':
        r['source_file_hashes']={'src/train.py':'0'*64};m['source_commit']=module._hash_json(r['source_file_hashes'])
    else:m[field]=r[field]='0'*64
    (out/'run_manifest.jsonl').write_text(json.dumps(m)+'\n');(out/'mini_report.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='expected frozen'):module.analyze([spec])


def test_retained_package_identities_use_actual_manifest_schemas(monkeypatch):
    monkeypatch.undo()
    module=api()
    for experiment in (module.ADAMW,module.SPECTRON):
        identity=module.expected_identity(experiment)
        assert 'src/train.py' in identity['source_file_hashes']
        assert identity['data_manifest_hash']=='b4fdd74b4c9359524b88d949f62c8ad0af3cb50b8663d4bc8711d49fa93ba19d'
