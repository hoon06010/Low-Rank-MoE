import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module():
    assert (ROOT/'scripts/seed401_spectron.py').exists(), 'dedicated Spectron entrypoint missing'
    from scripts import seed401_spectron
    return seed401_spectron

def test_fixed_configs():
    s=module()
    for c in ('HS','LL'):
        cfg=s.load_config(ROOT/f'configs/seed401_spectron/LR-{c}.json')
        assert cfg['model']['seed']==401
        assert cfg['training']['optimizer']=='spectron'
        assert cfg['training']['total_target_tokens']==8388608

def test_reject_adamw_and_changed_recipe(tmp_path):
    s=module()
    c=json.loads((ROOT/'configs/seed_repeat_c/seed-401/LR-HS.json').read_text())
    p=tmp_path/'LR-HS.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):s.load_config(p)
    c=json.loads((ROOT/'configs/seed401_spectron/LR-HS.json').read_text())
    c['training']['learning_rate']=0.001;p.write_text(json.dumps(c))
    with pytest.raises(ValueError):s.load_config(p)

def test_no_implicit_resource_budget():
    s=module()
    with pytest.raises(ValueError):s.validate_resources({},now=100)
    r=dict(allocated_at=90,stop_at=5500,quota_hours=2,budget_seconds=5400,save_reserve_seconds=1020)
    with pytest.raises(ValueError):s.validate_resources(r,now=100)
    r['stop_at']=5490
    assert s.validate_resources(r,now=100)==5400
    r['quota_hours']=float('nan')
    with pytest.raises(ValueError):s.validate_resources(r,now=100)

def test_missing_gpu_evidence_blocks_main(tmp_path):
    s=module()
    with pytest.raises(ValueError,match='pilot|evidence'):
        s.main_gate(tmp_path,tmp_path/'adamw')

def test_fingerprint_tensor_state():
    s=module()
    import torch
    a={'state':{0:{'momentum':torch.tensor([1.,2.])}},'param_groups':[{'lr':0.1}]}
    b={'state':{0:{'momentum':torch.tensor([1.,3.])}},'param_groups':[{'lr':0.1}]}
    assert s.state_hash(a)==s.state_hash(a)
    assert s.state_hash(a)!=s.state_hash(b)

def test_reservation_no_refund_and_no_auto_retry(tmp_path):
    s=module();r=dict(allocated_at=90,stop_at=5490,quota_hours=2,budget_seconds=5400,save_reserve_seconds=1020)
    row=s.reserve_resources(tmp_path,r,'pilot','runtime-a',now=100)
    assert row['reserved_seconds']==5400
    with pytest.raises(ValueError,match='unresolved'):s.reserve_resources(tmp_path,r,'pilot','runtime-b',now=100)

def test_portable_builder_exists():
    assert (ROOT/'scripts/seed401_spectron_package.py').exists(), 'portable package builder missing'

def test_spectron_snapshot_restore_integrity(tmp_path):
    s=module();src=tmp_path/'source';src.mkdir()
    (src/'spectron_ledger.json').write_text(json.dumps({'experiment_id':s.EXPERIMENT,'sessions':[{'status':'closed'}]}))
    (src/'evidence').write_text('verified');h=s.seal(src)
    s.restore_state(src,tmp_path/'restored',h)
    assert (tmp_path/'restored/evidence').read_text()=='verified'
    (src/'evidence').write_text('changed')
    with pytest.raises(ValueError,match='snapshot'):s.restore_state(src,tmp_path/'bad',h)

def test_final_requires_eight_dev_evaluations(tmp_path):
    s=module()
    r={'status':'completed','optimizer_step':2048,'trained_target_tokens':8388608}
    with pytest.raises(ValueError):s.verify_spectron_final(tmp_path,r)

def test_boundary_evaluation_is_due_after_resume(tmp_path):
    s=module()
    assert s.evaluation_due(tmp_path,1048576,'run','checkpoint')
    assert not s.evaluation_due(tmp_path,4096,'run','checkpoint')
    (tmp_path/'metrics.csv').write_text('run_id,split,checkpoint_id,trained_target_tokens\nrun,dev,checkpoint,1048576\n')
    assert not s.evaluation_due(tmp_path,1048576,'run','checkpoint')

def test_resume_run_id_mismatch_rejected():
    s=module()
    with pytest.raises(ValueError,match='run identity'):s.check_resume_run([{'run_id':'a'}],'b')
    s.check_resume_run([{'run_id':'a'}],'a')


def test_spectron_pilots_require_adamw_first(tmp_path):
    s=module()
    with pytest.raises(ValueError,match='AdamW'):s.adamw_gate(tmp_path)

@pytest.mark.parametrize('damage',['checkpoint','model','source'])
def test_adamw_anchor_gate_rejects_self_inconsistent_evidence(tmp_path,monkeypatch,damage):
    import shutil
    from test_analyze_seed_repeat_c import fixture
    from scripts import analyze_seed_repeat_c as collector
    s=module()
    monkeypatch.setattr(collector,'expected_identity',lambda _:{'source_file_hashes':{},'data_manifest_hash':'d'*64,'eval_revision':'e'*64,'tokenizer_revision':'tokenizer'})
    (tmp_path/'main').mkdir()
    for cell in ('HL','HS','LS','LL'):
        spec=fixture(tmp_path,cell=cell)
        shutil.copytree(spec['path'],tmp_path/'main'/f'LR-{cell}')
    s.adamw_gate(tmp_path)
    out=tmp_path/'main/LR-HL'
    if damage=='checkpoint':
        p=out/'checkpoints/latest.json';r=json.loads(p.read_text());r['checkpoint_id']='different-valid-checkpoint';p.write_text(json.dumps(r))
    else:
        p=out/'mini_report.json';r=json.loads(p.read_text())
        if damage=='model':r['model']['seed']=17
        else:r['source_file_hashes']={'fake':'a'*64}
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError):s.adamw_gate(tmp_path)

def test_execute_pins_gpu_zero_and_records_gate_failure(tmp_path,monkeypatch):
    import time
    s=module()
    assert s.GPU_DEVICE=='cuda:0'
    monkeypatch.setattr(s.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(s.torch.cuda,'get_device_name',lambda _: 'Tesla T4')
    monkeypatch.setattr(s,'runtime_id',lambda:'fixture-runtime')
    now=time.time();r=dict(allocated_at=now-1,stop_at=now+5399,quota_hours=2,budget_seconds=5400,save_reserve_seconds=1020)
    with pytest.raises(ValueError,match='AdamW'):s.execute('pilot',tmp_path/'state',tmp_path/'bundle',r,tmp_path/'missing')
    ledger=json.loads((tmp_path/'state/spectron_ledger.json').read_text())
    assert ledger['sessions'][0]['status']=='failed'
    assert ledger['sessions'][0]['reserved_seconds']==5400

def test_notebook_has_preparation_failure_journal():
    p=ROOT/'reports/seed401_spectron/package/seed401-spectron.ipynb'
    n=json.loads(p.read_text());source='\n'.join(''.join(c['source']) for c in n['cells'])
    assert 'setup_failed' in source and 'seed401_spectron_startup_journal.json' in source

def test_generated_setup_failure_preserves_external_reservation(tmp_path):
    n=json.loads((ROOT/'reports/seed401_spectron/package/seed401-spectron.ipynb').read_text())
    source=''.join(n['cells'][2]['source'])
    working=tmp_path/'working';working.mkdir();inputs=tmp_path/'input';inputs.mkdir()
    boot=tmp_path/'boot';boot.write_text('fixture-boot')
    source=source.replace('/kaggle/working',str(working)).replace('/kaggle/input',str(inputs)).replace('/proc/sys/kernel/random/boot_id',str(boot))
    env={'PHASE':'pilot','ALLOCATED_AT':100,'STARTUP_HISTORY_FROM':'','EXPECTED_MANIFEST':'missing','RESTORE_FROM':''}
    with pytest.raises(ValueError,match='Attach exactly one'):
        exec(compile(source,'generated-setup','exec'),env)
    journal=json.loads((working/'seed401_spectron_startup_journal.json').read_text())
    assert journal['events'][0]['status']=='setup_failed'
    assert journal['events'][0]['reserved_seconds']==5400

def test_generated_setup_keeps_local_events_over_imported_history(tmp_path):
    n=json.loads((ROOT/'reports/seed401_spectron/package/seed401-spectron.ipynb').read_text())
    source=''.join(n['cells'][2]['source'])
    working=tmp_path/'working';working.mkdir();inputs=tmp_path/'input';inputs.mkdir()
    boot=tmp_path/'boot';boot.write_text('fixture-boot')
    history=inputs/'history.json';history.write_text(json.dumps({'experiment_id':'seed401-spectron-fixed-recipe-v1','events':[{'status':'old_failure'}]}))
    source=source.replace('/kaggle/working',str(working)).replace('/kaggle/input',str(inputs)).replace('/proc/sys/kernel/random/boot_id',str(boot))
    env={'PHASE':'pilot','ALLOCATED_AT':100,'STARTUP_HISTORY_FROM':str(history),'EXPECTED_MANIFEST':'missing','RESTORE_FROM':''}
    for _ in range(2):
        with pytest.raises(ValueError,match='Attach exactly one'):
            exec(compile(source,'generated-setup','exec'),env)
    journal=json.loads((working/'seed401_spectron_startup_journal.json').read_text())
    assert [e['status'] for e in journal['events']]==['old_failure','setup_failed','setup_failed']
