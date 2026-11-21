import json
from pathlib import Path
import pytest
from scripts import seed401_recovery as recovery
from scripts import seed401_local as local


def test_exact_failure_rows_and_reservations():
    ledger=json.loads((Path(__file__).parents[1]/'reports/seed_repeat_c/seed-401/local/recovery-main-step128-ledger.json').read_text())
    recovery.validate_ledger(ledger)
    ledger['sessions'][1]['reserved_seconds']=0
    with pytest.raises(ValueError): recovery.validate_ledger(ledger)


def test_restore_rejects_wrong_snapshot_before_copy(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    (source/'snapshot.json').write_text('{}')
    with pytest.raises(ValueError,match='snapshot pin'):
        recovery.restore_recovery(source,tmp_path/'dest')
    assert not (tmp_path/'dest').exists()


def test_run_identity_rejects_mixed_run(tmp_path):
    payload={'run_id':'wanted','optimizer_step':128,'trained_target_tokens':524288,'checkpoint_id':'cp'}
    (tmp_path/'run_manifest.jsonl').write_text(json.dumps({'run_id':'other','checkpoint_step':128,'trained_target_tokens':524288,'status':'incomplete'}))
    with pytest.raises(ValueError,match='run identity'):
        recovery.verify_run_identity(tmp_path,payload)


def test_deadline_requires_save_reserve():
    with pytest.raises(ValueError): recovery.session_budget(1000,1100,1050,2)
    assert recovery.session_budget(1000,7000,1100,2)==5400


def test_boundary_evaluation_plan():
    assert recovery.missing_evaluation_steps(128,[])==[]
    assert recovery.missing_evaluation_steps(256,[])==[256]
    assert recovery.missing_evaluation_steps(2048,list(range(256,2048,256)))==[2048]
    with pytest.raises(ValueError,match='historical'):
        recovery.missing_evaluation_steps(512,[])

@pytest.mark.parametrize('step',[256,2048])
def test_interrupted_boundary_evaluation_repaired_without_training(tmp_path,monkeypatch,step):
    import torch
    from scripts import run_colab_mini as mini
    from scripts import seed_repeat_policy
    from src import train,model
    out=tmp_path/'out'; (out/'checkpoints').mkdir(parents=True)
    payload={'run_id':'run','checkpoint_id':'cp','optimizer_step':step,'trained_target_tokens':step*4096,'cursor':step*8,'epoch':0}
    cp=out/'checkpoints/cp.pt'; torch.save(payload,cp)
    (out/'checkpoints/latest.json').write_text(json.dumps({'filename':'cp.pt','sha256':local.sha(cp),'checkpoint_id':'cp'}))
    (out/'run_manifest.jsonl').write_text(json.dumps({'run_id':'run','checkpoint_id':'cp','checkpoint_step':step,'trained_target_tokens':step*4096,'status':'completed' if step==2048 else 'incomplete'}))
    config={'protocol_id':'test','model':{'config_id':'LR-HL','cell':'HL','seed':401},'training':{}}
    batches=[{'input_ids':[[1]],'attention_mask':[[1]]}]
    monkeypatch.setattr(mini,'_config',lambda p:config)
    monkeypatch.setattr(mini,'_validate_grid',lambda *a:None)
    monkeypatch.setattr(mini,'load_audited_bundle',lambda p:{'bundle_hash':'b','spec':{'manifest':{'tokenizer':{'revision':{'value':'t'}}}},'batches':{'train':batches,'dev':batches}})
    monkeypatch.setattr(seed_repeat_policy,'source_hashes',lambda:{})
    class FakeModel:
        def __call__(self,*a): return None,None,{}
        def parameters(self): return []
    monkeypatch.setattr(model,'MoETransformer',lambda config:FakeModel())
    events=[]
    class FakeSession:
        def __init__(self,m,*a,**k):
            self.model=m; self.run_id='run'; self.optimizer_step=step; self.trained_target_tokens=step*4096; self.training_config={}
        def restore_latest(self): events.append('restore')
        def result(self,status): return {**payload,'status':status}
        def fit(self,*a,**k): pytest.fail('evaluation recovery must never train')
        def _write_run_manifest(self,*a): events.append('manifest')
    monkeypatch.setattr(train,'TrainingSession',FakeSession)
    monkeypatch.setattr(mini,'evaluate',lambda *a,**k:events.append('evaluate') or {'loss_nats':2.,'eval_target_tokens':512})
    monkeypatch.setattr(mini,'append_routing_diagnostic',lambda *a:events.append('diagnostic'))
    for n in range(256,step,256):
        mini.write_metrics_csv(out/'metrics.csv',{'run_id':'run','protocol_id':'test','config_id':'LR-HL','family':'LR','cell':'HL','seed':401,'split':'dev','checkpoint_id':str(n),'trained_target_tokens':n*4096,'eval_target_tokens':512,'loss_nats':2.,'status':'incomplete','source_kind':'measured'})
    result=recovery.repair_boundary_evaluation(out,tmp_path/'config',tmp_path/'bundle',device='cpu')
    assert result['optimizer_step']==step
    assert events.count('evaluate')==1
    recovery.repair_boundary_evaluation(out,tmp_path/'config',tmp_path/'bundle',device='cpu')
    assert events.count('evaluate')==1
    if step==2048:
        report=json.loads((out/'mini_report.json').read_text())
        assert report['status']=='completed' and report['recovered_evaluation_only']


def test_restore_disk_shortage_does_not_copy(tmp_path,monkeypatch):
    from types import SimpleNamespace
    source=tmp_path/'source'; source.mkdir()
    ledger=json.loads((Path(__file__).parents[1]/'reports/seed_repeat_c/seed-401/local/recovery-main-step128-ledger.json').read_text())
    local.write(source/'ledger.json',ledger); local.seal(source)
    monkeypatch.setattr(recovery.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='disk'):
        recovery.restore_recovery(source,tmp_path/'destination',{'snapshot_sha256':local.sha(source/'snapshot.json')})
    assert not (tmp_path/'destination').exists()


def test_new_authorization_preserves_exhausted_historical_budget():
    ledger=json.loads((Path(__file__).parents[1]/'reports/seed_repeat_c/seed-401/local/recovery-main-step128-ledger.json').read_text())
    ledger['sessions'].append({'id':'past','runtime':'past','phase':'main','reserved_seconds':86400,'status':'closed'})
    before=sum(r['reserved_seconds'] for r in ledger['sessions'])
    assert local.remaining(ledger,'main')<0
    recovery.reserve_recovery(ledger,5400,'new')
    assert sum(r['reserved_seconds'] for r in ledger['sessions'])==before+5400
    assert ledger['limits']['main']==86400
    assert ledger['continuation_authorization']['total_user_time_cap_seconds'] is None


@pytest.mark.parametrize('step,stale_report',[(256,False),(2048,False),(2048,True)])
def test_real_training_session_boundary_restore(tmp_path,monkeypatch,step,stale_report):
    """Small synthetic model, real optimizer/model/checkpoint/evaluator; no GPU claims."""
    import torch
    from scripts import run_colab_mini as mini,seed_repeat_policy
    from src import train
    from src.model import MoETransformer
    from test_resume import CONFIG,TRAINING,BATCHES,_assert_nested_equal
    config={'protocol_id':'fixture','model':{**CONFIG,'seed':401,'cell':'HL','config_id':'LR-HL'},
            'training':{**TRAINING,'total_target_tokens':8388608}}
    identity={'protocol_id':'fixture','source_revision':train._json_hash({}),'data_manifest_hash':'a'*64,
      'tokenizer_revision':'fixture','eval_revision':train._json_hash(BATCHES),'source_kind':'synthetic_fixture'}
    batches=[BATCHES[0]]*16384
    out=tmp_path/'out'
    session=train.TrainingSession(MoETransformer(config['model']),config['model'],config['training'],batches,identity,out)
    session.fit(max_optimizer_steps=1)
    # Fixture boundary metadata: avoids 2048 unnecessary steps in a restoration test.
    session.optimizer_step=step;session.trained_target_tokens=step*4096;session.cursor=step*8
    session.scheduler.tokens_seen=step*4096
    session._save_checkpoint();session._write_run_manifest('completed' if step==2048 else 'incomplete','fixture interruption before evaluation')
    if stale_report:
        local.write(out/'mini_report.json',{'run_id':session.run_id,'optimizer_step':128,'trained_target_tokens':524288,'status':'incomplete','checkpoint_id':'old'})
    before_model={k:v.clone() for k,v in session.model.state_dict().items()}
    before_optimizer=session.optimizer.state_dict()
    for n in range(256,step,256):
        mini.write_metrics_csv(out/'metrics.csv',{'run_id':session.run_id,'protocol_id':'fixture','config_id':'LR-HL','family':'LR','cell':'HL','seed':401,'split':'dev','checkpoint_id':str(n),'trained_target_tokens':n*4096,'eval_target_tokens':8,'loss_nats':2.,'status':'incomplete','source_kind':'measured'})
    monkeypatch.setattr(mini,'_config',lambda p:config)
    monkeypatch.setattr(mini,'_validate_grid',lambda *a:None)
    monkeypatch.setattr(mini,'load_audited_bundle',lambda p:{'bundle_hash':'a'*64,'spec':{'manifest':{'tokenizer':{'revision':{'value':'fixture'}}}},'batches':{'train':batches,'dev':BATCHES}})
    monkeypatch.setattr(seed_repeat_policy,'source_hashes',lambda:{})
    original=train.TrainingSession
    restored=[]
    def synthetic_session(model,model_config,training,batches,identity,out,**kwargs):
        identity={**identity,'source_kind':'synthetic_fixture'}
        new=original(model,model_config,training,batches,identity,out,device='cpu')
        restored.append(new)
        return new
    monkeypatch.setattr(train,'TrainingSession',synthetic_session)
    monkeypatch.setattr(original,'fit',lambda *a,**k:pytest.fail('repair must not train'))
    result=recovery.repair_boundary_evaluation(out,tmp_path/'config',tmp_path/'bundle',device='cpu')
    assert result['optimizer_step']==step
    _assert_nested_equal(before_model,restored[-1].model.state_dict())
    _assert_nested_equal(before_optimizer,restored[-1].optimizer.state_dict())
    assert restored[-1].scheduler.tokens_seen==step*4096
    assert restored[-1].cursor==step*8
    if step==2048:
        report=json.loads((out/'mini_report.json').read_text())
        assert report['pilot'] is False and report['source_kind']=='measured' and report['precision']=='float32'
        assert report['loss_nats']>0 and report['status']=='completed'


def _sealed_recovery_fixture(path):
    """Real tensor checkpoint with scaled synthetic data; production pin machinery."""
    from src.train import TrainingSession
    from src.model import MoETransformer
    from test_resume import CONFIG,TRAINING,BATCHES,_identity
    path.mkdir(parents=True)
    ledger=json.loads((Path(__file__).parents[1]/'reports/seed_repeat_c/seed-401/local/recovery-main-step128-ledger.json').read_text())
    local.write(path/'ledger.json',ledger)
    config={**CONFIG,'seed':401,'cell':'HL','config_id':'LR-HL'}
    session=TrainingSession(MoETransformer(config),config,{**TRAINING,'total_target_tokens':8388608},[BATCHES[0]]*16384,_identity(),path/'main/LR-HL')
    session.fit(max_optimizer_steps=1)
    session.optimizer_step=128;session.trained_target_tokens=524288;session.cursor=1024;session.scheduler.tokens_seen=524288
    session._save_checkpoint();session._write_run_manifest('incomplete','synthetic fixture')
    local.write(path/'main/LR-HL/mini_report.json',session.result('incomplete'))
    local.seal(path)
    return session


def test_actual_checkpoint_restore_pin_then_next_runtime_roundtrip(tmp_path):
    from src.train import TrainingSession
    from src.model import MoETransformer
    from test_resume import BATCHES,_identity,_assert_nested_equal
    source=tmp_path/'saved-output'; initial=_sealed_recovery_fixture(source)
    first_pin=recovery.make_pin(source)
    first=tmp_path/'runtime-one'
    recovery.restore_recovery(source,first,pin=first_pin)
    resumed=TrainingSession(MoETransformer(initial.model_config),initial.model_config,initial.training_config,
        [BATCHES[0]]*16384,_identity(),first/'main/LR-HL')
    resumed.restore_latest()
    _assert_nested_equal(initial.model.state_dict(),resumed.model.state_dict())
    _assert_nested_equal(initial.optimizer.state_dict(),resumed.optimizer.state_dict())
    assert resumed.run_id==initial.run_id and resumed.optimizer_step==128
    ledger=json.loads((first/'ledger.json').read_text())
    reservation=recovery.reserve_recovery(ledger,5400,'actual-fixture-runtime-1')
    reservation['status']='closed';local.write(first/'ledger.json',ledger)
    # Emulate a later valid boundary to check that pins advance instead of reusing128.
    resumed.optimizer_step=160;resumed.trained_target_tokens=655360;resumed.cursor=1280;resumed.scheduler.tokens_seen=655360
    resumed._save_checkpoint();resumed._write_run_manifest('incomplete','next fixture boundary')
    local.write(first/'main/LR-HL/mini_report.json',resumed.result('incomplete'))
    (first/'closed-console.log').write_text('closed before snapshot\n')
    local.seal(first);next_pin=recovery.make_pin(first)
    assert next_pin['optimizer_step']==160 and next_pin['main_reserved_seconds']==18120
    assert next_pin['snapshot_sha256']!=first_pin['snapshot_sha256']
    second=tmp_path/'runtime-two'
    recovery.restore_recovery(first,second,pin=next_pin)
    assert recovery.make_pin(second)==next_pin
    assert json.loads((second/'ledger.json').read_text())['sessions'][1:3]==ledger['sessions'][1:3]
    assert (second/'closed-console.log').read_text()=='closed before snapshot\n'
    with pytest.raises(ValueError,match='snapshot pin'):
        recovery.restore_recovery(first,tmp_path/'stale-pin-target',pin=first_pin)


@pytest.mark.parametrize('failure',[False,True])
def test_operator_reserves_once_stops_and_seals(tmp_path,monkeypatch,failure):
    import torch
    from types import SimpleNamespace
    from scripts import run_colab_mini as mini
    source=tmp_path/'state';_sealed_recovery_fixture(source)
    config_report={'pilot_usable':True,'pilot_measurement_complete':True,'pilot_measured_steps':100,
      'optimizer_step':105,'trained_target_tokens':430080,'status':'incomplete',
      'elapsed_training_seconds_including_checkpoint':100,'evaluation_seconds_this_invocation':2}
    for cell in local.ORDER:
        local.write(source/'pilot'/f'LR-{cell}'/'mini_report.json',{**config_report,'model':{'seed':401,'cell':cell},'run_id':cell})
    now=[1100.]
    monkeypatch.setattr(recovery.time,'time',lambda:now[0])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'Tesla T4')
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(local,'validate_payload',lambda:None)
    original_read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'new-fixture-runtime' if str(p)=='/proc/sys/kernel/random/boot_id' else original_read(p,*a,**k))
    monkeypatch.setattr(recovery,'repair_boundary_evaluation',lambda *a:{'status':'incomplete'})
    calls=[]
    def bounded(config,bundle,out,**kwargs):
        calls.append((str(config),kwargs['resume']))
        assert kwargs['boundary_hook'](SimpleNamespace(epoch=0,cursor=1024,optimizer_step=128,trained_target_tokens=524288),'before_chunk')
        if failure: raise RuntimeError('original training error')
        now[0]=5081. # allocation1000 +5400 -1020 -300, safely stop before another chunk
        return {'status':'incomplete'}
    monkeypatch.setattr(mini,'run_colab_mini',bounded)
    if failure:
        with pytest.raises(RuntimeError,match='original training error'):
            recovery.run_recovery(10,1000,6400,source)
    else: recovery.run_recovery(10,1000,6400,source)
    ledger=json.loads((source/'ledger.json').read_text())
    assert len(calls)==1 and calls[0][1] is True
    assert ledger['sessions'][-1]['reserved_seconds']==5400
    assert ledger['sessions'][-1]['status']==('failed' if failure else 'closed')
    assert sum(r['reserved_seconds'] for r in ledger['sessions'] if r['phase']=='main')==18120
    assert (source/'snapshot.json').exists()
    if failure:
        assert 'original training error' in next(source.glob('failure-*.log')).read_text()
        with pytest.raises(ValueError,match='unresolved'): recovery.make_pin(source)
    else: assert recovery.make_pin(source)['main_reserved_seconds']==18120
