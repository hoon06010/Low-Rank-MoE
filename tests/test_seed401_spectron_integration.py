"""Actual CPU model/Spectron/checkpoint/evaluator; scaled fixture data only."""
import copy,json
from pathlib import Path
import pytest
from scripts import seed401_spectron as s
from src import data,train
from src.model import MoETransformer
from test_resume import CONFIG,TRAINING,BATCHES,_assert_nested_equal


def fixture(monkeypatch):
    config={'protocol_id':'fixture','model':{**CONFIG,'seed':401,'cell':'HS','config_id':'LR-HS'},
      'training':{**TRAINING,'total_target_tokens':4096,'optimizer':'spectron','recipe_id':s.EXPERIMENT,
      'spectron_momentum':.95,'spectron_ns_steps':5,'spectron_power_steps':1,'spectron_weight_decay':0.}}
    batches=[BATCHES[0]]*2048
    audited={'bundle_hash':'a'*64,'spec':{'purpose':'mini_exploration_only','context_length':4,'vocab_size':32,'manifest':{'tokenizer':{'revision':{'value':'fixture'}}}},'batches':{'train':batches,'dev':BATCHES}}
    monkeypatch.setattr(s,'load_config',lambda p:copy.deepcopy(config))
    monkeypatch.setattr(s,'load_audited_bundle',lambda p:audited)
    monkeypatch.setattr(data,'load_audited_bundle',lambda p:audited)
    monkeypatch.setattr(s,'GLOBAL_BATCH',2);monkeypatch.setattr(s,'ROW_TARGETS',2);monkeypatch.setattr(s,'EVAL_INTERVAL',512)
    monkeypatch.setattr(s,'runtime_id',lambda:'fixture-runtime')
    identity={'protocol_id':'fixture','source_revision':train._json_hash(s.source_identity()),'data_manifest_hash':'a'*64,
      'tokenizer_revision':'fixture','eval_revision':train._json_hash(BATCHES),'source_kind':'measured'}
    return config,batches,identity


@pytest.mark.parametrize('step',[256,2048])
def test_real_spectron_boundary_recovery_no_train(tmp_path,monkeypatch,step):
    config,batches,identity=fixture(monkeypatch)
    out=tmp_path/'out'
    session=train.TrainingSession(MoETransformer(config['model']),config['model'],config['training'],batches,identity,out,audit_bundle=tmp_path/'bundle')
    session.fit(max_optimizer_steps=1)
    session.optimizer_step=step;session.trained_target_tokens=step*2;session.cursor=step;session.scheduler.tokens_seen=step*2
    session._save_checkpoint();session._write_run_manifest('completed' if step==2048 else 'incomplete','fixture eval interruption')
    before=s.state_hash(session.optimizer.state_dict())
    model_hash=s.state_hash(session.model.state_dict())
    checkpoint=Path(s.pointer(out)[1]).read_bytes()
    monkeypatch.setattr(train.TrainingSession,'fit',lambda *a,**k:pytest.fail('recovery must not train'))
    # before_chunk false allows evaluation-only recovery at an unfinished boundary.
    result=s.run_spectron(tmp_path/'config',tmp_path/'bundle',out,device='cpu',resume=True,boundary_hook=lambda *_:False)
    assert result['optimizer_step']==step and result['optimizer_steps_this_invocation']==0
    assert result['optimizer_state_hash']==before and result['loss_nats']>0
    assert result['routing_diagnostic'] is not None
    assert (out/'routing_diagnostics.jsonl').exists()
    assert Path(s.pointer(out)[1]).read_bytes()==checkpoint
    restored=train.TrainingSession(MoETransformer(config['model']),config['model'],config['training'],batches,identity,out,audit_bundle=tmp_path/'bundle')
    restored.restore_latest()
    assert s.state_hash(restored.model.state_dict())==model_hash
    assert restored.scheduler.tokens_seen==step*2 and restored.cursor==step


def test_actual_spectron_pilot_optimizer_roundtrip(tmp_path,monkeypatch):
    config,_,_=fixture(monkeypatch)
    out=tmp_path/'pilot'
    result=s.run_spectron(tmp_path/'config',tmp_path/'bundle',out,device='cpu',pilot=True)
    assert result['optimizer_step']==105 and result['pilot_measured_steps']==100 and result['pilot_usable']
    checkpoint=Path(s.pointer(out)[1]).read_bytes()
    proof=s.run_spectron(tmp_path/'config',tmp_path/'bundle',out,device='cpu',resume=True,recovery_only=True)
    assert proof['optimizer_state_hash']==result['optimizer_state_hash']
    assert proof['run_id']==result['run_id'] and proof['optimizer_step']==105
    assert Path(s.pointer(out)[1]).read_bytes()==checkpoint
    # Identity strictness remains active in the real TrainingSession.
    config['training']['learning_rate']*=2
    with pytest.raises(ValueError,match='training_config_hash'):
        s.run_spectron(tmp_path/'config',tmp_path/'bundle',out,device='cpu',resume=True,recovery_only=True)
    config['training']['learning_rate']/=2
    hashes=s.source_identity();hashes['fixture-altered']='b'*64
    monkeypatch.setattr(s,'source_identity',lambda:hashes)
    with pytest.raises(ValueError,match='source_revision'):
        s.run_spectron(tmp_path/'config',tmp_path/'bundle',out,device='cpu',resume=True,recovery_only=True)
