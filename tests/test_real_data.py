"""Real-artifact boundary tests use explicit fixtures, not research evidence."""
import copy
import importlib
import json
from pathlib import Path

import pytest
import torch

from src.data import build_split_batches
from src.model import MoETransformer
from src.train import TrainingSession, _json_hash
from test_data import _ready_manifest
from test_training_smoke import CONFIG, TRAINING, BATCHES, _identity


def api():
    from src import data
    assert hasattr(data, 'load_audited_bundle'), 'audited artifact loader is not implemented'
    return data.load_audited_bundle


def bundle(tmp_path):
    raw = tmp_path / 'raw.txt'; raw.write_text('fixture raw bytes')
    tok = tmp_path / 'tokenizer.json'; tok.write_text('{}')
    import hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = _ready_manifest()
    manifest['dataset']['shards'][0].update(path='raw.txt',locally_verified_sha256=sha(raw))
    manifest['tokenizer']['artifact_checksums'][0].update(path='tokenizer.json',locally_verified_sha256=sha(tok))
    docs = [{'id':'a','text':'one','split':'train','group_hash':'a','text_sha256':'a'},
            {'id':'b','text':'two','split':'dev','group_hash':'b','text_sha256':'b'}]
    packed = {}
    for split in ('train','dev'):
        packed[split] = build_split_batches(docs, split=split, encode=lambda _: [1,2,3,4,5], eos_token_id=2, context_length=4)
    for split, rows in packed.items():
        path=tmp_path/f'{split}.json'; path.write_text(json.dumps(rows))
    spec={'schema_version':1,'purpose':'pipeline_smoke_only','manifest':manifest,
          'vocab_size':32,'context_length':4,'splits':{s:{'path':f'{s}.json','sha256':sha(tmp_path/f'{s}.json'),'content_hash':_json_hash(v),'valid_target_tokens':sum(x['valid_target_count'] for x in v)} for s,v in packed.items()}}
    path=tmp_path/'bundle.json';path.write_text(json.dumps(spec))
    return path, spec, packed


def test_loader_detects_changed_retained_labels(tmp_path):
    path,spec,packed=bundle(tmp_path)
    loaded=api()(path)
    assert loaded['batches']['dev']==packed['dev']
    p=tmp_path/'dev.json';rows=json.loads(p.read_text());rows[0]['labels'][0][0]=9;p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='checksum'):
        api()(path)


def test_loader_rejects_changed_raw_source(tmp_path):
    path,_,_=bundle(tmp_path)
    (tmp_path/'raw.txt').write_text('changed')
    with pytest.raises(ValueError,match='checksum'):
        api()(path)


def test_audited_measured_session_requires_matching_train_and_eval_identity(tmp_path):
    path,_,packed=bundle(tmp_path)
    loaded=api()(path)
    identity=_identity();identity.update(source_kind='measured',data_manifest_hash=loaded['bundle_hash'],eval_revision=_json_hash(packed['dev']),tokenizer_revision=loaded['spec']['manifest']['tokenizer']['revision']['value'])
    session=TrainingSession(MoETransformer(CONFIG),CONFIG,TRAINING,packed['train'],identity,tmp_path/'run',audit_bundle=path)
    assert session.identity['source_kind']=='measured'
    bad=copy.deepcopy(packed['train']);bad[0]['labels'][0][0]=7
    with pytest.raises(ValueError,match='training.*provenance'):
        TrainingSession(MoETransformer(CONFIG),CONFIG,TRAINING,bad,identity,tmp_path/'bad',audit_bundle=path)
    identity['eval_revision']='not-the-retained-evaluation'
    with pytest.raises(ValueError,match='evaluation.*provenance'):
        TrainingSession(MoETransformer(CONFIG),CONFIG,TRAINING,packed['train'],identity,tmp_path/'bad2',audit_bundle=path)


def test_unavailable_cuda_has_clear_gate(tmp_path):
    if torch.cuda.is_available(): pytest.skip('CUDA is available')
    with pytest.raises(ValueError,match='CUDA.*unavailable'):
        TrainingSession(MoETransformer(CONFIG),CONFIG,TRAINING,BATCHES,_identity(),tmp_path,device='cuda')


def test_near_duplicate_review_detects_similar_cross_split_text():
    import src.data as data
    assert hasattr(data,'sample_near_duplicates'), 'near-duplicate audit not implemented'
    text=' '.join(f'word{i}' for i in range(100))
    rows=[{'text':text,'split':'train','text_sha256':'a'}, {'text':text+' extra','split':'dev','text_sha256':'b'}]
    result=data.sample_near_duplicates(rows)
    assert result['pairs_checked']==1
    assert result['matches'][0]['jaccard']>.8


def test_bounded_packing_retains_exact_next_token_labels():
    import src.data as data
    assert hasattr(data,'pack_token_prefix'), 'bounded token packing not implemented'
    rows=data.pack_token_prefix(iter([3,4,5,6,7,8,9]),split='train',context_length=4,target_limit=4,pad_token_id=0)
    assert rows==[{'input_ids':[[3,4,5,6]],'labels':[[4,5,6,7]],'attention_mask':[[1,1,1,1]],'valid_target_count':4,'split':'train'}]
    rows=data.pack_token_prefix(iter([3,4]),split='dev',context_length=4,target_limit=4,pad_token_id=0)
    assert rows[0]['labels']==[[4,-100,-100,-100]]


def test_real_smoke_runner_resumes_and_evaluates_retained_dev(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('src.real_preflight') is not None, 'real preflight runner not implemented'
    from src.real_preflight import run_real_preflight
    path,_,packed=bundle(tmp_path)
    cfg={'protocol_id':'fixture-only','model':CONFIG,'training':dict(TRAINING,total_target_tokens=4,gradient_accumulation_steps=1),'warmup_steps':0,'measurement_steps':1}
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(cfg))
    report=run_real_preflight(config_path,path,tmp_path/'run')
    assert report['status']=='completed'
    assert report['eval_target_tokens']==5
    assert report['gpu_memory_bytes'] is None
    assert report['source_kind']=='measured'
    assert report['purpose']=='pipeline_smoke_only'


def test_manifest_schema_accepts_sampled_review():
    import jsonschema
    schema=json.loads(Path('data/manifests/schema.json').read_text())
    manifest=_ready_manifest()
    jsonschema.validate(manifest,schema)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires physical CUDA device')
def test_cuda_fit_evaluate_and_rng_restore(tmp_path):
    from src.evaluate import evaluate
    session=TrainingSession(MoETransformer(CONFIG),CONFIG,TRAINING,BATCHES,_identity(),tmp_path,device='cuda')
    assert session.fit()['status']=='completed'
    expected=torch.cuda.get_rng_state().clone()
    torch.rand(16,device='cuda')
    session.restore_latest()
    assert torch.equal(expected,torch.cuda.get_rng_state())
    assert evaluate(session.model,BATCHES,device='cuda')['eval_target_tokens']==4


def test_measured_tokenizer_identity_must_match_bundle(tmp_path):
    path,_,packed=bundle(tmp_path)
    loaded=api()(path)
    identity=_identity();identity.update(source_kind='measured',data_manifest_hash=loaded['bundle_hash'],eval_revision=_json_hash(packed['dev']),tokenizer_revision='wrong')
    with pytest.raises(ValueError,match='tokenizer.*provenance'):
        TrainingSession(MoETransformer(CONFIG),CONFIG,TRAINING,packed['train'],identity,tmp_path/'run',audit_bundle=path)
