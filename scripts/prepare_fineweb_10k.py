"""Prepare a bounded, pinned FineWeb-Edu setup sample from downloaded bytes.
Run from the project root: .venv/bin/python scripts/prepare_fineweb_10k.py
This never downloads data or authorizes GPU use.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
from collections import Counter
from datetime import datetime, timezone

import jsonschema
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from src.data import prepare_documents, document_order, sample_near_duplicates, pack_token_prefix, load_audited_bundle
from src.train import _sha256_file, _json_hash, _atomic_json

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/fineweb_10k'
OUT=ROOT/'data/prepared/fineweb_10k'
REV='84e8104e779e409e2267ac60609138e3dda2cbd2'
TOKENIZER_REV='607a30d783dfa663caf39e06633721c8d4cfcd7e'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'bundle.json').exists():
        raise ValueError('bundle already exists; preserve it and choose a new version before preparing again')
    meta=json.loads((RAW/'source_metadata.json').read_text())
    if meta['revision']!=REV: raise ValueError('source revision mismatch')
    source=RAW/'000_00000.parquet'
    digest=_sha256_file(source)
    if digest!=meta['file']['lfs']['oid']: raise ValueError('source digest mismatch')
    docs=[]
    for batch in pq.ParquetFile(source).iter_batches(batch_size=1000,columns=['id','url','text']):
        docs.extend(batch.to_pylist())
        if len(docs)>=10000: break
    docs=docs[:10000]
    if len(docs)!=10000: raise ValueError('expected 10,000 source documents')
    prepared,stats=prepare_documents(docs,seed=17,max_documents=10000)
    print('grouped',stats,flush=True)
    overlaps={}
    for key in ('text_sha256','group_hash','canonical_url','id'):
        memberships={}
        for d in prepared:
            if d.get(key): memberships.setdefault(str(d[key]),set()).add(d['split'])
        overlaps[key]=sum(len(v)>1 for v in memberships.values())
    if any(overlaps.values()): raise ValueError('cross-split identity overlap')
    near=sample_near_duplicates(prepared)
    print('near duplicate sampled pairs',near['pairs_checked'],'matches',len(near['matches']),flush=True)
    _atomic_json(OUT/'near_duplicate_audit.json',near)
    if near['matches']: raise ValueError('sampled near duplicates require review before readiness')
    tok=Tokenizer.from_file(str(RAW/'tokenizer/tokenizer.json'))
    eos=tok.token_to_id('<|endoftext|>')
    if eos!=50256 or tok.get_vocab_size()!=50257: raise ValueError('unexpected GPT-2 vocabulary')
    probe='A short English sentence.'
    encoded=tok.encode(probe,add_special_tokens=False).ids
    if tok.decode(encoded)!=probe or tok.encode('',add_special_tokens=False).ids:
        raise ValueError('tokenizer round trip or implicit token behavior mismatch')
    if tok.encode('<|endoftext|>',add_special_tokens=False).ids!=[eos]:
        raise ValueError('EOS encoding mismatch')
    split_records={};token_provenance={}
    for split,limit in [('train',65536),('dev',4096),('test',4096)]:
        consumed=[]
        def stream():
            for doc in document_order(prepared,seed=17,split=split):
                ids=tok.encode(doc['text'],add_special_tokens=False).ids
                consumed.append({'text_sha256':doc['text_sha256'],'body_tokens':len(ids),'eos_tokens':1})
                yield from ids
                yield eos
        rows=pack_token_prefix(stream(),split=split,context_length=512,target_limit=limit,pad_token_id=eos)
        path=OUT/f'{split}.json';_atomic_json(path,rows)
        count=sum(r['valid_target_count'] for r in rows)
        if count!=limit: raise ValueError(f'insufficient {split} tokens')
        split_records[split]={'path':path.name,'sha256':_sha256_file(path),'content_hash':_json_hash(rows),'valid_target_tokens':count}
        token_provenance[split]={'ordered_documents_touched':consumed,'target_limit':limit,
                                  'selection':'first target_limit+1 tokens of seeded document order, EOS after each body; final touched document may be truncated'}
    with (OUT/'documents.jsonl').open('w') as f:
        for doc in prepared: f.write(json.dumps(doc,ensure_ascii=False)+'\n')
    now=datetime.now(timezone.utc).isoformat()
    def revision(value): return {'value':value,'provenance':'locally_verified','locally_verified_at':now}
    def relative(path): return os.path.relpath(path,OUT)
    tokenizer_artifacts=[{'path':relative(p),'source_declared_sha256':None,'locally_verified_sha256':_sha256_file(p)} for p in sorted((RAW/'tokenizer').iterdir()) if p.is_file()]
    manifest={'schema_version':1,'manifest_id':'fineweb-10k-first-shard-setup-v1','status':'ready',
      'dataset':{'repository':'HuggingFaceFW/fineweb-edu','config':'sample-10BT','revision':revision(REV),
        'license':{'source_label':'ODC-BY-1.0','license_url':'https://opendatacommons.org/licenses/by/1-0/',
          'terms_url':'https://commoncrawl.org/terms-of-use','local_review_status':'reviewed',
          'review_notes':'Dataset card and linked terms reviewed 2026-09-26. Retain attribution and source URLs; database license does not replace individual content rights. Local bounded research setup only; no text redistribution. See plans/m9/data_terms.md.'},
        'source_splits':['train'],'shards':[{'path':relative(source),'source_declared_sha256':meta['file']['lfs']['oid'],
          'source_checksum_location':f'https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/{REV}/sample/10BT',
          'locally_verified_sha256':digest,'locally_verified_at':now,'verification_status':'match'}],
        'declared_shard_count':1,'declared_total_bytes':source.stat().st_size},
      'tokenizer':{'repository':'openai-community/gpt2','revision':revision(TOKENIZER_REV),'artifact_checksums':tokenizer_artifacts,
        'special_token_behavior':{'eos_token':'<|endoftext|>','bos_token':None,'pad_token':'<|endoftext|>','locally_tested':True,
          'notes':'encode(add_special_tokens=False); explicit EOS appended; no BOS inserted; padding uses EOS ID and ignored labels; round-trip and empty-input behavior checked.'}},
      'split_policy':{'method':'sha256_group_bucket','seed':17,'fractions':{'train':.98,'dev':.01,'test':.01},
        'identity_links':['document_id','canonical_url','normalized_text_sha256'],'split_isolation_checked':True,'near_duplicate_audit':'sampled'},
      'packing_policy':{'context_length':512,'eos_between_documents':True,'cross_split_packing':False,'labels_pre_shifted':True,'padding_label_id':-100,'document_context_after_eos':True},
      'actual_audit':{'status':'completed','performed_at':now,'documents_seen':stats['input_documents'],
        'missing_text_excluded':stats['missing_text'],'null_text_excluded':stats['null_text'],
        'non_string_text_excluded':stats['non_string_text'],'empty_text_excluded':stats['blank_text'],
        'duplicate_id_records':stats['duplicate_id_records'],'exact_duplicates_removed':stats['exact_duplicate_documents_removed'],
        'exact_cross_split_overlaps':overlaps['text_sha256'],'near_duplicate_review':'sampled',
        'notes':'Only first 10,000 rows of first pinned shard. Sampled 5-word shingle audit is not exhaustive and is limited to 4096 words/document. No population representativeness or full-corpus leakage claim.'}}
    jsonschema.validate(manifest,json.loads((ROOT/'data/manifests/schema.json').read_text()))
    audit={'statistics':stats,'split_documents':dict(Counter(d['split'] for d in prepared)),
           'cross_split_overlaps':overlaps,'source_rows':pq.ParquetFile(source).metadata.num_rows,
           'sampling':{'method':'first 10000 rows in pinned first shard','seed_for_group_split_and_order':17,'representative_of_full_corpus':False},
           'token_provenance':token_provenance,'tokenizer_probe_ids':encoded,
           'near_duplicate_review':'near_duplicate_audit.json'}
    _atomic_json(OUT/'audit.json',audit)
    artifacts=[{'path':relative(p),'sha256':_sha256_file(p)} for p in [OUT/'documents.jsonl',OUT/'audit.json',OUT/'near_duplicate_audit.json',RAW/'source_metadata.json',RAW/'dataset_card.md']]
    bundle={'schema_version':1,'purpose':'pipeline_smoke_only','manifest':manifest,'vocab_size':50257,'context_length':512,
            'splits':split_records,'audit_artifacts':artifacts}
    _atomic_json(OUT/'bundle.json',bundle)
    verified=load_audited_bundle(OUT/'bundle.json')
    print(json.dumps({'bundle_hash':verified['bundle_hash'],'split_documents':audit['split_documents'],
                      'retained_targets':{s:r['valid_target_tokens'] for s,r in split_records.items()}},indent=2),flush=True)


if __name__=='__main__':main()
