"""Prepare a bounded mini-exploration bundle from the pinned local sample.

Run from the project root with .venv/bin/python scripts/prepare_colab_mini.py.
The sample is the already-audited first 10,000 rows of one shard and is not
representative of FineWeb-Edu. This script does not download data or use a GPU.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import os
from datetime import datetime, timezone

import jsonschema
from tokenizers import Tokenizer

from src.data import document_order, pack_token_prefix, load_audited_bundle, normalize_text
from src.train import _atomic_json, _json_hash, _sha256_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/fineweb_10k'
SOURCE_BUNDLE = ROOT / 'data/prepared/fineweb_10k/bundle.json'
OUT = ROOT / 'data/prepared/colab_free_mini_v1'
REPORT = ROOT / 'reports/colab_free_mini/data_preparation.json'
TRAIN_TARGETS = 8_388_608
DEV_TARGETS = 65_536
TEST_TARGETS = 4_096


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError(f'{OUT} already contains files; preserving existing bundle')
    if REPORT.exists():
        raise ValueError(f'{REPORT} already exists; preserving existing report')

    # This rechecks the original parquet, tokenizer, retained files, packed
    # labels, and bundle artifact hashes before any source documents are used.
    source = load_audited_bundle(SOURCE_BUNDLE)
    original_bundle = json.loads(SOURCE_BUNDLE.read_text())
    docs_path = OUT / 'documents.jsonl'
    raw_docs = (SOURCE_BUNDLE.parent / 'documents.jsonl').read_bytes()
    docs_digest = hashlib.sha256(raw_docs).hexdigest()
    declared_doc = next(a['sha256'] for a in original_bundle['audit_artifacts']
                        if a['path'] == 'documents.jsonl')
    if docs_digest != declared_doc:
        raise ValueError('source documents.jsonl checksum mismatch')
    docs = [json.loads(line) for line in raw_docs.splitlines()]
    if not docs or any(not isinstance(d, dict) or not isinstance(d.get('text'), str)
                       or not d['text'].strip() for d in docs):
        raise ValueError('source documents contain invalid or empty text')
    for d in docs:
        normalized = normalize_text(d['text'])
        if hashlib.sha256(normalized.encode()).hexdigest() != d.get('text_sha256'):
            raise ValueError('source document content hash mismatch')
        if d.get('split') not in ('train', 'dev', 'test'):
            raise ValueError('source document has invalid split')

    identity_overlaps = {}
    for key in ('text_sha256', 'group_hash', 'canonical_url', 'id'):
        memberships = {}
        for doc in docs:
            if doc.get(key):
                memberships.setdefault(str(doc[key]), set()).add(doc['split'])
        identity_overlaps[key] = sum(len(splits) > 1 for splits in memberships.values())
    if any(identity_overlaps.values()):
        raise ValueError(f'cross-split identity overlap: {identity_overlaps}')

    tok_path = RAW / 'tokenizer/tokenizer.json'
    tok = Tokenizer.from_file(str(tok_path))
    eos = tok.token_to_id('<|endoftext|>')
    if eos != 50256 or tok.get_vocab_size() != 50257:
        raise ValueError('unexpected tokenizer vocabulary or EOS')
    split_records = {}
    token_provenance = {}
    for split, limit in (('train', TRAIN_TARGETS), ('dev', DEV_TARGETS)):
        available_docs = [d for d in docs if d['split'] == split]
        consumed = []

        def stream():
            for doc in document_order(available_docs, seed=17, split=split):
                ids = tok.encode(doc['text'], add_special_tokens=False).ids
                consumed.append({'text_sha256': doc['text_sha256'],
                                 'body_tokens': len(ids), 'eos_tokens': 1})
                yield from ids
                yield eos

        rows = pack_token_prefix(stream(), split=split, context_length=512,
                                 target_limit=limit, pad_token_id=eos)
        retained = sum(row['valid_target_count'] for row in rows)
        if retained != limit:
            raise ValueError(f'insufficient {split} coverage: requested {limit}, available {retained}')
        OUT.mkdir(parents=True, exist_ok=True)
        packed_path = OUT / f'{split}.json'
        _atomic_json(packed_path, rows)
        split_records[split] = {'path': packed_path.name,
                                'sha256': _sha256_file(packed_path),
                                'content_hash': _json_hash(rows),
                                'valid_target_tokens': retained}
        token_provenance[split] = {
            'ordered_documents_touched': consumed,
            'target_limit': limit,
            'selection': 'seeded order within the existing audited split; EOS appended per document; no document repetition',
        }

    # Keep the existing held-out test payload byte-for-byte. It is retained
    # only as provenance and must not be evaluated during the mini run.
    test_record = original_bundle['splits']['test']
    test_path = SOURCE_BUNDLE.parent / test_record['path']
    if _sha256_file(test_path) != test_record['sha256']:
        raise ValueError('existing test split checksum mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    test_out = OUT / 'test.json'
    test_out.write_bytes(test_path.read_bytes())
    actual_test_targets = sum(r['valid_target_count'] for r in source['batches']['test'])
    if actual_test_targets != TEST_TARGETS:
        raise ValueError(f'unexpected retained test target count: {actual_test_targets}')
    split_records['test'] = {'path': test_out.name, 'sha256': _sha256_file(test_out),
                             'content_hash': test_record['content_hash'],
                             'valid_target_tokens': TEST_TARGETS}
    token_provenance['test'] = {'source': 'existing audited test split retained byte-for-byte',
                                'evaluated': False, 'valid_target_tokens': TEST_TARGETS}

    now = datetime.now(timezone.utc).isoformat()
    manifest = json.loads(json.dumps(original_bundle['manifest']))
    manifest['manifest_id'] = 'fineweb-10k-colab-free-mini-exploration-v1'
    manifest['actual_audit']['notes'] += (
        ' Mini exploration reuses only the already-audited first 10,000 rows of '
        'the first pinned shard; not representative of the corpus or suitable '
        'for study claims. Test split retained but not evaluated.'
    )
    manifest['actual_audit']['performed_at'] = now
    audit = {
        'purpose': 'mini_exploration_only',
        'source_bundle_sha256': _sha256_file(SOURCE_BUNDLE),
        'source_documents_sha256': docs_digest,
        'source_documents_count': len(docs),
        'sampling': {'method': 'existing first 10000 rows of pinned first shard',
                     'representative_of_full_corpus': False},
        'split_isolation': {'inherited_from_source_audit': True,
                            'identity_overlaps': identity_overlaps,
                            'source_split_document_counts': {
                                s: sum(d['split'] == s for d in docs)
                                for s in ('train', 'dev', 'test')},
                            'new_train_dev_overlap': max(identity_overlaps.values(), default=0)},
        'token_provenance': token_provenance,
        'target_counts': {'train': TRAIN_TARGETS, 'dev': DEV_TARGETS,
                          'test_retained_not_evaluated': TEST_TARGETS},
    }
    docs_path.write_bytes(raw_docs)
    _atomic_json(OUT / 'audit.json', audit)
    near_path = OUT / 'near_duplicate_audit.json'
    near_source = SOURCE_BUNDLE.parent / 'near_duplicate_audit.json'
    near_path.write_bytes(near_source.read_bytes())
    source_artifacts = [
        {'path': 'documents.jsonl', 'sha256': _sha256_file(docs_path)},
        {'path': 'audit.json', 'sha256': _sha256_file(OUT / 'audit.json')},
        {'path': 'near_duplicate_audit.json', 'sha256': _sha256_file(near_path)},
        {'path': os.path.relpath(RAW / 'source_metadata.json', OUT),
         'sha256': _sha256_file(RAW / 'source_metadata.json')},
        {'path': os.path.relpath(RAW / 'dataset_card.md', OUT),
         'sha256': _sha256_file(RAW / 'dataset_card.md')},
    ]
    spec = {'schema_version': 1, 'purpose': 'mini_exploration_only',
            'manifest': manifest, 'vocab_size': 50257, 'context_length': 512,
            'splits': split_records, 'audit_artifacts': source_artifacts}
    jsonschema.validate(manifest, json.loads((ROOT / 'data/manifests/schema.json').read_text()))
    _atomic_json(OUT / 'bundle.json', spec)
    # Parent is extending the loader to accept this explicit non-study purpose.
    verified = load_audited_bundle(OUT / 'bundle.json')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(REPORT, {
        'purpose': 'mini_exploration_only', 'bundle_hash': verified['bundle_hash'],
        'bundle_path': os.path.relpath(OUT / 'bundle.json', ROOT),
        'source_bundle_sha256': audit['source_bundle_sha256'],
        'source_documents_sha256': docs_digest,
        'source_documents_count': len(docs),
        'sampling_representative_of_full_corpus': False,
        'target_counts': audit['target_counts'],
        'test_evaluated': False,
    })
    print(json.dumps({'bundle_hash': verified['bundle_hash'],
                      'source_documents_count': len(docs),
                      'target_counts': audit['target_counts']}, indent=2))


if __name__ == '__main__':
    main()
