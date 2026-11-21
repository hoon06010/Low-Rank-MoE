"""Execute on Colab after uploading payload + manifest to /content."""
from pathlib import Path
import hashlib
import json
import os
import re
import tarfile
import urllib.request

ROOT = Path('/content/low_rank_moe_mini')
MANIFEST = Path('/content/payload_manifest.json')
ARCHIVE = Path('/content/payload.tar.gz')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


spec = json.loads(MANIFEST.read_text())
if digest(ARCHIVE) != spec['payload_sha256']:
    raise ValueError('payload checksum mismatch')
if ROOT.exists() and any(ROOT.iterdir()):
    raise ValueError('bootstrap requires an empty project directory; preserve existing experiment')
if not hasattr(tarfile, 'data_filter'):
    raise RuntimeError('Python tar extraction filters are required')
ROOT.mkdir(exist_ok=True)
with tarfile.open(ARCHIVE, 'r:gz') as tar:
    expected = {f['path'] for f in spec['files']}
    members = tar.getmembers()
    if {m.name for m in members} != expected or len(members) != len(expected):
        raise ValueError('unexpected archive members')
    for member in members:
        target = (ROOT / member.name).resolve()
        if not member.isfile() or not target.is_relative_to(ROOT):
            raise ValueError('unsafe archive member')
    tar.extractall(ROOT, filter='data')
for record in spec['files']:
    if (ROOT / record['path']).stat().st_size != record['bytes'] or digest(ROOT / record['path']) != record['sha256']:
        raise ValueError('extracted file checksum mismatch')
for record in spec['remote_downloads']:
    if not re.fullmatch(r'https://huggingface\.co/datasets/HuggingFaceFW/fineweb-edu/resolve/[0-9a-f]{40}/sample/10BT/000_00000\.parquet', record['url']):
        raise ValueError('unexpected source URL')
    target = (ROOT / record['path']).resolve()
    if not target.is_relative_to(ROOT / 'data/raw'):
        raise ValueError('unsafe download destination')
    if target.exists() and digest(target) == record['sha256']:
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = Path('/content/000_00000.parquet')
    if staged.is_file() and digest(staged) == record['sha256']:
        os.link(staged, target)
        continue
    temporary = target.with_suffix('.download')
    print('Downloading pinned public FineWeb-Edu shard', flush=True)
    with urllib.request.urlopen(record['url'], timeout=60) as source, temporary.open('wb') as dest:
        while chunk := source.read(4 * 1024 * 1024):
            dest.write(chunk)
    if digest(temporary) != record['sha256']:
        raise ValueError('download checksum mismatch')
    temporary.replace(target)
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))
import torch
from src.data import load_audited_bundle
bundle = load_audited_bundle(ROOT / 'data/prepared/colab_free_mini_v1/bundle.json')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; GPU experiment not started')
print(json.dumps({'status': 'ready', 'purpose': bundle['spec']['purpose'],
    'bundle_hash': bundle['bundle_hash'], 'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'target_counts': {s: r['valid_target_tokens'] for s,r in bundle['spec']['splits'].items()}}, indent=2))
