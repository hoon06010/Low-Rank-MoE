"""Verify and prepare the separately frozen longer-training payload on free Colab."""
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import tarfile
import time
import urllib.request

ROOT = Path('/content/low_rank_moe_longer')
MANIFEST = Path('/content/longer_payload_manifest.json')
ARCHIVE = Path('/content/longer_payload.tar.gz')
BUNDLE = 'data/prepared/colab_free_mini_v1/bundle.json'
SCRIPTS = {'scripts/run_colab_mini.py', 'scripts/longer_training_remote.py', 'scripts/longer_training_bootstrap.py',
           'scripts/overnight_colab.py', 'scripts/overnight_store.py', 'overnight_free_colab.ipynb'}
CONFIG_DIRS = ('colab_free_mini', 'colab_stability_adamw', 'colab_stability_spectron', 'longer_training_b', 'longer_training_c')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_members(spec, members):
    paths = [record['path'] for record in spec['files']]
    if len(set(paths)) != len(paths) or len(members) != len(paths) or {m.name for m in members} != set(paths):
        raise ValueError('duplicate or unexpected archive members')
    for member in members:
        if not member.isfile() or not (ROOT/member.name).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('unsafe archive member')
    if not SCRIPTS.issubset(paths) or BUNDLE not in paths:
        raise ValueError('missing runtime scripts or audit bundle')
    for folder in ('longer_training_b', 'longer_training_c'):
        if not {f'configs/{folder}/LR-{cell}.json' for cell in ('HS','HL','LS','LL')}.issubset(paths):
            raise ValueError('incomplete longer-training grid')


def expected_paths(bundle, packaged_paths):
    allowed = {BUNDLE, *SCRIPTS}
    for path in packaged_paths:
        if re.fullmatch(r'src/[A-Za-z_][A-Za-z_0-9]*\.py', path):
            allowed.add(path)
    for folder in CONFIG_DIRS:
        allowed.update(f'configs/{folder}/LR-{cell}.json' for cell in ('HS','HL','LS','LL'))
    for record in [*bundle['audit_artifacts'], *bundle['splits'].values(), *bundle['manifest']['tokenizer']['artifact_checksums']]:
        resolved = (ROOT / Path(BUNDLE).parent / record['path']).resolve()
        if not resolved.is_relative_to(ROOT / 'data'):
            raise ValueError('bundle dependency outside data')
        allowed.add(str(resolved.relative_to(ROOT)))
    return allowed


def main():
    started = time.monotonic()
    # Fail closed if bootstrap hangs in a shard transfer. Include this elapsed cost in the controller budget.
    timeout = float(os.environ.get('LONGER_BOOTSTRAP_TIMEOUT', '900'))
    if not 0 < timeout <= 7200:
        raise ValueError('invalid bootstrap timeout')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bootstrap hard timeout')))
    signal.setitimer(signal.ITIMER_REAL, timeout)
    spec = json.loads(MANIFEST.read_text())
    if digest(ARCHIVE) != spec['payload_sha256'] or ARCHIVE.stat().st_size != spec['payload_bytes']:
        raise ValueError('payload checksum/size mismatch')
    if ROOT.exists() and any(ROOT.iterdir()):
        raise ValueError('bootstrap requires a new empty root; preserve existing runs')
    if not hasattr(tarfile, 'data_filter'):
        raise RuntimeError('Python tar extraction filters are required')
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        members = tar.getmembers()
        validate_members(spec, members)
        bundle = json.load(tar.extractfile(BUNDLE))
        paths = {m.name for m in members}
        if paths != expected_paths(bundle, paths):
            raise ValueError('archive differs from explicit source/config/bundle dependency allowlist')
        # Reserve raw shard declared size where known plus 2 GiB; full stage disk gate belongs to controller.
        raw_bytes = int(bundle['manifest']['dataset']['declared_total_bytes'])
        if raw_bytes <= 0:
            raise ValueError('missing audited raw shard size')
        if shutil.disk_usage('/content').free < sum(m.size for m in members) + raw_bytes + 2*1024**3:
            raise ValueError('insufficient bootstrap disk space')
        ROOT.mkdir(exist_ok=True)
        tar.extractall(ROOT, filter='data')
    for record in spec['files']:
        path = ROOT/record['path']
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('extracted checksum mismatch')
    if digest(ROOT/BUNDLE) != spec['bundle_sha256']:
        raise ValueError('bundle checksum mismatch')
    records = spec['remote_downloads']
    if len(records) != 1:
        raise ValueError('exactly one pinned shard required')
    record = records[0]
    if not re.fullmatch(r'https://huggingface\.co/datasets/HuggingFaceFW/fineweb-edu/resolve/[0-9a-f]{40}/sample/10BT/000_00000\.parquet', record['url']):
        raise ValueError('unexpected pinned source URL')
    if record['path'] != 'data/raw/fineweb_10k/000_00000.parquet':
        raise ValueError('unexpected raw shard destination')
    shard = bundle['manifest']['dataset']['shards'][0]
    revision = bundle['manifest']['dataset']['revision']['value']
    if record['sha256'] != shard['locally_verified_sha256'] or f'/resolve/{revision}/' not in record['url']:
        raise ValueError('remote source differs from audited bundle')
    target = ROOT/record['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = Path('/content/000_00000.parquet')
    if staged.is_file() and digest(staged) == record['sha256']:
        os.link(staged, target)
    else:
        temporary = target.with_suffix('.download')
        with urllib.request.urlopen(record['url'], timeout=60) as source, temporary.open('wb') as destination:
            while chunk := source.read(4*1024*1024):
                destination.write(chunk)
        if digest(temporary) != record['sha256']:
            raise ValueError('raw download checksum mismatch')
        temporary.replace(target)
    os.chdir(ROOT)
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    from src.data import load_audited_bundle
    audited = load_audited_bundle(ROOT/BUNDLE)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    result = {'status': 'ready', 'root': str(ROOT), 'payload_sha256': spec['payload_sha256'],
              'bundle_sha256': digest(ROOT/BUNDLE),
              'bundle_hash': audited['bundle_hash'], 'purpose': audited['spec']['purpose'],
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'python': platform.python_version(), 'precision': 'float32',
              'disk': dict(zip(('total','used','free'), shutil.disk_usage(ROOT))),
              'bootstrap_seconds': time.monotonic()-started,
              'target_counts': {s: r['valid_target_tokens'] for s,r in audited['spec']['splits'].items()},
              'scope': 'environment/data ready only; no A/B/C training or gate passed'}
    (ROOT/'longer_environment_ready.json').write_text(json.dumps(result, indent=2)+'\n')
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
