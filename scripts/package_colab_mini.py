"""Package an explicit experiment allowlist; omit raw shard and all credentials."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'data/prepared/colab_free_mini_v1/bundle.json'
OUTPUT = ROOT / 'reports/colab_free_mini'
MODES = ('mini', 'longer-training', 'seed-repeat')
LONGER_BUNDLE_SHA256 = 'b4fdd74b4c9359524b88d949f62c8ad0af3cb50b8663d4bc8711d49fa93ba19d'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='mini',
                        help='longer-training uses a separate output and includes frozen B/C configs')
    args = parser.parse_args(argv)
    repeat = args.mode == 'seed-repeat'
    longer = args.mode == 'longer-training'
    output = ROOT / 'reports/longer_training/package' if longer else OUTPUT
    if repeat:
        output = ROOT / 'reports/seed_repeat_c/package'
    if (longer or repeat) and digest(BUNDLE) != LONGER_BUNDLE_SHA256:
        raise ValueError('longer-training bundle hash differs from the frozen plan')
    spec = json.loads(BUNDLE.read_text())
    files = {BUNDLE, ROOT / 'scripts/run_colab_mini.py'}
    if repeat:
        files.add(ROOT / 'scripts/seed_repeat_policy.py')
        from scripts import seed_repeat_policy as policy
        files.update(ROOT / 'scripts' / name for name in ('run_seed_repeat_c.py', 'overnight_store.py', 'package_colab_mini.py'))
        files.add(ROOT / policy.POLICY)
        for seed in policy.SEEDS:
            for cell in policy.CELLS:
                path = ROOT / f'configs/seed_repeat_c/seed-{seed}/LR-{cell}.json'
                policy.validate_config(path, json.loads(path.read_text()))
                files.add(path)
    if longer:
        files.update(ROOT / 'scripts' / name for name in (
            'longer_training_remote.py', 'longer_training_bootstrap.py',
            'overnight_colab.py', 'overnight_store.py'))
        files.add(ROOT / 'overnight_free_colab.ipynb')
    files.update((ROOT / 'src').glob('*.py'))
    folders = ['colab_free_mini', 'colab_stability_adamw', 'colab_stability_spectron']
    if repeat:
        folders = []
    if longer:
        folders.extend(('longer_training_b', 'longer_training_c'))
    for folder in folders:
        configs = set((ROOT / 'configs' / folder).glob('*.json'))
        if folder.startswith('longer_training_') and {path.name for path in configs} != {
                'LR-HS.json', 'LR-HL.json', 'LR-LS.json', 'LR-LL.json'}:
            raise ValueError(f'longer-training requires exactly four configs: {folder}')
        files.update(configs)
    for record in [*spec['audit_artifacts'], *spec['splits'].values(),
                   *spec['manifest']['tokenizer']['artifact_checksums']]:
        path = (BUNDLE.parent / record['path']).resolve()
        if not path.is_relative_to(ROOT / 'data'):
            raise ValueError('data dependency is outside project data directory')
        if longer or repeat:
            expected = record.get('sha256') or record.get('locally_verified_sha256')
            if not expected or digest(path) != expected:
                raise ValueError(f'data dependency hash mismatch: {path}')
        files.add(path)
    remote_downloads = []
    revision = spec['manifest']['dataset']['revision']['value']
    for shard in spec['manifest']['dataset']['shards']:
        path = (BUNDLE.parent / shard['path']).resolve()
        if path != ROOT / 'data/raw/fineweb_10k/000_00000.parquet':
            raise ValueError('unexpected shard for bounded mini package')
        remote_downloads.append({
            'path': str(path.relative_to(ROOT)),
            'url': f'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{revision}/sample/10BT/000_00000.parquet',
            'sha256': shard['locally_verified_sha256'],
        })
    records = []
    for path in sorted(files):
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(ROOT):
            raise ValueError(f'unsafe or missing payload file: {path}')
        records.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size})
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'payload.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        for path in sorted(files):
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    manifest = {'files': records, 'remote_downloads': remote_downloads,
                'payload_sha256': digest(archive), 'payload_bytes': archive.stat().st_size,
                'bundle_sha256': digest(BUNDLE)}
    (output / 'payload_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'payload_bytes': manifest['payload_bytes'], 'files': len(records),
                      'payload_sha256': manifest['payload_sha256']}, indent=2))


if __name__ == '__main__':
    main()
