"""Frozen config policy for Stage-C initialization repeats (no training side effects)."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = 'stage-c-init-seed-repeat-v1'
ORDER_POLICY = 'fixed_audited_bundle_order'
SEEDS = (401, 402, 403)
CELLS = ('HL', 'HS', 'LS', 'LL')
BUNDLE = 'data/prepared/colab_free_mini_v1/bundle.json'
POLICY = 'configs/seed_repeat_c/experiment_manifest.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def validate_config(path, config):
    path = Path(path).resolve()
    root = ROOT / 'configs/seed_repeat_c'
    if not path.is_relative_to(root):
        return False
    manifest = json.loads((ROOT / POLICY).read_text())
    if manifest.get('experiment_id') != EXPERIMENT_ID or manifest.get('train_order_policy') != ORDER_POLICY:
        raise ValueError('invalid seed-repeat experiment manifest')
    seed, cell = config['model']['seed'], config['model']['cell']
    if type(seed) is not int or seed not in SEEDS or cell not in CELLS:
        raise ValueError('invalid seed-repeat seed or cell')
    expected = root / f'seed-{seed}/LR-{cell}.json'
    if path != expected:
        raise ValueError('seed-repeat config path does not match seed/cell')
    record = manifest['configs'].get(str(path.relative_to(ROOT)))
    if not record or record['sha256'] != digest(path) or record['config_json_hash'] != json_hash(config):
        raise ValueError('seed-repeat config hash mismatch')
    baseline = json.loads(json.dumps(config))
    baseline['model']['seed'] = 17
    if json_hash(baseline) != manifest['baseline_config_json_hashes'][f'LR-{cell}']:
        raise ValueError('seed-repeat config changes more than model.seed')
    return True


def source_hashes():
    paths = [*sorted((ROOT / 'src').glob('*.py'))]
    paths += [ROOT / 'scripts' / name for name in (
        'run_colab_mini.py', 'run_seed_repeat_c.py', 'seed_repeat_policy.py',
        'overnight_store.py', 'package_colab_mini.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}
