"""Verify a one-step restore from the penultimate LR-HL checkpoint in Colab."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import torch

root = Path('/content/low_rank_moe_mini')
source = root / 'results/colab_free_mini/train/LR-HL'
destination = root / 'results/colab_free_mini/recovery_check/LR-HL'
if destination.exists():
    raise ValueError('recovery check destination already exists')
checkpoints = sorted((source / 'checkpoints').glob('checkpoint-*.pt'), key=lambda p: p.stat().st_mtime_ns)
if len(checkpoints) < 2:
    raise ValueError('at least two checkpoints are required')
prior = checkpoints[-2]
payload = torch.load(prior, map_location='cpu', weights_only=False)
if payload['optimizer_step'] != 736 or payload['trained_target_tokens'] != 736 * 4096:
    raise ValueError('unexpected penultimate checkpoint cursor')
destination_checkpoint = destination / 'checkpoints'
destination_checkpoint.mkdir(parents=True)
os.link(prior, destination_checkpoint / prior.name)
with prior.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
(destination_checkpoint / 'latest.json').write_text(json.dumps({
    'filename': prior.name, 'checkpoint_id': payload['checkpoint_id'], 'sha256': digest,
}))
manifest = json.loads((source / 'run_manifest.jsonl').read_text().splitlines()[0])
manifest.update(status='incomplete', reason='separate recovery check',
                checkpoint_id=payload['checkpoint_id'], checkpoint_step=payload['optimizer_step'],
                trained_target_tokens=payload['trained_target_tokens'])
(destination / 'run_manifest.jsonl').write_text(json.dumps(manifest) + '\n')
command = [sys.executable, '-u', 'scripts/run_colab_mini.py',
    '--config', 'configs/colab_free_mini/LR-HL.json',
    '--bundle', 'data/prepared/colab_free_mini_v1/bundle.json',
    '--output', str(destination), '--device', 'cuda', '--resume',
    '--max-steps', '1', '--max-seconds', '120']
subprocess.run(command, cwd=root, check=True)
report = json.loads((destination / 'mini_report.json').read_text())
if report['optimizer_step'] != 737 or report['actual_tokens_total'] != 737 * 4096:
    raise ValueError('restored run did not advance one optimizer step')
result = {'status': 'passed', 'source_checkpoint': prior.name,
          'source_sha256': digest, 'restored_step': payload['optimizer_step'],
          'advanced_step': report['optimizer_step'], 'restored_tokens': payload['trained_target_tokens'],
          'advanced_tokens': report['actual_tokens_total'], 'purpose': 'recovery_check_only'}
(destination / 'recovery_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
