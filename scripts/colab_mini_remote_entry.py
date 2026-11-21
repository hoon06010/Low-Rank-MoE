"""Run one bounded mini-experiment invocation in the active Colab kernel."""
import os
from pathlib import Path
import subprocess
import sys

root = Path('/content/low_rank_moe_mini')
cell = os.environ.get('MINI_CELL', 'LR-HL')
phase = os.environ.get('MINI_PHASE', 'pilot')
if cell not in {'LR-HS', 'LR-HL', 'LR-LS', 'LR-LL'}:
    raise ValueError('unexpected mini cell')
if phase not in {'pilot', 'train'}:
    raise ValueError('unexpected mini phase')
command = [
    sys.executable, '-u', 'scripts/run_colab_mini.py',
    '--config', f'configs/colab_free_mini/{cell}.json',
    '--bundle', 'data/prepared/colab_free_mini_v1/bundle.json',
    '--output', f'results/colab_free_mini/{phase}/{cell}',
    '--device', 'cuda', '--max-seconds', '900' if phase == 'pilot' else '1800',
]
if phase == 'pilot':
    command.append('--pilot')
else:
    command.extend(['--max-steps', '2048'])
    if os.environ.get('MINI_RESUME') == '1':
        command.append('--resume')
print('Running', ' '.join(command), flush=True)
subprocess.run(command, cwd=root, check=True)
