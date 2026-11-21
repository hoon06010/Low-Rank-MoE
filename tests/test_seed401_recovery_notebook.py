"""The deliverable must be inert by default and contain the reviewed operator."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest


def test_notebook_is_explicit_resume_and_inert_by_default(tmp_path):
    from scripts.build_seed401_recovery_notebook import build
    root = tmp_path / 'root'
    (root / 'scripts').mkdir(parents=True)
    for name in ('prepare_seed401_local.py', 'seed401_local.py', 'seed401_recovery.py'):
        (root / 'scripts' / name).write_text('# fixture ' + name + '\n')
    destination = tmp_path / 'output'
    notebook, manifest = build(root, destination)
    nb = json.loads(notebook.read_text())
    codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for source in codes:
        ast.parse(source)
    settings = {}
    exec(codes[0], settings)
    assert settings['START_TRAINING'] is False
    assert settings['QUOTA_HOURS'] == 0
    assert settings['ALLOCATED_AT_ISO'] == ''
    assert settings['RESTORE_FROM'] == ''
    assert 'restore_recovery(' in '\n'.join(codes)
    assert "time.time()-300" not in '\n'.join(codes)
    assert "PHASE = \"pilot\"" not in '\n'.join(codes)
    metadata = json.loads(manifest.read_text())
    assert metadata['notebook_sha256'] == hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert set(metadata['embedded_sources']) == {
        'scripts/prepare_seed401_local.py', 'scripts/seed401_local.py', 'scripts/seed401_recovery.py'}
    for path, expected in metadata['embedded_sources'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected


def test_only_new_deliverables_written(tmp_path):
    from scripts.build_seed401_recovery_notebook import build
    root = tmp_path / 'root'
    (root / 'scripts').mkdir(parents=True)
    for name in ('prepare_seed401_local.py', 'seed401_local.py', 'seed401_recovery.py'):
        (root / 'scripts' / name).write_text('# fixture\n')
    destination = tmp_path / 'out'
    destination.mkdir()
    historical = destination / 'seed401-local.ipynb'
    historical.write_text('historical')
    build(root, destination)
    assert historical.read_text() == 'historical'
    assert sorted(p.name for p in destination.iterdir()) == [
        'seed401-local.ipynb', 'seed401-recovery-ready.ipynb', 'seed401-recovery-ready.manifest.json']


def test_interrupted_console_is_sealed_after_log_closes(tmp_path, monkeypatch):
    from scripts.build_seed401_recovery_notebook import build
    import subprocess
    root = tmp_path / 'root'
    (root / 'scripts').mkdir(parents=True)
    for name in ('prepare_seed401_local.py', 'seed401_local.py', 'seed401_recovery.py'):
        (root / 'scripts' / name).write_text('# fixture\n')
    notebook, _ = build(root, tmp_path / 'out')
    source = [''.join(c['source']) for c in json.loads(notebook.read_text())['cells'] if c['cell_type']=='code'][-1]
    class InterruptedStream:
        def __iter__(self):
            yield 'last visible output\n'
            raise RuntimeError('console interrupted')
        def close(self): pass
    events=[]
    process=SimpleNamespace(stdout=InterruptedStream(), terminate=lambda:events.append('terminated'),wait=lambda:0)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:process)
    local=SimpleNamespace(STATE=tmp_path,SESSION_SECONDS=5400,seal=lambda p:events.append('sealed'))
    recovery=SimpleNamespace(session_budget=lambda *a:5400)
    with pytest.raises(RuntimeError, match='console interrupted'):
        exec(source,dict(START_TRAINING=True,ALLOCATED_AT_ISO='2026-10-07T23:00:00+09:00',QUOTA_HOURS=20,
                         local=local,recovery=recovery,Path=Path,sys=sys,json=json))
    assert events==['terminated','sealed']
    assert next(tmp_path.glob('console-*.log')).read_text()=='last visible output\n'
