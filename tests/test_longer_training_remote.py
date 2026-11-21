"""CPU-only guard tests; these do not establish GPU readiness or recovery success."""
import hashlib
import json
from pathlib import Path
import tarfile
import io
import pytest
from scripts import longer_training_remote as remote
from scripts import longer_training_bootstrap as bootstrap


def test_session_budget_reserves_recovery_time():
    remote.validate_budget(1200, 1800, 600)
    for arguments in [(1200,1799,600),(float('nan'),7200,600),(-1,7200,600),(1200,7200,-1)]:
        with pytest.raises(ValueError):
            remote.validate_budget(*arguments)


def test_verified_pointer_rejects_tampering_and_escape(tmp_path):
    checkpoints=tmp_path/'checkpoints'; checkpoints.mkdir()
    target=checkpoints/'checkpoint-a.pt'; target.write_bytes(b'test')
    pointer={'filename':target.name,'sha256':hashlib.sha256(b'test').hexdigest(),'checkpoint_id':'a'}
    (checkpoints/'latest.json').write_text(json.dumps(pointer))
    assert remote.pointer_state(tmp_path)[1] == target
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        remote.pointer_state(tmp_path)
    pointer['filename']='../escape.pt'
    (checkpoints/'latest.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError,match='unsafe'):
        remote.pointer_state(tmp_path)


def test_recovery_rejects_archive_traversal(tmp_path,monkeypatch):
    monkeypatch.setattr(remote,'ROOT',tmp_path)
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as tar:
        member=tarfile.TarInfo('../escape'); member.size=1
        tar.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='unsafe'):
        remote.extract_recovery(archive,remote.digest(archive),tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_nested_restored_state_detects_rng_and_optimizer_difference():
    import numpy as np
    import torch
    a={'optimizer':{1:torch.tensor([1.0])},'rng':('x',np.array([1,2]))}
    b={'optimizer':{1:torch.tensor([1.0])},'rng':('x',np.array([1,2]))}
    assert remote.equal_state(a,b)
    b['rng'][1][0]=3
    assert not remote.equal_state(a,b)


def test_bootstrap_rejects_incomplete_grid():
    paths=[*bootstrap.SCRIPTS,bootstrap.BUNDLE]
    members=[tarfile.TarInfo(path) for path in paths]
    with pytest.raises(ValueError,match='incomplete'):
        bootstrap.validate_members({'files':[{'path':p} for p in paths]},members)


def test_timeout_child_kills_work_and_ignores_notebook_arguments(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(remote,'ROOT',tmp_path)
    monkeypatch.setenv('LONGER_ARGS','["--bad"]')
    result=remote.timed_child([sys.executable,'-c','import os,time; assert "LONGER_ARGS" not in os.environ; time.sleep(10)'],0.1,tmp_path/'log')
    assert result['status']=='hard_timeout'
    assert result['returncode']==-9
    assert result['wall_seconds']<3
