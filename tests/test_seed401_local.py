"""Safety regressions for the Drive-free seed401 operator."""
import json
from pathlib import Path
import pytest
from scripts import seed401_local as local


def test_reservations_preserve_old_failed_charge():
    state = local.new_state()
    assert local.remaining(state, 'pilot') == pytest.approx(5685.753700494766)
    local.reserve(state, 'pilot', 5400, 'runtime1')
    assert local.remaining(state, 'pilot') == pytest.approx(285.753700494766)
    state['sessions'][0]['status'] = 'closed'
    with pytest.raises(ValueError, match='budget'):
        local.reserve(state, 'pilot', 5400, 'runtime2')


def test_running_or_failed_session_blocks_reentry():
    state = local.new_state()
    row = local.reserve(state, 'main', 5400, 'runtime1')
    for status in ('running', 'failed'):
        row['status'] = status
        with pytest.raises(ValueError, match='unresolved'):
            local.reserve(state, 'main', 5400, 'runtime2')


def test_invalid_session_reservation():
    for seconds in (-1, 0, 5401, float('nan')):
        with pytest.raises(ValueError):
            local.reserve(local.new_state(), 'main', seconds, 'x')


def test_pilot_gate_requires_four_complete_fresh_pilots():
    reports = {c: {'pilot_usable': True, 'pilot_measurement_complete': True,
        'pilot_measured_steps': 100, 'optimizer_step': 105, 'trained_target_tokens': 430080,
        'status': 'incomplete', 'model': {'seed':401, 'cell':c},
        'elapsed_training_seconds_including_checkpoint': 100,
        'evaluation_seconds_this_invocation': 2, 'run_id':c} for c in local.ORDER}
    assert local.pilot_estimate(reports) > 8192
    reports['HL']['pilot_usable'] = False
    with pytest.raises(ValueError, match='pilot'):
        local.pilot_estimate(reports)


def test_checkpoint_hash_detects_damage(tmp_path):
    d = tmp_path / 'checkpoints'; d.mkdir()
    cp = d / 'checkpoint-1.pt'; cp.write_bytes(b'good')
    (d/'latest.json').write_text(json.dumps({'filename':cp.name,'sha256':local.sha(cp),'checkpoint_id':'1'}))
    assert local.pointer(tmp_path)[1] == cp
    cp.write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        local.pointer(tmp_path)


def test_restore_requires_empty_target_and_verified_snapshot(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'ledger.json').write_text(json.dumps(local.new_state()))
    (source/'example.txt').write_text('saved')
    local.seal(source)
    target = tmp_path/'target'
    local.restore(source, target)
    assert (target/'example.txt').read_text() == 'saved'
    with pytest.raises(ValueError, match='empty'):
        local.restore(source,target)
    (source/'example.txt').write_text('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        local.restore(source,tmp_path/'other')


def test_local_retention_keeps_latest_and_previous(tmp_path):
    import os
    d=tmp_path/'checkpoints'; d.mkdir()
    for i in range(4):
        p=d/f'checkpoint-{i}.pt';p.write_text(str(i));os.utime(p,(i+1,i+1))
    p=d/'checkpoint-3.pt'
    (d/'latest.json').write_text(json.dumps({'filename':p.name,'checkpoint_id':'3','sha256':local.sha(p)}))
    local.retain_two(tmp_path)
    assert sorted(p.name for p in d.glob('*.pt')) == ['checkpoint-2.pt','checkpoint-3.pt']


def test_failure_stops_before_next_cell_and_preserves_log(tmp_path, monkeypatch):
    import torch,time
    from scripts import run_colab_mini as mini
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'Tesla T4')
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    monkeypatch.setattr(local,'validate_payload',lambda:{'payload_sha256':'fixture'})
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'test-runtime' if str(p)=='/proc/sys/kernel/random/boot_id' else original(p,*a,**k))
    calls=[]
    def fail(config,*a,**k):
        calls.append(Path(config).stem)
        raise RuntimeError('original model failure')
    monkeypatch.setattr(mini,'run_colab_mini',fail)
    with pytest.raises(RuntimeError,match='original model failure'):
        local.run('pilot',10,time.time()-300,tmp_path)
    assert calls==['LR-HL']
    ledger=json.loads((tmp_path/'ledger.json').read_text())
    assert ledger['sessions'][0]['status']=='failed'
    assert ledger['sessions'][0]['reserved_seconds']==5400
    assert 'original model failure' in next(tmp_path.glob('failure-*.log')).read_text()
    assert (tmp_path/'snapshot.json').is_file()
    with pytest.raises(ValueError,match='unresolved'):
        local.run('pilot',10,time.time()-300,tmp_path)
    assert calls==['LR-HL']


def test_missing_pilots_prevent_main_training(tmp_path,monkeypatch):
    import torch,time
    from scripts import run_colab_mini as mini
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'Tesla T4')
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    monkeypatch.setattr(local,'validate_payload',lambda:{'payload_sha256':'fixture'})
    monkeypatch.setattr(mini,'run_colab_mini',lambda *a,**k:pytest.fail('main must not start'))
    with pytest.raises(FileNotFoundError):
        local.run('main',10,time.time()-300,tmp_path)
    assert not (tmp_path/'ledger.json').exists()
