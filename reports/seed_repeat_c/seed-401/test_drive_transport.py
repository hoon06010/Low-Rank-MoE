import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

@pytest.fixture
def transport():
    file = HERE / 'drive_transport.py'
    assert file.exists(), 'direct Drive transport implementation is missing'
    spec = importlib.util.spec_from_file_location('drive_transport', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class FakeDrive:
    def __init__(self):
        self.files = {}
        self.corrupt = False
        self.fail_upload = False
    def upload(self, folder, name, path):
        if self.fail_upload:
            raise OSError('interrupted upload')
        fid = str(len(self.files) + 1)
        self.files[fid] = (name, Path(path).read_bytes())
        return {'id': fid, 'name': name, 'size': str(len(self.files[fid][1]))}
    def download(self, fid, path):
        Path(path).write_bytes(b'corrupt' if self.corrupt else self.files[fid][1])
    def metadata(self, fid):
        name, data = self.files[fid]
        return {'id': fid, 'name': name, 'size': str(len(data))}
    def quota(self):
        return {'limit': str(10**12), 'usage': '0'}


def test_roundtrip_fresh_cache_and_identity(transport, tmp_path):
    backend = FakeDrive()
    run = tmp_path / 'run'
    transport.create_fixture(run)
    identity = {'seed': 401, 'phase': 'cpu-storage-fixture', 'source': 'frozen'}
    commit = transport.publish(backend, 'private-folder', run, identity, tmp_path/'publish-cache')
    pointer = json.loads(backend.files[commit['pointer_id']][1])
    assert pointer['receipt_id'] == commit['receipt_id']
    assert list(backend.files)[-1] == commit['pointer_id']
    output = tmp_path/'fresh-runtime'/'restored'
    receipt = transport.restore(backend, commit['pointer_id'], output, identity, tmp_path/'empty-cache')
    assert receipt['checkpoint_id'] == 'cpu-storage-fixture'
    assert (output/'checkpoints/fixture.pt').read_bytes() == (run/'checkpoints/fixture.pt').read_bytes()
    with pytest.raises(ValueError, match='identity'):
        transport.restore(backend, commit['pointer_id'], tmp_path/'wrong', {'seed':402}, tmp_path/'wrong-cache')
    assert not (tmp_path/'wrong').exists()


def test_corrupt_download_never_commits_pointer(transport, tmp_path):
    backend = FakeDrive(); backend.corrupt = True
    transport.create_fixture(tmp_path/'run')
    with pytest.raises(ValueError, match='integrity'):
        transport.publish(backend, 'folder', tmp_path/'run', {'seed':401}, tmp_path/'cache')
    assert all(not name.startswith('commit-') for name, _ in backend.files.values())


def test_interrupted_upload_preserves_input(transport, tmp_path):
    backend = FakeDrive(); backend.fail_upload = True
    transport.create_fixture(tmp_path/'run')
    with pytest.raises(OSError, match='interrupted'):
        transport.publish(backend, 'folder', tmp_path/'run', {'seed':401}, tmp_path/'cache')
    assert (tmp_path/'run/checkpoints/fixture.pt').exists()
    assert not backend.files


def test_restore_corrupt_remote_does_not_create_output(transport, tmp_path):
    backend = FakeDrive(); transport.create_fixture(tmp_path/'run')
    identity={'seed':401}
    commit=transport.publish(backend,'folder',tmp_path/'run',identity,tmp_path/'cache')
    backend.files[commit['archive_id']] = (backend.files[commit['archive_id']][0], b'broken')
    with pytest.raises(ValueError, match='integrity'):
        transport.restore(backend,commit['pointer_id'],tmp_path/'restored',identity,tmp_path/'fresh')
    assert not (tmp_path/'restored').exists()


def test_token_not_in_errors_or_instance(transport, monkeypatch):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','secret-test-token')
    client=transport.DriveAPI()
    assert 'secret-test-token' not in repr(client.__dict__)


def test_low_remote_quota_refuses_upload(transport, tmp_path):
    backend=FakeDrive(); backend.quota=lambda: {'limit':'1','usage':'1'}
    transport.create_fixture(tmp_path/'run')
    with pytest.raises(ValueError, match='quota'):
        transport.publish(backend,'folder',tmp_path/'run',{'seed':401},tmp_path/'cache')
    assert not backend.files

class Response:
    def __init__(self, status=200, body=b'', headers=None):
        import io
        self.status=status; self.headers=headers or {}; self.body=io.BytesIO(body)
    def read(self, *args): return self.body.read(*args)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self): self.body.close()

class FakeOpener:
    def __init__(self, responses): self.responses=list(responses); self.requests=[]
    def open(self, request, timeout):
        self.requests.append(request)
        item=self.responses.pop(0)
        if isinstance(item, Exception): raise item
        return item


def test_http_resumable_partial_progress_and_auth(transport, tmp_path, monkeypatch):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN', 'test-secret')
    client=transport.DriveAPI(); transport.CHUNK=4
    opener=FakeOpener([Response(headers={'Location':transport.UPLOAD+'&upload_id=fake'}),
        Response(308,headers={'Range':'bytes=0-1'}),
        Response(308,headers={'Range':'bytes=0-5'}),
        Response(200,b'{"id":"file_id"}')])
    client.opener=opener
    payload=tmp_path/'archive.tar'; payload.write_bytes(b'abcdefgh')
    assert client.upload('folder_id','archive.tar',payload)['id']=='file_id'
    assert [r.data for r in opener.requests[1:]]==[b'abcd',b'cdef',b'gh']
    assert all(r.get_header('Authorization')=='Bearer test-secret' for r in opener.requests)


def test_upload_location_cannot_exfiltrate_auth(transport, tmp_path, monkeypatch):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-secret')
    client=transport.DriveAPI()
    opener=FakeOpener([Response(headers={'Location':'https://attacker.invalid/upload'})])
    client.opener=opener
    payload=tmp_path/'archive.tar'; payload.write_bytes(b'archive')
    with pytest.raises(ValueError,match='endpoint'):
        client.upload('folder','archive.tar',payload)
    assert len(opener.requests)==1


def test_http_failure_redacts_token_and_body(transport, monkeypatch):
    from urllib.error import HTTPError
    import io
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-secret')
    client=transport.DriveAPI()
    client.opener=FakeOpener([HTTPError('https://www.googleapis.com/?token=test-secret',401,
        'test-secret',{},io.BytesIO(b'test-secret'))])
    with pytest.raises(RuntimeError,match='HTTP 401') as caught:
        client.quota()
    assert 'test-secret' not in str(caught.value)
    assert caught.value.__suppress_context__


def test_invalid_upload_progress_fails_closed(transport, tmp_path, monkeypatch):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-secret')
    client=transport.DriveAPI(); client.opener=FakeOpener([
        Response(headers={'Location':transport.UPLOAD}),Response(308,headers={'Range':'bytes=0-999'})])
    payload=tmp_path/'archive.tar'; payload.write_bytes(b'archive')
    with pytest.raises(ValueError,match='progress'):
        client.upload('folder','archive.tar',payload)


def test_http_download_truncation_discards_only_partial(transport, tmp_path, monkeypatch):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-secret')
    client=transport.DriveAPI(); client.opener=FakeOpener([
        Response(body=b'{"size":"20","id":"file"}'),Response(body=b'partial')])
    with pytest.raises(ValueError,match='integrity'):
        client.download('file',tmp_path/'download')
    assert not (tmp_path/'download').exists()


def test_restore_rejects_mutated_receipt_before_extract(transport,tmp_path):
    backend=FakeDrive(); transport.create_fixture(tmp_path/'run')
    commit=transport.publish(backend,'folder',tmp_path/'run',{'seed':401},tmp_path/'cache')
    name,data=backend.files[commit['receipt_id']]
    backend.files[commit['receipt_id']]=(name,data.replace(b'401',b'402'))
    with pytest.raises(ValueError,match='integrity'):
        transport.restore(backend,commit['pointer_id'],tmp_path/'restored',{'seed':401},tmp_path/'fresh')
    assert not (tmp_path/'restored').exists()


def test_quota_project_header_and_secret_redaction(transport, monkeypatch):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-secret')
    monkeypatch.setenv('GOOGLE_DRIVE_QUOTA_PROJECT','personal-project-123')
    client=transport.DriveAPI(); opener=FakeOpener([Response(body=b'{"storageQuota":{"limit":"1000","usage":"0"}}')]); client.opener=opener
    client.quota()
    assert opener.requests[0].get_header('X-goog-user-project')=='personal-project-123'


def test_operation_deadline_does_not_commit(transport,tmp_path):
    import time
    backend=FakeDrive(); backend.upload=lambda *args: time.sleep(1)
    transport.create_fixture(tmp_path/'run')
    with pytest.raises(TimeoutError):
        transport.publish(backend,'folder',tmp_path/'run',{'seed':401},tmp_path/'cache',timeout_seconds=.03)
    assert not backend.files


def test_existing_restore_cache_is_not_mutated(transport,tmp_path):
    backend=FakeDrive(); cache=tmp_path/'cache'; cache.mkdir(); sentinel=cache/'sentinel'; sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        transport.restore(backend,'id',tmp_path/'restored',{'seed':401},cache)
    assert sentinel.read_text()=='keep'
