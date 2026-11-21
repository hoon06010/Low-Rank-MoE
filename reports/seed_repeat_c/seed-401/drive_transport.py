"""Seed401 CPU storage preflight; not a GPU runner or persistence gate bypass.

Uses immutable Drive blob IDs. Credentials come only from the runtime environment.
Nothing is deleted, and a cache is never accepted as remote storage evidence.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.overnight_store import publish_snapshot, restore_snapshot, digest, deadline

API = 'https://www.googleapis.com/drive/v3'
UPLOAD = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable'
ID = re.compile(r'^[A-Za-z0-9_-]+$')
CHUNK = 8 * 1024**2


def safe_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError('invalid Drive file/folder ID')
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DriveAPI:
    def __init__(self, timeout_seconds=60):
        if not os.environ.get('GOOGLE_DRIVE_ACCESS_TOKEN'):
            raise ValueError('GOOGLE_DRIVE_ACCESS_TOKEN must be configured privately in runtime environment')
        if timeout_seconds <= 0:
            raise ValueError('invalid HTTP timeout')
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(NoRedirect())

    def _request(self, url, method='GET', data=None, headers=None):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.netloc != 'www.googleapis.com':
            raise ValueError('unexpected Drive API endpoint')
        token = os.environ.get('GOOGLE_DRIVE_ACCESS_TOKEN')
        if not token:
            raise ValueError('runtime Drive access token is unavailable')
        if any(char.isspace() for char in token):
            raise ValueError('runtime Drive access token has invalid format')
        quota_project = os.environ.get('GOOGLE_DRIVE_QUOTA_PROJECT')
        extra_headers = {}
        if quota_project:
            if not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', quota_project):
                raise ValueError('invalid Drive quota project ID')
            extra_headers['X-Goog-User-Project'] = quota_project
        request = urllib.request.Request(url, data=data, method=method,
            headers={**(headers or {}), **extra_headers, 'Authorization': 'Bearer ' + token})
        try:
            return self.opener.open(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as exc:
            if exc.code == 308 and method == 'PUT':
                return exc
            code = exc.code
            exc.close()
            raise RuntimeError(f'Drive HTTP {code}; credentials and response body omitted') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError('Drive request failed; credentials and URL omitted') from None

    @staticmethod
    def _json(response):
        with response:
            data = response.read(2 * 1024**2 + 1)
        if len(data) > 2 * 1024**2:
            raise ValueError('oversized Drive JSON response')
        return json.loads(data)

    def quota(self):
        return self._json(self._request(API + '/about?fields=storageQuota'))['storageQuota']

    def metadata(self, fid):
        return self._json(self._request(API + '/files/' + safe_id(fid) + '?fields=id,name,size,mimeType,trashed'))

    def upload(self, folder, name, path):
        safe_id(folder)
        if Path(name).name != name or not name:
            raise ValueError('unsafe upload name')
        path = Path(path)
        total = path.stat().st_size
        if not path.is_file() or path.is_symlink() or total <= 0:
            raise ValueError('upload requires nonempty regular file')
        metadata = json.dumps({'name': name, 'parents': [folder], 'mimeType': 'application/octet-stream'}).encode()
        with self._request(UPLOAD, 'POST', metadata, {
            'Content-Type': 'application/json; charset=UTF-8',
            'X-Upload-Content-Type': 'application/octet-stream',
            'X-Upload-Content-Length': str(total)}) as response:
            location = response.headers.get('Location')
        if not location:
            raise ValueError('missing resumable upload endpoint')
        offset = 0
        with path.open('rb') as source:
            while offset < total:
                source.seek(offset)
                block = source.read(min(CHUNK, total-offset))
                with self._request(location, 'PUT', block, {
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(len(block)),
                    'Content-Range': f'bytes {offset}-{offset+len(block)-1}/{total}'}) as response:
                    if response.status == 308:
                        match = re.fullmatch(r'bytes=0-(\d+)', response.headers.get('Range', ''))
                        received = int(match.group(1)) + 1 if match else 0
                        if not offset < received <= offset + len(block):
                            raise ValueError('resumable upload made invalid/no progress')
                        offset = received
                    else:
                        result = json.loads(response.read(2 * 1024**2))
                        if offset + len(block) != total:
                            raise ValueError('upload finalized before complete content')
                        safe_id(result.get('id'))
                        return result
        raise ValueError('upload did not finalize')

    def download(self, fid, path):
        path = Path(path)
        if path.exists():
            raise ValueError('download destination must be new')
        meta = self.metadata(fid)
        if meta.get('trashed') or 'size' not in meta:
            raise ValueError('remote file is unavailable or not a blob')
        expected_size = int(meta['size'])
        actual = 0
        try:
            with self._request(API + '/files/' + safe_id(fid) + '?alt=media') as response, path.open('xb') as target:
                while block := response.read(CHUNK):
                    actual += len(block)
                    if actual > expected_size:
                        raise ValueError('download exceeds remote blob size')
                    target.write(block)
            if actual != expected_size:
                raise ValueError('download size integrity mismatch')
        except BaseException:
            # Only this method's own newly created incomplete download is removed.
            path.unlink(missing_ok=True)
            raise


def new_cache(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def verified_download(backend, fid, target, sha256, byte_count):
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError('invalid remote blob byte count')
    if shutil.disk_usage(target.parent).free < byte_count + 1024**3:
        raise ValueError('insufficient local download space including reserve')
    meta = backend.metadata(fid)
    if int(meta.get('size', -1)) != byte_count or meta.get('trashed'):
        raise ValueError('remote metadata integrity mismatch')
    backend.download(fid, target)
    if target.stat().st_size != byte_count or digest(target) != sha256:
        raise ValueError('remote download integrity mismatch')


def upload_verified(backend, folder, path, cache):
    result = backend.upload(folder, path.name, path)
    fid = safe_id(result.get('id'))
    target = cache / ('readback-' + uuid.uuid4().hex)
    verified_download(backend, fid, target, digest(path), path.stat().st_size)
    return fid


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def publish(backend, folder, run, identity, cache, *, step=0, timeout_seconds=300):
    """Return only after archive, receipt and final immutable pointer readbacks pass."""
    safe_id(folder)
    with deadline(timeout_seconds):
        cache = new_cache(cache)
        record = publish_snapshot(Path(run), cache/'staged', identity, step=step, timeout_seconds=timeout_seconds)
        archive = cache/'staged'/record['archive']
        quota = backend.quota()
        limit = quota.get('limit')
        usage = quota.get('usage')
        if usage is None or limit is None:
            raise ValueError('remote quota limit/usage unavailable')
        if int(limit) - int(usage) < record['bytes'] + 1024**2:
            raise ValueError('insufficient remote quota')
        archive_id = upload_verified(backend, folder, archive, cache)
        remote_receipt = dict(record, archive_id=archive_id,
            evidence_kind='direct-drive-independent-download-sha256', created_at=time.time())
        receipt_path = cache/(record['generation']+'.json')
        write_json(receipt_path, remote_receipt)
        receipt_id = upload_verified(backend, folder, receipt_path, cache)
        pointer = {'generation': record['generation'], 'receipt_id': receipt_id,
            'receipt_sha256': digest(receipt_path), 'receipt_bytes': receipt_path.stat().st_size,
            'identity': identity, 'pointer_kind': 'immutable-commit-v1'}
        pointer_path = cache/('commit-'+record['generation']+'.json')
        write_json(pointer_path, pointer)
        pointer_id = upload_verified(backend, folder, pointer_path, cache)
        evidence = {**pointer, 'pointer_id': pointer_id, 'archive_id': archive_id,
            'archive_sha256': record['sha256'], 'archive_bytes': record['bytes'],
            'checkpoint_id': record['checkpoint_id'], 'checkpoint_sha256': record['checkpoint_sha256'],
            'remote_quota': quota, 'status': 'remote-publish-verified',
            'fresh_runtime_restore_verified': False}
        write_json(cache/'publish-evidence.json', evidence)
        return evidence


def download_json(backend, fid, target, max_bytes=2*1024**2):
    size=int(backend.metadata(fid).get('size', -1))
    if not 0 < size <= max_bytes:
        raise ValueError('invalid remote JSON size')
    backend.download(fid, target)
    if target.stat().st_size != size:
        raise ValueError('remote JSON size integrity mismatch')
    return json.loads(target.read_text())


def restore(backend, pointer_id, output, identity, cache, *, timeout_seconds=300):
    """Fetch an immutable commit by ID into empty cache; validate before extraction."""
    with deadline(timeout_seconds):
        output = Path(output)
        if output.exists():
            raise ValueError('restore output must be new')
        cache = new_cache(cache)
        pointer = download_json(backend, safe_id(pointer_id), cache/'commit.json')
        if pointer.get('identity') != identity or pointer.get('pointer_kind') != 'immutable-commit-v1':
            raise ValueError('commit identity/type mismatch')
        generation = pointer['generation']
        if not isinstance(generation, str) or not re.fullmatch(r'generation-\d{20}-[a-f0-9]{32}', generation):
            raise ValueError('unsafe generation')
        receipt_path = cache/(generation+'.json')
        verified_download(backend, safe_id(pointer['receipt_id']), receipt_path,
            pointer['receipt_sha256'], pointer['receipt_bytes'])
        record=json.loads(receipt_path.read_text())
        if record['identity'] != identity or record['generation'] != generation or record['archive'] != generation+'.tar':
            raise ValueError('receipt identity/path mismatch')
        verified_download(backend, safe_id(record['archive_id']), cache/record['archive'],
            record['sha256'], record['bytes'])
        write_json(cache/'latest.json', {'generation':generation})
        restored = restore_snapshot(cache, output, identity, timeout_seconds=timeout_seconds)
        write_json(cache/'restore-evidence.json', {'status':'fresh-cache-restore-verified',
            'pointer_id': pointer_id, 'checkpoint_id': restored['checkpoint_id'],
            'checkpoint_sha256': restored['checkpoint_sha256'], 'identity': identity,
            'step': restored['step'], 'cpu_fixture_only': identity.get('phase') == 'cpu-storage-fixture'})
        return restored


def create_fixture(output):
    """Synthetic bytes exercise storage; deliberately not a trainable torch checkpoint."""
    output=Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint=output/'checkpoints'; checkpoint.mkdir()
    payload=checkpoint/'fixture.pt'
    payload.write_bytes(b'seed401 synthetic CPU storage fixture\n'*1024)
    write_json(checkpoint/'latest.json', {'filename':'fixture.pt',
        'checkpoint_id':'cpu-storage-fixture', 'sha256':digest(payload)})
    (output/'run_manifest.jsonl').write_text(json.dumps({'run_id':'cpu-storage-fixture',
        'checkpoint_id':'cpu-storage-fixture','checkpoint_step':0,'trained_target_tokens':0,
        'status':'incomplete'})+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['fixture-publish','fresh-restore','quota'])
    parser.add_argument('--folder-id')
    parser.add_argument('--pointer-id')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--identity-file', type=Path)
    parser.add_argument('--timeout-seconds',type=float,default=300)
    args=parser.parse_args()
    backend=DriveAPI()
    if args.mode == 'quota':
        print(json.dumps(backend.quota(),sort_keys=True)); return
    if not args.work_dir or not args.identity_file:
        parser.error('--work-dir and --identity-file required')
    identity=json.loads(args.identity_file.read_text())
    if identity.get('seed') != 401 or identity.get('phase') != 'cpu-storage-fixture':
        raise ValueError('CLI restricted to seed401 CPU storage fixture')
    args.work_dir.mkdir(parents=True,exist_ok=False)
    if args.mode == 'fixture-publish':
        if not args.folder_id:
            parser.error('--folder-id required')
        run=args.work_dir/'fixture'; create_fixture(run)
        result=publish(backend,args.folder_id,run,identity,args.work_dir/'publish-cache',timeout_seconds=args.timeout_seconds)
    else:
        if not args.pointer_id:
            parser.error('--pointer-id required')
        result=restore(backend,args.pointer_id,args.work_dir/'restored',identity,args.work_dir/'download-cache',timeout_seconds=args.timeout_seconds)
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__ == '__main__':
    main()
