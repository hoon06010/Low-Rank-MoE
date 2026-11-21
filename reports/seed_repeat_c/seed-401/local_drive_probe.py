"""Local-only synthetic Drive probe; never sends credentials to Kaggle."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import ssl
import time
import urllib.parse
import urllib.request
import uuid
import certifi

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['publish', 'restore'])
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    credentials = json.loads((Path.home()/'.config/colab-cli/token.json').read_text())
    body = urllib.parse.urlencode(dict(grant_type='refresh_token',
        refresh_token=credentials['refresh_token'], client_id=credentials['client_id'],
        client_secret=credentials['client_secret'], scope='https://www.googleapis.com/auth/drive.file')).encode()
    request = urllib.request.Request('https://oauth2.googleapis.com/token', data=body)
    try:
        with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=certifi.where()), timeout=30) as response:
            token = json.load(response)
        if token.get('scope') != 'https://www.googleapis.com/auth/drive.file':
            raise ValueError('unexpected scope')
        os.environ['GOOGLE_DRIVE_ACCESS_TOKEN'] = token['access_token']
        spec = importlib.util.spec_from_file_location('transport', HERE/'drive_transport.py')
        transport = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transport)
        client = transport.DriveAPI()
        workspace = HERE/('local-drive-probe-'+uuid.uuid4().hex)
        workspace.mkdir()
        started = time.time()
        if args.mode == 'publish':
            identity = {'seed':401,'phase':'cpu-storage-fixture',
                'source_revision':'1d67b6451e97f946a065550e2f9a51a8240eb116dbacef9208656cf68132a646'}
            quota = client.quota()
            if int(quota['limit']) - int(quota['usage']) < 1024**3:
                raise ValueError('insufficient storage')
            folder = client._json(client._request(transport.API+'/files?fields=id,name,ownedByMe,permissions', 'POST',
                json.dumps({'name':'Low-Rank-MoE-seed401-direct-private-'+uuid.uuid4().hex[:8],
                    'mimeType':'application/vnd.google-apps.folder'}).encode(), {'Content-Type':'application/json'}))
            folder = client._json(client._request(transport.API+'/files/'+transport.safe_id(folder['id'])+
                '?fields=id,name,ownedByMe,permissions'))
            if not folder.get('ownedByMe') or any(p.get('role')!='owner' or p.get('type')!='user' for p in folder.get('permissions',[])):
                raise ValueError('folder private ownership not verified')
            transport.create_fixture(workspace/'fixture')
            result = transport.publish(client, folder['id'], workspace/'fixture', identity, workspace/'publish-cache')
            result.update(folder_id=folder['id'],folder_name=folder['name'],folder_private_owned_verified=True)
        else:
            if not args.evidence:
                parser.error('--evidence required')
            previous = json.loads(args.evidence.read_text())
            identity = previous['identity']
            result = transport.restore(client, previous['pointer_id'], workspace/'restored', identity, workspace/'download-cache')
            expected = b'seed401 synthetic CPU storage fixture\n'*1024
            if (workspace/'restored/checkpoints/fixture.pt').read_bytes()!=expected:
                raise ValueError('restored fixture bytes mismatch')
            result.update(pointer_id=previous['pointer_id'],folder_id=previous['folder_id'],restored_fixture_bytes_verified=True)
        result.update(probe_environment='local-Mac-CPU',mode=args.mode,elapsed_seconds=time.time()-started,
            kaggle_runtime_verified=False,whole_package_revalidated=False,cuda_recovery_verified=False,
            token_saved=False,token_transferred_to_kaggle=False,identity=identity)
        path=workspace/'local-probe-evidence.json'
        transport.write_json(path,result)
        print(json.dumps({'evidence':str(path),'folder_id':result['folder_id'],
            'pointer_id':result['pointer_id'],'mode':args.mode,'elapsed_seconds':result['elapsed_seconds']}))
    finally:
        os.environ.pop('GOOGLE_DRIVE_ACCESS_TOKEN',None)


if __name__=='__main__':
    main()
