"""Upload explicit seed401 payloads; refresh existing drive.file credentials in memory only."""
import argparse
import certifi
import importlib.util
import json
import os
from pathlib import Path
import ssl
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SCOPE = 'https://www.googleapis.com/auth/drive.file'


def fresh_token():
    credentials = json.loads((Path.home()/'.config/colab-cli/token.json').read_text())
    body = urllib.parse.urlencode(dict(grant_type='refresh_token', refresh_token=credentials['refresh_token'],
        client_id=credentials['client_id'], client_secret=credentials['client_secret'], scope=SCOPE)).encode()
    try:
        request = urllib.request.Request('https://oauth2.googleapis.com/token', data=body)
        with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=certifi.where()), timeout=30) as response:
            result = json.load(response)
    except Exception:
        raise RuntimeError('existing OAuth refresh failed; credential details omitted') from None
    if result.get('scope') != SCOPE or not 0 < result.get('expires_in',0) <= 3600:
        raise ValueError('unexpected scope or credential duration')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['upload', 'clipboard-token'])
    parser.add_argument('--path', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
    if args.mode == 'upload' and not folder_id:
        parser.error('GOOGLE_DRIVE_FOLDER_ID is required for upload')
    token = fresh_token()
    if args.mode == 'clipboard-token':
        subprocess.run(['/usr/bin/pbcopy'], input=token['access_token'].encode(), check=True)
        print(json.dumps({'destination':'macOS clipboard for private Kaggle GOOGLE_DRIVE_ACCESS_TOKEN paste',
            'scope':SCOPE,'expires_in':token['expires_in'],'token_printed':False,'token_saved_to_file':False}))
        return
    if not args.path or not args.evidence:
        parser.error('--path and --evidence required')
    path = args.path.resolve()
    allowed = [ROOT/'reports/seed_repeat_c/package', HERE/'deployment']
    if not any(path.is_relative_to(p.resolve()) for p in allowed) or path.is_symlink() or not path.is_file():
        raise ValueError('upload is restricted to explicit seed401 deployment files')
    if args.evidence.exists():
        raise ValueError('evidence destination must be new')
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['GOOGLE_DRIVE_ACCESS_TOKEN'] = token['access_token']
    try:
        spec=importlib.util.spec_from_file_location('transport',HERE/'drive_transport.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        client=module.DriveAPI()
        quota=client.quota()
        if int(quota['limit'])-int(quota['usage']) < path.stat().st_size+1024**3:
            raise ValueError('insufficient measured Drive quota')
        folder=client._json(client._request(module.API+'/files/'+folder_id+'?fields=id,name,ownedByMe,permissions'))
        if not folder.get('ownedByMe') or any(p.get('type')!='user' or p.get('role')!='owner' for p in folder.get('permissions',[])):
            raise ValueError('private owned folder required')
        started=time.time()
        with tempfile.TemporaryDirectory(prefix='seed401-deployment-') as cache:
            fid=module.upload_verified(client,folder_id,path,Path(cache))
        evidence={'file_id':fid,'folder_id':folder_id,'path':str(path.relative_to(ROOT)),
            'sha256':module.digest(path),'bytes':path.stat().st_size,'uploaded_at':time.time(),
            'elapsed_seconds':time.time()-started,'independent_readback_verified':True,
            'private_owned_folder_verified':True,'scope':SCOPE}
        args.evidence.parent.mkdir(parents=True,exist_ok=True)
        module.write_json(args.evidence,evidence)
        print(json.dumps(evidence,sort_keys=True))
    finally:
        os.environ.pop('GOOGLE_DRIVE_ACCESS_TOKEN',None)

if __name__=='__main__': main()
