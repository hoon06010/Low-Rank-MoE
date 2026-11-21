"""Prepare the frozen seed401 package from local/Kaggle input files, without Drive."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

PAYLOAD_HASH = '877e874871dfc74fa203a1243dfd1f7644685fdb6b892940f8cfed176f5cf268'
MANIFEST_HASH = '9205f64b56e78ecb231dedf3a0636b4838860cf81eb95af765a874ecfeee4d9e'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(input_dir, root, *, raw_source=None):
    input_dir, root = Path(input_dir), Path(root)
    found = [p for p in input_dir.rglob('payload_manifest.json') if sha(p)==MANIFEST_HASH]
    if not found:
        raise ValueError('Attach the private seed401-local-input dataset first')
    manifest_path = found[0]
    manifest = json.loads(manifest_path.read_text())
    if manifest['payload_sha256'] != PAYLOAD_HASH:
        raise ValueError('unexpected payload')
    root.mkdir(parents=True,exist_ok=True)
    expected = {r['path']:r for r in manifest['files']}
    archive = manifest_path.parent/'payload.tar.gz'
    if archive.exists():
        if sha(archive) != PAYLOAD_HASH:
            raise ValueError('payload checksum mismatch')
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if len(members)!=len(expected) or {m.name for m in members}!=set(expected):
                raise ValueError('archive member mismatch')
            for m in members:
                if not m.isfile() or not (root/m.name).resolve().is_relative_to(root.resolve()):
                    raise ValueError('unsafe archive member')
                if m.size!=expected[m.name]['bytes']:
                    raise ValueError('archive member size mismatch')
            tar.extractall(root,filter='data')
    else:
        candidates = [manifest_path.parent/'payload',manifest_path.parent]
        source = next((p for p in candidates if (p/'scripts/run_colab_mini.py').is_file()),None)
        if source is None:
            raise ValueError('missing payload archive or Kaggle-expanded payload')
        for name,r in expected.items():
            p=source/name
            if p.is_symlink() or sha(p)!=r['sha256']:
                raise ValueError('expanded payload checksum mismatch')
            dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    for name,r in expected.items():
        if sha(root/name)!=r['sha256']:
            raise ValueError('extracted payload checksum mismatch')
    shutil.copy2(manifest_path,root/'seed401_payload_manifest.json')
    for r in manifest['remote_downloads']:
        target=root/r['path']; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)==r['sha256']:
            continue
        if raw_source:
            source=Path(raw_source)
            if sha(source)!=r['sha256']:
                raise ValueError('raw source checksum mismatch')
            shutil.copy2(source,target)
        else:
            if shutil.disk_usage(root).free<10*1024**3:
                raise ValueError('at least 10 GiB free space needed')
            started=time.monotonic(); temporary=target.with_suffix('.download')
            with urllib.request.urlopen(r['url'],timeout=60) as response, temporary.open('wb') as stream:
                size=0
                while block:=response.read(8*1024**2):
                    size+=len(block)
                    if size>3*1024**3 or time.monotonic()-started>900:
                        raise TimeoutError('raw download exceeded bounded setup')
                    stream.write(block)
            if sha(temporary)!=r['sha256']:
                raise ValueError('raw download checksum mismatch')
            temporary.replace(target)
    return manifest
