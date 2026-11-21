"""Bounded local-only Colab archive transport; never imported by training payloads."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import time

CHUNK_BYTES = 32 * 1024 * 1024


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_manifest(spec, directory=None):
    parts = spec.get("parts", [])
    if not parts or not re.fullmatch("[0-9a-f]{64}", spec.get("archive_sha256", "")):
        raise ValueError("invalid archive manifest")
    directory = directory or str(PurePosixPath(parts[0].get("path", "")).parent)
    paths = set()
    total = 0
    for index, part in enumerate(parts):
        if part.get("index") != index:
            raise ValueError("part order must be contiguous from zero")
        size = part.get("bytes")
        if type(size) is not int or not 0 < size <= CHUNK_BYTES:
            raise ValueError("part size outside bounded transfer limit")
        path = part.get("path", "")
        if not path.startswith("/content/") or ".." in PurePosixPath(path).parts or path in paths:
            raise ValueError("unsafe or duplicate part path")
        if path != f"{directory}/part-{index:05d}":
            raise ValueError("part paths must be ordered siblings of the manifest")
        if not re.fullmatch("[0-9a-f]{64}", part.get("sha256", "")):
            raise ValueError("invalid part checksum")
        paths.add(path)
        total += size
    if type(spec.get("archive_bytes")) is not int or total != spec["archive_bytes"]:
        raise ValueError("archive size differs from sum of parts")
    return spec


def read_bytes(client, path, limit):
    value = client._request("GET", path, params={"content": "1"})
    if value.get("type") != "file":
        raise ValueError("remote object is not a file")
    text = value.get("content", "")
    if value.get("format") == "base64":
        text = "".join(text.split())
    if len(text) > (limit * 4 // 3 + 4096):
        raise ValueError("remote response exceeds size limit")
    if value.get("format") == "base64":
        result = base64.b64decode("".join(text.split()), validate=True)
    elif value.get("format") == "text":
        result = text.encode()
    else:
        raise ValueError("unsupported contents format")
    if len(result) > limit:
        raise ValueError("remote decoded content exceeds size limit")
    return result


def download(client, manifest_path, local, resume=False, expected_sha256=None):
    started = time.monotonic()
    local = Path(local)
    if local.is_symlink():
        raise ValueError("local archive cannot be a symlink")
    if resume and not re.fullmatch("[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("resume requires expected full archive sha256")
    if local.exists() and not resume:
        raise FileExistsError("local archive already exists")
    if resume and not local.is_file():
        raise FileNotFoundError("resume requires an existing partial archive")
    if not re.fullmatch(r"/content/(?:low_rank_moe_longer/transfer_[ABC]_(?:HL|HS|LS|LL)|longer_retrieved_HL_parts(?:_[A-Za-z0-9_-]+)?)/manifest\.json", manifest_path):
        raise ValueError("manifest path outside dedicated longer-training transfer directories")
    spec = validate_manifest(json.loads(read_bytes(client, manifest_path, 2*1024*1024)),
                             str(PurePosixPath(manifest_path).parent))
    if expected_sha256 is not None and expected_sha256 != spec["archive_sha256"]:
        raise ValueError("manifest differs from pinned expected archive hash")
    existing_bytes = local.stat().st_size if resume else 0
    if existing_bytes > spec["archive_bytes"]:
        raise ValueError("existing prefix exceeds archive size")
    if shutil.disk_usage(local.parent).free < spec["archive_bytes"] - existing_bytes + 256*1024*1024:
        raise ValueError("insufficient local disk space")
    full = hashlib.sha256()
    offset = 0
    with local.open("r+b" if resume else "xb") as destination:
        import fcntl
        fcntl.flock(destination.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.seek(0, 2) != existing_bytes:
            raise ValueError("local prefix changed before exclusive transfer lock")
        destination.seek(0)
        for part in spec["parts"]:
            prefix_size = min(part["bytes"], max(0, existing_bytes - offset))
            prefix = destination.read(prefix_size) if prefix_size else b""
            if prefix_size == part["bytes"]:
                if hashlib.sha256(prefix).hexdigest() != part["sha256"]:
                    raise ValueError("existing prefix checksum mismatch")
                full.update(prefix)
                offset += part["bytes"]
                continue
            content = read_bytes(client, part["path"], CHUNK_BYTES)
            if len(content) != part["bytes"] or hashlib.sha256(content).hexdigest() != part["sha256"]:
                raise ValueError("part checksum or size mismatch; partial archive retained")
            if prefix != content[:prefix_size]:
                raise ValueError("existing partial prefix differs from verified remote part")
            destination.write(content[prefix_size:])
            full.update(content)
            offset += part["bytes"]
    if local.stat().st_size != spec["archive_bytes"] or full.hexdigest() != spec["archive_sha256"]:
        raise ValueError("archive checksum or size mismatch; failed archive retained")
    return {"local":str(local), "sha256":full.hexdigest(), "bytes":local.stat().st_size,
            "parts":len(spec["parts"]), "resumed_bytes":existing_bytes,
            "elapsed_seconds":time.monotonic()-started}


def upload(client, local, remote_dir, expected_sha256):
    started = time.monotonic()
    local = Path(local)
    if digest(local) != expected_sha256:
        raise ValueError("local archive checksum differs from verified expected hash")
    if not re.fullmatch(r"/content/longer_retrieved_HL_parts(?:_[A-Za-z0-9_-]+)?", remote_dir):
        raise ValueError("upload requires dedicated HL recovery directory")
    try:
        client._request("GET", remote_dir, params={"content":"0"})
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("remote recovery directory already exists")
    client._request("PUT", remote_dir, json_data={"type":"directory"})
    parts = []
    full = hashlib.sha256()
    with local.open("rb") as source:
        while content := source.read(CHUNK_BYTES):
            index = len(parts)
            path = f"{remote_dir}/part-{index:05d}"
            sha = hashlib.sha256(content).hexdigest()
            full.update(content)
            client._request("PUT", path, json_data={"type":"file","format":"base64",
                            "name":PurePosixPath(path).name,"path":path,"content":base64.b64encode(content).decode(),"chunk":1})
            parts.append({"index":index,"path":path,"sha256":sha,"bytes":len(content)})
    if full.hexdigest() != expected_sha256:
        raise ValueError("archive changed during upload")
    spec = validate_manifest({"archive_sha256":expected_sha256,"archive_bytes":local.stat().st_size,"parts":parts})
    manifest_path = f"{remote_dir}/manifest.json"
    client._request("PUT", manifest_path, json_data={"type":"file","format":"text",
                    "name":"manifest.json","path":manifest_path,"content":json.dumps(spec),"chunk":1})
    return {"manifest":manifest_path,"elapsed_seconds":time.monotonic()-started,**spec}


def bounded_client(session_name, max_seconds):
    from colab_cli.contents import ContentsClient
    from colab_cli.state import StateStore
    import requests
    from urllib.parse import quote
    state = StateStore().get(session_name)
    if state is None:
        raise ValueError("unknown local Colab session")
    deadline = time.monotonic() + max_seconds
    class Client(ContentsClient):
        def __init__(self, state):
            super().__init__(state)
            self.http = requests.Session()

        def close(self):
            self.http.close()

        def _request(self, method, path, params=None, json_data=None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("transfer deadline exceeded")
            query = {"authuser":"0","colab-runtime-proxy-token":self.token,**(params or {})}
            url = f"{self.base_url}/api/contents/{quote(path.strip('/'),safe='/')}"
            try:
                with self.http.request(method,url,params=query,json=json_data,stream=True,
                                      timeout=(min(20,remaining),min(60,remaining))) as response:
                    if response.status_code == 404:
                        raise FileNotFoundError("remote path not found")
                    if not response.ok:
                        raise RuntimeError(f"contents HTTP status {response.status_code}")
                    chunks = bytearray()
                    for chunk in response.iter_content(1024*1024):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("transfer deadline exceeded")
                        chunks.extend(chunk)
                        if len(chunks) > 48*1024*1024:
                            raise ValueError("bounded contents response exceeded")
                    return json.loads(chunks)
            except requests.RequestException:
                raise RuntimeError("contents request failed; credentials suppressed") from None
    return Client(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("download","upload"))
    parser.add_argument("--session",required=True)
    parser.add_argument("--local",type=Path,required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--remote-dir",default="/content/longer_retrieved_HL_parts")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--resume",action="store_true",help="verify and extend an existing download using a pinned archive hash")
    parser.add_argument("--max-seconds",type=float,default=900)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 7200:
        parser.error("max-seconds must be within 0..7200")
    if args.operation == "download" and not args.manifest:
        parser.error("download requires --manifest")
    if args.operation == "upload" and not args.expected_sha256:
        parser.error("upload requires --expected-sha256")
    if args.resume and (args.operation != "download" or not args.expected_sha256):
        parser.error("--resume requires download and --expected-sha256")
    def timeout_handler(*_):
        raise TimeoutError("transfer hard deadline exceeded")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    client = None
    try:
        client = bounded_client(args.session,args.max_seconds)
        result = download(client,args.manifest,args.local,resume=args.resume,expected_sha256=args.expected_sha256) if args.operation == "download" else upload(client,args.local,args.remote_dir,args.expected_sha256)
        print(json.dumps(result,indent=2))
    except Exception as exc:
        # Never print request objects, URLs, or a traceback containing runtime tokens.
        print(json.dumps({"status":"failed","error_type":type(exc).__name__,"detail":"transfer stopped; retain partial files for diagnosis"}))
        raise SystemExit(1) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
