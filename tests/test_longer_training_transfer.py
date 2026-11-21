import base64
import hashlib
import json

import pytest

from scripts import longer_training_transfer as transfer


def manifest(parts):
    return {"archive_bytes":sum(map(len,parts)),"archive_sha256":hashlib.sha256(b"".join(parts)).hexdigest(),
            "parts":[{"index":i,"path":f"/content/low_rank_moe_longer/transfer_A_HL/part-{i:05d}","bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()} for i,data in enumerate(parts)]}


def test_manifest_rejects_reordered_parts_and_oversize():
    spec=manifest([b"abc",b"def"])
    spec["parts"].reverse()
    with pytest.raises(ValueError,match="order"):
        transfer.validate_manifest(spec)
    spec=manifest([b"abc"])
    spec["parts"][0]["bytes"]=transfer.CHUNK_BYTES+1
    with pytest.raises(ValueError,match="size"):
        transfer.validate_manifest(spec)


def test_download_validates_parts_full_hash_and_refuses_overwrite(tmp_path):
    parts=[b"abc",b"def"]
    spec=manifest(parts)
    class Client:
        def _request(self,method,path,**kwargs):
            content=json.dumps(spec).encode() if path.endswith("manifest.json") else parts[int(path[-5:])]
            return {"type":"file","format":"base64","content":base64.b64encode(content).decode()}
    target=tmp_path/"archive.tar"
    result=transfer.download(Client(),"/content/low_rank_moe_longer/transfer_A_HL/manifest.json",target)
    assert target.read_bytes()==b"abcdef" and result["sha256"]==spec["archive_sha256"]
    with pytest.raises(FileExistsError):
        transfer.download(Client(),"/content/low_rank_moe_longer/transfer_A_HL/manifest.json",target)
    spec["parts"][1]["sha256"]="0"*64
    with pytest.raises(ValueError,match="part checksum"):
        transfer.download(Client(),"/content/low_rank_moe_longer/transfer_A_HL/manifest.json",tmp_path/"bad.tar")


def test_download_rejects_wrong_full_hash(tmp_path):
    spec=manifest([b"abc"])
    spec["archive_sha256"]="0"*64
    class Client:
        def _request(self,method,path,**kwargs):
            value=json.dumps(spec).encode() if path.endswith("manifest.json") else b"abc"
            return {"type":"file","format":"base64","content":base64.b64encode(value).decode()}
    with pytest.raises(ValueError,match="archive checksum"):
        transfer.download(Client(),"/content/low_rank_moe_longer/transfer_A_HL/manifest.json",tmp_path/"bad.tar")


def test_upload_bounds_chunks_preserves_order_and_refuses_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer,"CHUNK_BYTES",3)
    archive=tmp_path/"archive.tar"
    archive.write_bytes(b"abcdefg")
    expected=hashlib.sha256(archive.read_bytes()).hexdigest()
    class Client:
        def __init__(self):
            self.objects={}
        def _request(self,method,path,**kwargs):
            if method=="GET":
                if path not in self.objects:
                    raise FileNotFoundError
                return self.objects[path]
            self.objects[path]=kwargs["json_data"]
            return self.objects[path]
    client=Client()
    result=transfer.upload(client,archive,"/content/longer_retrieved_HL_parts",expected)
    spec=json.loads(client.objects[result["manifest"]]["content"])
    assert [r["bytes"] for r in spec["parts"]]==[3,3,1]
    assert b"".join(base64.b64decode(client.objects[r["path"]]["content"]) for r in spec["parts"])==b"abcdefg"
    assert spec["archive_sha256"]==expected
    with pytest.raises(FileExistsError):
        transfer.upload(client,archive,"/content/longer_retrieved_HL_parts",expected)


def test_download_rejects_unrelated_remote_paths_before_request(tmp_path):
    class Client:
        def _request(self,*args,**kwargs):
            raise AssertionError("must reject before network request")
    with pytest.raises(ValueError,match="manifest path"):
        transfer.download(Client(),"/content/unrelated/manifest.json",tmp_path/"archive.tar")


def test_manifest_rejects_parts_from_different_directories():
    spec=manifest([b"abc",b"def"])
    spec["parts"][1]["path"]="/content/another/part-00001"
    with pytest.raises(ValueError,match="sibling"):
        transfer.validate_manifest(spec)


def test_read_bytes_accepts_wrapped_base64_within_decoded_limit():
    data=b"a"*262144
    encoded=base64.b64encode(data).decode()
    wrapped="\n".join(encoded[i:i+76] for i in range(0,len(encoded),76))
    class Client:
        def _request(self,*args,**kwargs):
            return {"type":"file","format":"base64","content":wrapped}
    assert transfer.read_bytes(Client(),"/content/part",len(data))==data


@pytest.mark.parametrize("prefix", [b"abc", b"abcde", b"abcdefghi"])
def test_resume_verifies_prefix_and_downloads_only_missing_parts(tmp_path, prefix):
    parts=[b"abc",b"def",b"ghi"]
    spec=manifest(parts)
    calls=[]
    class Client:
        def _request(self,method,path,**kwargs):
            calls.append(path)
            value=json.dumps(spec).encode() if path.endswith("manifest.json") else parts[int(path[-5:])]
            return {"type":"file","format":"base64","content":base64.b64encode(value).decode()}
    target=tmp_path/"archive.tar"
    target.write_bytes(prefix)
    result=transfer.download(Client(),"/content/low_rank_moe_longer/transfer_A_HL/manifest.json",target,
                             resume=True,expected_sha256=spec["archive_sha256"])
    assert target.read_bytes()==b"abcdefghi"
    assert result["resumed_bytes"]==len(prefix)
    assert not any(path.endswith("part-00000") for path in calls)
    assert len(calls)==1+(3-len(prefix)//3)


@pytest.mark.parametrize("prefix", [b"xbc", b"abcxe"])
def test_resume_rejects_corrupt_prefix_without_changing_file(tmp_path, prefix):
    parts=[b"abc",b"def"]
    spec=manifest(parts)
    class Client:
        def _request(self,method,path,**kwargs):
            value=json.dumps(spec).encode() if path.endswith("manifest.json") else parts[int(path[-5:])]
            return {"type":"file","format":"base64","content":base64.b64encode(value).decode()}
    target=tmp_path/"archive.tar"
    target.write_bytes(prefix)
    with pytest.raises(ValueError,match="prefix"):
        transfer.download(Client(),"/content/low_rank_moe_longer/transfer_A_HL/manifest.json",target,
                          resume=True,expected_sha256=spec["archive_sha256"])
    assert target.read_bytes()==prefix


def test_resume_requires_pin_and_rejects_wrong_manifest(tmp_path):
    spec=manifest([b"abc"])
    class Client:
        def _request(self,*args,**kwargs):
            return {"type":"file","format":"text","content":json.dumps(spec)}
    target=tmp_path/"archive.tar"
    target.write_bytes(b"a")
    for pin in (None,"0"*64):
        with pytest.raises(ValueError,match="expected|pinned"):
            transfer.download(Client(),"/content/low_rank_moe_longer/transfer_A_HL/manifest.json",target,
                              resume=True,expected_sha256=pin)
    assert target.read_bytes()==b"a"
