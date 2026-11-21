"""Frozen budgets and isolated, hash-verified longer-training payloads."""
import copy
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from scripts import package_colab_mini as package

ROOT = Path(__file__).resolve().parents[1]


def test_longer_training_configs_keep_recipe_and_only_extend_c_schedule():
    for cell in ("LR-HS", "LR-HL", "LR-LS", "LR-LL"):
        original = json.loads((ROOT / "configs/colab_stability_adamw" / f"{cell}.json").read_text())
        for stage, budget, warmup in (("b", 3145728, 157286), ("c", 8388608, 419430)):
            path = ROOT / f"configs/longer_training_{stage}" / f"{cell}.json"
            assert path.is_file(), f"missing frozen configuration: {path}"
            expected = copy.deepcopy(original)
            expected["training"].update(total_target_tokens=budget, warmup_target_tokens=warmup)
            assert json.loads(path.read_text()) == expected


def _fixture(tmp_path, monkeypatch):
    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return path
    dependency = write("data/prepared/train.json", "[]")
    record = {"path": "train.json", "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest()}
    spec = {"audit_artifacts": [], "splits": {"train": record}, "manifest": {
        "tokenizer": {"artifact_checksums": []},
        "dataset": {"revision": {"value": "pinned-revision"}, "shards": [{
            "path": "../raw/fineweb_10k/000_00000.parquet", "locally_verified_sha256": "a" * 64}]}}}
    bundle = write("data/prepared/bundle.json", json.dumps(spec))
    write("scripts/run_colab_mini.py", "# runner")
    write("scripts/longer_training_remote.py", "# remote controller")
    write("scripts/longer_training_bootstrap.py", "# bootstrap")
    write("scripts/overnight_colab.py", "# notebook controller")
    write("scripts/overnight_store.py", "# verified snapshots")
    write("overnight_free_colab.ipynb", "{}")
    write("src/train.py", "# source")
    for folder in ("colab_free_mini", "colab_stability_adamw", "colab_stability_spectron", "longer_training_b", "longer_training_c"):
        for cell in ("LR-HS", "LR-HL", "LR-LS", "LR-LL"):
            write(f"configs/{folder}/{cell}.json", "{}")
    old = write("reports/colab_free_mini/payload.tar.gz", "old payload")
    write("reports/colab_free_mini/payload_manifest.json", "old manifest")
    write("credentials.json", "secret")
    monkeypatch.setattr(package, "ROOT", tmp_path)
    monkeypatch.setattr(package, "BUNDLE", bundle)
    monkeypatch.setattr(package, "OUTPUT", old.parent)
    monkeypatch.setattr(package, "LONGER_BUNDLE_SHA256", hashlib.sha256(bundle.read_bytes()).hexdigest(), raising=False)
    return dependency, bundle


def test_longer_package_preserves_old_payload_and_verifies_archive(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    package.main(["--mode", "longer-training"])
    assert (tmp_path / "reports/colab_free_mini/payload.tar.gz").read_text() == "old payload"
    assert (tmp_path / "reports/colab_free_mini/payload_manifest.json").read_text() == "old manifest"
    output = tmp_path / "reports/longer_training/package"
    manifest = json.loads((output / "payload_manifest.json").read_text())
    archive = output / "payload.tar.gz"
    assert package.digest(archive) == manifest["payload_sha256"]
    records = {item["path"]: item for item in manifest["files"]}
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(records)
        for name, record in records.items():
            content = tar.extractfile(name).read()
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
            assert len(content) == record["bytes"]
    assert len([name for name in records if name.startswith("configs/longer_training_")]) == 8
    assert "data/prepared/train.json" in records
    assert "scripts/run_colab_mini.py" in records and "src/train.py" in records
    assert "scripts/longer_training_remote.py" in records
    assert "scripts/longer_training_bootstrap.py" in records
    assert "scripts/overnight_colab.py" in records
    assert "scripts/overnight_store.py" in records
    assert "overnight_free_colab.ipynb" in records
    assert "credentials.json" not in records
    assert not any(name.startswith("data/raw/") for name in records)


def test_longer_package_rejects_changed_bundle_and_dependency(tmp_path, monkeypatch):
    dependency, bundle = _fixture(tmp_path, monkeypatch)
    original = bundle.read_text()
    bundle.write_text(original + " ")
    with pytest.raises(ValueError, match="bundle.*hash"):
        package.main(["--mode", "longer-training"])
    bundle.write_text(original)
    dependency.write_text("corrupt")
    with pytest.raises(ValueError, match="dependency.*hash"):
        package.main(["--mode", "longer-training"])
    assert not (tmp_path / "reports/longer_training/package").exists()


def test_default_package_keeps_original_allowlist_and_output(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    package.main([])
    manifest = json.loads((tmp_path / "reports/colab_free_mini/payload_manifest.json").read_text())
    assert not any(row["path"].startswith("configs/longer_training_") for row in manifest["files"])
    assert not (tmp_path / "reports/longer_training").exists()
