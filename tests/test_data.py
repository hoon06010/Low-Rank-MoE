import pytest
import json
from copy import deepcopy
from pathlib import Path

from src.data import build_split_batches, prepare_documents, resume_batches, validate_ready_manifest


def _docs():
    return [
        {"id": "a", "url": "https://example.org/one#section", "text": "Alpha   text"},
        # URL links first -> second; normalized body links second -> third.
        {"id": "b", "url": "https://example.org/one/", "text": "Bridge text"},
        {"id": "c", "url": "https://example.org/three", "text": "Bridge\ntext"},
        {"id": "d", "url": "https://example.org/four", "text": "Independent text"},
        {"id": "empty", "url": "https://example.org/empty", "text": " \n\t "},
    ]


def test_transitive_id_url_and_body_links_assign_one_split():
    docs = [
        {"id": "shared-id", "url": "https://x.test/a", "text": "first"},
        {"id": "next", "url": "https://x.test/a/", "text": "second"},
        {"id": "third", "url": "https://x.test/c", "text": "second  \n"},
    ]
    prepared, stats = prepare_documents(docs, seed=23)
    assert len(prepared) == 2  # the exact normalized body copy is collapsed
    assert len({item["group_hash"] for item in prepared}) == 1
    assert len({item["split"] for item in prepared}) == 1
    assert stats["linked_components"] == 1
    assert stats["exact_duplicate_documents_removed"] == 1


def test_exact_duplicate_and_empty_document_handling_is_reported():
    docs = [
        {"id": "z", "url": "https://x.test/z", "text": "Same text"},
        {"id": "a", "url": "https://x.test/a", "text": "Same   text"},
        {"id": "blank", "text": "\n  "},
    ]
    prepared, stats = prepare_documents(docs)
    assert len(prepared) == 1
    assert stats == {
        "input_documents": 3,
        "missing_text": 0,
        "null_text": 0,
        "non_string_text": 0,
        "blank_text": 1,
        "exact_duplicate_documents_removed": 1,
        "duplicate_id_records": 0,
        "retained_documents": 1,
        "linked_components": 1,
        "largest_component_documents": 2,
    }


def test_split_isolation_and_input_order_invariance():
    docs = _docs()
    prepared, _ = prepare_documents(docs, seed=17)
    shuffled, _ = prepare_documents(list(reversed(docs)), seed=17)
    assert prepared == shuffled
    assignments = {item["text_sha256"]: item["split"] for item in prepared}
    assert assignments == {item["text_sha256"]: item["split"] for item in shuffled}

    # Every linked component appears in one split, even after transitive links.
    by_group = {}
    for item in prepared:
        by_group.setdefault(item["group_hash"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in by_group.values())
    assert all(item["normalized_text"] if "normalized_text" in item else item["text"] for item in prepared)


def test_split_local_eos_packing_pre_shifted_targets_and_padding():
    docs = [
        {"id": "first", "text": "a b", "split": "train", "group_hash": "g1", "text_sha256": "1"},
        {"id": "second", "text": "c d", "split": "train", "group_hash": "g2", "text_sha256": "2"},
        {"id": "heldout", "text": "x y", "split": "test", "group_hash": "g3", "text_sha256": "3"},
    ]
    token_ids = {"a b": [1, 2], "c d": [3, 4], "x y": [7, 8]}
    batches = build_split_batches(
        docs, split="train", encode=token_ids.__getitem__, eos_token_id=99,
        context_length=4, pad_token_id=0, seed=1,
    )
    assert len(batches) == 2
    assert batches[0] == {
        "input_ids": [[3, 4, 99, 1]],
        "labels": [[4, 99, 1, 2]],
        "attention_mask": [[1, 1, 1, 1]],
        "valid_target_count": 4,
        "split": "train",
    }
    assert batches[1] == {
        "input_ids": [[2, 99, 0, 0]],
        "labels": [[99, -100, -100, -100]],
        "attention_mask": [[1, 1, 0, 0]],
        "valid_target_count": 1,
        "split": "train",
    }
    assert 7 not in [token for batch in batches for row in batch["input_ids"] for token in row]
    assert sum(batch["valid_target_count"] for batch in batches) == 5


def test_deterministic_order_and_cursor_resume():
    docs = [
        {"id": str(i), "text": f"doc {i}", "split": "train", "group_hash": f"g{i}", "text_sha256": f"h{i}"}
        for i in range(11)
    ]
    batches = build_split_batches(
        docs, split="train", encode=lambda text: [int(text.split()[-1])],
        eos_token_id=99, context_length=2, seed=9,
    )
    replay = build_split_batches(
        list(reversed(docs)), split="train", encode=lambda text: [int(text.split()[-1])],
        eos_token_id=99, context_length=2, seed=9,
    )
    assert batches == replay
    full = list(resume_batches(batches, cursor=0))
    resumed = list(resume_batches(batches, cursor=2))
    assert resumed == full[2:]
    assert [index for index, _ in resumed] == list(range(2, len(batches)))
    assert list(resume_batches(batches, cursor=len(batches))) == []
    with pytest.raises(ValueError, match="cursor"):
        list(resume_batches(batches, cursor=len(batches) + 1))


def test_rejects_bad_inputs_and_split_names():
    with pytest.raises(ValueError, match="seed"):
        prepare_documents([], seed=-1)
    prepared, _ = prepare_documents(_docs())
    with pytest.raises(ValueError, match="split"):
        build_split_batches(prepared, split="validation", encode=lambda _: [], eos_token_id=1)
    with pytest.raises(ValueError, match="context_length"):
        build_split_batches(prepared, split="train", encode=lambda _: [], eos_token_id=1, context_length=0)
    with pytest.raises(TypeError, match="sequence"):
        build_split_batches(prepared, split="train", encode=lambda _: "bad", eos_token_id=1)


def test_repeated_document_ids_are_counted_and_linked():
    prepared, stats = prepare_documents([
        {"id": "same", "url": "https://x.test/1", "text": "text one"},
        {"id": "same", "url": "https://x.test/2", "text": "text two"},
    ])
    assert len(prepared) == 2
    assert len({item["group_hash"] for item in prepared}) == 1
    assert stats["duplicate_id_records"] == 1


def test_missing_null_non_string_and_blank_text_are_counted_separately():
    _, stats = prepare_documents([
        {"id": "missing"},
        {"id": "null", "text": None},
        {"id": "number", "text": 17},
        {"id": "blank", "text": " \n\t "},
        {"id": "valid", "text": "valid body"},
    ])
    assert stats["input_documents"] == 5
    assert stats["missing_text"] == 1
    assert stats["null_text"] == 1
    assert stats["non_string_text"] == 1
    assert stats["blank_text"] == 1
    assert stats["retained_documents"] == 1


def test_document_limit_stops_consuming_generator_at_limit_plus_one():
    seen = []

    def source():
        for i in range(20):
            seen.append(i)
            yield {"id": str(i), "text": f"doc {i}"}

    with pytest.raises(ValueError, match="max_documents=3"):
        prepare_documents(source(), max_documents=3)
    assert seen == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="max_documents"):
        prepare_documents([], max_documents=0)


def _ready_manifest():
    revision_time = "2026-09-25T00:00:00Z"
    digest = "a" * 64
    revision = {"value": "b" * 40, "provenance": "locally_verified", "locally_verified_at": revision_time}
    return {
        "schema_version": 1,
        "manifest_id": "synthetic-ready-fixture",
        "status": "ready",
        "dataset": {
            "repository": "owner/dataset", "config": "sample",
            "revision": deepcopy(revision),
            "license": {
                "source_label": "reviewed source label", "license_url": "https://example.test/license",
                "terms_url": "https://example.test/terms", "local_review_status": "reviewed",
            },
            "shards": [{
                "path": "shard-000.parquet", "source_declared_sha256": None,
                "locally_verified_sha256": digest, "locally_verified_at": revision_time,
                "verification_status": "match",
            }],
        },
        "tokenizer": {
            "repository": "owner/tokenizer", "revision": deepcopy(revision),
            "artifact_checksums": [{"path": "tokenizer.json", "source_declared_sha256": None,
                                    "locally_verified_sha256": digest}],
            "special_token_behavior": {
                "eos_token": "</s>", "bos_token": None, "pad_token": "<pad>",
                "locally_tested": True,
            },
        },
        "split_policy": {
            "method": "sha256_group_bucket", "seed": 17,
            "fractions": {"train": 0.98, "dev": 0.01, "test": 0.01},
            "identity_links": ["document_id", "canonical_url", "normalized_text_sha256"],
            "split_isolation_checked": True, "near_duplicate_audit": "sampled",
        },
        "packing_policy": {
            "context_length": 512, "eos_between_documents": True,
            "cross_split_packing": False, "labels_pre_shifted": True,
            "padding_label_id": -100, "document_context_after_eos": True,
        },
        "actual_audit": {
            "status": "completed", "performed_at": revision_time, "documents_seen": 200,
            "missing_text_excluded": 1, "null_text_excluded": 0,
            "non_string_text_excluded": 0, "empty_text_excluded": 2,
            "duplicate_id_records": 0, "exact_duplicates_removed": 3,
            "exact_cross_split_overlaps": 0, "near_duplicate_review": "sampled",
        },
    }


def test_ready_manifest_runtime_gate_accepts_only_complete_local_evidence():
    valid = _ready_manifest()
    validate_ready_manifest(valid)

    bad_mutations = [
        ("dataset revision tag", lambda m: m["dataset"]["revision"].update(value="v1.0.0")),
        ("dataset revision provenance", lambda m: m["dataset"]["revision"].update(provenance="source_provided")),
        ("dataset shard checksum", lambda m: m["dataset"]["shards"][0].update(locally_verified_sha256=None)),
        ("dataset terms", lambda m: m["dataset"]["license"].update(local_review_status="not_reviewed")),
        ("tokenizer revision", lambda m: m["tokenizer"]["revision"].update(provenance="unverified")),
        ("tokenizer artifact checksum", lambda m: m["tokenizer"]["artifact_checksums"][0].update(locally_verified_sha256=None)),
        ("tokenizer behavior", lambda m: m["tokenizer"]["special_token_behavior"].update(locally_tested=False)),
        ("split isolation", lambda m: m["split_policy"].update(split_isolation_checked=False)),
        ("audit completion", lambda m: m["actual_audit"].update(status="in_progress")),
        ("audit count", lambda m: m["actual_audit"].update(documents_seen=None)),
        ("cross-split overlap", lambda m: m["actual_audit"].update(exact_cross_split_overlaps=1)),
        ("near duplicate review", lambda m: m["actual_audit"].update(near_duplicate_review="not_performed")),
    ]
    for label, mutate in bad_mutations:
        candidate = deepcopy(valid)
        mutate(candidate)
        with pytest.raises(ValueError):
            validate_ready_manifest(candidate)


def test_schema_has_a_ready_only_evidence_gate():
    schema_path = Path(__file__).resolve().parents[1] / "data/manifests/schema.json"
    schema = json.loads(schema_path.read_text())
    gates = schema["allOf"]
    assert len(gates) == 1
    gate = gates[0]
    assert gate["if"]["properties"]["status"]["const"] == "ready"
    then = gate["then"]["properties"]
    assert then["dataset"]["properties"]["revision"]["properties"]["provenance"]["const"] == "locally_verified"
    assert then["tokenizer"]["properties"]["special_token_behavior"]["properties"]["locally_tested"]["const"] is True
    assert then["split_policy"]["properties"]["split_isolation_checked"]["const"] is True
    assert then["actual_audit"]["properties"]["exact_cross_split_overlaps"]["const"] == 0
