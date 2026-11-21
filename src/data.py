"""Deterministic document grouping, split assignment, and token packing.

This module has no dataset or tokenizer dependency. Callers provide documents
and an encoder so preparation logic can be tested with synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from itertools import islice
from urllib.parse import urlsplit, urlunsplit


SPLITS = ("train", "dev", "test")
_SPLIT_BUCKETS = 10_000
DEFAULT_MAX_DOCUMENTS = 100_000
_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace for exact duplicate comparison."""
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def canonicalize_url(url: str | None) -> str | None:
    """Normalize URL case/default ports and drop fragments for duplicate links."""
    if not url or not isinstance(url, str):
        return None
    value = url.strip()
    try:
        parts = urlsplit(value)
        if not parts.scheme or not parts.netloc:
            return value or None
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        if not host:
            return value or None
        try:
            port = parts.port
        except ValueError:
            return value or None
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        return urlunsplit((scheme, host, path, parts.query, ""))
    except ValueError:
        return value or None


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _doc_id(document: dict) -> str | None:
    value = document.get("id")
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _stable_document_key(document: dict) -> str:
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return _sha256(encoded)


def _group_split(group_hash: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_hash}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % _SPLIT_BUCKETS
    if bucket < 9_800:
        return "train"
    if bucket < 9_900:
        return "dev"
    return "test"


def prepare_documents(
    documents: Iterable[dict], *, seed: int = 17, max_documents: int = DEFAULT_MAX_DOCUMENTS
) -> tuple[list[dict], dict[str, int]]:
    """Group linked documents, assign each group a deterministic split, dedup.

    Identity edges connect equal nonempty document IDs, canonical URLs, or
    normalized-body SHA-256 values. Missing, null, non-string, and blank text
    are counted separately and excluded. At most ``max_documents`` inputs are
    materialized; larger streams fail before the remaining input is consumed.
    Exact normalized body duplicates are represented once, with a stable
    metadata winner; non-identical members of a linked component are retained.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(max_documents) is not int or max_documents <= 0:
        raise ValueError("max_documents must be a positive integer")
    raw_documents = list(islice(iter(documents), max_documents + 1))
    if len(raw_documents) > max_documents:
        raise ValueError(f"document count exceeds max_documents={max_documents}; use streaming preparation for larger sources")
    valid: list[dict] = []
    missing_text = null_text = non_string_text = blank_text = 0
    for raw in raw_documents:
        if not isinstance(raw, dict):
            raise TypeError("each document must be a mapping")
        if "text" not in raw:
            missing_text += 1
            continue
        if raw["text"] is None:
            null_text += 1
            continue
        if not isinstance(raw["text"], str):
            non_string_text += 1
            continue
        if not normalize_text(raw["text"]):
            blank_text += 1
            continue
        normalized = normalize_text(raw["text"])
        item = dict(raw)
        item["normalized_text"] = normalized
        item["text_sha256"] = _sha256(normalized)
        item["canonical_url"] = canonicalize_url(raw.get("url"))
        valid.append(item)

    union_find = _UnionFind(len(valid))
    seen: dict[tuple[str, str], int] = {}
    id_counts = Counter(doc_id for item in valid if (doc_id := _doc_id(item)) is not None)
    for index, item in enumerate(valid):
        keys = []
        doc_id = _doc_id(item)
        if doc_id is not None:
            keys.append(("id", doc_id))
        if item["canonical_url"]:
            keys.append(("url", item["canonical_url"]))
        keys.append(("text", item["text_sha256"]))
        for key in keys:
            prior = seen.setdefault(key, index)
            union_find.union(index, prior)

    components: dict[int, list[int]] = {}
    for index in range(len(valid)):
        components.setdefault(union_find.find(index), []).append(index)

    output: list[dict] = []
    duplicate_count = 0
    group_sizes: list[int] = []
    for indices in components.values():
        members = [valid[index] for index in indices]
        # The component signature is a function of its linked identities, never
        # iteration order or the union-find root chosen by insertion order.
        identity_keys = set()
        for member in members:
            identity_keys.add(f"text:{member['text_sha256']}")
            if _doc_id(member) is not None:
                identity_keys.add(f"id:{_doc_id(member)}")
            if member["canonical_url"]:
                identity_keys.add(f"url:{member['canonical_url']}")
        group_hash = _sha256("\n".join(sorted(identity_keys)))
        split = _group_split(group_hash, seed)
        group_sizes.append(len(members))

        # Collapse exact normalized text copies, while retaining different text
        # records connected through a shared ID or URL.
        by_text: dict[str, list[dict]] = {}
        for member in members:
            by_text.setdefault(member["normalized_text"], []).append(member)
        for text, copies in by_text.items():
            winner = min(copies, key=_stable_document_key)
            duplicate_count += len(copies) - 1
            record = {k: v for k, v in winner.items() if k != "normalized_text"}
            record.update(group_hash=group_hash, split=split)
            output.append(record)

    output.sort(key=lambda item: (item["split"], item["group_hash"], item["text_sha256"], _stable_document_key(item)))
    return output, {
        "input_documents": len(raw_documents),
        "missing_text": missing_text,
        "null_text": null_text,
        "non_string_text": non_string_text,
        "blank_text": blank_text,
        "exact_duplicate_documents_removed": duplicate_count,
        "duplicate_id_records": sum(count - 1 for count in id_counts.values() if count > 1),
        "retained_documents": len(output),
        "linked_components": len(components),
        "largest_component_documents": max(group_sizes, default=0),
    }


def validate_ready_manifest(manifest: dict) -> None:
    """Validate the evidence gate required before a manifest can be ``ready``.

    This pure-standard-library check deliberately covers readiness conditions
    even when a JSON Schema validator is not installed. It raises ``ValueError``
    with the first missing or unverified evidence item.
    """
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def require_time(value: Any, label: str) -> None:
        require(isinstance(value, str) and bool(value), f"{label} must be a timestamp")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
        require(parsed.tzinfo is not None, f"{label} must include a timezone")

    require(isinstance(manifest, dict), "manifest must be an object")
    require(manifest.get("status") == "ready", "manifest status must be ready")
    dataset = manifest.get("dataset")
    tokenizer = manifest.get("tokenizer")
    require(isinstance(dataset, dict), "dataset metadata is required")
    require(isinstance(tokenizer, dict), "tokenizer metadata is required")

    def check_revision(record: Any, label: str) -> None:
        require(isinstance(record, dict), f"{label} revision is required")
        value = record.get("value")
        require(isinstance(value, str) and _IMMUTABLE_REVISION.fullmatch(value) is not None,
                f"{label} revision must be an immutable commit or digest")
        require(record.get("provenance") == "locally_verified", f"{label} revision must be locally verified")
        require_time(record.get("locally_verified_at"), f"{label} revision verification time")

    check_revision(dataset.get("revision"), "dataset")
    shards = dataset.get("shards")
    require(isinstance(shards, list) and len(shards) > 0, "ready manifest requires declared dataset shards")
    for index, shard in enumerate(shards):
        label = f"dataset shard {index}"
        require(isinstance(shard, dict), f"{label} must be an object")
        require(bool(shard.get("path")), f"{label} path is required")
        require(_SHA256.fullmatch(shard.get("locally_verified_sha256") or "") is not None,
                f"{label} requires a locally verified SHA-256")
        require(shard.get("verification_status") == "match", f"{label} checksum must match")
        require_time(shard.get("locally_verified_at"), f"{label} verification time")

    license_info = dataset.get("license")
    require(isinstance(license_info, dict), "dataset license metadata is required")
    require(license_info.get("local_review_status") == "reviewed", "applicable dataset terms must be reviewed")
    require(bool(license_info.get("source_label")), "dataset license source label is required")
    for link_name in ("license_url", "terms_url"):
        link = license_info.get(link_name)
        parsed = urlsplit(link) if isinstance(link, str) else None
        require(parsed is not None and parsed.scheme in {"https", "http"} and bool(parsed.netloc),
                f"dataset {link_name} must be an HTTP(S) URL")

    check_revision(tokenizer.get("revision"), "tokenizer")
    artifacts = tokenizer.get("artifact_checksums")
    require(isinstance(artifacts, list) and len(artifacts) > 0,
            "ready manifest requires tokenizer artifacts")
    for index, artifact in enumerate(artifacts):
        require(isinstance(artifact, dict), f"tokenizer artifact {index} must be an object")
        require(bool(artifact.get("path")), f"tokenizer artifact {index} path is required")
        require(_SHA256.fullmatch(artifact.get("locally_verified_sha256") or "") is not None,
                f"tokenizer artifact {index} requires a locally verified SHA-256")
    special_tokens = tokenizer.get("special_token_behavior")
    require(isinstance(special_tokens, dict) and special_tokens.get("locally_tested") is True,
            "tokenizer special-token behavior must be tested locally")

    split_policy = manifest.get("split_policy")
    require(isinstance(split_policy, dict) and split_policy.get("split_isolation_checked") is True,
            "split isolation must be checked")
    audit = manifest.get("actual_audit")
    require(isinstance(audit, dict) and audit.get("status") == "completed",
            "actual data audit must be completed")
    require_time(audit.get("performed_at"), "actual audit time")
    for field in (
        "documents_seen", "missing_text_excluded", "null_text_excluded",
        "non_string_text_excluded", "empty_text_excluded", "duplicate_id_records",
        "exact_duplicates_removed", "exact_cross_split_overlaps",
    ):
        require(type(audit.get(field)) is int and audit[field] >= 0,
                f"actual audit requires a nonnegative {field} count")
    require(audit["exact_cross_split_overlaps"] == 0, "exact cross-split overlap count must be zero")
    require(audit.get("near_duplicate_review") in {"sampled", "completed"},
            "near-duplicate review must be sampled or completed")


def document_order(documents: Iterable[dict], *, seed: int, split: str) -> list[dict]:
    """Return stable hash order for one already assigned split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    selected = [dict(d) for d in documents if d.get("split") == split]
    return sorted(
        selected,
        key=lambda d: (
            hashlib.sha256(f"{seed}:{split}:{d['group_hash']}:{d['text_sha256']}".encode()).hexdigest(),
            d["text_sha256"],
        ),
    )


def build_split_batches(
    documents: Iterable[dict],
    *,
    split: str,
    encode: Callable[[str], Sequence[int]],
    eos_token_id: int,
    context_length: int = 512,
    pad_token_id: int = 0,
    seed: int = 17,
) -> list[dict[str, list[list[int]] | int | str]]:
    """Tokenize and pack only one split; labels are pre-shifted exactly once.

    Each document gets one EOS token. Documents may share causal context inside
    the split, with EOS marking boundaries. A label can use the following token
    just beyond the context block. The final stream token has no next target.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if type(eos_token_id) is not int or type(pad_token_id) is not int:
        raise ValueError("EOS and padding token IDs must be integers")

    stream: list[int] = []
    ordered = document_order(documents, seed=seed, split=split)
    for document in ordered:
        tokens = encode(document.get("text", ""))
        if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
            raise TypeError("encode(text) must return a sequence of token IDs")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("token IDs must be nonnegative integers")
        stream.extend(tokens)
        stream.append(eos_token_id)

    batches: list[dict[str, list[list[int]] | int | str]] = []
    for start in range(0, len(stream), context_length):
        inputs = stream[start : start + context_length]
        next_tokens = stream[start + 1 : start + context_length + 1]
        real_length = len(inputs)
        labels = next_tokens + [-100] * (context_length - len(next_tokens))
        input_ids = inputs + [pad_token_id] * (context_length - real_length)
        attention_mask = [1] * real_length + [0] * (context_length - real_length)
        # Labels at padded input positions and positions without a next token
        # are ignored. For a short final block, every non-final real position
        # still has its next token; the final stream token is masked.
        labels = labels[:context_length]
        valid_target_count = sum(label != -100 for label in labels)
        batches.append({
            "input_ids": [input_ids],
            "labels": [labels],
            "attention_mask": [attention_mask],
            "valid_target_count": valid_target_count,
            "split": split,
        })
    return batches


def resume_batches(
    batches: Sequence[dict], *, cursor: int = 0
) -> Iterator[tuple[int, dict]]:
    """Yield batches beginning at a stable zero-based batch cursor."""
    if type(cursor) is not int or cursor < 0 or cursor > len(batches):
        raise ValueError("cursor must be between zero and the number of batches")
    yield from enumerate(batches[cursor:], start=cursor)


def sample_near_duplicates(documents: Sequence[dict], *, sample_per_split: int = 128, threshold: float = .8) -> dict:
    """Deterministic bounded cross-split 5-word shingle review, not exhaustive."""
    if sample_per_split <= 0 or not 0 < threshold <= 1:
        raise ValueError('invalid near-duplicate audit parameters')
    samples = {}
    for split in SPLITS:
        selected = sorted((d for d in documents if d['split'] == split), key=lambda d: d['text_sha256'])[:sample_per_split]
        samples[split] = []
        for doc in selected:
            words = normalize_text(doc['text']).lower().split()[:4096]
            shingles = {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}
            if not shingles:
                shingles = {tuple(words)}
            samples[split].append((doc['text_sha256'], shingles))
    matches, checked = [], 0
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:
            for a, x in samples[left]:
                for b, y in samples[right]:
                    checked += 1
                    score = len(x & y) / len(x | y)
                    if score >= threshold:
                        matches.append({'left_split':left,'right_split':right,'left_hash':a,'right_hash':b,'jaccard':score})
    return {'status':'sampled','method':'normalized_lowercase_5_word_shingle_jaccard',
            'selection':'lexicographically smallest normalized-text SHA256 per split',
            'sample_per_split':sample_per_split,'max_words_per_document':4096,
            'sample_counts':{s:len(v) for s,v in samples.items()},'threshold':threshold,
            'pairs_checked':checked,'matches':matches,'exhaustive':False}


def load_audited_bundle(path) -> dict:
    """Rehash retained source/tokenizer and packed labels before real execution.

    Bundle hashes establish reproducibility and corruption detection; they are
    not authentication against an adversary who can rewrite all local files.
    """
    from pathlib import Path
    path = Path(path).resolve()
    raw = path.read_bytes()
    spec = json.loads(raw)
    if spec.get('schema_version') != 1 or spec.get('purpose') not in {'pipeline_smoke_only', 'mini_exploration_only'}:
        raise ValueError('unsupported bundle version or purpose')
    manifest = spec['manifest']
    validate_ready_manifest(manifest)

    def verify(relative, expected):
        artifact = (path.parent / relative).resolve()
        if not artifact.is_file():
            raise ValueError(f'missing retained artifact: {relative}')
        with artifact.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'artifact checksum mismatch: {relative}')
        return artifact

    for shard in manifest['dataset']['shards']:
        verify(shard['path'], shard['locally_verified_sha256'])
    for artifact in manifest['tokenizer']['artifact_checksums']:
        verify(artifact['path'], artifact['locally_verified_sha256'])
    for artifact in spec.get('audit_artifacts', []):
        verify(artifact['path'], artifact['sha256'])
    batches = {}
    for split, record in spec['splits'].items():
        if split not in SPLITS:
            raise ValueError('unknown split')
        rows = json.loads(verify(record['path'], record['sha256']).read_text())
        canonical = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != record['content_hash']:
            raise ValueError('packed content hash mismatch')
        total = 0
        for row in rows:
            if row.get('split') != split:
                raise ValueError('packed split mismatch')
            inputs, labels, masks = (row[k] for k in ('input_ids','labels','attention_mask'))
            if len(inputs) != len(labels) or len(inputs) != len(masks) or not inputs:
                raise ValueError('invalid packed batch shape')
            count = 0
            for x,y,m in zip(inputs, labels, masks):
                if len(x) != spec['context_length'] or len(y) != len(x) or len(m) != len(x):
                    raise ValueError('invalid packed row shape')
                if any(type(v) is not int or not 0 <= v < spec['vocab_size'] for v in x):
                    raise ValueError('input token outside vocabulary')
                for j,(label,mask) in enumerate(zip(y,m)):
                    if mask not in (0,1) or (not mask and label != -100):
                        raise ValueError('invalid packed padding')
                    if type(label) is not int or (label != -100 and not 0 <= label < spec['vocab_size']):
                        raise ValueError('label outside vocabulary')
                    if label != -100 and j+1 < len(x) and m[j+1] and label != x[j+1]:
                        raise ValueError('labels are not next-token targets')
                    count += label != -100
            if count != row['valid_target_count']:
                raise ValueError('retained target count mismatch')
            total += count
        if total <= 0 or total != record['valid_target_tokens']:
            raise ValueError('retained split target count mismatch')
        batches[split] = rows
    if not {'train','dev'}.issubset(batches):
        raise ValueError('retained training and evaluation targets are required')
    return {'spec':spec,'bundle_hash':hashlib.sha256(raw).hexdigest(),'batches':batches}


def pack_token_prefix(tokens: Iterable[int], *, split: str, context_length: int,
                      target_limit: int, pad_token_id: int) -> list[dict]:
    """Consume at most target_limit+1 tokens; retain the lookahead target."""
    if split not in SPLITS or context_length <= 0 or target_limit <= 0:
        raise ValueError('invalid bounded packing settings')
    stream = list(islice(tokens, target_limit + 1))
    if any(type(t) is not int or t < 0 for t in stream):
        raise ValueError('invalid token ID')
    rows = []
    for start in range(0, max(0,len(stream)-1), context_length):
        x = stream[start:min(start+context_length,len(stream)-1)]
        y = stream[start+1:start+len(x)+1]
        real = len(x)
        rows.append({'input_ids':[x+[pad_token_id]*(context_length-real)],
                     'labels':[y+[-100]*(context_length-real)],
                     'attention_mask':[[1]*real+[0]*(context_length-real)],
                     'valid_target_count':real,'split':split})
    return rows
