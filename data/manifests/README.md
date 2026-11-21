# Data manifest and audit boundary

`schema.json` defines the dataset, tokenizer, split, packing, and audit fields. A manifest records source declarations separately from local verification. A tag or commit reported by a hosting service is not a locally verified revision until the exact resolved artifact is fetched and checked. A hosting pointer checksum belongs in `source_declared_sha256`; the file digest computed from downloaded bytes belongs in `locally_verified_sha256`.

## Current candidate metadata

The current candidate is the English FineWeb-Edu `sample-10BT` configuration. The task's read-only source metadata review identified tag `v1.0.0`, resolved commit `84e8104e779e409e2267ac60609138e3dda2cbd2`, an ODC-BY v1.0 label alongside Common Crawl terms, a source-provided train split, 14 shards, and an approximate total size of 28.5 GB. The relevant source pages are the [configuration file listing](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/v1.0.0/sample/10BT) and [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/v1.0.0/README.md).

One hosting pointer lists SHA-256 `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871` for the first shard. This is source-provided metadata only; the shard was not downloaded and its local bytes were not checked. The dataset has not been downloaded, and no document-level audit, split construction, near-duplicate review, or token-count check has been run. The source's train split is the input pool for the proposed local 98/1/1 train/dev/test grouping; it is not evidence that the local splits have already been built.

The tokenizer candidate is [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e) at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. That revision is a source-provided candidate. Tokenizer artifacts/checksums, special-token behavior, and compatibility with the final data path have not been locally verified.

These notes describe candidate metadata, not a completed data audit or an approved dataset manifest. `status: ready` is gated in `schema.json`: dataset and tokenizer revisions must be immutable commit/digest values marked locally verified; every declared shard and tokenizer artifact needs a locally verified checksum; applicable terms must be reviewed; tokenizer special-token behavior and split isolation must be checked; and the completed audit must provide counts with zero exact cross-split overlaps. `src.data.validate_ready_manifest` applies the same readiness checks using only the Python standard library. Record failed or incomplete checks explicitly. Do not represent unknown document counts, overlap counts, or checksums as zero.

## Bounded preparation

`prepare_documents` defaults to `max_documents=100000`. It reads at most one record beyond that limit before rejecting the input, so an oversized iterable is not fully materialized accidentally. Missing text, null text, other non-string text, and blank/whitespace-only text have separate counters (`missing_text`, `null_text`, `non_string_text`, `blank_text`). Exact duplicate bodies are counted separately. The manifest's actual-audit fields use the corresponding excluded-record counts and must remain null until measured.

The full `sample-10BT` source is approximately 28.5 GB across 14 shards. Running connected-component grouping across it needs a streaming or external-sort preparation path and a deliberate resource plan; raising the in-memory document limit does not make full-dataset preparation safe. M4 implements and tests bounded toy/audit samples only.

## Preparation contract

- `src.data.prepare_documents` normalizes text with Unicode NFKC and collapsed whitespace, links records that share a nonempty document ID, canonical URL, or normalized-text SHA-256, then assigns the connected component as a unit using a seeded SHA-256 bucket. Exact normalized-text copies are reduced to one stable representative; unusable body categories are counted separately. Repeated IDs are reported as `duplicate_id_records` and link the records into one group; they are retained for review unless exact text deduplication removes a copy.
- Hash buckets target 98% train, 1% dev, and 1% test in expectation. Small corpora can have different realized proportions. Split membership is independent of input iteration order.
- Tokenize each split separately. `build_split_batches` appends EOS after each retained document and never concatenates across splits. Documents may share causal context after EOS, as specified by the experiment plan.
- Each returned batch has shape `[B,T]` (the current builder emits one packed row per batch). Labels are already next-token targets at the matching input position. The final stream position with no next token and every padded position receive `-100`; `attention_mask` is one for real input positions and zero for padding. `valid_target_count` is the count of labels other than `-100`.
- `resume_batches(..., cursor=i)` resumes at batch index `i` in the deterministic packed sequence. Persist the split, seed, source/data identity, tokenizer identity, and cursor with training state; a cursor alone does not identify the data stream.

Exact duplicate grouping prevents the same normalized body from being assigned across splits. It does not find near duplicates, establish document quality, validate licenses, or prove absence of every form of evaluation leakage. Those audits must be recorded separately before real-data training.
