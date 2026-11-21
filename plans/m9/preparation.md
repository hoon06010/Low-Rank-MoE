# M9 bounded real-data preparation — 2026-09-26

Authorized: prepare approximately 10,000 FineWeb-Edu sample-10BT documents, GPT-2 tokenizer, grouped 98/1/1 split, real-data local smoke training and resumable CPU/CUDA execution with measurement instrumentation. No rental, payment, or main experiment is authorized.

Implementation order:
1. Pin public source and tokenizer revisions; fetch bounded source data, retain byte hashes and sampling provenance; audit exact and sampled near-duplicate leakage and tokenizer behavior.
2. Retain packed split inputs and next-token labels with hashes. Validate these artifacts before measured TrainingSession construction; preserve rejection of unaudited measured inputs.
3. Add CPU/CUDA FP32 device support, used-device RNG restore, synchronized pilot timing and memory measurement. No CUDA result may be reported without CUDA execution.
4. Verify failure gates and uninterrupted versus restored training; run a tiny real-text CPU smoke, retain manifests/logs, document the bounded sample and hardware limits.

Use existing data grouping and trainer rather than replacing them. Prepared text is a setup sample, not a confirmatory dataset. Full-corpus preparation and near-duplicate completeness are out of scope. All original research plans and historical verification reports remain preserved.
