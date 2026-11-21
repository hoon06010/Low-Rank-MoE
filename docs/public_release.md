# Public release scope

This release contains research code, configuration, numerical results and selected technical documents.
Personal paths, Drive file/folder identifiers, private service URLs, cloud project identifiers, operator approval diaries and personal storage records were removed.
Operational histories and original source files remain in a separate private archive and are not part of this repository's Git history.
Historical drive_runner.py is an archival example: its private review identifiers are redacted and its example operator is not a production account.
deploy_private.py requires GOOGLE_DRIVE_FOLDER_ID for uploads. GOOGLE_DRIVE_QUOTA_PROJECT may be supplied by the caller; personal project defaults are removed.
No credentials, data shards, checkpoints or virtual environments are distributed.
IMPORT_MANIFEST.json lists only public file paths, byte counts and SHA-256 hashes. It does not contain Drive IDs or hashes of private source versions.
Training results were preserved. The original GPU experiments were not rerun for this release.
