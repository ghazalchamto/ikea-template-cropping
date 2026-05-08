# QA Tests

Dedicated test cases for the production-label QA workflow.

These tests focus on the end-to-end contract that QA testers rely on:

1. Uploaded labels are cached server-side (`file_hash`).
2. Extractor state can be restored from cache after refresh.
3. Validation can run by `file_hash` without re-uploading bytes.

## Run

From repo root:

```bash
.venv/bin/python -m pytest tests/qa -v
```

Or run only the hash-based validation contract:

```bash
.venv/bin/python -m pytest tests/qa/test_validation_hash_flow.py -v
```

