# Sealed evaluation protocol

1. Verify `BLIND_SEAL.sha256` before ingestion and after scoring.
2. Ingest each `sample_id` into an isolated memory namespace.
3. Run every question exactly once with the frozen answer model/configuration; no retries except transport failures recorded as failures.
4. Run one independent Judge pass only after answers are immutable.
5. Perform deterministic answer-contract checks and stratified human review.
6. Never repair this directory after seeing scores. Any fix creates a new RC and a new holdout.
7. Report the full 500 research track and the 250 permissive-source track separately.
