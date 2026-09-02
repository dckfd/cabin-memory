# Recovery33 memory snapshot

`recovery33-memory.tar.gz` is a deterministic, read-only export of Docker volume `tdai-memory-core-cockpit-zh-rc52-v7-recovery33` after the final evaluation.

The snapshot is approximately 29 MB before compression and includes the vector database, conversations, records, metadata databases, profiles and checkpoint metadata. It contains benchmark-derived memory only and inherits the full dataset's research/non-commercial restrictions.

Verify it with:

```bash
(cd artifacts/memory && sha256sum -c SHA256SUMS)
```

Restore it without overwriting an existing volume with:

```bash
./scripts/release/import-recovery33-memory.sh
```
