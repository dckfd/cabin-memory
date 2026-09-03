# Release provenance

## Snapshot boundary

- Snapshot date: `2026-09-03` (`Asia/Shanghai`)
- Release label: `cockpit-zh-rc53-recovery33-harness`
- Evaluation dataset: `cockpit-zh-public-mix-500-v7`
- Memory volume: `tdai-memory-core-cockpit-zh-rc52-v7-recovery33`
- Independent Judge: `483/500` (`96.6%`)
- Human-adjusted review: `489/500` (`97.8%`)
- Memory rebuilt: `false`

## Source baselines

- Outer repository upstream: `https://github.com/TencentCloud/TencentDB-Agent-Memory.git`
- Outer branch at capture: `experiment/cockpit-memory-v13-20260824`
- Outer baseline commit: `05eaead4c38d83426d2b6981bf6378bbf16fe2eb`
- Runtime source branch at capture: `cockpit-memory-rc1-20260827`
- Runtime source baseline commit: `3aac5359b0a81060b90fb286b88dab907b8d387e`

The release contains the complete selected working-tree files, including changes made after those baseline commits. The commit created in this clean release repository is the authoritative snapshot identifier.

RC53 starts from published repository commit `64c632fd366ed212c05ac405f708418a2f2ae656`.
It reuses the frozen Recovery33 memory read-only and regenerates retrieval, answers and
independent Judge outputs. The v7 questions are the same set used during earlier debugging;
this release demonstrates reproducibility/regression behavior, not unseen-holdout generalization.

## Included

- Outer tracked source files as present in the working tree.
- Runtime `MemoryCore`, `MemoryProxy`, SDK, deployment examples and tests from `third_party/tencentdb-agent-memory-v2`.
- `benchmarks/framework_eval` source and sealed public-mix-v7 challenge.
- Recovery33 RC52 and RC53 full-500 audit artifacts.
- Portable cockpit JSONL schemas, adapter, validator and fail-closed RC runner.
- A read-only export of the stopped Recovery33 Docker volume.

## Excluded

- All source `.git` directories and original histories.
- `node_modules`, build caches, Python bytecode and virtual environments.
- Real `.env`, admin keys, proxy configuration and runtime secrets.
- Unrelated downloaded datasets, model weights and historical run directories.
- The running container image; it can be rebuilt from the included source and Docker files.

## Integrity

Run `sha256sum -c SHA256SUMS` from the repository root. The memory archive also has a local checksum file under `artifacts/memory/`.
