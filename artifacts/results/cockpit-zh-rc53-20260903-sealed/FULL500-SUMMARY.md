# Cockpit Chinese RC53 — sealed full-500 result

- Run date: 2026-09-03 (Asia/Shanghai)
- Dataset: `cockpit-zh-public-mix-500-v7` (same frozen 500-question set as Recovery33)
- Memory: frozen Recovery33 L0/L1/L2/L3 reused read-only; no seed or rebuild
- Retrieval/answer code: RC53 repository tag is the authoritative source identity
- Answer model: DeepSeek V4 Flash
- Independent Judge: DeepSeek V4 Pro, one pass, no retry of negative labels

## Result

| Metric | Result |
| --- | ---: |
| Judge accuracy | **96.6% (483/500)** |
| Judge call errors | **0** |
| Empty answers | **0** |
| Temporal reasoning | **100/100 (100%)** |
| Knowledge update | **195/200 (97.5%)** |
| Multi-session | **144/150 (96%)** |
| Single-session preference | **44/50 (88%)** |

Compared with the frozen Recovery33 result (446/500, 89.2%), the fresh run is +37
questions and +7.4 percentage points. At the per-question label level, 42 previous
negatives became positive and five previous positives became negative. Model and
Judge calls are nondeterministic, so this comparison is not a paired deterministic
proof of every change.

## Route and evidence audit

- Structured deterministic answers: 242/500, with 242/242 Judge positives.
- DeepSeek Flash answers: 258/500, with 241/258 Judge positives.
- New owner/event-chain retrieval short circuit: 100 questions, 100/100 Judge positives.
- Gold evidence-ID complete retrieval: 474/500. Evidence IDs were used only after the
  run for this audit and were never consumed by retrieval or answering.
- Every prediction records `uses_gold_or_evidence_ids=false`.
- All retrieval, prediction and Judge files contain 500 unique, aligned question IDs.

## Human review

All 17 Judge negatives and 20 stratified Judge positives were manually reviewed. Six
negative labels are conservative Judge false negatives: the answer gave the requested
`distance > rating > rest facilities` order and also stated the applicable low-battery
threshold. The answer is substantively correct and non-contradictory. The remaining 11
are genuine answer errors. Human-adjusted accuracy is therefore **489/500 (97.8%)**.

True residual errors:

- Completed-navigation aggregation/counting: 6.
- Final vehicle-inspection appointment chain: 3.
- Multi-field terminal settings with mixed changed/unchanged values: 2.

See `MANUAL-REVIEW.md` for question IDs and review policy.

## Interpretation boundary

This is a fresh retrieval, answer and independent-Judge run, but it uses the same v7
question set that informed earlier debugging. It validates artifact reproducibility and
regression behavior; it is not an unseen generalization estimate. A new untouched
Chinese cockpit holdout remains required before a production launch claim.

## Release verification

- Python structured retrieval/answer tests: 76 passed.
- Outer Node.js 22 tests: 96 passed across 10 files.
- Outer TypeScript/plugin build: passed.
- Dataset contract validation: 50 conversations, 500 questions, zero errors.
- Result-directory and repository SHA256 verification: passed.
- Published-file credential scan: no API key found.
