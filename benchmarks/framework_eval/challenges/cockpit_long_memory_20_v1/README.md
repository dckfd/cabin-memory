# Cockpit Long Memory Challenge 20 (V1)

This is a small, original diagnostic set for testing whether an in-car memory
agent can do more than recover one explicit slot.  It is not an official score
for CarMem, LongMemEval, KVRET, or SLURP.

## Scope

- 4 isolated driver histories;
- 44 timestamped sessions and 97 messages;
- 20 questions in Chinese and English;
- 6 knowledge-update, 9 multi-session, 3 temporal-reasoning, and 2
  preference questions;
- 4 of the 20 questions are deliberately unanswerable;
- every answerable question has message-level evidence IDs.

The set covers state replacement, cancellation/rescheduling, interval-valid
aliases, conditional preferences, multi-speaker constraints, frequency
aggregation, personalized recommendation, and abstention.  The questions are
designed to miss the current deterministic cockpit slot compiler, so a normal
run should exercise the configured answer model rather than report another
zero-model slot score.

## Provenance

All conversations, names, entities, questions, and answers in this directory
were written specifically for this repository.  No source utterance was copied
from another dataset.

The design taxonomy was informed by:

- [CarMem](https://github.com/johanneskirmayr/CarMem): in-car preference
  extraction, maintenance, negation, and retrieval;
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval): multi-session
  reasoning, knowledge updates, temporal reasoning, preference use, and
  abstention.

The public CarMem repository was inspected at commit
`a08f8affdecad0ed092b33e9bf68e4e1928be36e`.  It did not contain an explicit
license file at the time of inspection, which is another reason this challenge
uses newly written content rather than redistributing CarMem rows.

## Files and regeneration

- `conversations.jsonl`: LongMemEval-compatible conversation histories;
- `questions.jsonl`: questions, reference answers, timestamps, and evidence;
- `manifest.json`: immutable dataset summary and source inspirations.
- `selection.json`: all 20 frozen QA IDs for the common evaluation runner.

Regenerate deterministically from the checked-in source definition:

```bash
python3 -m \
  benchmarks.framework_eval.datasets.prepare_cockpit_long_memory_challenge
```

The existing `longmemeval` dataset adapter and official task prompts can read
this directory via `--dataset-root`; no existing benchmark data, prompt, or
Judge implementation needs to be changed.

## Interpretation

Twenty curated questions are sufficient for a fast failure diagnostic, not for
a publication-grade accuracy claim.  Report each question, retrieval evidence,
answer route, model usage, and Judge decision.  Do not merge these numbers with
official LongMemEval or CarMem leaderboards.
