# RC53 stratified human review

Review policy: accept an answer when it contains every requested value and no factual
contradiction. Extra source-grounded condition text is not an error unless it changes or
invalidates the requested result. Review did not trigger regeneration or a second Judge
call.

## All 17 Judge negatives

### Judge false negatives — substantively correct (6)

- `p07-q06`, `p08-q06`, `p11-q06`, `p22-q06`, `p28-q06`, `p50-q06`: each answer gives
  distance first, rating second and rest facilities third. The additional low-battery
  threshold is consistent with the question's current battery level and does not
  contradict the reference answer.

### True answer errors (11)

- Aggregation/counting: `p02-q01`, `p05-q01`, `p07-q01`, `p10-q01`, `p11-q01`,
  `p12-q01`.
- Final appointment chain: `p07-q05`, `p08-q05`, `p43-q05`.
- Mixed changed/unchanged terminal settings: `p34-q10`, `p35-q10`.

## Stratified positive audit (20)

Five positive answers from each category were checked against the question and gold
answer. All 20 were accepted.

- Knowledge update: `p01-q02`, `p01-q05`, `p01-q08`, `p01-q10`, `p02-q02`.
- Multi-session: `p01-q01`, `p01-q04`, `p01-q07`, `p02-q04`, `p02-q07`.
- Single-session preference: `p01-q06`, `p02-q06`, `p03-q06`, `p04-q06`, `p05-q06`.
- Temporal reasoning: `p01-q03`, `p01-q09`, `p02-q03`, `p02-q09`, `p03-q03`.

## Reported metrics

- Official independent-Judge score: 483/500 (96.6%).
- Human-adjusted score: 489/500 (97.8%).
- Human review does not replace or rewrite the immutable Judge artifact; both values are
  reported to make evaluator strictness visible.
