/** Coverage-first Reciprocal Rank Fusion for multi-query retrieval. */
export function coverageFirstRrfMerge<T extends { id: string; score: number }>(
  lists: T[][],
  limit: number,
  rrfK = 60,
  minPerList = 1,
): T[] {
  const aggregate = new Map<string, { item: T; score: number }>();
  for (const list of lists) {
    list.forEach((item, rank) => {
      const contribution = 1 / (rrfK + rank + 1);
      const current = aggregate.get(item.id);
      if (current) current.score += contribution;
      else aggregate.set(item.id, { item, score: contribution });
    });
  }

  // Reserve a bounded number of distinct results for every source/facet. This
  // prevents a broad document that ranks in every branch from starving a
  // narrow keyword hit, semantic hit, or necessary second-hop fact.
  const selected: T[] = [];
  const seen = new Set<string>();
  const floor = Math.max(0, Math.floor(minPerList));
  for (let round = 0; round < floor; round++) {
    for (const list of lists) {
      const candidate = list.find((item) => !seen.has(item.id));
      if (!candidate) continue;
      seen.add(candidate.id);
      selected.push({ ...candidate, score: aggregate.get(candidate.id)?.score ?? candidate.score });
      if (selected.length >= limit) return selected;
    }
  }

  const remaining = [...aggregate.values()]
    .filter(({ item }) => !seen.has(item.id))
    .sort((a, b) => b.score - a.score)
    .map(({ item, score }) => ({ ...item, score }));
  return selected.concat(remaining).slice(0, limit);
}
