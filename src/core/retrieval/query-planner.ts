/**
 * Deterministic, model-free query decomposition for multi-evidence retrieval.
 *
 * The planner deliberately uses no dataset vocabulary and never generates
 * facts. It preserves the original query and adds a small number of clauses
 * that can retrieve the separate events needed by comparison, ordering,
 * duration, cause/effect, and other multi-hop questions.
 */

const SPLIT_RE = /\s+(?:and|or|then|while|whereas|before|after|because|but)\s+|[;；]|(?:以及|并且|然后|而且|但是|之前|之后|因为|同时)/giu;
const WRAPPER_RE = /^(?:please\s+)?(?:tell\s+me\s+|do\s+you\s+remember\s+|can\s+you\s+recall\s+)?/iu;

export interface RetrievalQueryPlan {
  original: string;
  queries: string[];
  decomposed: boolean;
}

function normalizeQuery(value: string): string {
  return value.replace(/\s+/gu, " ").trim();
}

function usefulClause(value: string): boolean {
  const compact = value.replace(/[^\p{L}\p{N}]/gu, "");
  return compact.length >= 3;
}

/** Return at most `maxQueries` stable, de-duplicated retrieval queries. */
export function planRetrievalQueries(query: string, maxQueries = 4): RetrievalQueryPlan {
  const original = normalizeQuery(query);
  if (!original || maxQueries <= 1) {
    return { original, queries: original ? [original] : [], decomposed: false };
  }

  const candidates: string[] = [original];
  const unwrapped = normalizeQuery(original.replace(WRAPPER_RE, ""));
  if (unwrapped && unwrapped !== original) candidates.push(unwrapped);

  // `between A and B` is a common generic two-event form. Retain both sides
  // rather than letting the ordinary `and` split discard the relation anchor.
  const between = unwrapped.match(/\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)/iu);
  if (between) {
    candidates.push(between[1], between[2]);
  }

  for (const clause of unwrapped.split(SPLIT_RE)) {
    const cleaned = normalizeQuery(clause.replace(/^[,，:：?.!]+|[,，:：?.!]+$/gu, ""));
    if (usefulClause(cleaned)) candidates.push(cleaned);
  }

  const seen = new Set<string>();
  const queries: string[] = [];
  for (const candidate of candidates) {
    const normalized = normalizeQuery(candidate);
    const key = normalized.replace(/[,，:：;；?.!]+$/gu, "").toLocaleLowerCase();
    if (!usefulClause(normalized) || seen.has(key)) continue;
    seen.add(key);
    queries.push(normalized);
    if (queries.length >= Math.max(1, maxQueries)) break;
  }

  return { original, queries, decomposed: queries.length > 1 };
}
