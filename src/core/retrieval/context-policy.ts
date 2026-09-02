/** Decide whether adjacent conversation turns are likely to help a query. */
export function shouldExpandConversationContext(query: string): boolean {
  const normalized = query.replace(/\s+/gu, " ").trim();
  if (!normalized) return false;

  // Binary questions should be decided from direct supporting/contradicting
  // evidence. Nearby plans or hypotheticals often flip a correct Yes/No.
  if (/^(?:did|does|do|is|are|was|were|has|have|had|can|could|would|will|should)\b/iu.test(normalized)) {
    return false;
  }
  if (/^(?:是否|有没有|是不是|能否|会不会)/u.test(normalized)) return false;

  // Counts and fixed-cardinality lists are especially vulnerable to evidence
  // from adjacent but distinct events being accidentally included.
  if (/^how\s+(?:many|much)\b/iu.test(normalized)) return false;
  if (/^(?:多少|(?:有)?几(?:个|次|件|种)?)/u.test(normalized)) return false;
  if (/\b(?:the|which|what)\s+(?:two|three|four|five)\b/iu.test(normalized)) return false;
  if (/^(?:what|which)\b.*\b(?:events|activities|places|locations|items|things|examples|names|songs|movies|books)\b/iu.test(normalized)) {
    return false;
  }
  if (/(?:哪|什么)(?:两|三|四|五)(?:个|件|次|种)/u.test(normalized)) return false;

  return true;
}

/** Prefer consensus ranking for questions with an explicitly bounded answer set. */
export function shouldPreferConsensusFusion(query: string): boolean {
  const normalized = query.replace(/\s+/gu, " ").trim();
  if (/^how\s+(?:many|much)\b/iu.test(normalized)) return true;
  if (/^(?:多少|(?:有)?几(?:个|次|件|种)?)/u.test(normalized)) return true;
  if (/\b(?:first|second|third|fourth|fifth)\b.*\b(?:first|second|third|fourth|fifth)\b/iu.test(normalized)) return true;
  if (/(?:第一|第二|第三|第四|第五).*(?:第一|第二|第三|第四|第五)/u.test(normalized)) return true;
  return false;
}
