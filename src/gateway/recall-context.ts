/** Compose the standalone recall payload without dropping dynamic L1 evidence. */
export function composeRecallContext(parts: {
  dynamicL1?: string;
  l0?: string;
  stable?: string;
}): string {
  return [
    parts.dynamicL1 ? `## Retrieved structured memory\n${parts.dynamicL1}` : "",
    parts.l0 ? `## Retrieved conversation evidence\n${parts.l0}` : "",
    parts.stable ? `## Stable memory context\n${parts.stable}` : "",
  ].filter(Boolean).join("\n\n");
}
