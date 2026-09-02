/**
 * Fail-closed merge policy for the independent cockpit atomic compiler.
 *
 * The compiler may add complete atomic memories and may replace a defective
 * draft only when its complete replacements cover the draft's explicit
 * transition lineage. Existing complete drafts are never removed or mutated.
 */

import type { ExtractedMemory } from "./l1-writer.js";

export interface CockpitContractReviewMergeResult {
  memories: ExtractedMemory[];
  reviewedComplete: number;
  added: number;
  replacedDefective: number;
}

function metadataOf(memory: ExtractedMemory): Record<string, unknown> {
  return memory.metadata as Record<string, unknown>;
}

function stringValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0))];
}

function qualityStatus(memory: ExtractedMemory): string | undefined {
  const quality = metadataOf(memory).construction_quality;
  return quality && typeof quality === "object" && !Array.isArray(quality)
    ? stringValue((quality as Record<string, unknown>).status)
    : undefined;
}

function stateIdentity(memory: ExtractedMemory): string | undefined {
  const metadata = metadataOf(memory);
  const stateKey = stringValue(metadata.state_key);
  const episodeKey = stringValue(metadata.episode_key);
  if (!stateKey || !episodeKey) return undefined;
  return [
    episodeKey,
    stateKey,
    stringValue(metadata.valid_from) ?? "",
    stringValue(metadata.valid_to) ?? "",
  ].join("\u0000");
}

function hasSharedSource(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const rightIds = new Set(right.source_message_ids);
  return left.source_message_ids.some((id) => rightIds.has(id));
}

function replacementCandidates(
  defective: ExtractedMemory,
  reviewedComplete: ExtractedMemory[],
): ExtractedMemory[] {
  const metadata = metadataOf(defective);
  const episodeKey = stringValue(metadata.episode_key);
  const domain = stringValue(metadata.domain);
  const slot = stringValue(metadata.slot);
  const oldRefs = new Set(stringArray(metadata.supersedes));

  return reviewedComplete.filter((candidate) => {
    if (!hasSharedSource(defective, candidate)) return false;
    const candidateMetadata = metadataOf(candidate);
    if (episodeKey && stringValue(candidateMetadata.episode_key) === episodeKey) return true;
    if (domain && slot
      && stringValue(candidateMetadata.domain) === domain
      && stringValue(candidateMetadata.slot) === slot) return true;
    return stringArray(candidateMetadata.supersedes).some((reference) => oldRefs.has(reference));
  });
}

function replacementsCoverDefect(
  defective: ExtractedMemory,
  candidates: ExtractedMemory[],
): boolean {
  if (candidates.length === 0) return false;
  const oldRefs = stringArray(metadataOf(defective).supersedes);
  if (oldRefs.length === 0) return true;
  const coveredRefs = new Set(candidates.flatMap((candidate) =>
    stringArray(metadataOf(candidate).supersedes)
  ));
  return oldRefs.every((reference) => coveredRefs.has(reference));
}

/**
 * Merge independent compiler output without trusting it to rewrite an
 * already-complete memory. Only deterministically validated
 * `construction_quality=complete` compiler rows participate.
 */
export function mergeCockpitContractReview(
  draft: ExtractedMemory[],
  reviewed: ExtractedMemory[],
): CockpitContractReviewMergeResult {
  const reviewedComplete = reviewed.filter((memory) => qualityStatus(memory) === "complete");
  const completeDraftIdentities = new Set(
    draft
      .filter((memory) => qualityStatus(memory) === "complete")
      .map(stateIdentity)
      .filter((identity): identity is string => Boolean(identity)),
  );

  const additions: ExtractedMemory[] = [];
  const additionIdentities = new Set<string>();
  for (const memory of reviewedComplete) {
    const identity = stateIdentity(memory);
    if (!identity || completeDraftIdentities.has(identity) || additionIdentities.has(identity)) continue;
    additionIdentities.add(identity);
    additions.push(memory);
  }

  let replacedDefective = 0;
  const retainedDraft = draft.filter((memory) => {
    if (qualityStatus(memory) === "complete") return true;
    const candidates = replacementCandidates(memory, additions);
    if (!replacementsCoverDefect(memory, candidates)) return true;
    replacedDefective += 1;
    return false;
  });

  return {
    memories: [...retainedDraft, ...additions],
    reviewedComplete: reviewedComplete.length,
    added: additions.length,
    replacedDefective,
  };
}
