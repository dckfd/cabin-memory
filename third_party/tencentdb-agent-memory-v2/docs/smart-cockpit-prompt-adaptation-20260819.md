# Smart-cockpit prompt adaptation

Date: 2026-08-19

Production hardening update: 2026-08-26

This change adds a configurable cockpit prompt family without changing benchmark
questions, judges, reference answers, or third-party dataset content.

## Configuration

- MemoryCore prompt mode: cockpit, chat, or code.
- Context-offload domain: smart-cockpit or generic.
- MemoryProxy recall domain: smart-cockpit or generic.
- Cockpit recall defaults to deterministic `temporalQueryMode: cockpit_v1` and
  `timezone: Asia/Shanghai`. Production callers should pass the vehicle/user
  IANA timezone per request; `TDAI_MEMORY_TIMEZONE` is the deployment fallback.
- The global-image deployment scripts default to cockpit / smart-cockpit.
  Library defaults remain chat / generic for backward compatibility.

## Adapted memory flow

1. L1 extraction reconstructs fragmented user commands from adjacent assistant
   clarification while keeping only user messages as fact sources.
2. L1 dedup compares user, occupant, vehicle, seat/zone, entity, time, and action
   state before merging.
3. L2 scene memory groups navigation, comfort, media, communication, schedule,
   charging, and parking without turning every command into a permanent profile.
4. L3 persona stores only evidence-backed stable preferences and interaction
   rules, with privacy, safety, vehicle, and occupant scope.
5. Offload L1 distinguishes requested, confirmed, executed, verified, failed,
   and cancelled tool states.
6. L1.5 treats an atomic completed command as short, but preserves multi-turn
   disambiguation, safety confirmation, interruption, and correction as stateful
   tasks.
7. L2 MMD records only observed state transitions and exact cockpit slots.
8. Recall and MMD injection explicitly mark memory as historical evidence rather
   than a new action; live vehicle state and sensitive authorization must be
   revalidated.
9. L1 dense and FTS dedup candidates now carry lifecycle/scope metadata into the
   conflict prompt. Update/merge writes preserve old metadata, apply newer
   evidence, then apply explicit merged metadata instead of silently dropping
   vehicle, occupant, seat, event-time or action-state fields.
10. Cockpit episodic metadata separates `mentioned_at` from event time, anchors
    relative expressions to the source-message timestamp, and supports
    `episode_key`, `supersedes`, `source_session_id` and evidence roles. New L1
    records start at version 1 and merged records increment the highest target
    version.
11. Recall resolves bounded relative-day phrases such as “昨天上午” and
    “前天晚上” against request-time metadata before keyword/vector search. It
    injects a small dynamic absolute-time envelope, adds no LLM call, and leaves
    retrieval-dependent phrases such as “上次那个” to L0/L1 search.
12. The compatibility `/recall` endpoint now returns dynamic L1/query-time
    evidence and stable system context separately, while its legacy `context`
    field contains both. Previously that field omitted the recalled L1 evidence.
13. Zero-config Core instances now select local keyword recall when no embedding
    provider exists. An explicitly configured hybrid strategy stays strict and
    reports its dependency error instead of silently changing behavior.
14. Proxy `recallL1: true` is now effective again. In smart-cockpit mode it
    performs deterministic selective recall only for history, preference,
    elliptical-reference, and past/current relative-time turns. Fully specified
    current commands make zero recall calls; future-only phrases receive an
    absolute clock envelope without a history lookup.
15. Selective Proxy recall injects at most Top-3 L1 records, caps each record at
    600 characters and all record bodies at 1,800 characters, and tells a weak
    main model to make one L0 fallback when L1 misses. Request metadata accepts
    `x-tdai-request-time` and `x-tdai-timezone`; both are validated and fall back
    to gateway receive time and the configured IANA timezone.
16. Cockpit dynamic L1 recall is self-agent only by default, so a triggered turn
    makes one data-plane search and cannot mix short-lived vehicle events from
    an imported agent namespace. Generic mode retains cross-agent recall.

## Generalization controls

- No dataset-specific answer or benchmark question appears in a prompt.
- Domain behavior is based on generic failure classes: fragmented slots,
  corrections, negation, ASR ambiguity, multi-candidate entities, occupant/seat
  scope, stale state, action verification, and safety authorization.
- Generic and coding prompt modes remain selectable for non-cockpit use.
- Both offload implementations import the same L1/L1.5/L2 prompt source.

## Static token audit

Using o200k_base, the six construction system prompts together changed from
9,400 to 6,309 tokens, a reduction of 3,091 tokens (32.9 percent). The detailed
breakdown is in benchmarks/results/smart-cockpit-prompt-audit-20260819.json.
L1 extraction grew by 850 tokens because it now defines short-command evidence
and scope rules; the larger L2/L3 and MMD prompts were rewritten more compactly.

Cockpit recall guidance adds 100 tokens on the Core runtime. On the Proxy path,
the memory-tools and active-recall guides add 141 and 210 tokens respectively;
these are stable system blocks designed for session-level prompt caching.
The selective L1 block is dynamic but appears only on memory-dependent turns
and is bounded independently from the stable cached prefix.

## Public cockpit benchmark snapshot

- The workspace already contains normalized KVRET and SLURP evaluation sets.
- All 254 official CAR-Bench task records are saved under
  `benchmarks/data/CAR-Bench`, covering base, disambiguation, and hallucination
  splits. The large navigation mock tables are omitted because they are not
  needed for prompt-case analysis or lightweight offline regression.
- Dataset files were JSON-validated and recorded with SHA-256 checksums. No
  external model call was used while preparing the snapshot.

## Deployment

- MemoryCore image: `agentmemory/memory-core:2.0.0-smart-cockpit-prompts-v1`
- MemoryProxy image: `agentmemory/memory-proxy:2.0.0-smart-cockpit-prompts-v1`
- Both containers report healthy after rollout.
- The deployed Core configuration uses `promptMode: cockpit`; the Proxy uses
  `domainProfile: smart-cockpit`.
- Previous image tags and the pre-adaptation source snapshot are retained for
  rollback.

## Verification

- MemoryCore: 27 tests passed (9 files).
- MemoryProxy: 23 tests passed (including live-pipeline registration, zero-call,
  time-only, Top-K budget, tenant routing, and L1-miss fallback cases).
- MemoryCore plugin build passed.
- No external model or judge call was used for prompt adaptation or regression.

The pre-change source recovery point is stored under
backups/tencentdb-agent-memory-v2-pre-cockpit-prompts-20260819.
