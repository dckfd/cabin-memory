import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { prepareCockpitQuery } from "../../third_party/tencentdb-agent-memory-v2/MemoryProxy/src/injection/cockpit-query.js";
import { TdaiL1RecallInjector } from "../../third_party/tencentdb-agent-memory-v2/MemoryProxy/src/injection/injectors/tdai-l1-recall-injector.js";
import type { AgentContext } from "../../third_party/tencentdb-agent-memory-v2/MemoryProxy/src/injection/types.js";
import type { TdaiMemoryConfig } from "../../third_party/tencentdb-agent-memory-v2/MemoryProxy/src/tdai/types.js";

type Question = {
  question_id: string;
  conversation_id: string;
  text: string;
  evidence_ids: string[];
  metadata?: { question_date?: string; is_abstention?: boolean };
};
type Isolation = {
  team_id: string;
  user_id: string;
  service_id: string;
  conversations: Record<string, { agent_id: string; task_id: string }>;
};
type Variant = { name: string; text: string };

function argument(name: string, fallback = ""): string {
  const index = process.argv.indexOf(name);
  return index >= 0 ? String(process.argv[index + 1] ?? "") : fallback;
}

const sourcePath = resolve(argument("--source"));
const isolationPath = resolve(argument("--isolation"));
const outputPath = resolve(argument("--output"));
const core = argument("--core", "http://127.0.0.1:18421").replace(/\/+$/, "");
const concurrency = Number(argument("--concurrency", "4"));
if (!sourcePath || !isolationPath || !outputPath) {
  throw new Error("--source, --isolation, and --output are required");
}
if (existsSync(outputPath)) throw new Error(`refusing to overwrite ${outputPath}`);

const questions = readFileSync(sourcePath, "utf8").split("\n").filter(Boolean)
  .map((line) => (JSON.parse(line) as { question: Question }).question);
const isolation = JSON.parse(readFileSync(isolationPath, "utf8")) as Isolation;
if (questions.length !== 20) throw new Error(`expected 20 frozen questions, found ${questions.length}`);

const config: TdaiMemoryConfig = {
  enabled: true,
  endpoint: core,
  apiKey: "local-benchmark-bearer",
  serviceId: isolation.service_id,
  writeL0: false,
  recallL1: true,
  injectL2L3: false,
  l1Limit: 8,
  l2Limit: 3,
  timeoutMs: 15_000,
};
const recall = new TdaiL1RecallInjector(config, null, 8, 6, null, {
  domainProfile: "smart-cockpit",
  timezone: "Asia/Shanghai",
  maxCharsPerMemory: 900,
  maxTotalChars: 3_600,
});

function variants(text: string): Variant[] {
  const cjk = /[\u3400-\u9fff]/u.test(text);
  const stripped = text.replace(/[，。！？、,.!?]/gu, " ").replace(/\s+/gu, " ").trim();
  const spaced = text.replace(/([，。！？、,.!?])/gu, " $1 ").replace(/\s+/gu, " ").trim();
  const values: Variant[] = cjk ? [
    { name: "original", text },
    { name: "polite_prefix", text: `请帮我回忆一下，${text}` },
    { name: "asr_filler", text: `嗯，那个，我想问一下，${text}` },
    { name: "quoted", text: `“${text}”` },
    { name: "concise_suffix", text: `${text} 请直接给出结论。` },
    { name: "punctuation_spaced", text: spaced },
    { name: "punctuation_removed", text: stripped },
    { name: "memory_prefix", text: `根据之前车内对话，${text}` },
    { name: "spoken_wrapper", text: `麻烦查一下以前的记录，我的问题是：${text}` },
    { name: "asr_pause", text: text.replace(/[，。！？]/gu, " 呃 ").trim() },
  ] : [
    { name: "original", text },
    { name: "polite_prefix", text: `Please recall this for me: ${text}` },
    { name: "asr_filler", text: `Um, well, I wanted to ask: ${text}` },
    { name: "quoted", text: `\"${text}\"` },
    { name: "concise_suffix", text: `${text} Give only the conclusion.` },
    { name: "punctuation_spaced", text: spaced },
    { name: "punctuation_removed", text: stripped },
    { name: "memory_prefix", text: `Based on our previous in-car conversations, ${text}` },
    { name: "spoken_wrapper", text: `Could you check the earlier records? My question is: ${text}` },
    { name: "case_normalized", text: text.toLocaleLowerCase("en-US") },
  ];
  if (new Set(values.map((item) => item.text)).size !== 10) {
    throw new Error(`variant collision for: ${text}`);
  }
  return values;
}

function context(question: Question, variant: Variant): AgentContext {
  const scope = isolation.conversations[question.conversation_id];
  if (!scope) throw new Error(`missing isolation scope for ${question.conversation_id}`);
  return {
    messages: [{ role: "user", blocks: [{ type: "text", content: variant.text }] }],
    requestParams: {},
    metadata: {
      protocol: "openai",
      traceId: `robustness:${question.question_id}:${variant.name}`,
      keyId: "robustness-no-user-secret",
      modelId: "recall-only",
      stream: false,
      agentSource: "cockpit",
      requestTime: question.metadata?.question_date,
      timezone: "Asia/Shanghai",
      custom: { session: {
        team_id: isolation.team_id,
        user_id: isolation.user_id,
        agent_id: scope.agent_id,
        task_id: scope.task_id,
        session_id: `robustness-${question.question_id.replace(/[^a-z0-9]+/gi, "-")}-${variant.name}`,
        space_id: isolation.service_id,
      } },
    },
  };
}

function citedEvidenceIds(text: string): string[] {
  return [...new Set(text.match(/cockpit-hard-(?:cn|en)-\d{2}-s\d{2}:\d{3}/g) ?? [])];
}

const cases = questions.flatMap((question) => variants(question.text).map((variant) => ({ question, variant })));
const results = new Array<Record<string, unknown>>(cases.length);
let cursor = 0;
const wallStarted = performance.now();
await Promise.all(Array.from({ length: concurrency }, async () => {
  while (true) {
    const index = cursor++;
    if (index >= cases.length) return;
    const { question, variant } = cases[index];
    const started = performance.now();
    try {
      const prepared = prepareCockpitQuery(variant.text, {
        requestTime: question.metadata?.question_date,
        timezone: "Asia/Shanghai",
      });
      const blocks = await recall.execute(context(question, variant));
      const injected = blocks.map((block) => block.content).join("\n");
      const metadata = blocks[0]?.metadata ?? {};
      const ids = citedEvidenceIds(injected);
      const expected = question.evidence_ids ?? [];
      const ratio = expected.length === 0
        ? 1
        : expected.filter((id) => ids.includes(id)).length / expected.length;
      results[index] = {
        qa_id: question.question_id,
        variant: variant.name,
        query: variant.text,
        route_reasons: prepared.reasons,
        should_search_memory: prepared.shouldSearchMemory,
        mode: metadata.mode ?? null,
        evidence_sufficient: metadata.evidenceSufficient ?? null,
        evidence_status_reasons: metadata.evidenceStatusReasons ?? [],
        retrieval_attempts: metadata.retrievalAttempts ?? 0,
        retrieval_errors: metadata.retrievalErrors ?? 0,
        supplementary_retrieval_attempts: metadata.supplementaryRetrievalAttempts ?? 0,
        supplementary_retrieval_errors: metadata.supplementaryRetrievalErrors ?? 0,
        expected_evidence_ids: expected,
        final_evidence_ids: ids,
        final_evidence_recall: ratio,
        injected_chars: injected.length,
        latency_ms: Math.round((performance.now() - started) * 10) / 10,
        error: null,
      };
    } catch (error) {
      results[index] = {
        qa_id: question.question_id,
        variant: variant.name,
        query: variant.text,
        latency_ms: Math.round((performance.now() - started) * 10) / 10,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
}));

const successful = results.filter((row) => row.error === null);
const answerable = successful.filter((row) => (row.expected_evidence_ids as string[]).length > 0);
const ratios = answerable.map((row) => Number(row.final_evidence_recall));
const baseRisk = new Map(questions.map((question) => [
  question.question_id,
  prepareCockpitQuery(question.text, {
    requestTime: question.metadata?.question_date,
    timezone: "Asia/Shanghai",
  }).reasons,
]));
for (const row of successful) {
  const expectedReasons = baseRisk.get(String(row.qa_id)) ?? [];
  const actualReasons = row.route_reasons as string[];
  row.route_preserved = expectedReasons.every((reason) => actualReasons.includes(reason));
}
const report = {
  schema_version: 1,
  protocol: "cockpit-hard20-semantic-preserving-query-robustness-v1",
  scope: "200 deterministic text/ASR-like query variants; recall only; no answer model; no user credential",
  base_questions: questions.length,
  variants_per_question: 10,
  count: results.length,
  completed: successful.length,
  error_count: results.length - successful.length,
  answerable_case_count: answerable.length,
  exact_evidence_recall_count: answerable.filter((row) => row.final_evidence_recall === 1).length,
  mean_evidence_recall: ratios.reduce((sum, value) => sum + value, 0) / Math.max(ratios.length, 1),
  route_preserved_count: successful.filter((row) => row.route_preserved === true).length,
  wall_ms: Math.round((performance.now() - wallStarted) * 10) / 10,
  credential_files_read: false,
  results,
};
writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
console.log(JSON.stringify({
  completed: `${report.completed}/${report.count}`,
  exact_evidence_recall: `${report.exact_evidence_recall_count}/${report.answerable_case_count}`,
  mean_evidence_recall: report.mean_evidence_recall,
  route_preserved: `${report.route_preserved_count}/${report.count}`,
  wall_ms: report.wall_ms,
}, null, 2));
