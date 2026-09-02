from collections import Counter
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest import mock

from benchmarks.framework_eval.adapters.base import MemoryAdapter, UnsupportedCapabilityError
from benchmarks.framework_eval.adapters.bm25 import BM25Adapter
from benchmarks.framework_eval.adapters.full_context import FullContextAdapter
from benchmarks.framework_eval.adapters.tencentdb_http import (
    TencentDBHTTPAdapter,
    _annotate_v3_relative_time,
    _boost_v3_explicit_source_dates,
    _format_v3_content,
    _focus_tokens,
    _humanize_v3_source_time,
    _is_temporal_question,
    _rank_v3_ledger_facts,
    _resolved_v3_relative_times,
    _scoped_v3_session_id,
    _should_focus_question,
    _v3_message_body,
    _v3_source_role,
    _v3_source_ids,
    _v3_source_timestamp,
    _v3_transport_role,
)
from benchmarks.framework_eval.tencentdb_ledger import (
    _deterministic_pair_event_facts,
    _event_rollup_input,
    _json_array,
    _relation_rollup_input,
    _session_prompt,
    _validate_event_rollup_facts,
    _validate_relation_rollup_facts,
    _validate_rollup_facts,
    _validate_session_facts,
)
from benchmarks.framework_eval.answering import AnswerConfig, OpenAIAnswerer
from benchmarks.framework_eval.bridges.hindsight_bridge import (
    _missing_pending_rows,
    _operation_groups,
    _session_item,
)
from benchmarks.framework_eval.datasets.locomo_refined import LoCoMoRefinedDataset
from benchmarks.framework_eval.datasets.longmemeval import LongMemEvalDataset
from benchmarks.framework_eval.datasets.kvret_memory import KVRETMemoryDataset
from benchmarks.framework_eval.datasets.slurp_memory import SLURPMemoryDataset
from benchmarks.framework_eval.judges.base import DatasetJudge, JudgeConfig
from benchmarks.framework_eval.judges.kvret_memory import slot_value_matches
from benchmarks.framework_eval.judges.slurp_memory import SLURPMemoryJudge
from benchmarks.framework_eval.slurp_extract_answer import extract_fragment_reply
from benchmarks.framework_eval.judges.longmemeval import (
    OpenAIJudgeLLM,
    official_prompt,
    parse_yes_no,
)
from benchmarks.framework_eval.native_runner import NativeAnswerRunner
from benchmarks.framework_eval.plugins import PluginCatalog
from benchmarks.framework_eval.registry import load_registry
from benchmarks.framework_eval.runner import RetrievalRunner
from benchmarks.framework_eval.planner import build_plan, load_profiles
from benchmarks.framework_eval.schema import (
    ContentPart,
    MemoryAnswer,
    MemoryHit,
    Message,
    Question,
    Session,
)
from benchmarks.framework_eval.validator import validate_retrieval
from benchmarks.framework_eval.sources import verify_locked_sources


ROOT = Path(__file__).resolve().parents[2]


class FrameworkEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = LoCoMoRefinedDataset(ROOT / "LoCoMo_refined")

    def test_dataset_preserves_sessions_and_evidence(self):
        conversation = self.dataset.conversation("conv-26")
        self.assertEqual(19, len(conversation.sessions))
        self.assertEqual("D1:1", conversation.sessions[0].messages[0].message_id)
        question = self.dataset.questions({"conv-26"})[0]
        self.assertEqual(("D1:3",), question.evidence_ids)

    def test_full_context_contains_gold_source(self):
        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        adapter = FullContextAdapter()
        adapter.ingest(conversation)
        hits = adapter.search(question, limit=8)
        self.assertTrue(any("D1:3" in hit.source_ids for hit in hits))

    def test_bm25_returns_ranked_hits(self):
        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        adapter = BM25Adapter()
        adapter.ingest(conversation)
        hits = adapter.search(question, limit=8)
        self.assertTrue(hits)
        self.assertTrue(all(hits[index].score >= hits[index + 1].score
                            for index in range(len(hits) - 1)))
        self.assertTrue(hits[0].metadata.get("timestamp"))

    def test_context_rendering_keeps_temporal_provenance(self):
        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        adapter = BM25Adapter()
        adapter.ingest(conversation)
        context = adapter.render_hits(adapter.search(question, limit=1))
        self.assertIn("time=", context)

    def test_answer_config_uses_project_prompt_template(self):
        config = AnswerConfig("http://unused", "", "model", system_template="RULES\n{context}")
        payload = OpenAIAnswerer(config).build_payload("When?", "Evidence")
        self.assertEqual("RULES\nEvidence", payload["messages"][0]["content"])

    def test_tencentdb_v3_preserves_turn_provenance_in_content(self):
        message = self.dataset.conversation("conv-26").sessions[0].messages[2]
        formatted = _format_v3_content(message)
        self.assertIn("[D1:3]", formatted)
        self.assertIn("[source_time=2023-05-08T13:56:00Z]", formatted)
        self.assertIn("[source_role=", formatted)
        self.assertIn("Caroline:", formatted)
        self.assertEqual(["D1:3"], _v3_source_ids(formatted))
        self.assertEqual(
            "2023-05-08T13:56:00Z", _v3_source_timestamp(formatted)
        )
        self.assertEqual([], _v3_source_ids("memory without a source envelope"))

    def test_tencentdb_v3_preserves_tool_role_through_two_role_transport(self):
        message = Message(
            "tool-1",
            "tool",
            "Climate controller reports rear-left setpoint 22°C.",
            speaker="vehicle_control",
            timestamp="2026-08-26T09:05:00+08:00",
        )
        formatted = _format_v3_content(message)

        self.assertIn("[source_role=tool]", formatted)
        self.assertEqual("tool", _v3_source_role(formatted))
        self.assertEqual(
            "Climate controller reports rear-left setpoint 22°C.",
            _v3_message_body(formatted),
        )
        self.assertEqual(
            "assistant", _v3_transport_role("tool", extract_l1=True)
        )
        self.assertEqual(
            "assistant", _v3_transport_role("system", extract_l1=True)
        )
        self.assertEqual("user", _v3_transport_role("user", extract_l1=True))
        self.assertEqual(
            "assistant", _v3_transport_role("user", extract_l1=False)
        )
        self.assertEqual(
            "legacy body", _v3_message_body("[legacy-1] Driver: legacy body")
        )

    def test_tencentdb_ledger_parser_keeps_only_attributed_source_facts(self):
        session = self.dataset.conversation("conv-26").sessions[0]
        raw = _json_array("""```json
        [
          {"subject":"Caroline","category":"event","topic":"support group",
           "statement":"Caroline attended a support group.","values":[],
           "event_date":"2023-05-07","source_ids":["D1:3","D1:2"]},
          {"subject":"Unknown","category":"event","topic":"bad",
           "statement":"Unsupported.","values":[],"event_date":"",
           "source_ids":["D1:3"]}
        ]
        ```""")
        facts = _validate_session_facts(raw, session)
        self.assertEqual(1, len(facts))
        self.assertEqual(["D1:3"], facts[0]["source_ids"])

    def test_tencentdb_ledger_prompt_exposes_supplied_visual_descriptions(self):
        conversation = self.dataset.conversation("conv-26")
        session = next(
            item for item in conversation.sessions
            if any(message.message_id == "D17:12" for message in item.messages)
        )
        prompt = _session_prompt(session)
        self.assertIn("[Image caption:", prompt)
        self.assertIn("painting of a sunset with a pink sky", prompt)

    def test_tencentdb_ledger_falls_back_to_dialogue_roles(self):
        session = Session(
            session_id="role-only-session",
            timestamp="2026-01-01T00:00:00Z",
            messages=(
                Message("u1", "user", "Find a nearby gas station."),
                Message("a1", "assistant", "I selected Shell on Main Street."),
            ),
        )
        prompt = _session_prompt(session)
        self.assertIn("User: Find a nearby gas station.", prompt)
        self.assertIn("Assistant: I selected Shell on Main Street.", prompt)
        facts = _validate_session_facts([{
            "subject": "Assistant",
            "category": "travel",
            "topic": "selected destination",
            "statement": "The assistant selected Shell on Main Street.",
            "values": ["Shell", "Main Street"],
            "event_date": "",
            "source_ids": ["a1", "u1"],
        }], session)
        self.assertEqual(["a1"], facts[0]["source_ids"])

    def test_tencentdb_ledger_ranking_groups_exact_target_facts(self):
        facts = (
            {
                "subject": "Melanie", "category": "instrument",
                "topic": "musical instrument played",
                "statement": "Melanie plays the violin.",
                "values": ["violin"], "source_ids": ["D2:5"],
            },
            {
                "subject": "Melanie", "category": "instrument",
                "topic": "musical instrument played",
                "statement": "Melanie plays the clarinet.",
                "values": ["clarinet"], "source_ids": ["D15:26"],
            },
            {
                "subject": "Caroline", "category": "instrument",
                "topic": "musical instrument played",
                "statement": "Caroline plays the piano.",
                "values": ["piano"], "source_ids": ["D18:1"],
            },
        )
        ranked = _rank_v3_ledger_facts(
            facts, "What instruments does Melanie play?", ("Caroline", "Melanie")
        )
        statements = [fact["statement"] for _score, fact in ranked]
        self.assertEqual(
            ["Melanie plays the violin.", "Melanie plays the clarinet."],
            statements,
        )

    def test_tencentdb_ledger_ranking_maps_irregular_family_plurals(self):
        facts = (
            {
                "subject": "Melanie", "category": "family",
                "topic": "family members",
                "statement": "Melanie has a daughter and a son.",
                "values": ["daughter", "son"], "source_ids": ["D1:1"],
                "fact_kind": "rollup",
            },
            {
                "subject": "Melanie", "category": "emotion",
                "topic": "children after an accident",
                "statement": "Melanie's children were scared for their brother.",
                "values": ["children", "brother"], "source_ids": ["D2:1"],
            },
            {
                "subject": "Melanie", "category": "career",
                "topic": "job", "statement": "Melanie changed jobs.",
                "values": [], "source_ids": ["D3:1"],
            },
        )
        ranked = _rank_v3_ledger_facts(
            facts, "How many children does Melanie have?", ("Melanie",)
        )
        categories = [fact["category"] for _score, fact in ranked[:2]]
        self.assertIn("family", categories)
        self.assertIn("emotion", categories)

    def test_tencentdb_ledger_ranking_maps_while_to_event(self):
        facts = (
            {
                "subject": "Melanie", "category": "travel",
                "topic": "camping location",
                "statement": "Melanie camped in the mountains.",
                "values": ["mountains"], "source_ids": ["D1:1"],
            },
            {
                "subject": "Melanie", "category": "event",
                "topic": "camping activities",
                "statement": "While camping, Melanie explored and hiked.",
                "values": ["explored", "hiked"], "source_ids": ["D1:2"],
            },
        )
        ranked = _rank_v3_ledger_facts(
            facts, "What did Melanie do while camping?", ("Melanie",)
        )
        self.assertEqual("D1:2", ranked[0][1]["source_ids"][0])

    def test_tencentdb_ledger_rollup_requires_multiple_grounded_sources(self):
        atomic = [
            {"subject": "Melanie", "source_ids": ["D2:5"]},
            {"subject": "Melanie", "source_ids": ["D15:26"]},
        ]
        raw = [{
            "subject": "Melanie", "category": "instrument",
            "topic": "instruments played",
            "statement": "Melanie plays violin and clarinet.",
            "values": ["violin", "clarinet"], "event_date": "",
            "source_ids": ["D2:5", "D15:26", "invented"],
            "derived": True, "derivation": "union of two instrument facts",
        }, {
            "subject": "Melanie", "category": "pet", "topic": "pet",
            "statement": "Single-source restatement.", "values": [],
            "event_date": "", "source_ids": ["D2:5"], "derived": True,
        }]
        rollups = _validate_rollup_facts(raw, "Melanie", atomic)
        self.assertEqual(1, len(rollups))
        self.assertEqual(["D2:5", "D15:26"], rollups[0]["source_ids"])
        self.assertEqual("rollup", rollups[0]["fact_kind"])
        reasoning = _validate_rollup_facts(
            raw[:1], "Melanie", atomic, fact_kind="reasoning_rollup"
        )
        self.assertEqual("reasoning_rollup", reasoning[0]["fact_kind"])

    def test_tencentdb_relation_rollup_links_two_attributed_actors(self):
        conversation = self.dataset.conversation("conv-26")
        atomic = [
            {
                "subject": "Caroline", "category": "book",
                "statement": "Caroline recommended Becoming Nicole.",
                "source_ids": ["D7:11"],
            },
            {
                "subject": "Melanie", "category": "book",
                "statement": "Melanie read a book Caroline recommended.",
                "source_ids": ["D17:10"],
            },
        ]
        payload = _relation_rollup_input(conversation, atomic)
        selected = {row["source_id"] for row in payload["source_messages"]}
        self.assertTrue({"D7:11", "D17:10"}.issubset(selected))
        raw = [{
            "subject": "Melanie", "category": "book",
            "topic": "book recommended by Caroline",
            "statement": "Melanie read Becoming Nicole after Caroline recommended it.",
            "values": ["Becoming Nicole"], "event_date": "",
            "source_ids": ["D7:11", "D17:10"],
            "participants": ["Caroline", "Melanie"],
            "relation_type": "recommendation", "confidence": 0.95,
            "derived": True, "derivation": "unique cross-actor reference",
        }]
        facts = _validate_relation_rollup_facts(raw, conversation, payload)
        self.assertEqual(1, len(facts))
        self.assertEqual("relation_rollup", facts[0]["fact_kind"])
        self.assertEqual(["D7:11", "D17:10"], facts[0]["source_ids"])
        rejected = dict(raw[0], source_ids=["D7:11"], confidence=0.99)
        self.assertEqual(
            [], _validate_relation_rollup_facts([rejected], conversation, payload)
        )

    def test_tencentdb_event_rollup_requires_auditable_distinct_groups(self):
        conversation = self.dataset.conversation("conv-26")
        atomic = [
            {
                "subject": "Melanie", "category": "travel",
                "topic": "beach camping", "values": ["beach", "camping"],
                "statement": "Melanie camped at the beach.",
                "source_ids": ["D6:16"],
            },
            {
                "subject": "Melanie", "category": "travel",
                "topic": "beach outing", "values": ["beach"],
                "statement": "Melanie later went to the beach with her kids.",
                "source_ids": ["D10:8"],
            },
        ]
        payload = _event_rollup_input(conversation, atomic)
        beach_cluster = next(
            cluster for cluster in payload["candidate_clusters"]
            if cluster["shared_anchor"] == "beach"
        )
        raw = [{
            "candidate_cluster_id": beach_cluster["candidate_cluster_id"],
            "decision": "accept",
            "subject": "Melanie", "category": "travel",
            "topic": "2023 beach visits",
            "statement": "Melanie went to the beach twice in 2023.",
            "values": ["beach"], "event_date": "2023",
            "source_ids": ["D6:16", "D10:8"],
            "memory_kind": "event_count",
            "event_groups": [
                {"event_key": "July beach camping", "event_date": "2023-07-06",
                 "source_ids": ["D6:16"]},
                {"event_key": "later July beach outing", "event_date": "2023-07-20",
                 "source_ids": ["D10:8"]},
            ],
            "event_count": 2, "projection_horizon": "",
            "confidence": 0.95, "derived": True,
            "derivation": "two separately dated occurrences",
        }]
        facts = _validate_event_rollup_facts(raw, conversation, payload)
        self.assertEqual(1, len(facts))
        self.assertEqual("event_count", facts[0]["memory_kind"])
        self.assertIn("2", facts[0]["values"])
        rejected = dict(raw[0], event_count=3)
        self.assertEqual(
            [], _validate_event_rollup_facts([rejected], conversation, payload)
        )
        deterministic = _deterministic_pair_event_facts(payload, [])
        beach_count = next(
            fact for fact in deterministic
            if fact["candidate_cluster_id"] == beach_cluster["candidate_cluster_id"]
        )
        self.assertEqual(2, beach_count["event_count"])
        self.assertEqual(["D10:8", "D6:16"], beach_count["source_ids"])

    def test_tencentdb_ledger_ranking_prefers_audited_memory_by_intent(self):
        facts = (
            {
                "subject": "Melanie", "category": "book",
                "topic": "book recommended by Caroline",
                "statement": "Melanie read Becoming Nicole after Caroline recommended it.",
                "values": ["Becoming Nicole"],
                "source_ids": ["D7:11", "D17:10"],
                "fact_kind": "relation_rollup", "relation_type": "recommendation",
                "participants": ["Caroline", "Melanie"],
            },
            {
                "subject": "Caroline", "category": "book",
                "topic": "book", "statement": "Caroline read another book.",
                "values": ["another book"], "source_ids": ["D3:1"],
            },
        )
        ranked = _rank_v3_ledger_facts(
            facts, "What book did Caroline recommend?", ("Caroline", "Melanie")
        )
        self.assertEqual("relation_rollup", ranked[0][1]["fact_kind"])

    def test_tencentdb_event_identity_does_not_overprecision_when_questions(self):
        facts = (
            {
                "subject": "Caroline", "category": "event",
                "topic": "conference attendance",
                "statement": "Caroline attended a conference in July 2023.",
                "values": ["conference", "July 2023"], "source_ids": ["D1:1"],
            },
            {
                "subject": "Caroline", "category": "event",
                "topic": "conference identity",
                "statement": "The conference was on July 10, 2023.",
                "values": ["conference"], "source_ids": ["D1:1", "D2:1"],
                "fact_kind": "event_rollup", "memory_kind": "event_identity",
            },
        )
        when_ranked = _rank_v3_ledger_facts(
            facts, "When did Caroline attend the conference?", ("Caroline",)
        )
        self.assertTrue(all(
            fact.get("memory_kind") != "event_identity"
            for _score, fact in when_ranked
        ))
        identity_ranked = _rank_v3_ledger_facts(
            facts, "Was this the same or another conference?", ("Caroline",)
        )
        self.assertTrue(any(
            fact.get("memory_kind") == "event_identity"
            for _score, fact in identity_ranked
        ))

    def test_tencentdb_adapter_renders_ledger_as_one_source_grounded_hit(self):
        facts = [
            {
                "subject": "Melanie", "category": "instrument",
                "topic": "musical instrument played",
                "statement": "Melanie plays the violin.",
                "values": ["violin"], "event_date": "",
                "source_ids": ["D2:5"],
            },
            {
                "subject": "Melanie", "category": "instrument",
                "topic": "musical instrument played",
                "statement": "Melanie plays the clarinet.",
                "values": ["clarinet"], "event_date": "",
                "source_ids": ["D15:26"],
            },
        ]
        manifest = {
            "conversations": {
                "conv-26": {
                    "agent_id": "agent-caroline", "task_id": "task-caroline",
                    "perspectives": {
                        "Caroline": {
                            "agent_id": "agent-caroline",
                            "task_id": "task-caroline",
                        },
                        "Melanie": {
                            "agent_id": "agent-melanie",
                            "task_id": "task-melanie",
                        },
                    },
                },
            },
        }
        ledger = {"conversations": {"conv-26": {"facts": facts}}}
        with tempfile.TemporaryDirectory() as directory:
            isolation_path = Path(directory) / "isolation.json"
            ledger_path = Path(directory) / "ledger.json"
            isolation_path.write_text(json.dumps(manifest), encoding="utf-8")
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(isolation_path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_LEDGER_PATH": str(ledger_path),
                "TDAI_EVAL_LEDGER_FACT_RESULTS": "4",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                row = adapter._v3_ledger_row(
                    "conv-26", "What instruments does Melanie play?"
                )
        self.assertIsNotNone(row)
        self.assertEqual("L1X", row["metadata"]["level"])
        self.assertEqual(["D2:5", "D15:26"], row["source_ids"])
        self.assertIn("Melanie plays the violin", row["content"])
        self.assertIn("Melanie plays the clarinet", row["content"])

    def test_tencentdb_ingest_can_normalize_relative_time_once(self):
        manifest = {
            "conversations": {
                "conv-26": {"agent_id": "agent-26", "task_id": "task-26"},
            },
        }
        session = self.dataset.conversation("conv-26").sessions[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
                "TDAI_EVAL_INGEST_RESOLVE_RELATIVE_TIME": "true",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(
                    adapter, "_post", return_value={"code": 0, "data": {}}
                ) as post:
                    adapter.ingest_session("conv-26", session)
        body = post.call_args.args[1]
        content = next(
            item["content"] for item in body["messages"]
            if "LGBTQ support group yesterday" in item["content"]
        )
        self.assertIn('"yesterday" = 7 May 2023', content)
        self.assertEqual(1, content.count("[resolved_relative_time:"))

    def test_hindsight_bridge_preserves_session_time_and_source_lineage(self):
        session = self.dataset.conversation("conv-26").sessions[0]
        item, lineage = _session_item("conv-26", session.to_dict())
        self.assertEqual("2023-05-08T13:56:00+00:00", item["timestamp"])
        self.assertEqual("conv-26-session-001", item["document_id"])
        self.assertIn("D1:3", lineage["source_ids"])
        self.assertIn('\"speaker\": \"Caroline\"', item["content"])

    def test_hindsight_resume_submits_only_missing_documents(self):
        pending = [
            {"item": {"document_id": f"session-{index}"}}
            for index in range(1, 5)
        ]
        missing = _missing_pending_rows(pending, {"session-1", "session-3"})
        self.assertEqual(
            ["session-2", "session-4"],
            [row["item"]["document_id"] for row in missing],
        )

    def test_hindsight_readiness_classifies_operations(self):
        operations = [
            {"id": "done", "status": "completed"},
            {"id": "queued", "status": "pending"},
            {"id": "bad", "status": "failed"},
        ]
        active, failed = _operation_groups(operations)
        self.assertEqual(["queued"], [row["id"] for row in active])
        self.assertEqual(["bad"], [row["id"] for row in failed])

    def test_tencentdb_v3_transport_session_is_tenant_scoped(self):
        common = {
            "source_session_id": "session-001",
            "conversation_id": "conversation-1",
            "team_id": "team-1",
            "user_id": "user-1",
            "task_id": "task-1",
        }
        first = _scoped_v3_session_id(agent_id="agent-1", **common)
        repeated = _scoped_v3_session_id(agent_id="agent-1", **common)
        other_agent = _scoped_v3_session_id(agent_id="agent-2", **common)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_agent)
        self.assertTrue(first.startswith("session-001--"))

    def test_tencentdb_isolation_manifest_loads_run_level_tenancy(self):
        manifest = {
            "team_id": "team-run",
            "user_id": "user-run",
            "service_id": "service-run",
            "conversations": {
                "conv-26": {"agent_id": "agent-26", "task_id": "task-26"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict(
                "os.environ",
                {"TDAI_EVAL_ISOLATION_MAP": str(path)},
                clear=False,
            ):
                for name in ("TDAI_EVAL_TEAM_ID", "TDAI_EVAL_USER_ID", "TDAI_EVAL_SERVICE_ID"):
                    os.environ.pop(name, None)
                adapter = TencentDBHTTPAdapter("http://unused")
            self.assertEqual(
                ("team-run", "agent-26", "user-run", "task-26"),
                adapter._isolation("conv-26"),
            )
            self.assertEqual("service-run", adapter.service_id)

    def test_tencentdb_v3_ingests_each_human_from_an_isolated_perspective(self):
        manifest = {
            "team_id": "team-run",
            "user_id": "user-run",
            "service_id": "service-run",
            "conversations": {
                "conv-26": {
                    "agent_id": "agent-26",
                    "task_id": "legacy-task",
                    "perspectives": {
                        "Caroline": {
                            "agent_id": "agent-caroline",
                            "task_id": "task-caroline",
                        },
                        "Melanie": {
                            "agent_id": "agent-melanie",
                            "task_id": "task-melanie",
                        },
                    },
                },
            },
        }
        conversation = self.dataset.conversation("conv-26")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            environment = {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "auto",
                "TDAI_EVAL_USER_KEY": "",
            }
            with mock.patch.dict("os.environ", environment, clear=False):
                for name in (
                    "TDAI_EVAL_TEAM_ID",
                    "TDAI_EVAL_USER_ID",
                    "TDAI_EVAL_SERVICE_ID",
                ):
                    os.environ.pop(name, None)
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(
                    adapter, "_post", return_value={"code": 0, "data": {}}
                ) as post:
                    adapter.prepare(conversation)
                    adapter.ingest_session(
                        conversation.conversation_id, conversation.sessions[0]
                    )

        calls = [
            call.args[1]
            for call in post.call_args_list
            if call.args[0] == "/v3/conversation/add"
        ]
        self.assertEqual(2, len(calls))
        by_task = {body["task_id"]: body for body in calls}
        self.assertEqual(
            ["user", "assistant"],
            [item["role"] for item in by_task["task-caroline"]["messages"][:2]],
        )
        self.assertEqual(
            ["assistant", "user"],
            [item["role"] for item in by_task["task-melanie"]["messages"][:2]],
        )
        self.assertNotEqual(
            by_task["task-caroline"]["session_id"],
            by_task["task-melanie"]["session_id"],
        )
        self.assertIn(
            "[source_time=2023-05-08T13:56:00Z]",
            by_task["task-melanie"]["messages"][0]["content"],
        )

    def test_tencentdb_v3_auto_provisions_and_persists_missing_perspective(self):
        manifest = {
            "team_id": "team-run",
            "user_id": "user-run",
            "service_id": "service-run",
            "conversations": {
                "conv-26": {"agent_id": "agent-26", "task_id": "task-caroline"},
            },
        }
        conversation = self.dataset.conversation("conv-26")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            environment = {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "auto",
                "TDAI_EVAL_USER_KEY": "",
            }
            with mock.patch.dict("os.environ", environment, clear=False):
                for name in (
                    "TDAI_EVAL_TEAM_ID",
                    "TDAI_EVAL_USER_ID",
                    "TDAI_EVAL_SERVICE_ID",
                ):
                    os.environ.pop(name, None)
                adapter = TencentDBHTTPAdapter("http://unused")
                def provision(endpoint, _body):
                    if endpoint.endswith("/agent/create"):
                        return {"code": 0, "data": {"agent_id": "agent-melanie"}}
                    return {"code": 0, "data": {"task_id": "task-melanie"}}

                with mock.patch.object(adapter, "_post", side_effect=provision) as post:
                    adapter.prepare(conversation)
            persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(2, post.call_count)
        self.assertEqual("/v2/agent/create", post.call_args_list[0].args[0])
        self.assertEqual("/v2/task/create", post.call_args_list[1].args[0])
        self.assertEqual(
            ["agent-melanie"], post.call_args_list[1].args[1]["agent_ids"],
        )
        scopes = persisted["conversations"]["conv-26"]["perspectives"]
        self.assertEqual("agent-26", scopes["Caroline"]["agent_id"])
        self.assertEqual("task-caroline", scopes["Caroline"]["task_id"])
        self.assertEqual("agent-melanie", scopes["Melanie"]["agent_id"])
        self.assertEqual("task-melanie", scopes["Melanie"]["task_id"])

    def test_tencentdb_v3_task_provisioning_uses_user_key_when_available(self):
        with mock.patch.dict("os.environ", {
            "TDAI_HTTP_API_VERSION": "v3",
            "TDAI_EVAL_USER_KEY": "user-secret",
        }, clear=False):
            adapter = TencentDBHTTPAdapter("http://unused")
            with mock.patch.object(adapter, "_post", return_value={
                "code": 0,
                "data": {"task_id": "task-new"},
            }) as post:
                task_id = adapter._create_perspective_task(
                    "conv-26",
                    "Melanie",
                    team_id="team-run",
                    agent_id="agent-26",
                    user_id="user-run",
                )

        self.assertEqual("task-new", task_id)
        self.assertEqual("/v3/meta/task/create", post.call_args.args[0])
        self.assertEqual(
            [{"agent_id": "agent-26", "role_in_task": "memory-perspective"}],
            post.call_args.args[1]["linked_agents"],
        )

    def test_tencentdb_single_mode_ignores_configured_perspectives(self):
        manifest = {
            "conversations": {
                "conv-26": {
                    "agent_id": "agent-26",
                    "task_id": "task-single",
                    "perspectives": {
                        "Caroline": {"task_id": "task-caroline"},
                        "Melanie": {"task_id": "task-melanie"},
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                perspectives = adapter._perspective_isolations("conv-26")

        self.assertEqual(1, len(perspectives))
        self.assertEqual("", perspectives[0].speaker)
        self.assertEqual("task-single", perspectives[0].task_id)

    def test_tencentdb_v3_search_merges_perspectives_and_restores_source_time(self):
        manifest = {
            "conversations": {
                "conv-26": {
                    "agent_id": "agent-26",
                    "perspectives": {
                        "Caroline": {
                            "agent_id": "agent-caroline",
                            "task_id": "task-caroline",
                        },
                        "Melanie": {
                            "agent_id": "agent-melanie",
                            "task_id": "task-melanie",
                        },
                    },
                },
            },
        }
        question = self.dataset.questions({"conv-26"})[0]
        source_content = (
            "[D1:3] [source_time=2023-05-08T13:56:00Z] Caroline: source"
        )

        def fake_post(endpoint, body):
            if endpoint == "/v3/atomic/search":
                owner = "Caroline" if body["task_id"] == "task-caroline" else "Melanie"
                return {"data": {"items": [{
                    "id": f"memory-{owner}",
                    "content": f"{owner} memory",
                    "score": 0.9,
                    "type": "persona",
                }]}}
            return {"data": {"messages": [{
                "id": f"message-{body['task_id']}",
                "content": source_content,
                "score": 0.8,
                "role": "user",
                "timestamp": "backend-time",
            }]}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_MEMORY_LAYERS": "L0,L1",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    hits = adapter.search(question, limit=10)

        self.assertEqual(3, len(hits))
        raw = next(hit for hit in hits if hit.metadata["level"] == "L0")
        self.assertEqual(("D1:3",), raw.source_ids)
        self.assertEqual("2023-05-08T13:56:00Z", raw.metadata["timestamp"])
        self.assertIn("time=2023-05-08T13:56:00Z", adapter.render_hits([raw]))

    def test_tencentdb_v3_l0_window_expands_only_the_anchor_session(self):
        manifest = {
            "team_id": "team-run",
            "user_id": "user-run",
            "conversations": {
                "conv-26": {"agent_id": "agent-26", "task_id": "task-26"},
            },
        }
        question = self.dataset.questions({"conv-26"})[0]
        source_rows = [
            {
                "id": "message-1",
                "session_id": "session-1",
                "role": "user",
                "content": "[D1:1] Alice: earlier context",
                "timestamp": "2026-01-01T00:00:00.001Z",
            },
            {
                "id": "message-2",
                "session_id": "session-1",
                "role": "assistant",
                "content": "[D1:2] Bob: matching question",
                "timestamp": "2026-01-01T00:00:00.002Z",
            },
            {
                "id": "message-3",
                "session_id": "session-1",
                "role": "user",
                "content": "[D1:3] Alice: adjacent answer",
                "timestamp": "2026-01-01T00:00:00.003Z",
            },
            {
                "id": "message-4",
                "session_id": "session-2",
                "role": "user",
                "content": "[D2:1] Alice: unrelated session",
                "timestamp": "2026-01-02T00:00:00.001Z",
            },
        ]

        def fake_post(endpoint, body):
            if endpoint == "/v3/conversation/search":
                self.assertEqual(8, body["limit"])
                return {"data": {"messages": [{
                    "id": "message-2",
                    "content": source_rows[1]["content"],
                    "score": 0.9,
                    "role": "assistant",
                    "timestamp": "backend-time",
                }]}}
            if endpoint == "/v3/conversation/query":
                return {"code": 0, "data": {
                    "messages": source_rows,
                    "total": len(source_rows),
                }}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
                "TDAI_EVAL_MEMORY_LAYERS": "L0",
                "TDAI_EVAL_L0_WINDOW_RADIUS": "1",
                "TDAI_EVAL_CANDIDATE_MULTIPLIER": "2",
                "TDAI_EVAL_L0_MARK_ANCHORS": "true",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    hits = adapter.search(question, limit=4)

        self.assertEqual(1, len(hits))
        self.assertEqual(("D1:1", "D1:2", "D1:3"), hits[0].source_ids)
        self.assertIn("adjacent answer", hits[0].content)
        self.assertIn("[retrieval_anchor] [D1:2]", hits[0].content)
        self.assertNotIn("unrelated session", hits[0].content)
        self.assertEqual("window", hits[0].metadata["role"])
        self.assertEqual(3, hits[0].metadata["window_message_count"])

    def test_tencentdb_v3_resolves_unambiguous_relative_dates(self):
        timestamp = "2023-07-15T13:51:00Z"
        resolved = dict(_resolved_v3_relative_times(
            "I went yesterday, last Friday, and last week.", timestamp
        ))
        self.assertEqual("14 July 2023", resolved["yesterday"])
        self.assertEqual("14 July 2023", resolved["last friday"])
        self.assertEqual(
            "3 July 2023 to 9 July 2023", resolved["last week"]
        )
        annotated = _annotate_v3_relative_time(
            "The road trip was this past weekend.",
            "2023-10-20T18:55:00Z",
        )
        self.assertIn(
            '"this past weekend" = 14 October 2023 to 15 October 2023',
            annotated,
        )
        self.assertIn("session_date = 20 October 2023", annotated)
        self.assertEqual(
            "[D1:1] [source_date=15 July 2023] Alice: hello",
            _humanize_v3_source_time(
                "[D1:1] [source_time=2023-07-15T13:51:00Z] Alice: hello",
                "2023-07-15T13:51:00Z",
            ),
        )

    def test_tencentdb_explicit_source_date_boost_is_exact_and_l0_only(self):
        rows = [{
            "content": "target",
            "score": 0.03,
            "metadata": {"level": "L0", "timestamp": "2024-01-02T08:00:00Z"},
        }, {
            "content": "other",
            "score": 0.04,
            "metadata": {"level": "L0", "timestamp": "2024-01-03T08:00:00Z"},
        }, {
            "content": "summary mentioning 2024-01-02",
            "score": 0.05,
            "metadata": {"level": "L1"},
        }]
        _boost_v3_explicit_source_dates(rows, "interaction logged on 2024-01-02")
        self.assertEqual(1.03, rows[0]["score"])
        self.assertEqual("2024-01-02", rows[0]["metadata"]["explicit_source_date_boost"])
        self.assertEqual(0.04, rows[1]["score"])
        self.assertEqual(0.05, rows[2]["score"])

    def test_tencentdb_explicit_date_candidate_starts_at_session_first_turn(self):
        manifest = {
            "conversations": {
                "conv-test": {"agent_id": "agent-1", "task_id": "task-1"},
            },
        }
        source_rows = [
            {
                "id": "target-1", "session_id": "target", "role": "user",
                "content": "[D1] [source_time=2024-01-02T08:00:00Z] first turn",
                "timestamp": "2026-01-01T00:00:00.001Z",
            },
            {
                "id": "target-2", "session_id": "target", "role": "assistant",
                "content": "[D2] [source_time=2024-01-02T08:00:00Z] matching topic",
                "timestamp": "2026-01-01T00:00:00.002Z",
            },
            {
                "id": "other-1", "session_id": "other", "role": "user",
                "content": "[D3] [source_time=2024-01-03T08:00:00Z] matching topic",
                "timestamp": "2026-01-01T00:00:00.003Z",
            },
        ]

        def fake_post(endpoint, _body):
            if endpoint == "/v3/conversation/query":
                return {"code": 0, "data": {
                    "messages": source_rows, "total": len(source_rows),
                }}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
                "TDAI_EVAL_MEMORY_LAYERS": "L0",
                "TDAI_EVAL_L0_EXPLICIT_DATE_RESULTS": "1",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    rows = adapter._v3_l0_explicit_date_rows(
                        "conv-test",
                        "interaction logged on 2024-01-02 matching topic",
                    )

        self.assertEqual(1, len(rows))
        self.assertEqual("target-1", rows[0]["metadata"]["backend_message_id"])
        self.assertEqual("target", rows[0]["metadata"]["session_id"])
        self.assertEqual(
            "2024-01-02",
            rows[0]["metadata"]["explicit_source_date_candidate"],
        )

    def test_tencentdb_v3_relative_dates_leave_ambiguous_phrases_unchanged(self):
        self.assertEqual(
            [],
            _resolved_v3_relative_times(
                "I did this recently and a few weeks ago.",
                "2023-07-15T13:51:00Z",
            ),
        )
        self.assertTrue(_is_temporal_question("When did Alice leave?"))
        self.assertTrue(_is_temporal_question("What year did Alice leave?"))
        self.assertFalse(_is_temporal_question(
            "What setback did Alice face in October?"
        ))
        modes = frozenset({"aggregate", "when", "emotion", "inference"})
        self.assertTrue(_should_focus_question(
            "What instruments does Alice play?", modes
        ))
        self.assertTrue(_should_focus_question(
            "How did Alice feel after the trip?", modes
        ))
        self.assertTrue(_should_focus_question(
            "When did Alice leave?", modes
        ))
        self.assertFalse(_should_focus_question(
            "Where did Oliver hide his bone?", modes
        ))
        self.assertTrue(_should_focus_question(
            "Would Alice pursue writing?", modes
        ))
        self.assertEqual(
            {"child", "read", "book", "name", "pet", "attend"},
            set(_focus_tokens(
                "children reading books named pets seen many times"
            )),
        )
        self.assertEqual(
            {"emotion"},
            set(_focus_tokens("feel appreciated grateful thankful")),
        )

    def test_tencentdb_v3_focus_repeats_anchor_and_adjacent_answer(self):
        manifest = {
            "conversations": {
                "conv-test": {"agent_id": "agent-1", "task_id": "task-1"},
            },
        }
        question = Question(
            "q1", "conv-test", "What pets does Alice have?", ("two cats",)
        )
        source_rows = [
            {
                "id": "message-1", "session_id": "session-1",
                "role": "assistant",
                "content": (
                    "[D1:1] [source_time=2023-07-15T13:51:00Z] "
                    "Bob: What pets does Alice have?"
                ),
                "timestamp": "2026-01-01T00:00:00.001Z",
            },
            {
                "id": "message-2", "session_id": "session-1",
                "role": "user",
                "content": (
                    "[D1:2] [source_time=2023-07-15T13:51:00Z] "
                    "Alice: I have two cats."
                ),
                "timestamp": "2026-01-01T00:00:00.002Z",
            },
        ]

        def fake_post(endpoint, _body):
            if endpoint == "/v3/conversation/search":
                return {"data": {"messages": [{
                    "id": "message-1",
                    "content": source_rows[0]["content"],
                    "score": 0.9,
                    "role": "assistant",
                    "timestamp": "backend-time",
                }]}}
            if endpoint == "/v3/conversation/query":
                return {"code": 0, "data": {
                    "messages": source_rows, "total": len(source_rows),
                }}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
                "TDAI_EVAL_MEMORY_LAYERS": "L0",
                "TDAI_EVAL_L0_WINDOW_RADIUS": "1",
                "TDAI_EVAL_L0_FOCUS_ANCHORS": "1",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    hits = adapter.search(question, limit=2)

        self.assertEqual(2, len(hits))
        self.assertEqual("focus", hits[-1].metadata["role"])
        self.assertIn("What pets does Alice have?", hits[-1].content)
        self.assertIn("I have two cats.", hits[-1].content)
        self.assertEqual(("D1:1", "D1:2"), hits[-1].source_ids)

    def test_tencentdb_v3_dynamic_selection_can_reserve_l0_anchors(self):
        manifest = {
            "conversations": {
                "conv-26": {"agent_id": "agent-26", "task_id": "task-26"},
            },
        }
        question = self.dataset.questions({"conv-26"})[0]

        def fake_post(endpoint, _body):
            if endpoint == "/v3/atomic/search":
                return {"data": {"items": [
                    {"id": f"l1-{index}", "content": f"memory {index}",
                     "score": 1.0 - index / 10, "type": "fact"}
                    for index in range(4)
                ]}}
            if endpoint == "/v3/conversation/search":
                return {"data": {"messages": [
                    {"id": f"l0-{index}", "content": f"[D1:{index}] raw {index}",
                     "score": 0.1 - index / 100, "role": "user"}
                    for index in range(2)
                ]}}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
                "TDAI_EVAL_MEMORY_LAYERS": "L0,L1",
                "TDAI_EVAL_L0_WINDOW_RADIUS": "0",
                "TDAI_EVAL_L0_MIN_RESULTS": "0",
                "TDAI_EVAL_L0_MIN_FRACTION": "0.5",
                "TDAI_EVAL_L0_FIRST": "true",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    hits = adapter.search(question, limit=3)

        self.assertEqual(
            {"L0": 2, "L1": 1},
            dict(Counter(hit.metadata["level"] for hit in hits)),
        )
        self.assertEqual("L0", hits[0].metadata["level"])

    def test_tencentdb_v3_session_bm25_adds_distinct_lexical_window(self):
        manifest = {
            "conversations": {
                "conv-test": {"agent_id": "agent-1", "task_id": "task-1"},
            },
        }
        question = Question(
            "q1", "conv-test", "What pets does Alice have?", ("two cats",)
        )
        source_rows = [
            {
                "id": "semantic-only",
                "session_id": "session-1",
                "role": "user",
                "content": "[D1:1] Alice: unrelated semantic result",
                "timestamp": "2026-01-01T00:00:00.001Z",
            },
            {
                "id": "lexical-question",
                "session_id": "session-2",
                "role": "assistant",
                "content": "[D2:1] Bob: What pets does Alice have?",
                "timestamp": "2026-01-02T00:00:00.001Z",
            },
            {
                "id": "lexical-answer",
                "session_id": "session-2",
                "role": "user",
                "content": "[D2:2] Alice: two cats",
                "timestamp": "2026-01-02T00:00:00.002Z",
            },
        ]

        def fake_post(endpoint, _body):
            if endpoint == "/v3/conversation/search":
                return {"data": {"messages": [{
                    "id": "semantic-only",
                    "content": source_rows[0]["content"],
                    "score": 0.032,
                    "role": "user",
                    "timestamp": "backend-time",
                }]}}
            if endpoint == "/v3/conversation/query":
                return {"code": 0, "data": {
                    "messages": source_rows,
                    "total": len(source_rows),
                }}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
                "TDAI_EVAL_MEMORY_LAYERS": "L0",
                "TDAI_EVAL_L0_WINDOW_RADIUS": "1",
                "TDAI_EVAL_L0_MIN_RESULTS": "1",
                "TDAI_EVAL_L0_SESSION_BM25_RESULTS": "1",
                "TDAI_EVAL_L0_SESSION_BM25_WEIGHT": "3",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    hits = adapter.search(question, limit=1)

        self.assertEqual(1, len(hits))
        self.assertEqual(("D2:1", "D2:2"), hits[0].source_ids)
        self.assertIn("two cats", hits[0].content)
        self.assertEqual("session_bm25", hits[0].metadata["retrieval_strategy"])

    def test_tencentdb_v3_l0_quota_can_diversify_semantic_sessions(self):
        manifest = {
            "conversations": {
                "conv-test": {"agent_id": "agent-1", "task_id": "task-1"},
            },
        }
        question = Question("q1", "conv-test", "question", ("answer",))
        source_rows = [
            {"id": "s1-a", "session_id": "session-1", "role": "user",
             "content": "[D1:1] Alice: first", "timestamp": "2026-01-01T00:00:00.001Z"},
            {"id": "s1-b", "session_id": "session-1", "role": "user",
             "content": "[D1:2] Alice: second", "timestamp": "2026-01-01T00:00:00.002Z"},
            {"id": "s2-a", "session_id": "session-2", "role": "user",
             "content": "[D2:1] Alice: third", "timestamp": "2026-01-02T00:00:00.001Z"},
        ]

        def fake_post(endpoint, _body):
            if endpoint == "/v3/atomic/search":
                return {"data": {"items": [{
                    "id": "l1", "content": "memory", "score": 0.9,
                    "type": "fact",
                }]}}
            if endpoint == "/v3/conversation/search":
                return {"data": {"messages": [
                    {"id": row["id"], "content": row["content"],
                     "score": score, "role": row["role"],
                     "timestamp": row["timestamp"]}
                    for row, score in zip(source_rows, (0.3, 0.2, 0.1))
                ]}}
            if endpoint == "/v3/conversation/query":
                return {"code": 0, "data": {
                    "messages": source_rows, "total": len(source_rows),
                }}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_PERSPECTIVE_MODE": "single",
                "TDAI_EVAL_MEMORY_LAYERS": "L0,L1",
                "TDAI_EVAL_L0_WINDOW_RADIUS": "0",
                "TDAI_EVAL_L0_MIN_RESULTS": "2",
                "TDAI_EVAL_L0_DIVERSIFY_SESSIONS": "true",
                "TDAI_EVAL_L0_SESSION_BM25_RESULTS": "0",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    hits = adapter.search(question, limit=3)

        l0_sources = {
            hit.source_ids for hit in hits if hit.metadata["level"] == "L0"
        }
        self.assertEqual({("D1:1",), ("D2:1",)}, l0_sources)

    def test_tencentdb_v3_search_includes_l2_and_l3_from_both_agents(self):
        manifest = {
            "conversations": {
                "conv-26": {
                    "agent_id": "agent-caroline",
                    "task_id": "task-caroline",
                    "perspectives": {
                        "Caroline": {
                            "agent_id": "agent-caroline",
                            "task_id": "task-caroline",
                        },
                        "Melanie": {
                            "agent_id": "agent-melanie",
                            "task_id": "task-melanie",
                        },
                    },
                },
            },
        }
        question = self.dataset.questions({"conv-26"})[0]

        def fake_post(endpoint, body):
            owner = (
                "Caroline" if body.get("agent_id") == "agent-caroline"
                else "Melanie"
            )
            if endpoint == "/v3/atomic/search":
                return {"data": {"items": [{
                    "id": f"memory-{owner}",
                    "content": f"{owner} atomic memory",
                    "score": 0.9,
                    "type": "persona",
                }]}}
            if endpoint == "/v3/conversation/search":
                return {"data": {"messages": []}}
            if endpoint == "/v3/scenario/ls":
                return {"data": {"entries": [{
                    "path": f"life-{owner}.md",
                    "summary": f"{owner} travel and charity events",
                    "version": 1,
                }]}}
            if endpoint == "/v3/scenario/read":
                return {"data": {
                    "content": f"# {owner} life\n\n{owner} joined a charity race.",
                    "version": 1,
                }}
            if endpoint == "/v3/core/read":
                return {"data": {
                    "content": f"# Persona\n\n{owner} is an active volunteer.",
                    "version": 1,
                }}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_MEMORY_LAYERS": "L0,L1,L2,L3",
                "TDAI_EVAL_L2_RESULTS": "2",
                "TDAI_EVAL_L3_RESULTS": "2",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post):
                    hits = adapter.search(question, limit=10)

        levels = Counter(hit.metadata["level"] for hit in hits)
        self.assertEqual({"L1": 2, "L2": 2, "L3": 2}, dict(levels))
        for level in ("L2", "L3"):
            self.assertEqual(
                {"Caroline", "Melanie"},
                {
                    hit.metadata["perspective_owner"]
                    for hit in hits if hit.metadata["level"] == level
                },
            )

    def test_tencentdb_v3_full_layer_readiness_requires_profiles_per_agent(self):
        manifest = {
            "conversations": {
                "conv-26": {
                    "agent_id": "agent-caroline",
                    "task_id": "task-caroline",
                    "perspectives": {
                        "Caroline": {
                            "agent_id": "agent-caroline",
                            "task_id": "task-caroline",
                        },
                        "Melanie": {
                            "agent_id": "agent-melanie",
                            "task_id": "task-melanie",
                        },
                    },
                },
            },
        }

        def fake_post(endpoint, _body):
            if endpoint == "/v2/pipeline/status":
                idle = {"queued": 0, "running": 0}
                return {"code": 0, "data": {
                    "l1": idle, "l2": idle, "l3": idle,
                }}
            if endpoint in {"/v3/scenario/count", "/v3/core/count"}:
                return {"code": 0, "data": {"total": 1}}
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TDAI_EVAL_ISOLATION_MAP": str(path),
                "TDAI_HTTP_API_VERSION": "v3",
                "TDAI_EVAL_MEMORY_LAYERS": "L0,L1,L2,L3",
                "TDAI_EVAL_READY_SETTLE_SECONDS": "0",
            }, clear=False):
                adapter = TencentDBHTTPAdapter("http://unused")
                with mock.patch.object(adapter, "_post", side_effect=fake_post) as post:
                    adapter.wait_until_ready("conv-26", timeout=1)

        endpoints = [call.args[0] for call in post.call_args_list]
        self.assertEqual(2, endpoints.count("/v3/scenario/count"))
        self.assertEqual(2, endpoints.count("/v3/core/count"))

    def test_tencentdb_v3_readiness_rejects_dead_lettered_memory_tasks(self):
        idle = {"queued": 0, "running": 0}
        status = {"code": 0, "data": {
            "l1": idle,
            "l2": idle,
            "l3": idle,
            "worker": {
                "tasksDeadLettered": 1,
                "deadLetterCount": 1,
            },
        }}
        with mock.patch.dict("os.environ", {
            "TDAI_HTTP_API_VERSION": "v3",
            "TDAI_EVAL_MEMORY_LAYERS": "L1",
            "TDAI_EVAL_READY_SETTLE_SECONDS": "0",
        }, clear=False):
            adapter = TencentDBHTTPAdapter("http://unused")
            with mock.patch.object(adapter, "_post", return_value=status):
                with self.assertRaisesRegex(RuntimeError, "dead-letter"):
                    adapter.wait_until_ready("conv-26", timeout=1)

    def test_runner_writes_canonical_jsonl(self):
        conversation = self.dataset.conversation("conv-26")
        questions = self.dataset.questions({"conv-26"})[:2]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            report = RetrievalRunner(BM25Adapter(), output).run([conversation], questions)
            self.assertEqual(2, report["questions"])
            self.assertEqual(2, len(output.read_text(encoding="utf-8").splitlines()))
            validation = validate_retrieval(output, expected_count=2)
            self.assertTrue(validation["pass"])

    def test_runner_resume_does_not_duplicate_questions(self):
        conversation = self.dataset.conversation("conv-26")
        questions = self.dataset.questions({"conv-26"})[:2]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            RetrievalRunner(BM25Adapter(), output).run([conversation], questions)
            report = RetrievalRunner(BM25Adapter(), output, resume=True).run(
                [conversation], questions
            )
            self.assertEqual(0, report["questions"])
            self.assertEqual(2, report["skipped"])
            self.assertEqual(2, len(output.read_text(encoding="utf-8").splitlines()))

    def test_runner_resume_skips_completed_conversation_before_ingest(self):
        class CountingAdapter(BM25Adapter):
            def __init__(self):
                super().__init__()
                self.ingest_count = 0

            def ingest_session(self, conversation_id: str, session: Session) -> None:
                self.ingest_count += 1
                super().ingest_session(conversation_id, session)

        conversation = self.dataset.conversation("conv-26")
        questions = self.dataset.questions({"conv-26"})[:1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            RetrievalRunner(BM25Adapter(), output).run([conversation], questions)
            adapter = CountingAdapter()
            report = RetrievalRunner(adapter, output, resume=True).run(
                [conversation], questions
            )
            self.assertEqual(0, adapter.ingest_count)
            self.assertEqual(1, report["skipped"])

    def test_runner_can_reuse_prepared_store_without_ingest(self):
        class PreparedAdapter(MemoryAdapter):
            adapter_id = "prepared-test"
            capabilities = frozenset({"search"})

            def ingest_session(self, conversation_id: str, session: Session) -> None:
                raise AssertionError("prepared store must not be ingested again")

            def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
                return [MemoryHit("prepared evidence", 1.0)]

        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            report = RetrievalRunner(
                PreparedAdapter(), output, ingest=False
            ).run([conversation], [question])
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1, report["questions"])
            self.assertTrue(row["metrics"]["store_reused"])

    def test_runner_reports_readiness_separately_from_ingest(self):
        class PreparedAsyncAdapter(MemoryAdapter):
            adapter_id = "prepared-async-test"
            capabilities = frozenset({"search", "wait_until_ready"})

            def __init__(self):
                self.waited_for: tuple[str, float] | None = None

            def wait_until_ready(self, conversation_id: str, *, timeout: float) -> None:
                self.waited_for = (conversation_id, timeout)

            def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
                return [MemoryHit("prepared evidence", 1.0)]

        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            adapter = PreparedAsyncAdapter()
            RetrievalRunner(
                adapter,
                output,
                ingest=False,
                ready_timeout=12.5,
            ).run([conversation], [question])
            row = json.loads(output.read_text(encoding="utf-8"))
            metrics = row["metrics"]
            self.assertEqual((conversation.conversation_id, 12.5), adapter.waited_for)
            self.assertEqual(0.0, metrics["conversation_ingest_seconds"])
            self.assertGreaterEqual(metrics["readiness_seconds"], 0.0)
            self.assertEqual(
                metrics["readiness_seconds"],
                metrics["conversation_lifecycle_seconds"],
            )

    def test_runner_enforces_context_budget(self):
        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            RetrievalRunner(
                FullContextAdapter(), output, max_context_chars=1000
            ).run([conversation], [question])
            import json
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1000, row["metrics"]["context_chars"])
            self.assertTrue(row["metrics"]["context_truncated"])

    def test_runner_can_ingest_all_without_searching(self):
        class IngestOnlyAdapter(MemoryAdapter):
            capabilities = frozenset({"ingest"})

            def __init__(self):
                self.session_count = 0

            def ingest_session(self, conversation_id: str, session: Session) -> None:
                self.session_count += 1

            def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
                raise AssertionError("ingest-only mode must not search")

        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ingest-only.jsonl"
            adapter = IngestOnlyAdapter()
            result = RetrievalRunner(
                adapter, output, ingest_only=True
            ).run([conversation], [question])
        self.assertEqual(len(conversation.sessions), adapter.session_count)
        self.assertEqual(1, result["ingested_conversations"])
        self.assertEqual(0, result["questions"])

    def test_registry_marks_closed_implementation(self):
        registry = load_registry(ROOT / "benchmarks/framework_eval/frameworks.json")
        self.assertEqual("not-open-source", registry["memorax"].status)

    def test_longmemeval_normalization(self):
        dataset = LongMemEvalDataset(ROOT / "benchmarks/data/LongMemEval/normalized")
        question = dataset.questions()[0]
        conversation = dataset.conversation(question.conversation_id)
        self.assertTrue(conversation.sessions)
        self.assertTrue(question.evidence_ids)

    def test_kvret_memory_is_balanced_and_source_grounded(self):
        dataset = KVRETMemoryDataset(ROOT / "benchmarks/data/KVRET/normalized")
        conversations = dataset.conversations()
        questions = dataset.questions()
        self.assertEqual(6, len(conversations))
        self.assertEqual({23}, {len(row.sessions) for row in conversations})
        self.assertEqual(138, len(questions))
        self.assertEqual(
            {"calendar": 46, "navigation": 46, "weather": 46},
            dict(Counter(question.category for question in questions)),
        )
        self.assertEqual(
            138,
            len({(question.conversation_id, question.text) for question in questions}),
        )
        navigation = dataset.question("kvret-memory#q0014")
        self.assertEqual("Panda Express", navigation.answers[0])
        self.assertIn("842 Arrowhead Way", navigation.answers)
        self.assertEqual(("K00S015T01", "K00S015T04"), navigation.evidence_ids)
        self.assertEqual([], dataset.validate())

    def test_kvret_slot_judge_normalizes_surface_and_time_variants(self):
        self.assertTrue(slot_value_matches("The answer is 7:00 p.m.", "7 pm", exact=False))
        self.assertTrue(slot_value_matches("11:00 UTC", "11 am", exact=True))
        self.assertTrue(slot_value_matches("Seatlle", "Seattle", exact=True))
        self.assertTrue(slot_value_matches("Valero", "valero", exact=True))
        self.assertFalse(slot_value_matches("Chevron", "Valero", exact=False))

    def test_slurp_memory_is_same_speaker_fragmented_and_grounded(self):
        dataset = SLURPMemoryDataset(ROOT / "benchmarks/data/SLURP/normalized")
        conversations = dataset.conversations()
        questions = dataset.questions()
        self.assertEqual(6, len(conversations))
        self.assertEqual({23}, {len(row.sessions) for row in conversations})
        self.assertEqual(138, len(questions))
        self.assertEqual(18, len({question.category for question in questions}))
        self.assertEqual(
            138,
            len({question.metadata["source_slurp_id"] for question in questions}),
        )
        for conversation in conversations:
            self.assertEqual(
                1,
                len({
                    session.metadata["source_speaker_hash"]
                    for session in conversation.sessions
                }),
            )
            for session in conversation.sessions:
                self.assertEqual(
                    ["user", "assistant", "user"],
                    [message.role for message in session.messages],
                )
        first = questions[0]
        session = conversations[0].sessions[0]
        self.assertEqual(tuple(message.message_id for message in session.messages), first.evidence_ids)
        self.assertNotIn(first.answers[0].casefold(), session.messages[0].content.casefold())
        self.assertEqual(first.answers[0], session.messages[2].content)
        self.assertEqual([], dataset.validate())

    def test_slurp_memory_judge_is_deterministic(self):
        judge = SLURPMemoryJudge(dataset_root=ROOT / "benchmarks/data/SLURP/normalized")
        self.assertFalse(judge.requires_model)
        self.assertEqual("slurp_memory_deterministic_entity", judge.judge_id)

    def test_slurp_fragment_reader_uses_question_anchor_not_evidence_ids(self):
        question = (
            'During the driver voice interaction logged on 2025-01-02 that '
            'began "is my alarm set for", what did the driver reply when the '
            'car assistant asked for the "timeofday" detail?'
        )
        context = """[1 score=0.9]
[S9T01] [source_time=2025-01-01T08:00:00Z] Driver: is my alarm set for Friday
[S9T02] [source_time=2025-01-01T08:00:05Z] Car Assistant: What time?
[S9T03] [source_time=2025-01-01T08:00:10Z] Driver: evening

[2 score=0.8]
[S2T02] [source_time=2025-01-02T08:00:05Z] Car Assistant: What should I use for the timeofday?
[S2T03] [source_time=2025-01-02T08:00:10Z] Driver: morning
[S2T01] [source_time=2025-01-02T08:00:00Z] Driver: is my alarm set for tomorrow
"""
        self.assertEqual("morning", extract_fragment_reply(question, context))

        full_context = """[1 sources=session-2,S2T01,S2T02,S2T03 time=2025-01-02T08:00:00Z]
[S2T01] Driver: is my alarm set for tomorrow
[S2T02] Car Assistant: What should I use for the timeofday?
[S2T03] Driver: morning
"""
        self.assertEqual(
            "morning", extract_fragment_reply(question, full_context)
        )

        bm25_context = """[1 score=9 sources=S2T01,session-2 time=2025-01-02T08:00:00Z]
Driver: is my alarm set for tomorrow

[2 score=8 sources=S2T02,session-2 time=2025-01-02T08:00:05Z]
Car Assistant: What should I use for the timeofday?

[3 score=7 sources=S2T03,session-2 time=2025-01-02T08:00:10Z]
Driver: morning
"""
        self.assertEqual("morning", extract_fragment_reply(question, bm25_context))

    def test_plan_is_non_executing_and_marks_unavailable_framework(self):
        profiles = load_profiles(ROOT / "benchmarks/framework_eval/profiles.json")
        plan = build_plan(
            root=ROOT,
            profiles=profiles,
            framework_ids=["bm25", "memorax"],
            datasets=["locomo_refined"],
            split="gate4",
            answer_model="qwen3.8-max",
            judge="qwen3-14b-refined",
            track="unified",
        )
        self.assertEqual("plan-only", plan["execution_policy"])
        self.assertEqual("adapter-ready", plan["runs"][0]["status"])
        self.assertIn("not-open-source", plan["runs"][1]["blockers"])

    def test_downloaded_sources_match_lock(self):
        result = verify_locked_sources(
            ROOT, ROOT / "benchmarks/framework_eval/sources.lock.json"
        )
        self.assertEqual(14, result["source_count"])
        self.assertTrue(result["pass"])

    def test_plugin_catalog_discovers_datasets_frameworks_and_judges(self):
        catalog = PluginCatalog(
            root=ROOT,
            framework_profiles=ROOT / "benchmarks/framework_eval/profiles.json",
            dataset_profiles=ROOT / "benchmarks/framework_eval/datasets.json",
        )
        self.assertIsInstance(catalog.create_memory_adapter("bm25"), BM25Adapter)
        self.assertIsInstance(catalog.create_dataset("locomo_refined"), LoCoMoRefinedDataset)
        self.assertIsInstance(catalog.create_dataset("slurp_memory"), SLURPMemoryDataset)
        self.assertIsInstance(catalog.create_judge("locomo_refined"), DatasetJudge)
        self.assertIsInstance(catalog.create_judge("slurp_memory"), SLURPMemoryJudge)
        self.assertIsInstance(catalog.create_dataset("kvret_memory"), KVRETMemoryDataset)
        self.assertIsInstance(catalog.create_judge("kvret_memory"), DatasetJudge)

    def test_multimodal_content_parts_survive_normalization_and_retrieval(self):
        conversation = self.dataset.conversation("conv-26")
        image_message = next(
            message
            for session in conversation.sessions
            for message in session.messages
            if message.parts
        )
        self.assertEqual("image", image_message.parts[0].type)
        self.assertTrue(image_message.parts[0].uri.startswith("https://"))
        adapter = FullContextAdapter()
        adapter.ingest(conversation)
        question = self.dataset.questions({"conv-26"})[0]
        hits = adapter.search(question, limit=8)
        self.assertTrue(any(hit.parts for hit in hits))
        serialized = [part for hit in hits for part in hit.to_dict()["parts"]]
        self.assertTrue(any(part["uri"] == image_message.parts[0].uri for part in serialized))

    def test_capability_contract_rejects_unsupported_operation(self):
        with self.assertRaises(UnsupportedCapabilityError):
            FullContextAdapter().answer(
                Question("q", "c", "question", ("answer",)), limit=1
            )

    def test_native_answer_runner_writes_separate_track(self):
        class AnswerAdapter(MemoryAdapter):
            adapter_id = "answer-test"
            capabilities = frozenset({"ingest", "answer"})

            def ingest_session(self, conversation_id: str, session: Session) -> None:
                pass

            def answer(self, question: Question, *, limit: int) -> MemoryAnswer:
                return MemoryAnswer("native answer", source_ids=("D1:1",))

        conversation = self.dataset.conversation("conv-26")
        question = self.dataset.questions({"conv-26"})[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            result = NativeAnswerRunner(AnswerAdapter(), output).run(
                [conversation], [question]
            )
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1, result["questions"])
            self.assertEqual("native", row["answer_track"])
            self.assertEqual("native answer", row["predicted_answer"])

    def test_longmemeval_judge_prompt_and_parser_are_strict(self):
        prompt = official_prompt(
            question="When?", answers=("Tuesday",), response="Tuesday",
            category="temporal-reasoning", abstention=False,
        )
        self.assertIn("off-by-one", prompt)
        self.assertTrue(parse_yes_no("Yes."))
        self.assertFalse(parse_yes_no("no"))
        with self.assertRaises(ValueError):
            parse_yes_no("probably")

    def test_deepseek_v4_judge_disables_thinking_with_current_protocol(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "yes"}}],
            "usage": {"completion_tokens": 1},
        }).encode("utf-8")
        config = JudgeConfig(
            metrics=("llm",),
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
        )
        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            text, _ = OpenAIJudgeLLM().complete("judge this", config)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("yes", text)
        self.assertEqual({"type": "disabled"}, payload["thinking"])

    def test_multimodal_answer_payload_is_opt_in(self):
        image = ContentPart(type="image", uri="https://example.test/evidence.png")
        text_answerer = OpenAIAnswerer(AnswerConfig("http://unused", "", "model"))
        multimodal_answerer = OpenAIAnswerer(AnswerConfig(
            "http://unused", "", "model", multimodal=True
        ))
        text_payload = text_answerer.build_payload("question", "context", (image,))
        image_payload = multimodal_answerer.build_payload("question", "context", (image,))
        self.assertIsInstance(text_payload["messages"][1]["content"], str)
        self.assertEqual(
            "image_url", image_payload["messages"][1]["content"][1]["type"]
        )


if __name__ == "__main__":
    unittest.main()
