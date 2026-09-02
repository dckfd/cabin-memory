from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .adapters.adaptive_policy import (
    decide_l0_fast_path,
    decide_l0_fast_path_v2,
    score_l0_candidate,
)
from .adapters.base import MemoryAdapter
from .adapters.tencentdb_http import TencentDBHTTPAdapter
from .runner import RetrievalRunner
from .schema import Conversation, MemoryHit, Message, Question, Session


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_policy_accepts_specific_single_fact_anchor(self):
        decision = decide_l0_fast_path(
            'What destination was selected for the "coffee shop" request?',
            "Driver: find a coffee shop\nAssistant: Starbucks was selected.",
            top_score=0.9,
            second_score=0.4,
        )
        self.assertFalse(decision.fallback)

    def test_policy_expands_update_and_multifact_questions(self):
        decision = decide_l0_fast_path(
            "What is the latest destination after the route was updated?",
            "The latest destination after the route was updated is the airport.",
            top_score=0.9,
            second_score=0.3,
        )
        self.assertTrue(decision.fallback)
        self.assertEqual("complex_or_update_query", decision.reason)

    def test_v2_candidate_rerank_promotes_exact_date_and_quote(self):
        query = (
            'On 2025-01-07, which destination matched the "coffee shop" request?'
        )
        dense_first = score_l0_candidate(
            query,
            "[source_time=2025-01-06] Driver: route to a restaurant",
            backend_rank=0,
        )
        exact_second = score_l0_candidate(
            query,
            "[source_time=2025-01-07] Driver: coffee shop destination",
            backend_rank=1,
        )
        self.assertGreater(exact_second.score, dense_first.score)
        self.assertEqual(1.0, exact_second.critical_slot_coverage)

    def test_v2_policy_requires_critical_slots_and_complete_reply_pair(self):
        missing_date = decide_l0_fast_path_v2(
            'On 2025-01-07, what matched the "coffee shop" request?',
            "Driver: coffee shop request matched Starbucks.",
            top_score=0.9,
            second_score=0.3,
        )
        incomplete_pair = decide_l0_fast_path_v2(
            'What did the driver reply about the "temperature"?',
            "Car Assistant: What temperature should I use?",
            top_score=0.9,
            second_score=0.3,
        )
        self.assertEqual("missing_critical_slots", missing_date.reason)
        self.assertTrue(missing_date.fallback)
        self.assertEqual("incomplete_dialogue_pair", incomplete_pair.reason)
        self.assertTrue(incomplete_pair.fallback)

    def _adapter(self, directory: str, **environment: str) -> TencentDBHTTPAdapter:
        manifest = {
            "team_id": "team",
            "user_id": "user",
            "conversations": {
                "conv": {"agent_id": "agent", "task_id": "task"},
            },
        }
        path = Path(directory) / "isolation.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        values = {
            "TDAI_EVAL_ISOLATION_MAP": str(path),
            "TDAI_HTTP_API_VERSION": "v3",
            "TDAI_EVAL_PERSPECTIVE_MODE": "single",
            "TDAI_EVAL_MEMORY_LAYERS": "L0,L1",
            "TDAI_EVAL_RETRIEVAL_POLICY": "adaptive",
            "TDAI_EVAL_L0_WINDOW_RADIUS": "0",
            "TDAI_EVAL_ADAPTIVE_FALLBACK_L2_K": "0",
            "TDAI_EVAL_ADAPTIVE_FALLBACK_L3_K": "0",
            "TDAI_EVAL_L23_SCHEDULE": "disabled",
            "TDAI_EVAL_L23_READINESS_MODE": "required",
            **environment,
        }
        patcher = mock.patch.dict("os.environ", values, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        return TencentDBHTTPAdapter("http://unused")

    def test_fast_route_does_not_call_l1(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1,L2,L3",
                TDAI_EVAL_ADAPTIVE_FALLBACK_L2_K="1",
                TDAI_EVAL_ADAPTIVE_FALLBACK_L3_K="1",
                TDAI_EVAL_ADAPTIVE_LAZY_LAYER_READINESS="true",
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        "id": "m1",
                        "content": "[S1] Driver: coffee shop route is Starbucks",
                        "score": 0.9,
                        "role": "user",
                    }, {
                        "id": "m2",
                        "content": "[S2] Driver: unrelated weather",
                        "score": 0.3,
                        "role": "user",
                    }]}}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv", 'Which route matched the "coffee shop" request?',
                ("Starbucks",),
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post) as post, \
                    mock.patch.object(
                        adapter, "_wait_until_ready_layers"
                    ) as readiness:
                adapter.wait_until_ready("conv", timeout=30)
                hits = adapter.search(question, limit=5)
        self.assertEqual(1, len(hits))
        self.assertEqual("fast", hits[0].metadata["retrieval_route"])
        self.assertEqual(1, hits[0].metadata["adaptive_search_calls"])
        self.assertNotIn(
            "/v3/atomic/search", [call.args[0] for call in post.call_args_list]
        )
        self.assertFalse(any(
            call.args[0].startswith(("/v3/scenario/", "/v3/core/"))
            for call in post.call_args_list
        ))
        readiness.assert_not_called()

    def test_v2_adapter_reranks_broad_l0_pool_before_top1(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_ADAPTIVE_POLICY_VERSION="v2",
                TDAI_EVAL_ADAPTIVE_SLOT_RERANK="true",
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        "id": "wrong",
                        "content": (
                            "[W1] [source_time=2025-01-06T08:00:00Z] "
                            "Driver: restaurant route"
                        ),
                        "score": 0.95,
                        "role": "user",
                    }, {
                        "id": "exact",
                        "content": (
                            "[E1] [source_time=2025-01-07T08:00:00Z] "
                            "Driver: coffee shop destination was Starbucks"
                        ),
                        "score": 0.70,
                        "role": "user",
                    }]}}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv",
                'On 2025-01-07, which destination matched the "coffee shop" request?',
                ("Starbucks",),
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=5)

        self.assertEqual(("E1",), hits[0].source_ids)
        self.assertEqual("v2", hits[0].metadata["adaptive_policy_version"])
        self.assertEqual(1, hits[0].metadata["adapter_rerank_position"])

    def test_local_window_cap_keeps_anchor_and_nearest_response(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0",
                TDAI_EVAL_L0_WINDOW_BEFORE="4",
                TDAI_EVAL_L0_WINDOW_AFTER="4",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="3",
            )
            history = [{
                "id": f"b{index}",
                "session_id": "source-session",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"[S{index}] Driver: turn {index}",
                "timestamp": f"2026-01-01T08:00:{index:02d}Z",
            } for index in range(9)]
            history[4]["content"] = "[S4] Driver: target command"

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        **history[4], "score": 0.9,
                    }, {
                        **history[0], "score": 0.1,
                    }]}}
                if endpoint == "/v3/conversation/query":
                    return {"data": {"messages": history, "total": len(history)}}
                raise AssertionError(endpoint)

            question = Question("q", "conv", "target command", ("target",))
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=5)

        self.assertEqual(3, hits[0].metadata["window_message_count"])
        self.assertTrue(hits[0].metadata["window_truncated"])
        self.assertIn("[S3]", hits[0].content)
        self.assertIn("[S4]", hits[0].content)
        self.assertIn("[S5]", hits[0].content)
        self.assertNotIn("[S2]", hits[0].content)

    def test_short_session_keeps_complete_slot_filling_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0",
                TDAI_EVAL_L0_WINDOW_BEFORE="1",
                TDAI_EVAL_L0_WINDOW_AFTER="1",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="8",
            )
            history = [{
                "id": f"b{index}",
                "session_id": "source-session",
                "role": "assistant",
                "content": f"[S{index}] turn {index}",
                "timestamp": f"2026-01-01T08:00:{index:02d}Z",
            } for index in range(6)]
            history[0]["content"] = "[S0] Driver: weather in Seattle"
            history[3]["content"] = (
                "[S3] Car Assistant: two day forecast"
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        **history[3], "score": 0.9,
                    }]}}
                if endpoint == "/v3/conversation/query":
                    return {
                        "data": {"messages": history, "total": len(history)}
                    }
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv", 'forecast for "two day"', ("Seattle",)
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=5)

        self.assertEqual(6, hits[0].metadata["window_message_count"])
        self.assertTrue(hits[0].metadata["window_full_short_session"])
        self.assertFalse(hits[0].metadata["window_truncated"])
        self.assertIn("[S0] Driver: weather in Seattle", hits[0].content)
        self.assertIn(
            "[S3] Car Assistant: two day forecast", hits[0].content
        )

    def test_conversation_batch_without_trace_skips_unsafe_window_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_WRITE_POLICY="cockpit_episode_v2",
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_TYPED_EPISODE_HEADERS="false",
                TDAI_EVAL_CONSTRUCTION_TRACE="",
                TDAI_EVAL_L0_WINDOW_BEFORE="2",
                TDAI_EVAL_L0_WINDOW_AFTER="12",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="8",
            )
            history = [{
                "id": "first",
                "session_id": "shared-transport",
                "role": "user",
                "content": "[S01T01] Driver: navigate home",
                "timestamp": "2026-01-01T08:00:00Z",
            }, {
                "id": "second",
                "session_id": "shared-transport",
                "role": "user",
                "content": "[S02T01] Driver: play jazz",
                "timestamp": "2026-01-02T08:00:00Z",
            }]

            with mock.patch.object(adapter, "_post", return_value={
                "data": {"messages": history, "total": len(history)},
            }):
                expanded = adapter._expand_v3_l0_windows(
                    "conv",
                    [{
                        "content": history[0]["content"],
                        "score": 0.9,
                        "source_ids": ["S01T01"],
                        "metadata": {
                            "level": "L0", "backend_message_id": "first",
                        },
                    }],
                    query="navigate home",
                )

        self.assertEqual("[S01T01] Driver: navigate home", expanded[0]["content"])
        self.assertNotIn("play jazz", expanded[0]["content"])
        self.assertEqual(
            "missing_source_session_provenance",
            expanded[0]["metadata"]["window_expansion_skipped"],
        )

    def test_low_confidence_route_fetches_top3_and_l1(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        "id": "m1", "content": "[S1] weather is sunny",
                        "score": 0.5, "role": "assistant",
                    }, {
                        "id": "m2", "content": "[S2] play some music",
                        "score": 0.49, "role": "user",
                    }]}}
                if endpoint == "/v3/atomic/search":
                    return {"data": {"items": [{
                        "id": "l1", "content": "Driver prefers the airport route",
                        "score": 0.8, "type": "preference",
                    }]}}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv", "Which destination matches my commute preference?",
                ("airport",),
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post) as post:
                hits = adapter.search(question, limit=5)
        self.assertEqual("fallback", hits[0].metadata["retrieval_route"])
        self.assertEqual(2, hits[0].metadata["adaptive_search_calls"])
        self.assertEqual(
            {"L0", "L1"}, {hit.metadata["level"] for hit in hits}
        )
        self.assertIn(
            "/v3/atomic/search", [call.args[0] for call in post.call_args_list]
        )

    def test_low_confidence_route_adds_l2_and_l3_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1,L2,L3",
                TDAI_EVAL_ADAPTIVE_FALLBACK_L2_K="1",
                TDAI_EVAL_ADAPTIVE_FALLBACK_L3_K="1",
                TDAI_EVAL_ADAPTIVE_FALLBACK_CONTEXT_CHARS="2600",
                TDAI_EVAL_L23_SCHEDULE="buffered_dirty_event",
                TDAI_EVAL_ADAPTIVE_LAZY_LAYER_READINESS="true",
                TDAI_EVAL_ADAPTIVE_LAYER_WAIT_BUDGET_SECONDS="30",
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        "id": "m1", "content": "[S1] weather is sunny",
                        "score": 0.5, "role": "assistant",
                    }, {
                        "id": "m2", "content": "[S2] play some music",
                        "score": 0.49, "role": "user",
                    }]}}
                if endpoint == "/v3/atomic/search":
                    return {"data": {"items": [{
                        "id": "l1", "content": "Driver prefers quiet routes",
                        "score": 0.8, "type": "preference",
                    }]}}
                if endpoint == "/v3/scenario/ls":
                    return {"data": {"entries": [{
                        "path": "commute.md", "summary": "commute preferences",
                    }]}}
                if endpoint == "/v3/scenario/read":
                    return {"data": {
                        "content": "Use the airport route for the morning commute.",
                        "version": 2,
                    }}
                if endpoint == "/v3/core/read":
                    return {"data": {
                        "content": "The driver values predictable travel time.",
                        "version": 3,
                    }}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv", "What is the latest commute preference and why?",
                ("airport",),
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post) as post, \
                    mock.patch.object(
                        adapter, "_wait_until_ready_layers"
                    ) as readiness:
                adapter.wait_until_ready("conv", timeout=30)
                hits = adapter.search(question, limit=7)

        self.assertEqual(
            {"L0", "L1", "L2", "L3"},
            {hit.metadata["level"] for hit in hits},
        )
        self.assertEqual(2, hits[0].metadata["adaptive_profile_hits"])
        self.assertEqual(
            ["L2", "L3"], hits[0].metadata["adaptive_profile_levels"]
        )
        self.assertEqual(2600, hits[0].metadata[
            "retrieval_context_budget_chars"
        ])
        endpoints = [call.args[0] for call in post.call_args_list]
        self.assertIn("/v3/scenario/ls", endpoints)
        self.assertIn("/v3/core/read", endpoints)
        self.assertEqual(
            [frozenset({"L1"}), frozenset({"L0", "L1", "L2", "L3"})],
            [call.kwargs["layers"] for call in readiness.call_args_list],
        )
        self.assertEqual(
            ["L1", "L2", "L3"],
            hits[0].metadata["adaptive_readiness_layers"],
        )

    def test_zero_wait_budget_skips_profiles_that_are_still_building(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1,L2,L3",
                TDAI_EVAL_ADAPTIVE_FALLBACK_L2_K="1",
                TDAI_EVAL_ADAPTIVE_FALLBACK_L3_K="1",
                TDAI_EVAL_L23_SCHEDULE="buffered_dirty_event",
                TDAI_EVAL_ADAPTIVE_LAZY_LAYER_READINESS="true",
                TDAI_EVAL_ADAPTIVE_LAYER_WAIT_BUDGET_SECONDS="0",
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        "id": "m1", "content": "[S1] weather is sunny",
                        "score": 0.5, "role": "assistant",
                    }, {
                        "id": "m2", "content": "[S2] play some music",
                        "score": 0.49, "role": "user",
                    }]}}
                if endpoint == "/v3/atomic/search":
                    return {"data": {"items": []}}
                if endpoint == "/v3/scenario/count":
                    return {"data": {"total": 0}}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv", "What is my latest commute preference and why?",
                ("airport",),
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post) as post, \
                    mock.patch.object(
                        adapter, "_wait_until_ready_layers"
                    ) as readiness:
                adapter.wait_until_ready("conv", timeout=30)
                hits = adapter.search(question, limit=7)

        readiness.assert_not_called()
        endpoints = [call.args[0] for call in post.call_args_list]
        self.assertIn("/v3/scenario/count", endpoints)
        self.assertNotIn("/v3/scenario/ls", endpoints)
        self.assertEqual(
            ["L2", "L3"], hits[0].metadata["adaptive_unready_layers"]
        )

    def test_runner_enforces_route_specific_context_budget(self):
        class RoutedAdapter(MemoryAdapter):
            adapter_id = "routed"
            capabilities = frozenset({"search"})

            def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
                return [MemoryHit(
                    "x" * 200,
                    1.0,
                    metadata={
                        "retrieval_policy": "adaptive_l0_first_v1",
                        "retrieval_route": "fast",
                        "retrieval_context_budget_chars": 80,
                    },
                )]

        conversation = Conversation("conv", (Session(
            "session", "", (Message("m", "user", "hello"),),
        ),))
        question = Question("q", "conv", "question", ("answer",))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            RetrievalRunner(
                RoutedAdapter(), output, ingest=False, max_context_chars=1000
            ).run([conversation], [question])
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(80, row["metrics"]["context_budget_chars"])
        self.assertEqual(80, row["metrics"]["context_chars"])
        self.assertTrue(row["metrics"]["context_truncated"])

    def test_runner_completes_current_message_with_bounded_overflow(self):
        class RoutedAdapter(MemoryAdapter):
            adapter_id = "routed"
            capabilities = frozenset({"search"})

            def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
                return [MemoryHit(
                    "first\ncritical answer\ntrailing",
                    1.0,
                    metadata={
                        "retrieval_context_budget_chars": 30,
                        "retrieval_context_line_overflow_chars": 20,
                    },
                )]

        conversation = Conversation("conv", (Session(
            "session", "", (Message("m", "user", "hello"),),
        ),))
        question = Question("q", "conv", "question", ("answer",))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retrieval.jsonl"
            RetrievalRunner(RoutedAdapter(), output, ingest=False).run(
                [conversation], [question]
            )
            row = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(30, row["metrics"]["context_budget_chars"])
        self.assertGreater(row["metrics"]["context_chars"], 30)
        self.assertLessEqual(row["metrics"]["context_budget_overflow_chars"], 20)
        self.assertEqual(
            "completed_current_line", row["metrics"]["context_boundary_action"]
        )
        self.assertTrue(row["context"].endswith("critical answer\n"))
        self.assertNotIn("trailing", row["context"])

    def test_relative_query_time_adds_pre_topk_source_time_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0",
                TDAI_EVAL_TEMPORAL_QUERY_MODE="interval_v1",
                TDAI_EVAL_TEMPORAL_QUERY_RESULTS="2",
                TDAI_EVAL_L0_WINDOW_BEFORE="1",
                TDAI_EVAL_L0_WINDOW_AFTER="1",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="8",
            )
            history = [{
                "id": "old-user",
                "session_id": "old-session",
                "role": "user",
                "content": (
                    "[OLD1] [source_time=2026-08-22T09:00:00+08:00] "
                    "Driver: 导航去旧机场"
                ),
                "timestamp": "2026-08-22T09:00:00+08:00",
            }, {
                "id": "target-user",
                "session_id": "target-session",
                "role": "user",
                "content": (
                    "[TARGET1] [source_time=2026-08-23T09:00:00+08:00] "
                    "Driver: 导航去虹桥机场"
                ),
                "timestamp": "2026-08-23T09:00:00+08:00",
            }, {
                "id": "target-assistant",
                "session_id": "target-session",
                "role": "assistant",
                "content": (
                    "[TARGET2] [source_time=2026-08-23T09:00:01+08:00] "
                    "Car Assistant: 已开始导航到虹桥机场"
                ),
                "timestamp": "2026-08-23T09:00:01+08:00",
            }]
            backend_queries: list[str] = []

            def fake_post(endpoint, body):
                if endpoint == "/v3/conversation/search":
                    backend_queries.append(body["query"])
                    return {"data": {"messages": [{
                        **history[0], "score": 0.95,
                    }]}}
                if endpoint == "/v3/conversation/query":
                    return {"data": {"messages": history, "total": len(history)}}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv", "我昨天上午让你导航去哪来着？", ("虹桥机场",),
                metadata={
                    "query_time": "2026-08-24T15:30:00+08:00",
                    "timezone": "Asia/Shanghai",
                },
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=3)

        self.assertIn("虹桥机场", hits[0].content)
        self.assertEqual("interval_v1", hits[0].metadata["temporal_query_mode"])
        self.assertEqual(1, hits[0].metadata["temporal_candidate_hits"])
        self.assertEqual(
            ["mentioned_at"], hits[0].metadata["temporal_match_dimensions"]
        )
        self.assertIn("normalized_query_time", backend_queries[0])
        self.assertIn("2026-08-23", backend_queries[0])

    def test_relative_query_matches_event_time_distinct_from_mention_time(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0",
                TDAI_EVAL_TEMPORAL_QUERY_MODE="interval_v1",
                TDAI_EVAL_TEMPORAL_QUERY_RESULTS="2",
            )
            history = [{
                "id": "planned",
                "session_id": "plan-session",
                "role": "user",
                "content": (
                    "[PLAN1] [source_time=2026-08-22T20:00:00+08:00] "
                    "Driver: 明天上午导航去虹桥机场"
                ),
                "timestamp": "2026-08-22T20:00:00+08:00",
            }]

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": []}}
                if endpoint == "/v3/conversation/query":
                    return {"data": {"messages": history, "total": 1}}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv", "我昨天上午计划导航去哪？", ("虹桥机场",),
                metadata={
                    "query_time": "2026-08-24T15:30:00+08:00",
                    "timezone": "Asia/Shanghai",
                },
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=2)

        self.assertEqual(1, len(hits))
        self.assertIn("虹桥机场", hits[0].content)
        self.assertEqual(
            ["event_time"], hits[0].metadata["temporal_match_dimensions"]
        )

    def test_relative_query_without_anchor_does_not_guess_or_scan_history(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0",
                TDAI_EVAL_TEMPORAL_QUERY_MODE="interval_v1",
            )
            posted: list[tuple[str, dict]] = []

            def fake_post(endpoint, body):
                posted.append((endpoint, body))
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        "id": "m1", "session_id": "s1", "role": "user",
                        "content": "[S1] Driver: old route", "score": 0.9,
                    }]}}
                raise AssertionError(endpoint)

            question = Question("q", "conv", "昨天导航去了哪里？", ("机场",))
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=2)

        self.assertEqual(1, len(hits))
        self.assertEqual("昨天导航去了哪里？", posted[0][1]["query"])
        self.assertNotIn(
            "/v3/conversation/query", [endpoint for endpoint, _ in posted]
        )
        self.assertEqual(0, hits[0].metadata["temporal_candidate_hits"])

    def test_temporal_interaction_anchor_keeps_long_episode_start_and_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0",
                TDAI_EVAL_TEMPORAL_QUERY_MODE="interval_v1",
                TDAI_EVAL_TEMPORAL_QUERY_RESULTS="2",
                TDAI_EVAL_L0_WINDOW_BEFORE="2",
                TDAI_EVAL_L0_WINDOW_AFTER="12",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="8",
            )
            history = []
            contents = [
                "Driver: route to the nearest hospital",
                "Car Assistant: Stanford Childrens Health is three miles away",
                "Driver: yes",
                "Car Assistant: selecting a fast route",
                "Driver: okay",
                "Car Assistant: the route has no traffic",
                "Driver: please continue",
                "Car Assistant: Stanford Childrens Health was selected",
                "Driver: thank you",
                "Car Assistant: you're welcome",
            ]
            for index, content in enumerate(contents, 1):
                history.append({
                    "id": f"backend-{index}",
                    "session_id": "long-session",
                    "role": "user" if index % 2 else "assistant",
                    "content": (
                        f"[T{index:02d}] "
                        "[source_time=2024-01-19T08:00:00Z] "
                        f"{content}"
                    ),
                    "timestamp": f"2024-01-19T08:00:{index:02d}Z",
                })

            def fake_post(endpoint, _body):
                if endpoint == "/v3/conversation/search":
                    return {"data": {"messages": [{
                        **history[8], "score": 0.95,
                    }]}}
                if endpoint == "/v3/conversation/query":
                    return {"data": {"messages": history, "total": len(history)}}
                raise AssertionError(endpoint)

            question = Question(
                "q", "conv",
                (
                    "During the vehicle interaction from the day before "
                    "yesterday, which destination was selected for the "
                    "\"hospital\" request?"
                ),
                ("Stanford Childrens Health",),
                metadata={"query_time": "2024-01-21T15:30:00Z"},
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=3)

        self.assertIn("[T01]", hits[0].content)
        self.assertIn("[T08]", hits[0].content)
        self.assertNotIn("[T09]", hits[0].content)
        self.assertEqual(8, hits[0].metadata["window_message_count"])
        self.assertEqual(
            ("quote:hospital", "date:20240119"),
            hits[0].metadata["temporal_anchor_relevance"][
                "matched_critical_slots"
            ],
        )

    def test_confident_temporal_candidate_skips_backend_ann_search(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0",
                TDAI_EVAL_TEMPORAL_QUERY_MODE="interval_v1",
                TDAI_EVAL_TEMPORAL_SHORT_CIRCUIT="true",
                TDAI_EVAL_L0_WINDOW_AFTER="4",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="8",
            )
            history = [{
                "id": "target",
                "session_id": "target-session",
                "role": "user",
                "content": (
                    "[T01] [source_time=2026-08-23T09:00:00+08:00] "
                    "Driver: 导航去虹桥机场"
                ),
                "timestamp": "2026-08-23T09:00:00+08:00",
            }, {
                "id": "reply",
                "session_id": "target-session",
                "role": "assistant",
                "content": (
                    "[T02] [source_time=2026-08-23T09:00:01+08:00] "
                    "Car Assistant: 已开始导航去虹桥机场"
                ),
                "timestamp": "2026-08-23T09:00:01+08:00",
            }]
            endpoints: list[str] = []

            def fake_post(endpoint, _body):
                endpoints.append(endpoint)
                if endpoint == "/v3/conversation/query":
                    return {"data": {"messages": history, "total": 2}}
                raise AssertionError(f"unexpected backend call: {endpoint}")

            question = Question(
                "q", "conv", "我昨天让你导航去\"虹桥机场\"了吗？",
                ("虹桥机场",),
                metadata={
                    "query_time": "2026-08-24T15:30:00+08:00",
                    "timezone": "Asia/Shanghai",
                },
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                hits = adapter.search(question, limit=3)

        self.assertIn("虹桥机场", hits[0].content)
        self.assertEqual(0, hits[0].metadata["adaptive_search_calls"])
        self.assertTrue(hits[0].metadata["temporal_short_circuit_used"])
        self.assertNotIn("/v3/conversation/search", endpoints)

    def test_typed_episode_trace_skips_every_backend_read(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "construction.jsonl"
            trace_path.write_text(json.dumps({
                "conversation_id": "conv",
                "session_id": "nav-session",
                "agent_id": "agent",
                "pipeline_user_rounds": 0,
                "decision": {"extract_l1": False},
                "source_ids": ["N1", "N2", "N3"],
                "typed_cockpit_episode": {
                    "schema_version": 1,
                    "scene": "navigation",
                    "intent": "navigation.set_destination",
                    "state": "confirmed",
                    "slots": {"destination": "虹桥机场"},
                    "aliases": [],
                    "source_ids": ["N1", "N2"],
                    "confidence": 0.995,
                    "selection_actor": "assistant",
                    "mentioned_at": "2026-08-23T09:00:00+08:00",
                    "request_text": "find an airport",
                    "transitions": [{
                        "action": "structured_slot",
                        "value": "虹桥机场",
                        "source_id": "N1",
                        "actor": "user",
                        "sequence": 0,
                    }, {
                        "action": "navigate",
                        "value": "虹桥机场",
                        "source_id": "N2",
                        "actor": "assistant",
                        "sequence": 1,
                    }],
                },
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            adapter = self._adapter(
                directory,
                TDAI_EVAL_CONSTRUCTION_TRACE=str(trace_path),
                TDAI_EVAL_L1_WRITE_POLICY="cockpit_episode_v2",
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_TYPED_COCKPIT_EPISODES="true",
                TDAI_EVAL_TYPED_EPISODE_SHORT_CIRCUIT="true",
                TDAI_EVAL_TEMPORAL_QUERY_MODE="interval_v1",
            )
            question = Question(
                "q", "conv",
                (
                    "During yesterday's vehicle interaction, which destination "
                    "did the car assistant select for the driver's "
                    "\"airport\" request?"
                ),
                ("虹桥机场",),
                metadata={
                    "query_time": "2026-08-24T15:30:00+08:00",
                    "timezone": "Asia/Shanghai",
                },
            )
            with mock.patch.object(
                adapter, "_post", side_effect=AssertionError("backend read")
            ):
                hits = adapter.search(question, limit=3)
                chinese_hits = adapter.search(Question(
                    "q-zh", "conv", "我昨天上午让你导航去哪儿了？",
                    ("虹桥机场",),
                    metadata={
                        "query_time": "2026-08-24T15:30:00+08:00",
                        "timezone": "Asia/Shanghai",
                    },
                ), limit=3)

        self.assertEqual(1, len(hits))
        self.assertEqual(("N1", "N2", "N3"), hits[0].source_ids)
        self.assertEqual("L0T", hits[0].metadata["level"])
        self.assertEqual(
            "typed_episode_trace_v1",
            hits[0].metadata["retrieval_strategy"],
        )
        self.assertEqual(0, hits[0].metadata["adaptive_search_calls"])
        self.assertTrue(
            hits[0].metadata["typed_episode_short_circuit_used"]
        )
        self.assertFalse(hits[0].metadata["temporal_short_circuit_used"])
        self.assertEqual("虹桥机场", (
            chinese_hits[0].metadata["typed_cockpit_episode"]["slots"]
            ["destination"]
        ))
        self.assertTrue(
            chinese_hits[0].metadata["typed_episode_short_circuit_used"]
        )

    def test_typed_episode_sqlite_survives_restart_without_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "construction.jsonl"
            index_path = Path(directory) / "typed-episodes.sqlite3"
            environment = {
                "TDAI_EVAL_CONSTRUCTION_TRACE": str(trace_path),
                "TDAI_EVAL_TYPED_EPISODE_INDEX_PATH": str(index_path),
                "TDAI_EVAL_TYPED_EPISODE_INDEX_REQUIRED": "true",
                "TDAI_EVAL_L1_WRITE_POLICY": "cockpit_episode_v2",
                "TDAI_EVAL_L1_BATCH_MODE": "conversation",
                "TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS": "true",
                "TDAI_EVAL_TYPED_COCKPIT_EPISODES": "true",
                "TDAI_EVAL_TYPED_EPISODE_SHORT_CIRCUIT": "true",
                "TDAI_EVAL_TEMPORAL_QUERY_MODE": "interval_v1",
            }
            writer = self._adapter(directory, **environment)
            session = Session(
                "nav-session",
                "2026-08-23T09:00:00+08:00",
                (
                    Message(
                        "N1", "user", "Find an airport.", speaker="Driver",
                        timestamp="2026-08-23T09:00:00+08:00",
                    ),
                    Message(
                        "N2", "assistant",
                        "Navigating to Hongqiao Airport.",
                        speaker="Car Assistant",
                        timestamp="2026-08-23T09:00:05+08:00",
                    ),
                ),
                {"source_domain": "navigation"},
            )
            with mock.patch.object(
                writer, "_post", return_value={"code": 0, "data": {}}
            ):
                writer.ingest_session("conv", session)
            writer_metrics = writer.construction_metrics()
            trace_path.unlink()

            reader = self._adapter(directory, **environment)
            question = Question(
                "q", "conv",
                (
                    "During yesterday's vehicle interaction, which destination "
                    "did the car assistant select for the driver's "
                    '"airport" request?'
                ),
                ("Hongqiao Airport",),
                metadata={
                    "query_time": "2026-08-24T15:30:00+08:00",
                    "timezone": "Asia/Shanghai",
                },
            )
            with mock.patch.object(
                reader, "_post", side_effect=AssertionError("backend read")
            ):
                hits = reader.search(question, limit=3)
            reader_metrics = reader.construction_metrics()

        self.assertEqual(1, writer_metrics["typed_episode_index_writes"])
        self.assertTrue(writer_metrics["typed_episode_index_available"])
        self.assertEqual(1, len(hits))
        self.assertIn("Hongqiao Airport", hits[0].content)
        self.assertEqual(
            "typed_episode_sqlite_v1",
            hits[0].metadata["retrieval_strategy"],
        )
        self.assertEqual(1, hits[0].metadata["typed_episode_record_revision"])
        self.assertEqual(1, reader_metrics["typed_episode_index_reads"])

    def test_bounded_relative_day_does_not_force_adaptive_fallback(self):
        decision = decide_l0_fast_path_v2(
            (
                "During the vehicle interaction from the day before "
                "yesterday, what destination matched the \"shopping center\"?\n"
                "[normalized_query_time: day before yesterday=2024-01-03 (day)]"
            ),
            (
                "[source_time=2024-01-03T08:00:00Z] Driver: Find a shopping "
                "center. Car Assistant: Navigating to Stanford Shopping Center."
            ),
            top_score=1.0,
            second_score=0.2,
        )

        self.assertFalse(decision.complex_query)
        self.assertFalse(decision.fallback)
        self.assertEqual((), decision.missing_critical_slots)

    def test_quoted_now_slot_is_not_a_latest_state_operator(self):
        decision = decide_l0_fast_path_v2(
            (
                "During last Monday's vehicle interaction, which location "
                "did the driver specify for the forecast on \"right now\"?\n"
                "[normalized_query_time: last Monday=2024-01-29 (day)]"
            ),
            (
                "[source_time=2024-01-29T08:00:00Z] Driver: San Mateo "
                "right now. Car Assistant: It is windy in San Mateo."
            ),
            top_score=1.9,
            second_score=0.3,
        )

        self.assertFalse(decision.complex_query)
        self.assertFalse(decision.fallback)
        self.assertTrue(decision.quoted_anchor_match)

    def test_possessive_apostrophe_is_not_a_quoted_critical_slot(self):
        decision = decide_l0_fast_path_v2(
            (
                "During yesterday's vehicle interaction, what time was the "
                "\"swimming\" reminder?\n"
                "[normalized_query_time: yesterday=2024-01-01 (day)]"
            ),
            (
                "[source_time=2024-01-01T08:00:00Z] Driver: Schedule swimming "
                "at 3 pm."
            ),
            top_score=1.0,
            second_score=0.1,
        )

        self.assertNotIn(
            "quote:svehicleinteractionwhattimewastheswimming",
            decision.missing_critical_slots,
        )
        self.assertEqual((), decision.missing_critical_slots)


if __name__ == "__main__":
    unittest.main()
