from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .schema import Conversation, Message, Question, Session
from .provision_tencentdb import load_reused_principal
from .selective_memory_ablation import (
    DEFAULT_PROTOCOL,
    SelectiveMemoryAblation,
    _selected_protocol,
    arm_environment,
    construction_environment,
    merge_retrieval_shards,
    resolve_arm,
    resolve_store_groups,
    shard_selection_manifest,
    select_smoke_manifest,
    summarize_arm,
    validate_execution_subset,
)


class _Dataset:
    def __init__(self, questions):
        self._questions = questions

    def questions(self):
        return list(self._questions)

    def conversation(self, conversation_id):
        return Conversation(str(conversation_id), (Session(
            f"{conversation_id}-session", "",
            (Message("message", "user", "hello"),),
        ),))


class SelectiveMemoryAblationTests(unittest.TestCase):
    def test_selective_profile_serializes_agent_scoped_l23_workers(self):
        profile = (
            Path(__file__).parent / "configs" /
            "tencentdb_selective_async_l23.env"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "export MEMORY_PIPELINE_WORKER_CONCURRENCY=1", profile
        )
        self.assertIn("export TDAI_EVAL_L1_BATCH_MODE=conversation", profile)
        self.assertIn(
            "export TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS=true", profile
        )
        self.assertIn(
            "export TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES=1", profile
        )
        self.assertIn(
            "export TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL=l0_only", profile
        )
        self.assertIn(
            "export TDAI_EVAL_L1_ZERO_OUTPUT_WAIT_FOR_REPAIR=false", profile
        )
        self.assertIn(
            'export MEMORY_LLM_MODEL="${TDAI_EVAL_CONSTRUCTION_MODEL:-qwen3.6-flash}"',
            profile,
        )
        self.assertIn("export MEMORY_MAX_MESSAGES_PER_EXTRACTION=40", profile)
        self.assertIn("export MEMORY_PIPELINE_L2_DELAY_SECONDS=180", profile)

    def test_buffered_profiles_run_each_dataset_end_to_end(self):
        events = []

        class _Child:
            def __init__(self, dataset_id):
                self.dataset_id = dataset_id

            def ingest(self, *, smoke_only):
                events.append((self.dataset_id, "ingest", smoke_only))

            def retrieve(self, *, smoke):
                events.append((self.dataset_id, "retrieve", smoke))

            def answer(self, *, smoke):
                events.append((self.dataset_id, "answer", smoke))

            def score(self, *, smoke):
                events.append((self.dataset_id, "score", smoke))

        runner = object.__new__(SelectiveMemoryAblation)
        runner.datasets = {"first": None, "second": None}
        runner.stores = {
            "selective": {"l23_schedule": "buffered_dirty_event"},
        }
        runner._dataset_runner = lambda dataset_id: _Child(dataset_id)

        runner.run_end_to_end(smoke=True)

        self.assertEqual([
            ("first", "ingest", True),
            ("first", "retrieve", True),
            ("first", "answer", True),
            ("first", "score", True),
            ("second", "ingest", True),
            ("second", "retrieve", True),
            ("second", "answer", True),
            ("second", "score", True),
        ], events)

    def test_split_profile_stage_requires_one_dataset(self):
        runner = object.__new__(SelectiveMemoryAblation)
        runner.datasets = {"first": None, "second": None}
        runner.stores = {
            "selective": {"l23_schedule": "buffered_dirty_event"},
        }

        with self.assertRaisesRegex(RuntimeError, "exactly one --dataset"):
            runner.require_safe_standalone_stage("ingest")

    def test_reused_principal_loads_key_without_copying_it_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "isolation.json"
            key = root / ".user-key"
            manifest.write_text(json.dumps({
                "team_id": "team",
                "user_id": "user",
                "service_id": "default",
            }), encoding="utf-8")
            key.write_text("private-value\n", encoding="utf-8")

            loaded = load_reused_principal(
                manifest, key, service_id="default"
            )

        self.assertEqual(("team", "user", "private-value"), loaded)

    def test_protocol_resolves_five_paired_arms_and_two_stores(self):
        protocol = _selected_protocol(DEFAULT_PROTOCOL, phase=1)
        arms = [resolve_arm(value) for value in protocol["arms"]]
        groups = resolve_store_groups(arms)

        self.assertEqual(5, len(arms))
        self.assertEqual({"full", "selective"}, set(groups))
        self.assertEqual(
            "buffered_dirty_event", groups["selective"]["l23_schedule"]
        )
        adaptive = next(
            value for value in arms
            if value["id"] == "selective-l1-async-l23"
        )
        self.assertEqual(7, adaptive["resolved_top_k"])
        self.assertEqual(2600, adaptive["resolved_max_context_chars"])
        full = next(value for value in arms if value["id"] == "full-l0-l1")
        self.assertEqual(1, full["retrieval"]["window_before"])
        self.assertEqual(12, full["retrieval"]["window_after"])

    def test_filtered_execution_is_validated_against_full_frozen_plan(self):
        frozen = {
            "protocol_id": "p",
            "phase": 1,
            "datasets": [{"dataset": "d1"}, {"dataset": "d2"}],
            "arms": [{"id": "a1"}, {"id": "a2"}],
        }
        validate_execution_subset(frozen, {
            "protocol_id": "p",
            "datasets": [{"dataset": "d1"}],
            "arms": [{"id": "a2"}],
        }, phase=1)

        with self.assertRaisesRegex(RuntimeError, "absent from frozen"):
            validate_execution_subset(frozen, {
                "protocol_id": "p",
                "datasets": [{"dataset": "d3"}],
                "arms": [{"id": "a1"}],
            }, phase=1)
        with self.assertRaisesRegex(RuntimeError, "different phase"):
            validate_execution_subset(frozen, {
                "protocol_id": "p",
                "datasets": [{"dataset": "d1"}],
                "arms": [{"id": "a1"}],
            }, phase=2)

    def test_async_arm_environment_keeps_profiles_on_fallback_only(self):
        protocol = _selected_protocol(DEFAULT_PROTOCOL, phase=1)
        arm = next(
            value for value in protocol["arms"]
            if value["id"] == "selective-l1-async-l23"
        )
        environment = arm_environment(arm)

        self.assertEqual("adaptive", environment["TDAI_EVAL_RETRIEVAL_POLICY"])
        self.assertEqual("1", environment["TDAI_EVAL_ADAPTIVE_FAST_L0_K"])
        self.assertEqual("1", environment["TDAI_EVAL_ADAPTIVE_FALLBACK_L2_K"])
        self.assertEqual("1", environment["TDAI_EVAL_ADAPTIVE_FALLBACK_L3_K"])
        self.assertEqual("true", environment[
            "TDAI_EVAL_ADAPTIVE_LAZY_LAYER_READINESS"
        ])
        self.assertEqual(
            "conversation", environment["TDAI_EVAL_L1_BATCH_MODE"]
        )
        self.assertEqual("true", environment[
            "TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS"
        ])
        self.assertEqual("1", environment[
            "TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES"
        ])
        self.assertEqual("l0_only", environment[
            "TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL"
        ])
        self.assertEqual("false", environment[
            "TDAI_EVAL_L1_ZERO_OUTPUT_WAIT_FOR_REPAIR"
        ])
        self.assertEqual("0", environment[
            "TDAI_EVAL_ADAPTIVE_LAYER_WAIT_BUDGET_SECONDS"
        ])
        self.assertEqual("1", environment[
            "TDAI_EVAL_L0_EXPLICIT_DATE_RESULTS"
        ])
        self.assertEqual("true", environment[
            "TDAI_EVAL_L0_EXPLICIT_DATE_BOOST"
        ])
        self.assertEqual("dirty_only", environment[
            "TDAI_EVAL_L23_READINESS_MODE"
        ])

    def test_selective_construction_environment_enables_lossless_compaction(self):
        environment = construction_environment({
            "l1_policy": "cockpit_selective_v1",
            "l23_schedule": "buffered_dirty_event",
        })

        self.assertEqual("conversation", environment[
            "TDAI_EVAL_L1_BATCH_MODE"
        ])
        self.assertEqual("true", environment[
            "TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS"
        ])
        self.assertEqual("1", environment[
            "TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES"
        ])
        self.assertEqual("l0_only", environment[
            "TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL"
        ])
        self.assertEqual("false", environment[
            "TDAI_EVAL_L1_ZERO_OUTPUT_WAIT_FOR_REPAIR"
        ])

    def test_smoke_selection_round_robins_categories_without_labels(self):
        questions = [
            Question(f"q{index}", f"c{index // 2}", "question", ("secret",),
                     category=category, evidence_ids=("gold",))
            for index, category in enumerate(("a", "a", "b", "b", "c", "c"))
        ]
        manifest = {
            "protocol_id": "p",
            "dataset_id": "d",
            "question_ids": [item.question_id for item in questions],
        }
        selected = select_smoke_manifest(
            _Dataset(questions), manifest, count=3, seed="seed"
        )
        categories = {
            next(item.category for item in questions if item.question_id == qa_id)
            for qa_id in selected["question_ids"]
        }

        self.assertEqual({"a", "b", "c"}, categories)
        self.assertFalse(selected["leakage_controls"][
            "answers_read_for_selection"
        ])
        self.assertFalse(selected["leakage_controls"][
            "evidence_ids_read_for_selection"
        ])

    def test_smoke_selection_minimizes_ingested_conversations(self):
        questions = [
            Question(f"compact-{category}", "compact", "question", ("secret",),
                     category=category, evidence_ids=("gold",))
            for category in ("a", "b", "c", "d")
        ]
        questions.extend(
            Question(f"spread-{category}", f"spread-{category}", "question",
                     ("secret",), category=category, evidence_ids=("gold",))
            for category in ("a", "b", "c", "d")
        )
        selected = select_smoke_manifest(
            _Dataset(questions), {
                "protocol_id": "p",
                "dataset_id": "d",
                "question_ids": [item.question_id for item in questions],
            },
            count=4,
            seed="seed",
        )

        self.assertEqual(["compact"], selected["conversation_ids"])
        self.assertEqual(1, selected["counts"]["sessions"])

    def test_merge_retrieval_requires_exact_paired_question_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(json.dumps({
                "question": {"question_id": "q1"},
            }) + "\n", encoding="utf-8")
            second.write_text(json.dumps({
                "question": {"question_id": "q2"},
            }) + "\n", encoding="utf-8")
            output = root / "merged.jsonl"
            result = merge_retrieval_shards(
                [first, second], {"q1", "q2"}, output
            )

            self.assertEqual(2, result["questions"])
            with self.assertRaisesRegex(RuntimeError, "merge mismatch"):
                merge_retrieval_shards([first], {"q1", "q2"}, output)

    def test_retrieval_shard_manifest_only_contains_local_conversations(self):
        questions = [
            Question("q1", "c1", "question", ("one",)),
            Question("q2", "c1", "question", ("two",)),
            Question("q3", "c2", "question", ("three",)),
        ]
        manifest = shard_selection_manifest(
            {
                "protocol_id": "p",
                "dataset_id": "d",
                "question_ids": ["q1", "q2", "q3"],
            },
            _Dataset(questions),
            ["c2"],
            shard_index=1,
            shard_count=2,
        )

        self.assertEqual(["q3"], manifest["question_ids"])
        self.assertEqual(["c2"], manifest["conversation_ids"])
        self.assertEqual(1, manifest["shard_index"])
        self.assertEqual(2, manifest["shard_count"])

    def test_retrieval_shard_manifest_rejects_unselected_conversation(self):
        with self.assertRaisesRegex(RuntimeError, "no selected questions"):
            shard_selection_manifest(
                {
                    "protocol_id": "p",
                    "dataset_id": "d",
                    "question_ids": ["q1"],
                },
                _Dataset([Question("q1", "c1", "question", ("one",))]),
                ["c2"],
                shard_index=0,
                shard_count=1,
            )

    def test_summary_keeps_unavailable_construction_tokens_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retrieval = root / "retrieval.jsonl"
            predictions = root / "predictions.jsonl"
            score = root / "score.json"
            traces = root / "traces"
            traces.mkdir()
            retrieval.write_text(json.dumps({
                "question": {"question_id": "q1"},
                "metrics": {
                    "context_chars": 100,
                    "search_seconds": 0.2,
                    "evidence_recall": 1.0,
                    "retrieval_route": "fast",
                },
            }) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps({
                "qa_id": "q1",
                "usage": {"prompt_tokens": 30, "total_tokens": 35},
            }) + "\n", encoding="utf-8")
            score.write_text(json.dumps({
                "metrics": {
                    "contains": {"mean": 1.0},
                    "exact": {"mean": 1.0},
                },
            }), encoding="utf-8")
            (traces / "one.jsonl").write_text(json.dumps({
                "conversation_id": "c",
                "session_id": "s",
                "agent_id": "a",
                "decision": {"extract_l1": False, "source_characters": 12},
            }) + "\n", encoding="utf-8")
            row = summarize_arm(
                dataset_id="slurp_memory",
                arm={"id": "arm", "store_group": "selective"},
                retrieval_path=retrieval,
                predictions_path=predictions,
                score_path=score,
                trace_dir=traces,
            )

        self.assertEqual(1.0, row["accuracy"])
        self.assertEqual(30.0, row["mean_answer_prompt_tokens"])
        self.assertEqual(0.0, row["construction"]["selection_rate"])
        self.assertIsNone(row["construction"]["actual_total_tokens"])


if __name__ == "__main__":
    unittest.main()
