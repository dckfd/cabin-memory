from __future__ import annotations

import json
import unittest
from pathlib import Path

from .adapters.full_context import FullContextAdapter
from .answering import AnswerConfig, OpenAIAnswerer
from .datasets.longmemeval import LongMemEvalDataset
from .datasets.prepare_cockpit_long_memory_challenge import build


ROOT = Path(__file__).resolve().parent
CHALLENGE_ROOT = ROOT / "challenges" / "cockpit_long_memory_20_v1"


class CockpitLongMemoryChallengeTest(unittest.TestCase):
    def test_checked_in_dataset_matches_generator_and_has_grounded_evidence(self):
        conversations, questions = build()
        self.assertEqual(4, len(conversations))
        self.assertEqual(44, sum(row["session_count"] for row in conversations))
        self.assertEqual(97, sum(row["message_count"] for row in conversations))
        self.assertEqual(20, len(questions))
        self.assertEqual(4, sum(row["is_abstention"] for row in questions))
        self.assertEqual(
            {
                "knowledge-update": 6,
                "multi-session": 9,
                "single-session-preference": 2,
                "temporal-reasoning": 3,
            },
            {
                category: sum(row["category"] == category for row in questions)
                for category in {row["category"] for row in questions}
            },
        )

        conversation_text = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in conversations
        )
        question_text = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in questions
        )
        self.assertEqual(
            conversation_text,
            (CHALLENGE_ROOT / "conversations.jsonl").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            question_text,
            (CHALLENGE_ROOT / "questions.jsonl").read_text(encoding="utf-8"),
        )
        selection = json.loads(
            (CHALLENGE_ROOT / "selection.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [row["qa_id"] for row in questions], selection["question_ids"]
        )
        self.assertEqual(
            [row["sample_id"] for row in conversations],
            selection["conversation_ids"],
        )

        message_ids = {
            f"{session['source_session_id']}:{index:03d}"
            for conversation in conversations
            for session in conversation["sessions"]
            for index, _ in enumerate(session["messages"], 1)
        }
        evidence_ids = {
            value for question in questions for value in question["answer_session_ids"]
        }
        self.assertTrue(evidence_ids)
        self.assertFalse(evidence_ids - message_ids)
        self.assertTrue(all(
            question["answer_session_ids"]
            for question in questions if not question["is_abstention"]
        ))

    def test_existing_longmemeval_adapter_reads_challenge(self):
        dataset = LongMemEvalDataset(CHALLENGE_ROOT)
        self.assertEqual(4, len(dataset.conversations()))
        self.assertEqual(20, len(dataset.questions()))
        message_ids = {
            message.message_id
            for conversation in dataset.conversations()
            for session in conversation.sessions
            for message in session.messages
        }
        self.assertFalse({
            source
            for question in dataset.questions()
            for source in question.evidence_ids
            if source not in message_ids
        })

    def test_hard_questions_do_not_use_current_deterministic_slot_compiler(self):
        dataset = LongMemEvalDataset(CHALLENGE_ROOT)
        adapter = FullContextAdapter()
        for conversation in dataset.conversations():
            adapter.ingest(conversation)
        answerer = OpenAIAnswerer(AnswerConfig(
            base_url="http://invalid",
            api_key="unused",
            model="unused",
            temporal_query_mode="interval_v1",
            temporal_default_timezone="Asia/Shanghai",
            deterministic_slot_mode="cockpit_v1",
        ))
        resolved = []
        for question in dataset.questions():
            hits = adapter.search(question, limit=100)
            candidate = answerer.deterministic_answer(
                question.text,
                adapter.render_hits(hits),
                question.metadata,
                [hit.to_dict() for hit in hits],
            )
            if candidate is not None:
                resolved.append(question.question_id)
        self.assertEqual([], resolved)


if __name__ == "__main__":
    unittest.main()
