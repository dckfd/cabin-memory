from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

from .selective_memory_protocol import select_question_ids, validate_arms


def _question(question_id: str, conversation_id: str, category: str):
    return SimpleNamespace(
        question_id=question_id,
        conversation_id=conversation_id,
        category=category,
    )


class SelectiveMemoryProtocolTests(unittest.TestCase):
    def test_conversation_sampling_never_slices_a_history(self):
        questions = [
            _question(f"q{group}-{index}", f"c{group}", "command")
            for group in range(8)
            for index in range(group + 2)
        ]
        selected, conversations = select_question_ids(
            questions,
            strategy="conversation_hash_until",
            target=20,
            seed="stable",
            minimum=15,
            maximum=25,
        )
        selected_set = set(selected)
        for conversation_id in conversations:
            expected = {
                item.question_id for item in questions
                if item.conversation_id == conversation_id
            }
            self.assertTrue(expected.issubset(selected_set))

    def test_stratified_sampling_is_exact_and_deterministic(self):
        questions = [
            _question(f"a-{index}", f"ca-{index}", "a") for index in range(60)
        ] + [
            _question(f"b-{index}", f"cb-{index}", "b") for index in range(40)
        ]
        first, _ = select_question_ids(
            questions,
            strategy="stratified_question_hash",
            target=50,
            seed="stable",
            minimum=50,
            maximum=50,
        )
        second, _ = select_question_ids(
            list(reversed(questions)),
            strategy="stratified_question_hash",
            target=50,
            seed="stable",
            minimum=50,
            maximum=50,
        )
        categories = Counter(value.split("-", 1)[0] for value in first)
        self.assertEqual({"a": 30, "b": 20}, dict(categories))
        self.assertEqual(first, second)

    def test_arm_contract_rejects_missing_ablation(self):
        with self.assertRaisesRegex(ValueError, "exactly these arms"):
            validate_arms([{
                "id": "full-l0-l1",
                "construction": {"l1_policy": "all"},
                "retrieval": {"policy": "fixed"},
            }])


if __name__ == "__main__":
    unittest.main()
