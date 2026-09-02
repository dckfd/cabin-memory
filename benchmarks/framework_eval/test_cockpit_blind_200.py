from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarks.framework_eval.datasets.longmemeval import LongMemEvalDataset


ROOT = Path(__file__).resolve().parent / "challenges" / "cockpit_blind_200_v1"


def test_blind_200_is_sealed_and_balanced() -> None:
    dataset = LongMemEvalDataset(ROOT)
    questions = dataset.questions()
    assert len(dataset.conversations()) == 25
    assert len(questions) == 200
    assert len({question.question_id for question in questions}) == 200
    assert sum(bool(question.metadata["is_abstention"]) for question in questions) == 25
    raw = [json.loads(line) for line in (ROOT / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert Counter(row["metadata"]["ability"] for row in raw) == {
        "aggregation-frequency": 25,
        "latest-final-update": 25,
        "two-date-validity": 25,
        "multi-person-cross-session": 25,
        "update-cancel-negation": 25,
        "conditional-priority": 25,
        "insufficient-evidence-abstention": 25,
        "multi-target-final-state": 25,
    }


def test_blind_200_evidence_ids_exist() -> None:
    conversations = [json.loads(line) for line in (ROOT / "conversations.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [json.loads(line) for line in (ROOT / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    evidence_ids = {
        f"{session['source_session_id']}:{index:03d}"
        for conversation in conversations
        for session in conversation["sessions"]
        for index, _ in enumerate(session["messages"], 1)
    }
    assert all(item in evidence_ids for row in questions for item in row["answer_session_ids"])
