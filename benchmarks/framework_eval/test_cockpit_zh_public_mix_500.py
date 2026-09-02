from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmarks.framework_eval.datasets.longmemeval import LongMemEvalDataset


ROOT = Path(__file__).resolve().parent / "challenges" / "cockpit_zh_public_mix_500_v4"


def rows(name: str, root: Path = ROOT) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_public_mix_500_loads_and_is_balanced() -> None:
    dataset = LongMemEvalDataset(ROOT)
    questions = rows("questions.jsonl")
    assert len(dataset.conversations()) == 50
    assert len(dataset.questions()) == len(questions) == 500
    assert sum(bool(row["is_abstention"]) for row in questions) == 50
    assert set(Counter(row["metadata"]["ability"] for row in questions).values()) == {50}
    assert set(Counter(row["metadata"]["surface_style"] for row in questions).values()) == {100}
    assert Counter(row["metadata"]["license_class"] for row in questions) == {
        "permissive-public-source": 250,
        "noncommercial-mixed-public-source": 250,
    }


def test_public_mix_500_event_chain_links_are_complete() -> None:
    conversations = rows("conversations.jsonl")
    questions = rows("questions.jsonl")
    evidence_ids = {
        f"{session['source_session_id']}:{index:03d}"
        for conversation in conversations
        for session in conversation["sessions"]
        for index, _ in enumerate(session["messages"], 1)
    }
    expected = {
        "aggregation-frequency": 5,
        "latest-final-update": 2,
        "two-date-validity": 3,
        "multi-person-cross-session": 3,
        "final-cancellation": 3,
        "conditional-priority": 1,
        "insufficient-evidence-abstention": 0,
        "multi-target-final-state": 2,
        "cutoff-state": 2,
        "correction-retained-constraint": 2,
    }
    for row in questions:
        links = row["answer_session_ids"]
        assert len(links) == expected[row["metadata"]["ability"]]
        assert all(link in evidence_ids for link in links)
        assert all(link.startswith(row["sample_id"] + "-s") for link in links)


def test_public_mix_500_lineage_and_license_tracks_are_isolated() -> None:
    conversations = rows("conversations.jsonl")
    lineage = rows("source_lineage.jsonl")
    refs = {row["ref_id"]: row for row in rows("source_refs.jsonl")}
    assert len(lineage) == 2100
    assert all(ref_id in refs for row in lineage for ref_id in row["source_ref_ids"])
    for index, conversation in enumerate(conversations, 1):
        sample = conversation["sample_id"]
        datasets = {
            refs[ref_id]["dataset"]
            for row in lineage
            if row["sample_id"] == sample
            for ref_id in row["source_ref_ids"]
        }
        assert datasets == ({"CrossWOZ"} if index <= 25 else {"RiSAWOZ", "DuRecDial"})

    subset = ROOT / "permissive-source-250"
    subset_questions = rows("questions.jsonl", subset)
    subset_registry = json.loads((subset / "source_registry.json").read_text(encoding="utf-8"))
    assert len(subset_questions) == 250
    assert {item["dataset"] for item in subset_registry["sources"]} == {"CrossWOZ"}


def test_public_mix_500_top_level_seal_covers_every_artifact() -> None:
    sealed = {}
    for row in (ROOT / "BLIND_SEAL.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = row.split("  ", 1)
        sealed[relative] = expected
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != ROOT / "BLIND_SEAL.sha256"
    }
    assert set(sealed) == actual
    for relative, expected in sealed.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
