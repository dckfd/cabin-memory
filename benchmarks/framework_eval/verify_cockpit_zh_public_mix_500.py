#!/usr/bin/env python3
"""Independent structural, provenance, event-chain and overlap audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


ABILITIES = {
    "aggregation-frequency",
    "latest-final-update",
    "two-date-validity",
    "multi-person-cross-session",
    "final-cancellation",
    "conditional-priority",
    "insufficient-evidence-abstention",
    "multi-target-final-state",
    "cutoff-state",
    "correction-retained-constraint",
}

EXPECTED_EVIDENCE = {
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(value: object) -> str:
    return "".join(re.findall(r"[0-9a-z\u3400-\u9fff]+", str(value).lower()))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(root: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    rows = (root / "BLIND_SEAL.sha256").read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for row in rows:
        expected, relative = row.split("  ", 1)
        if relative in seen:
            errors.append(f"duplicate seal path: {relative}")
            continue
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            errors.append(f"sealed file missing: {relative}")
        elif digest(path) != expected:
            errors.append(f"sealed file hash mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "BLIND_SEAL.sha256"
    }
    missing = sorted(actual - seen)
    if missing:
        errors.extend(f"file absent from top-level seal: {value}" for value in missing)
    return len(rows), errors


def flatten_messages(conversations: Iterable[dict]) -> tuple[dict[str, dict], dict[str, str]]:
    messages: dict[str, dict] = {}
    owners: dict[str, str] = {}
    for conversation in conversations:
        sample_id = str(conversation["sample_id"])
        for session in conversation["sessions"]:
            session_id = str(session["source_session_id"])
            for index, message in enumerate(session["messages"], 1):
                message_id = f"{session_id}:{index:03d}"
                if message_id in messages:
                    raise AssertionError(f"duplicate message id: {message_id}")
                messages[message_id] = message
                owners[message_id] = sample_id
    return messages, owners


def require_in(text: str, values: Iterable[object], row_id: str, label: str) -> None:
    for value in values:
        needle = normalized(value)
        if needle and needle not in text:
            raise AssertionError(f"{row_id}: missing {label} value {value!r}")


def verify_chain(row: dict, evidence_text: str, conversation_text: str) -> None:
    row_id = str(row["qa_id"])
    metadata = row["metadata"]
    ability = str(metadata["ability"])
    spec = metadata["chain_spec"]
    answer = normalized(row["answer"][0])
    require_in(evidence_text, metadata["required_evidence_values"], row_id, "evidence")

    if ability == "aggregation-frequency":
        counts = Counter(spec["events"])
        winner, count = counts.most_common(1)[0]
        assert winner == spec["winner"] and count == spec["count"] == 3
        require_in(answer, [winner, count], row_id, "answer")
    elif ability == "latest-final-update":
        assert len(spec["updates"]) == 2 and spec["updates"][0] != spec["updates"][1]
        require_in(answer, spec["updates"], row_id, "answer")
    elif ability == "two-date-validity":
        assert spec["queries"]["2026-03-16"] == spec["temporary"]
        assert spec["queries"]["2026-03-20"] == spec["base"]
        require_in(answer, spec["queries"].values(), row_id, "answer")
    elif ability == "multi-person-cross-session":
        assert len(spec["bindings"]) == 3
        require_in(answer, list(spec["bindings"].keys()) + list(spec["bindings"].values()), row_id, "answer")
    elif ability == "final-cancellation":
        assert spec["final"] is None and spec["replacement_after_cancel"] is False
        require_in(answer, [spec["rescheduled"].split("@", 1)[1], "没有", "取消"], row_id, "answer")
    elif ability == "conditional-priority":
        assert spec["current"] < spec["threshold"]
        assert spec["active_order"] == ["距离", "评分", "休息设施"]
        require_in(answer, spec["active_order"], row_id, "answer")
    elif ability == "insufficient-evidence-abstention":
        assert row["is_abstention"] is True and not row["answer_session_ids"]
        assert normalized(spec["requested_field"]) not in conversation_text
        require_in(answer, [spec["owner"], "无法确定"], row_id, "answer")
    elif ability == "multi-target-final-state":
        assert set(spec["initial"]) == set(spec["final"]) == {"午饭地点", "过夜地点", "会合地点"}
        require_in(answer, spec["final"].values(), row_id, "answer")
        stale_only = set(spec["initial"].values()) - set(spec["final"].values())
        if any(normalized(value) in answer for value in stale_only):
            raise AssertionError(f"{row_id}: stale alias leaked into final answer")
    elif ability == "cutoff-state":
        events = sorted(spec["events"], key=lambda item: item["date"])
        eligible = [item for item in events if item["date"] <= spec["cutoff"]]
        assert eligible and eligible[-1]["value"] == spec["answer"]
        require_in(answer, [spec["answer"]], row_id, "answer")
        newer = events[-1]["value"]
        if (
            newer != spec["answer"]
            and normalized(newer) not in normalized(spec["answer"])
            and normalized(newer) in answer
        ):
            raise AssertionError(f"{row_id}: post-cutoff value leaked into answer")
    elif ability == "correction-retained-constraint":
        assert spec["old_valid"] is False and spec["old_content"] != spec["new_content"]
        require_in(answer, [spec["old_content"], spec["new_content"], spec["retained"]["音量上限"], "避开收费道路", "失效"], row_id, "answer")
    else:
        raise AssertionError(f"{row_id}: unknown ability {ability}")


def verify_temporal_anchors(
    conversations: list[dict], questions: list[dict],
    lineage_by_id: dict[str, dict], ref_by_id: dict[str, dict],
) -> dict:
    """Reject co-mentioned alternatives masquerading as dated updates."""
    question_by_key = {
        (row["sample_id"], row["metadata"]["ability"]): row
        for row in questions
    }
    datasets: Counter[str] = Counter()
    for conversation in conversations:
        sample_id = str(conversation["sample_id"])
        old_session, new_session = conversation["sessions"][:2]
        old_message_id = f"{old_session['source_session_id']}:001"
        new_message_id = f"{new_session['source_session_id']}:001"
        old_ids = lineage_by_id[old_message_id]["source_ref_ids"]
        new_ids = lineage_by_id[new_message_id]["source_ref_ids"]
        assert len(old_ids) == len(new_ids) == 1, f"{sample_id}: anchor ref cardinality"
        old = ref_by_id[old_ids[0]]
        new = ref_by_id[new_ids[0]]
        assert old["dataset"] == new["dataset"], f"{sample_id}: anchor dataset mismatch"
        assert old["record_id"] == new["record_id"], f"{sample_id}: anchor record mismatch"
        assert old["field"] == new["field"], f"{sample_id}: anchor field mismatch"
        assert old["turn_id"] != new["turn_id"], f"{sample_id}: values co-mentioned in one turn"
        assert old["value"] != new["value"], f"{sample_id}: anchor values are identical"
        assert normalized(old["source_excerpt"]) != normalized(new["source_excerpt"]), (
            f"{sample_id}: anchor excerpts are identical"
        )

        latest = question_by_key[(sample_id, "latest-final-update")]["metadata"]["chain_spec"]
        cutoff = question_by_key[(sample_id, "cutoff-state")]["metadata"]["chain_spec"]
        assert latest["field"] == old["field"]
        assert latest["updates"] == [old["value"], new["value"]]
        assert [event["value"] for event in cutoff["events"]] == [old["value"], new["value"]]
        assert cutoff["answer"] == old["value"]
        datasets[old["dataset"]] += 1
    return {
        "anchors": sum(datasets.values()),
        "distinct_source_turns": sum(datasets.values()),
        "dataset_distribution": dict(sorted(datasets.items())),
        "same_turn_pairs": 0,
        "same_excerpt_pairs": 0,
    }


def iter_prior_files(roots: Iterable[Path], dataset: Path) -> Iterable[Path]:
    dataset = dataset.resolve()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("questions.jsonl"):
            resolved = path.resolve()
            if dataset in resolved.parents:
                continue
            yield path


def prior_overlap(roots: Iterable[Path], dataset: Path, questions: list[dict]) -> dict:
    current = {normalized(row["question"]) for row in questions}
    prior: set[str] = set()
    files = 0
    rows = 0
    parse_errors: list[str] = []
    for path in sorted(set(iter_prior_files(roots, dataset))):
        files += 1
        try:
            for row in read_jsonl(path):
                question = row.get("question")
                if question:
                    prior.add(normalized(question))
                    rows += 1
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            parse_errors.append(f"{path}: {error}")
    return {
        "files": files,
        "rows": rows,
        "unique_normalized_questions": len(prior),
        "normalized_overlap_count": len(current & prior),
        "parse_errors": parse_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conversations = read_jsonl(args.dataset / "conversations.jsonl")
    questions = read_jsonl(args.dataset / "questions.jsonl")
    lineage = read_jsonl(args.dataset / "source_lineage.jsonl")
    refs = read_jsonl(args.dataset / "source_refs.jsonl")
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((args.dataset / "source_registry.json").read_text(encoding="utf-8"))
    messages, owners = flatten_messages(conversations)
    ref_by_id = {row["ref_id"]: row for row in refs}
    lineage_by_id = {row["message_id"]: row for row in lineage}

    assert len(conversations) == 50 and len(questions) == 500
    assert len(messages) == len(lineage) == 2100
    assert len(ref_by_id) == len(refs)
    assert set(lineage_by_id) == set(messages)
    assert set(manifest["ability_distribution"]) == ABILITIES
    assert set(registry) >= {"schema_version", "sources", "excluded"}
    assert len({row["qa_id"] for row in questions}) == 500
    assert len({normalized(row["question"]) for row in questions}) == 500

    for ref in refs:
        require_in(normalized(ref["source_excerpt"]), [ref["value"]], ref["ref_id"], "source excerpt")
    for row in lineage:
        for ref_id in row["source_ref_ids"]:
            assert ref_id in ref_by_id, f"missing source ref {ref_id}"

    temporal_anchor_audit = verify_temporal_anchors(
        conversations, questions, lineage_by_id, ref_by_id,
    )

    evidence_distribution: Counter[str] = Counter()
    for row in questions:
        ability = row["metadata"]["ability"]
        evidence = row["answer_session_ids"]
        assert len(evidence) == EXPECTED_EVIDENCE[ability]
        assert all(value in messages for value in evidence)
        assert all(owners[value] == row["sample_id"] for value in evidence)
        evidence_text = normalized(" ".join(messages[value]["content"] for value in evidence))
        conversation_text = normalized(" ".join(
            message["content"] for message_id, message in messages.items()
            if owners[message_id] == row["sample_id"]
        ))
        verify_chain(row, evidence_text, conversation_text)
        evidence_distribution[ability] += len(evidence)

    for index, conversation in enumerate(conversations, 1):
        source_ids = {
            ref_by_id[ref_id]["dataset"]
            for row in lineage
            if row["sample_id"] == conversation["sample_id"]
            for ref_id in row["source_ref_ids"]
        }
        if index <= 25:
            assert source_ids == {"CrossWOZ"}
        else:
            assert source_ids == {"RiSAWOZ", "DuRecDial"}

    seal_rows, seal_errors = verify_seal(args.dataset)
    overlap = prior_overlap(args.prior_root, args.dataset, questions)
    errors = list(seal_errors)
    if overlap["normalized_overlap_count"]:
        errors.append(f"prior normalized question overlap: {overlap['normalized_overlap_count']}")
    if overlap["parse_errors"]:
        errors.extend(overlap["parse_errors"])

    summary = {
        "dataset_id": manifest["dataset_id"],
        "status": "PASS" if not errors else "FAIL",
        "counts": {
            "conversations": len(conversations),
            "sessions": sum(row["session_count"] for row in conversations),
            "messages": len(messages),
            "questions": len(questions),
            "source_refs": len(refs),
            "lineage_rows": len(lineage),
            "seal_rows": seal_rows,
        },
        "ability_distribution": dict(sorted(Counter(row["metadata"]["ability"] for row in questions).items())),
        "style_distribution": dict(sorted(Counter(row["metadata"]["surface_style"] for row in questions).items())),
        "temporal_anchor_audit": temporal_anchor_audit,
        "evidence_links_by_ability": dict(sorted(evidence_distribution.items())),
        "prior_question_overlap": overlap,
        "seal_errors": seal_errors,
        "errors": errors,
        "answer_or_judge_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
