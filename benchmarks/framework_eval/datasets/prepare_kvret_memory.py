from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SOURCE_URL = "https://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip"
NAVIGATION_FALLBACK_QUESTION = (
    "During the vehicle interaction logged on {interaction_date}, what destination "
    'did the driver mention in their "{anchor}" request?'
)
CALENDAR_ASSISTANT_QUESTION = (
    "During the vehicle interaction logged on {interaction_date}, what time did "
    'the car assistant report for the "{anchor}" event?'
)
DOMAIN_SPECS = (
    {
        "source": "calendar",
        "category": "calendar",
        "anchor_slot": "event",
        "answer_slot": "time",
        "question": (
            "During the vehicle interaction logged on {interaction_date}, what "
            'time did the driver specify for the "{anchor}" reminder?'
        ),
    },
    {
        "source": "location information",
        "category": "navigation",
        "anchor_slot": "poi_type",
        "answer_slot": "poi",
        "question": (
            "During the vehicle interaction logged on {interaction_date}, which "
            'destination did the car assistant select for the driver\'s '
            '"{anchor}" request?'
        ),
    },
    {
        "source": "weekly forecast",
        "category": "weather",
        "anchor_slot": "date",
        "answer_slot": "location",
        "question": (
            "During the vehicle interaction logged on {interaction_date}, which "
            'location did the driver specify for the forecast on "{anchor}"?'
        ),
    },
)


def _surface(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _match_key(value: Any) -> str:
    return "".join(re.findall(r"[a-z0-9]+", _surface(value).casefold()))


def _dialogue_slots(row: dict[str, Any]) -> dict[str, str]:
    slots: dict[str, str] = {}
    for turn in row.get("dialogue") or []:
        if str(turn.get("turn") or "").casefold() != "assistant":
            continue
        for key, value in ((turn.get("data") or {}).get("slots") or {}).items():
            rendered = _surface(value)
            if rendered:
                slots[str(key)] = rendered
    return slots


def _matching_turns(row: dict[str, Any], value: str) -> list[int]:
    target = _match_key(value)
    if not target:
        return []
    return [
        index
        for index, turn in enumerate(row.get("dialogue") or [], 1)
        if target in _match_key((turn.get("data") or {}).get("utterance"))
    ]


def _selected_navigation_poi(row: dict[str, Any]) -> tuple[str, list[int]]:
    """Return the final uniquely named KB destination in an assistant turn.

    KVRET ``slots.poi`` records a user constraint and can be a generic value
    such as ``restaurant``; it is not necessarily the destination selected by
    the assistant. The selected entity is grounded instead in the scenario KB
    and the last assistant response that names exactly one KB POI.
    """
    items = (((row.get("scenario") or {}).get("kb") or {}).get("items") or [])
    poi_values = list(dict.fromkeys(
        _surface(item.get("poi")) for item in items if _surface(item.get("poi"))
    ))
    grounded: list[tuple[int, str]] = []
    for turn_index, turn in enumerate(row.get("dialogue") or [], 1):
        if str(turn.get("turn") or "").casefold() != "assistant":
            continue
        utterance_key = _match_key((turn.get("data") or {}).get("utterance"))
        matches = [
            poi for poi in poi_values
            if _match_key(poi) and _match_key(poi) in utterance_key
        ]
        # Prefer the most specific entity if one KB name contains another.
        matches = [
            poi for poi in matches
            if not any(
                _match_key(poi) != _match_key(other)
                and _match_key(poi) in _match_key(other)
                for other in matches
            )
        ]
        if len(matches) == 1:
            grounded.append((turn_index, matches[0]))
    if not grounded:
        return "", []
    turn_index, answer = grounded[-1]
    return answer, [turn_index]


def _navigation_answer_aliases(
    row: dict[str, Any], answer: str, answer_turns: list[int],
) -> list[str]:
    aliases = [answer]
    if not answer_turns:
        return aliases
    items = (((row.get("scenario") or {}).get("kb") or {}).get("items") or [])
    selected_turn_text = " ".join(
        _surface((row["dialogue"][index - 1].get("data") or {}).get("utterance"))
        for index in answer_turns
    )
    for item in items:
        if _match_key(item.get("poi")) != _match_key(answer):
            continue
        address = _surface(item.get("address"))
        if address and _match_key(address) in _match_key(selected_turn_text):
            aliases.append(address)
        break
    return list(dict.fromkeys(aliases))


def _domain_candidates(
    rows: list[dict[str, Any]], spec: dict[str, str], count: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in sorted(
        rows, key=lambda item: str((item.get("scenario") or {}).get("uuid") or "")
    ):
        kb = ((row.get("scenario") or {}).get("kb") or {})
        if str(kb.get("kb_title") or "") != spec["source"]:
            continue
        slots = _dialogue_slots(row)
        anchor = slots.get(spec["anchor_slot"], "")
        slot_answer = slots.get(spec["answer_slot"], "")
        slot_answer_turns = _matching_turns(row, slot_answer)
        anchor_turns = _matching_turns(row, anchor)
        if (
            not anchor
            or not slot_answer
            or _match_key(anchor) == _match_key(slot_answer)
            or not anchor_turns
            or not slot_answer_turns
        ):
            continue
        answer = slot_answer
        answers = [slot_answer]
        answer_turns = slot_answer_turns
        answer_slot = spec["answer_slot"]
        question_template = spec["question"]
        if spec["category"] == "navigation":
            selected_answer, selected_turns = _selected_navigation_poi(row)
            if selected_answer and selected_turns:
                answer = selected_answer
                answers = _navigation_answer_aliases(
                    row, selected_answer, selected_turns
                )
                answer_turns = selected_turns
                answer_slot = "selected_kb_poi"
            else:
                question_template = NAVIGATION_FALLBACK_QUESTION
        elif spec["category"] == "calendar":
            answer_actor = str(
                (row["dialogue"][answer_turns[0] - 1]).get("turn") or ""
            ).casefold()
            if answer_actor == "assistant":
                question_template = CALENDAR_ASSISTANT_QUESTION
        candidates.append({
            "row": row,
            "source_split": str(row.get("_source_split") or "unknown"),
            "category": spec["category"],
            "anchor_slot": spec["anchor_slot"],
            "answer_slot": answer_slot,
            "anchor": anchor,
            "answer": answer,
            "answers": answers,
            "question_template": question_template,
            "evidence_turns": list(dict.fromkeys((anchor_turns[0], answer_turns[0]))),
        })
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise ValueError(
            f"KVRET domain {spec['source']!r} yielded {len(candidates)} valid "
            f"examples; expected {count}"
        )
    return candidates


def derive_kvret_memory(
    rows: list[dict[str, Any]], *, groups: int = 6, per_domain: int = 46,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic long-memory task without consulting model output.

    Independent official KVRET test dialogues are grouped into synthetic vehicle
    histories.  This transformation measures memory for fragmented commands; it
    is not the official KVRET response-generation task and must be reported as a
    derived benchmark.
    """
    if groups <= 0 or per_domain <= 0:
        raise ValueError("groups and per_domain must be positive")
    selected_by_domain = [
        _domain_candidates(rows, spec, per_domain) for spec in DOMAIN_SPECS
    ]
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for domain_index, candidates in enumerate(selected_by_domain):
        for rank, candidate in enumerate(candidates):
            # Offsets distribute 46 examples/domain as 23 total sessions into
            # each of six vehicle histories: 8/8/7 domain counts rotate evenly.
            group_index = (rank + 2 * domain_index) % groups
            buckets[group_index].append({
                **candidate,
                "domain_index": domain_index,
                "domain_rank": rank,
            })

    conversations: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    question_index = 0
    start = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    for group_index in range(groups):
        conversation_id = f"kvret-memory-{group_index:02d}"
        sessions: list[dict[str, Any]] = []
        candidates = sorted(
            buckets[group_index],
            key=lambda item: (item["domain_rank"], item["domain_index"]),
        )
        for session_index, candidate in enumerate(candidates, 1):
            source = candidate["row"]
            source_uuid = str((source.get("scenario") or {}).get("uuid") or "")
            timestamp = start + timedelta(days=session_index - 1)
            rendered_timestamp = timestamp.isoformat().replace("+00:00", "Z")
            session_id = f"{conversation_id}-session-{session_index:03d}"
            messages: list[dict[str, Any]] = []
            turn_to_source: dict[int, str] = {}
            for turn_index, turn in enumerate(source.get("dialogue") or [], 1):
                actor = str(turn.get("turn") or "").casefold()
                message_id = f"K{group_index:02d}S{session_index:03d}T{turn_index:02d}"
                turn_to_source[turn_index] = message_id
                messages.append({
                    "message_id": message_id,
                    "role": "user" if actor == "driver" else "assistant",
                    "speaker": "Driver" if actor == "driver" else "Car Assistant",
                    "content": _surface((turn.get("data") or {}).get("utterance")),
                    "timestamp": rendered_timestamp,
                    "metadata": {
                        "source_uuid": source_uuid,
                        "source_turn": turn_index,
                        "kvret_actor": actor,
                    },
                })
            sessions.append({
                "session_id": session_id,
                "timestamp": rendered_timestamp,
                "messages": messages,
                "metadata": {
                    "source_uuid": source_uuid,
                    "source_split": candidate["source_split"],
                    "source_domain": candidate["category"],
                    "derived_session_index": session_index,
                },
            })
            questions.append({
                "qa_id": f"kvret-memory#q{question_index:04d}",
                "sample_id": conversation_id,
                "question": candidate["question_template"].format(
                    anchor=candidate["anchor"],
                    interaction_date=timestamp.date().isoformat(),
                ),
                "answer": candidate["answers"],
                "category": candidate["category"],
                "evidence": [
                    turn_to_source[index] for index in candidate["evidence_turns"]
                ],
                "metadata": {
                    "derived": True,
                    "source_split": candidate["source_split"],
                    "source_uuid": source_uuid,
                    "anchor_slot": candidate["anchor_slot"],
                    "answer_slot": candidate["answer_slot"],
                    "anchor_value": candidate["anchor"],
                    "interaction_date": timestamp.date().isoformat(),
                },
            })
            question_index += 1
        conversations.append({
            "sample_id": conversation_id,
            "sessions": sessions,
            "metadata": {
                "scenario": "smart-cockpit-fragmented-command-memory",
                "derived": True,
                "single_user_agent_roles": True,
                "source_dataset": "Stanford KVRET",
            },
        })

    questions.sort(key=lambda item: (item["sample_id"], item["qa_id"]))
    expected = len(DOMAIN_SPECS) * per_domain
    if len(questions) != expected:
        raise AssertionError(f"derived {len(questions)} questions, expected {expected}")
    if any(len(row["sessions"]) != expected // groups for row in conversations):
        raise AssertionError("vehicle histories are not evenly balanced")
    return conversations, questions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive a low-token smart-cockpit memory benchmark from KVRET"
    )
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help=(
            "Official KVRET JSON split. Repeat to build a larger deterministic "
            "research slice from train/dev/test."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--per-domain", type=int, default=46)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_paths = [path.resolve() for path in args.input]
    source_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    for source_path in source_paths:
        source_bytes = source_path.read_bytes()
        split_match = re.search(
            r"kvret_(train|dev|test)_public", source_path.name.casefold()
        )
        source_split = split_match.group(1) if split_match else source_path.stem
        for row in json.loads(source_bytes):
            copied = dict(row)
            copied["_source_split"] = source_split
            source_rows.append(copied)
        source_files.append({
            "path": str(source_path),
            "split": source_split,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        })
    conversations, questions = derive_kvret_memory(
        source_rows, groups=args.groups, per_domain=args.per_domain
    )
    output = args.output_dir.resolve()
    _write_jsonl(output / "conversations.jsonl", conversations)
    _write_jsonl(output / "questions.jsonl", questions)
    manifest = {
        "schema_version": 1,
        "dataset_id": "kvret_memory",
        "name": f"KVRET-Memory-{len(questions)}",
        "kind": "derived-memory-evaluation",
        "official_kvret_metric": False,
        "source_dataset": "Stanford KVRET",
        "source_url": SOURCE_URL,
        "source_split": "+".join(row["split"] for row in source_files),
        "source_files": source_files,
        "license_note": (
            "The Stanford download is public but the archive does not include an "
            "explicit dataset license; use this derived artifact for research "
            "evaluation and verify rights separately for redistribution/commercial use."
        ),
        "generation": {
            "uses_llm": False,
            "uses_reference_response_generation": False,
            "uses_dialogue_slot_annotations": True,
            "groups": args.groups,
            "questions_per_domain": args.per_domain,
            "selection": "UUID sort plus fixed domain templates",
        },
        "conversations": len(conversations),
        "sessions": sum(len(row["sessions"]) for row in conversations),
        "messages": sum(
            len(session["messages"])
            for row in conversations
            for session in row["sessions"]
        ),
        "questions": len(questions),
        "categories": dict(Counter(row["category"] for row in questions)),
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
