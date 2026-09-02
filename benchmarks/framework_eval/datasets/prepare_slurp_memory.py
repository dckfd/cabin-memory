from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SOURCE_URL = "https://github.com/pswietojanski/slurp"
TEXT_LICENSE = "CC-BY-4.0"
AUDIO_LICENSE = "CC-BY-NC-4.0"


def _surface(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _match_key(value: Any) -> str:
    return "".join(re.findall(r"[a-z0-9]+", _surface(value).casefold()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entity_surface(row: dict[str, Any], entity: dict[str, Any]) -> str:
    tokens = row.get("tokens") or []
    indices = [int(value) for value in entity.get("span") or []]
    if not indices or any(value < 0 or value >= len(tokens) for value in indices):
        return ""
    return _surface(" ".join(str(tokens[value].get("surface") or "") for value in indices))


def _fragment_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    """Move one grounded entity into a deterministic clarification turn.

    SLURP is a single-command SLU corpus. For a memory evaluation, the last
    usable entity is removed from the original command and supplied as the
    driver's answer to a generic assistant clarification. No fact is generated
    or changed; the original sentence and target span remain in metadata.
    """
    tokens = row.get("tokens") or []
    candidates: list[dict[str, Any]] = []
    for entity in row.get("entities") or []:
        indices = sorted({int(value) for value in entity.get("span") or []})
        if not indices or any(value < 0 or value >= len(tokens) for value in indices):
            continue
        answer = _entity_surface(row, entity)
        retained = [
            str(token.get("surface") or "")
            for index, token in enumerate(tokens)
            if index not in set(indices)
        ]
        fragmented = _surface(" ".join(retained))
        if (
            not answer
            or len(_match_key(answer)) < 2
            # A target at the beginning often leaves a generic, unnatural
            # fragment (for example, "pink is all we need" -> "is all we
            # need"). Prefer commands whose intent-bearing prefix survives.
            or indices[0] == 0
            or len(re.findall(r"[a-z0-9]+", fragmented.casefold())) < 3
            or _match_key(answer) in _match_key(fragmented)
        ):
            continue
        candidates.append({
            "answer": answer,
            "fragmented_sentence": fragmented,
            "entity_type": str(entity.get("type") or "").strip(),
            "entity_span": indices,
            "span_start": indices[0],
        })
    if not candidates:
        return None
    # A trailing value produces the most natural clarification in most SLURP
    # commands (time, date, place, person, device, and so on).
    return max(candidates, key=lambda item: (item["span_start"], len(item["entity_span"])))


def _recording_priority(recording: dict[str, Any]) -> tuple[Any, ...]:
    filename = str(recording.get("file") or "")
    return (
        0 if str(recording.get("status") or "").casefold() == "correct" else 1,
        float(recording.get("ent_wer") or 0),
        float(recording.get("wer") or 0),
        0 if "-headset." in filename else 1,
        filename,
    )


def _speaker_candidates(
    rows: list[dict[str, Any]], metadata: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_speaker: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        fragment = _fragment_candidate(row)
        if fragment is None:
            continue
        slurp_id = str(row.get("slurp_id") or "")
        recording_metadata = (metadata.get(slurp_id) or {}).get("recordings") or {}
        candidates_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for recording in row.get("recordings") or []:
            filename = str(recording.get("file") or "")
            speaker_id = str((recording_metadata.get(filename) or {}).get("usrid") or "")
            if speaker_id:
                candidates_by_speaker[speaker_id].append(recording)
        for speaker_id, recordings in candidates_by_speaker.items():
            recording = min(recordings, key=_recording_priority)
            by_speaker[speaker_id][slurp_id] = {
                "row": row,
                "speaker_id": speaker_id,
                "recording": recording,
                **fragment,
            }
    return {
        speaker: list(values.values())
        for speaker, values in by_speaker.items()
    }


def _diverse_select(
    candidates: list[dict[str, Any]], *, count: int, used_ids: set[str],
) -> list[dict[str, Any]]:
    remaining = [
        candidate for candidate in candidates
        if str(candidate["row"]["slurp_id"]) not in used_ids
    ]
    selected: list[dict[str, Any]] = []
    scenario_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    while remaining and len(selected) < count:
        candidate = min(
            remaining,
            key=lambda item: (
                scenario_counts[str(item["row"].get("scenario") or "")],
                entity_counts[item["entity_type"]],
                intent_counts[str(item["row"].get("intent") or "")],
                float(item["recording"].get("ent_wer") or 0),
                float(item["recording"].get("wer") or 0),
                str(item["row"].get("scenario") or ""),
                str(item["row"].get("intent") or ""),
                int(item["row"]["slurp_id"]),
            ),
        )
        remaining.remove(candidate)
        selected.append(candidate)
        scenario_counts[str(candidate["row"].get("scenario") or "")] += 1
        entity_counts[candidate["entity_type"]] += 1
        intent_counts[str(candidate["row"].get("intent") or "")] += 1
    return selected


def _select_speaker_histories(
    rows: list[dict[str, Any]], metadata: dict[str, Any], *, groups: int,
    sessions_per_group: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    candidates = _speaker_candidates(rows, metadata)
    ranked_speakers = sorted(candidates, key=lambda value: (-len(candidates[value]), value))
    histories: list[tuple[str, list[dict[str, Any]]]] = []
    used_ids: set[str] = set()
    for speaker_id in ranked_speakers:
        selected = _diverse_select(
            candidates[speaker_id], count=sessions_per_group, used_ids=used_ids,
        )
        if len(selected) != sessions_per_group:
            continue
        histories.append((speaker_id, selected))
        used_ids.update(str(item["row"]["slurp_id"]) for item in selected)
        if len(histories) == groups:
            break
    if len(histories) != groups:
        raise ValueError(
            f"SLURP yielded {len(histories)} speaker histories; expected {groups}"
        )
    return histories


def _select_all_eligible_speaker_histories(
    rows: list[dict[str, Any]], metadata: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Assign every traceable eligible command to exactly one real speaker.

    A SLURP command may have recordings from multiple speakers.  Greedy set
    cover keeps histories as large as possible (and therefore reduces the
    number of isolated memory namespaces) while preserving the invariant that
    every history belongs to one source speaker and every source utterance is
    used at most once.  Ties are resolved by speaker id for reproducibility.
    """
    candidates = _speaker_candidates(rows, metadata)
    ids_by_speaker = {
        speaker: {
            str(candidate["row"]["slurp_id"])
            for candidate in values
        }
        for speaker, values in candidates.items()
    }
    remaining = set().union(*ids_by_speaker.values()) if ids_by_speaker else set()
    histories: list[tuple[str, list[dict[str, Any]]]] = []
    while remaining:
        available = [
            speaker for speaker, source_ids in ids_by_speaker.items()
            if source_ids & remaining
        ]
        if not available:
            raise ValueError(
                f"unable to assign {len(remaining)} eligible SLURP utterances"
            )
        speaker_id = min(
            available,
            key=lambda value: (-len(ids_by_speaker[value] & remaining), value),
        )
        selected = [
            candidate for candidate in candidates[speaker_id]
            if str(candidate["row"]["slurp_id"]) in remaining
        ]
        histories.append((speaker_id, selected))
        remaining.difference_update(
            str(candidate["row"]["slurp_id"]) for candidate in selected
        )
    return histories


def derive_slurp_memory(
    rows: list[dict[str, Any]], metadata: dict[str, Any], *, groups: int = 6,
    sessions_per_group: int = 23,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create same-speaker fragmented-command histories without an LLM."""
    if groups <= 0 or sessions_per_group <= 0:
        raise ValueError("groups and sessions_per_group must be positive")
    histories = _select_speaker_histories(
        rows, metadata, groups=groups, sessions_per_group=sessions_per_group,
    )
    return _render_slurp_memory_histories(
        histories,
        conversation_prefix="slurp-memory",
        question_prefix="slurp-memory",
    )


def derive_slurp_memory_all_eligible(
    rows: list[dict[str, Any]], metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the exhaustive source-speaker-grounded test-split track."""
    histories = _select_all_eligible_speaker_histories(rows, metadata)
    return _render_slurp_memory_histories(
        histories,
        conversation_prefix="slurp-memory-full",
        question_prefix="slurp-memory-full",
    )


def _render_slurp_memory_histories(
    histories: list[tuple[str, list[dict[str, Any]]]], *,
    conversation_prefix: str,
    question_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversations: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    question_index = 0
    start = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    for group_index, (speaker_id, selected) in enumerate(histories):
        conversation_id = f"{conversation_prefix}-{group_index:02d}"
        speaker_hash = hashlib.sha256(speaker_id.encode()).hexdigest()[:12]
        # Hash order prevents scenario clusters while remaining reproducible.
        selected.sort(key=lambda item: hashlib.sha256(
            f"{speaker_id}:{item['row']['slurp_id']}".encode()
        ).hexdigest())
        sessions: list[dict[str, Any]] = []
        for session_index, candidate in enumerate(selected, 1):
            row = candidate["row"]
            timestamp = start + timedelta(days=session_index - 1)
            rendered_timestamp = timestamp.isoformat().replace("+00:00", "Z")
            session_id = f"{conversation_id}-session-{session_index:03d}"
            prefix = f"S{group_index:02d}S{session_index:03d}"
            slot_label = candidate["entity_type"].replace("_", " ")
            anchor_tokens = candidate["fragmented_sentence"].split()[:5]
            anchor = " ".join(anchor_tokens)
            messages = [
                {
                    "message_id": f"{prefix}T01",
                    "role": "user",
                    "speaker": "Driver",
                    "content": candidate["fragmented_sentence"],
                    "timestamp": rendered_timestamp,
                    "metadata": {
                        "source_slurp_id": int(row["slurp_id"]),
                        "source_sentence": row["sentence"],
                        "source_recording": candidate["recording"].get("file", ""),
                        "fragment_role": "command_without_target_entity",
                    },
                },
                {
                    "message_id": f"{prefix}T02",
                    "role": "assistant",
                    "speaker": "Car Assistant",
                    "content": f"What should I use for the {slot_label}?",
                    "timestamp": (timestamp + timedelta(seconds=5)).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "metadata": {
                        "source_slurp_id": int(row["slurp_id"]),
                        "fragment_role": "deterministic_clarification",
                    },
                },
                {
                    "message_id": f"{prefix}T03",
                    "role": "user",
                    "speaker": "Driver",
                    "content": candidate["answer"],
                    "timestamp": (timestamp + timedelta(seconds=10)).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "metadata": {
                        "source_slurp_id": int(row["slurp_id"]),
                        "fragment_role": "target_entity_reply",
                    },
                },
            ]
            sessions.append({
                "session_id": session_id,
                "timestamp": rendered_timestamp,
                "messages": messages,
                "metadata": {
                    "source_slurp_id": int(row["slurp_id"]),
                    "source_scenario": row.get("scenario", ""),
                    "source_intent": row.get("intent", ""),
                    "source_speaker_hash": speaker_hash,
                    "derived_fragmentation": True,
                },
            })
            questions.append({
                "qa_id": f"{question_prefix}#q{question_index:04d}",
                "sample_id": conversation_id,
                "question": (
                    f"During the driver voice interaction logged on "
                    f"{timestamp.date().isoformat()} that began \"{anchor}\", what "
                    f"did the driver reply when the car assistant asked for the "
                    f"\"{slot_label}\" detail?"
                ),
                "answer": [candidate["answer"]],
                "category": str(row.get("scenario") or ""),
                "evidence": [f"{prefix}T01", f"{prefix}T02", f"{prefix}T03"],
                "metadata": {
                    "derived": True,
                    "source_split": "test",
                    "source_slurp_id": int(row["slurp_id"]),
                    "source_speaker_hash": speaker_hash,
                    "scenario": row.get("scenario", ""),
                    "intent": row.get("intent", ""),
                    "action": row.get("action", ""),
                    "entity_type": candidate["entity_type"],
                    "entity_span": candidate["entity_span"],
                    "original_sentence": row.get("sentence", ""),
                    "interaction_date": timestamp.date().isoformat(),
                },
            })
            question_index += 1
        conversations.append({
            "sample_id": conversation_id,
            "sessions": sessions,
            "metadata": {
                "scenario": "smart-cockpit-fragmented-voice-command-memory",
                "derived": True,
                "single_user_agent_roles": True,
                "source_dataset": "SLURP",
                "source_speaker_hash": speaker_hash,
            },
        })
    questions.sort(key=lambda item: (item["sample_id"], item["qa_id"]))
    return conversations, questions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive a same-speaker fragmented memory task from SLURP",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--sessions-per-group", type=int, default=23)
    parser.add_argument(
        "--all-eligible", action="store_true",
        help=(
            "Use every fragment-eligible test utterance with traceable source "
            "speaker metadata exactly once"
        ),
    )
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if args.all_eligible:
        conversations, questions = derive_slurp_memory_all_eligible(rows, metadata)
    else:
        conversations, questions = derive_slurp_memory(
            rows, metadata, groups=args.groups,
            sessions_per_group=args.sessions_per_group,
        )
    _write_jsonl(args.output_dir / "conversations.jsonl", conversations)
    _write_jsonl(args.output_dir / "questions.jsonl", questions)
    categories = Counter(str(row["category"]) for row in questions)
    manifest = {
        "schema_version": 1,
        "dataset_id": "slurp_memory",
        "name": f"SLURP-Memory-{len(questions)}",
        "version": (
            f"slurp-memory-full-{len(questions)}-v1"
            if args.all_eligible else "slurp-memory-138-v1"
        ),
        "kind": "derived-memory-evaluation",
        "official_slurp_metric": False,
        "source_dataset": "SLURP",
        "source_url": SOURCE_URL,
        "source_split": "test",
        "source_file": str(args.input.resolve()),
        "source_sha256": _file_sha256(args.input),
        "metadata_sha256": _file_sha256(args.metadata),
        "licenses": {"text": TEXT_LICENSE, "audio": AUDIO_LICENSE},
        "audio_downloaded": False,
        "generation": {
            "uses_llm": False,
            "uses_reference_response_generation": False,
            "uses_entity_annotations": True,
            "same_source_speaker_per_history": True,
            "fragmentation": "command-without-target / clarification / target reply",
            "selection_mode": (
                "all-traceable-fragment-eligible" if args.all_eligible
                else "balanced-fixed-size"
            ),
            "groups": len(conversations),
            "sessions_per_group": (
                None if args.all_eligible else args.sessions_per_group
            ),
        },
        "conversations": len(conversations),
        "sessions": sum(len(row["sessions"]) for row in conversations),
        "messages": sum(
            len(session["messages"])
            for row in conversations for session in row["sessions"]
        ),
        "questions": len(questions),
        "categories": dict(sorted(categories.items())),
        "source_speakers": len({
            row["metadata"]["source_speaker_hash"] for row in conversations
        }),
        "unique_source_utterances": len({
            row["metadata"]["source_slurp_id"] for row in questions
        }),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
