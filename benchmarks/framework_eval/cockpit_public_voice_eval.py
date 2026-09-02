from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from .cockpit_voice_shadow import run_shadow


_TEXT_UNITS = re.compile(r"[\u3400-\u9fff]|[a-z0-9]", re.IGNORECASE)
_PROTECTED_TOKEN = re.compile(r"[a-z]+(?:[-_][a-z0-9]+)*|\d+(?:\.\d+)?", re.IGNORECASE)


def normalize_asr_text(value: str) -> str:
    """Return language-neutral CER units for Mandarin/mixed cockpit speech."""
    return "".join(_TEXT_UNITS.findall(str(value or "").casefold()))


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for left_index, left in enumerate(reference, 1):
        current = [left_index]
        for right_index, right in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_asr_text(reference)
    actual = normalize_asr_text(hypothesis)
    if not expected:
        raise ValueError("reference transcript has no scoreable characters")
    return edit_distance(expected, actual) / len(expected)


def _quantile(values: Sequence[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(proportion * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _term_recall(reference: str, hypothesis: str, critical_terms: Iterable[str]) -> tuple[int, int]:
    expected = [normalize_asr_text(term) for term in critical_terms]
    expected = [term for term in expected if term]
    actual = normalize_asr_text(hypothesis)
    return sum(term in actual for term in expected), len(expected)


def _protected_tokens(reference: str) -> list[str]:
    return [token.casefold() for token in _PROTECTED_TOKEN.findall(reference)]


def _event_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        clip_id = str(row.get("clip_id") or f"clip-{index:05d}")
        hypothesis = str(row.get("hypothesis") or "").strip()
        base = {
            "namespace": str(row.get("namespace") or "public-voice-shadow"),
            "conversation_id": str(row.get("conversation_id") or "aishell5-dev"),
            "utterance_id": clip_id,
            "speaker_id": str(row.get("speaker_id") or "unknown-speaker"),
            "speaker_role": str(row.get("speaker_role") or "driver"),
            "started_at": str(row.get("started_at") or "2026-08-27T00:00:00Z"),
            "ended_at": str(row.get("ended_at") or "2026-08-27T00:00:01Z"),
            "source_system": str(row.get("source_system") or "public-asr-shadow"),
            "vehicle_id": str(row.get("vehicle_id") or "aishell5-public"),
            "seat": str(row.get("seat") or "unknown"),
            "transcript_confidence": row.get("transcript_confidence"),
            "trace_id": str(row.get("trace_id") or clip_id),
        }
        partial = dict(base, revision=0, is_final=False, text=hypothesis[: max(1, len(hypothesis) // 2)])
        final = dict(base, revision=1, is_final=True, text=hypothesis)
        events.extend((partial, final, dict(final)))
    return events


def evaluate_public_voice_rows(
    rows: Iterable[Mapping[str, Any]], db_path: Path,
) -> dict[str, Any]:
    source = [dict(row) for row in rows]
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(source):
        clip_id = str(row.get("clip_id") or f"clip-{index:05d}")
        reference = str(row.get("reference") or "")
        hypothesis = str(row.get("hypothesis") or "")
        cer = character_error_rate(reference, hypothesis)
        critical_hit, critical_total = _term_recall(
            reference, hypothesis, row.get("critical_terms") or (),
        )
        protected = _protected_tokens(reference)
        protected_hit, protected_total = _term_recall(reference, hypothesis, protected)
        scored.append({
            "clip_id": clip_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "cer": cer,
            "exact": normalize_asr_text(reference) == normalize_asr_text(hypothesis),
            "critical_terms_hit": critical_hit,
            "critical_terms_total": critical_total,
            "protected_tokens_hit": protected_hit,
            "protected_tokens_total": protected_total,
            "audio_path": str(row.get("audio_path") or ""),
            "speaker_id": str(row.get("speaker_id") or "unknown-speaker"),
            "started_at": str(row.get("started_at") or ""),
            "ended_at": str(row.get("ended_at") or ""),
        })

    shadow = run_shadow(_event_rows(source), db_path)
    reasons = Counter(str(row["reason"]) for row in shadow["rows"])
    accepted = [row for row in shadow["rows"] if row["status"] == "accepted"]
    lineage_complete = sum(
        bool((row.get("memory_write") or {}).get("metadata", {}).get("source_utterance_id"))
        and bool((row.get("memory_write") or {}).get("metadata", {}).get("speaker_id"))
        and bool((row.get("memory_write") or {}).get("timestamp"))
        for row in accepted
    )
    cers = [float(row["cer"]) for row in scored]
    term_hit = sum(int(row["critical_terms_hit"]) for row in scored)
    term_total = sum(int(row["critical_terms_total"]) for row in scored)
    protected_hit = sum(int(row["protected_tokens_hit"]) for row in scored)
    protected_total = sum(int(row["protected_tokens_total"]) for row in scored)
    count = len(scored)
    gates = {
        "partials_never_written": reasons["partial_transcript"] == count,
        "finals_written_exactly_once": len(accepted) == count,
        "duplicate_finals_idempotent": reasons["idempotent_duplicate"] == count,
        "no_rejected_events": shadow["rejected_count"] == 0,
        "lineage_complete": lineage_complete == count,
    }
    return {
        "schema_version": 1,
        "protocol": "cockpit-public-real-audio-shadow-v1",
        "scope": "ASR and final-transcript ingestion only; no live memory writes",
        "clips": count,
        "asr": {
            "exact": sum(bool(row["exact"]) for row in scored),
            "mean_cer": mean(cers) if cers else 0.0,
            "median_cer": median(cers) if cers else 0.0,
            "p95_cer": _quantile(cers, 0.95),
            "max_cer": max(cers, default=0.0),
            "critical_term_recall": term_hit / term_total if term_total else None,
            "critical_terms": {"hit": term_hit, "total": term_total},
            "protected_token_recall": protected_hit / protected_total if protected_total else None,
            "protected_tokens": {"hit": protected_hit, "total": protected_total},
        },
        "ingestion": {
            "events": shadow["event_count"],
            "accepted": shadow["accepted_count"],
            "ignored": shadow["ignored_count"],
            "rejected": shadow["rejected_count"],
            "reasons": dict(sorted(reasons.items())),
            "lineage_complete": lineage_complete,
        },
        "gates": gates,
        "gate_pass": all(gates.values()),
        "rows": scored,
        "shadow_rows": shadow["rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = evaluate_public_voice_rows(rows, args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    compact = {key: value for key, value in summary.items() if key not in {"rows", "shadow_rows"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
