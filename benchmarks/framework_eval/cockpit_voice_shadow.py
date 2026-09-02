from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping


_SPEAKER_ROLES = {"driver", "passenger", "assistant"}


@dataclass(frozen=True)
class VoiceTranscriptEvent:
    namespace: str
    conversation_id: str
    utterance_id: str
    revision: int
    is_final: bool
    text: str
    speaker_id: str
    speaker_role: str
    started_at: str
    ended_at: str
    source_system: str = "asr"
    vehicle_id: str = ""
    seat: str = ""
    transcript_confidence: float | None = None
    trace_id: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VoiceTranscriptEvent":
        event = cls(
            namespace=str(value.get("namespace") or "").strip(),
            conversation_id=str(value.get("conversation_id") or "").strip(),
            utterance_id=str(value.get("utterance_id") or "").strip(),
            revision=int(value.get("revision", -1)),
            is_final=value.get("is_final") is True,
            text=str(value.get("text") or "").strip(),
            speaker_id=str(value.get("speaker_id") or "").strip(),
            speaker_role=str(value.get("speaker_role") or "").strip().lower(),
            started_at=str(value.get("started_at") or "").strip(),
            ended_at=str(value.get("ended_at") or "").strip(),
            source_system=str(value.get("source_system") or "asr").strip(),
            vehicle_id=str(value.get("vehicle_id") or "").strip(),
            seat=str(value.get("seat") or "").strip(),
            transcript_confidence=(
                None
                if value.get("transcript_confidence") is None
                else float(value["transcript_confidence"])
            ),
            trace_id=str(value.get("trace_id") or "").strip(),
        )
        event.validate()
        return event

    def validate(self) -> None:
        required = {
            "namespace": self.namespace,
            "conversation_id": self.conversation_id,
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing required fields: {','.join(missing)}")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.speaker_role not in _SPEAKER_ROLES:
            raise ValueError(f"unsupported speaker_role: {self.speaker_role}")
        if self.is_final and not self.text:
            raise ValueError("final transcript text must be non-empty")
        if self.transcript_confidence is not None and not 0 <= self.transcript_confidence <= 1:
            raise ValueError("transcript_confidence must be between 0 and 1")
        start = _parse_time(self.started_at)
        end = _parse_time(self.ended_at)
        if end < start:
            raise ValueError("ended_at precedes started_at")


@dataclass(frozen=True)
class ShadowDecision:
    status: str
    reason: str
    memory_write: dict[str, Any] | None = None
    superseded_revision: int | None = None


class VoiceShadowGate:
    """Durable final-transcript/idempotency gate; it never calls MemoryCore."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS voice_event_revisions (
              namespace TEXT NOT NULL,
              conversation_id TEXT NOT NULL,
              utterance_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              fingerprint TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              accepted_at TEXT NOT NULL,
              superseded_by_revision INTEGER,
              PRIMARY KEY(namespace, conversation_id, utterance_id, revision)
            );
            CREATE TABLE IF NOT EXISTS voice_event_heads (
              namespace TEXT NOT NULL,
              conversation_id TEXT NOT NULL,
              utterance_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              fingerprint TEXT NOT NULL,
              PRIMARY KEY(namespace, conversation_id, utterance_id)
            );
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def accept(self, event: VoiceTranscriptEvent) -> ShadowDecision:
        if not event.is_final:
            return ShadowDecision("ignored", "partial_transcript")

        payload = _canonical_payload(event)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        key = (event.namespace, event.conversation_id, event.utterance_id)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            head = self._db.execute(
                """SELECT revision, fingerprint FROM voice_event_heads
                   WHERE namespace=? AND conversation_id=? AND utterance_id=?""",
                key,
            ).fetchone()
            if head is not None and event.revision < int(head["revision"]):
                self._db.rollback()
                return ShadowDecision("ignored", "stale_revision")
            if head is not None and event.revision == int(head["revision"]):
                self._db.rollback()
                if fingerprint == str(head["fingerprint"]):
                    return ShadowDecision("ignored", "idempotent_duplicate")
                return ShadowDecision("rejected", "same_revision_payload_conflict")

            previous = int(head["revision"]) if head is not None else None
            now = datetime.now(timezone.utc).isoformat()
            self._db.execute(
                """INSERT INTO voice_event_revisions
                   (namespace, conversation_id, utterance_id, revision,
                    fingerprint, payload_json, accepted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (*key, event.revision, fingerprint, payload, now),
            )
            if previous is not None:
                self._db.execute(
                    """UPDATE voice_event_revisions SET superseded_by_revision=?
                       WHERE namespace=? AND conversation_id=? AND utterance_id=?
                         AND revision=?""",
                    (event.revision, *key, previous),
                )
            self._db.execute(
                """INSERT INTO voice_event_heads
                   (namespace, conversation_id, utterance_id, revision, fingerprint)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(namespace, conversation_id, utterance_id)
                   DO UPDATE SET revision=excluded.revision,
                                 fingerprint=excluded.fingerprint""",
                (*key, event.revision, fingerprint),
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        return ShadowDecision(
            "accepted",
            "final_transcript",
            memory_write=_memory_write(event, fingerprint),
            superseded_revision=previous,
        )


def run_shadow(events: Iterable[Mapping[str, Any]], db_path: Path) -> dict[str, Any]:
    gate = VoiceShadowGate(db_path)
    rows: list[dict[str, Any]] = []
    try:
        for index, raw in enumerate(events):
            try:
                event = VoiceTranscriptEvent.from_mapping(raw)
                decision = gate.accept(event)
                rows.append({
                    "index": index,
                    "utterance_id": event.utterance_id,
                    "revision": event.revision,
                    **asdict(decision),
                })
            except (TypeError, ValueError) as error:
                rows.append({
                    "index": index,
                    "utterance_id": str(raw.get("utterance_id") or ""),
                    "revision": raw.get("revision"),
                    "status": "rejected",
                    "reason": f"invalid_event:{error}",
                    "memory_write": None,
                    "superseded_revision": None,
                })
    finally:
        gate.close()
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("accepted", "ignored", "rejected")
    }
    return {
        "schema_version": 1,
        "mode": "voice-final-transcript-shadow-no-memory-writes",
        "event_count": len(rows),
        **{f"{name}_count": count for name, count in counts.items()},
        "rows": rows,
    }


def _memory_write(event: VoiceTranscriptEvent, fingerprint: str) -> dict[str, Any]:
    return {
        "message_id": f"voice:{event.utterance_id}:r{event.revision}",
        "conversation_id": event.conversation_id,
        "speaker": event.speaker_id,
        "role": "assistant" if event.speaker_role == "assistant" else "user",
        "content": event.text,
        "timestamp": event.ended_at,
        "metadata": {
            "source": "voice_final_transcript",
            "source_system": event.source_system,
            "source_utterance_id": event.utterance_id,
            "source_revision": event.revision,
            "source_fingerprint": fingerprint,
            "speaker_id": event.speaker_id,
            "speaker_role": event.speaker_role,
            "vehicle_id": event.vehicle_id,
            "seat": event.seat,
            "started_at": event.started_at,
            "ended_at": event.ended_at,
            "transcript_confidence": event.transcript_confidence,
            "trace_id": event.trace_id,
        },
    }


def _canonical_payload(event: VoiceTranscriptEvent) -> str:
    return json.dumps(asdict(event), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid RFC3339 timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp requires timezone: {value}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    events = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = run_shadow(events, args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False))
    return 1 if summary["rejected_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
