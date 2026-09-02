from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .cockpit_voice_shadow import VoiceShadowGate, VoiceTranscriptEvent, run_shadow


def _event(**changes):
    value = {
        "namespace": "vehicle-42/driver-li",
        "conversation_id": "trip-20260827",
        "utterance_id": "utt-001",
        "revision": 1,
        "is_final": True,
        "text": "导航去公司地下车库",
        "speaker_id": "driver-li",
        "speaker_role": "driver",
        "started_at": "2026-08-27T08:00:00+08:00",
        "ended_at": "2026-08-27T08:00:02+08:00",
        "vehicle_id": "vehicle-42",
        "seat": "driver",
        "transcript_confidence": 0.98,
    }
    value.update(changes)
    return value


class CockpitVoiceShadowTests(unittest.TestCase):
    def test_partial_is_ignored_and_never_becomes_a_memory_write(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_shadow([_event(is_final=False)], Path(directory) / "ledger.db")
        self.assertEqual(0, summary["accepted_count"])
        self.assertEqual("partial_transcript", summary["rows"][0]["reason"])
        self.assertIsNone(summary["rows"][0]["memory_write"])

    def test_replay_is_idempotent_and_same_revision_conflict_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "ledger.db"
            gate = VoiceShadowGate(db)
            event = VoiceTranscriptEvent.from_mapping(_event())
            self.assertEqual("accepted", gate.accept(event).status)
            self.assertEqual("idempotent_duplicate", gate.accept(event).reason)
            conflict = VoiceTranscriptEvent.from_mapping(_event(text="导航去机场"))
            self.assertEqual("same_revision_payload_conflict", gate.accept(conflict).reason)
            gate.close()

    def test_higher_final_revision_supersedes_and_lower_revision_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = VoiceShadowGate(Path(directory) / "ledger.db")
            first = VoiceTranscriptEvent.from_mapping(_event())
            correction = VoiceTranscriptEvent.from_mapping(_event(revision=2, text="改去虹桥机场T2"))
            gate.accept(first)
            accepted = gate.accept(correction)
            stale = gate.accept(first)
            gate.close()
        self.assertEqual(1, accepted.superseded_revision)
        self.assertEqual("stale_revision", stale.reason)

    def test_memory_projection_preserves_speaker_time_revision_and_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = VoiceShadowGate(Path(directory) / "ledger.db")
            decision = gate.accept(VoiceTranscriptEvent.from_mapping(_event()))
            gate.close()
        write = decision.memory_write or {}
        self.assertEqual("driver-li", write["speaker"])
        self.assertEqual("2026-08-27T08:00:02+08:00", write["timestamp"])
        self.assertEqual(1, write["metadata"]["source_revision"])
        self.assertEqual("utt-001", write["metadata"]["source_utterance_id"])
        self.assertNotIn("audio", write)

    def test_invalid_speaker_time_and_confidence_are_rejected(self):
        cases = [
            _event(speaker_role="unknown"),
            _event(ended_at="2026-08-27T07:59:59+08:00"),
            _event(transcript_confidence=1.2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = run_shadow(cases, Path(directory) / "ledger.db")
        self.assertEqual(3, summary["rejected_count"])
        self.assertTrue(all(row["memory_write"] is None for row in summary["rows"]))


if __name__ == "__main__":
    unittest.main()
