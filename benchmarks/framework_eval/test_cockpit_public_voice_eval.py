from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .cockpit_public_voice_eval import (
    character_error_rate,
    evaluate_public_voice_rows,
    normalize_asr_text,
)


def _row(**changes):
    row = {
        "clip_id": "dev-001-0001",
        "reference": "导航去浦东机场 T1。",
        "hypothesis": "导航去浦东机场T1",
        "critical_terms": ["浦东机场", "T1"],
        "speaker_id": "speaker-01",
        "started_at": "2026-08-27T08:00:00+08:00",
        "ended_at": "2026-08-27T08:00:02+08:00",
    }
    row.update(changes)
    return row


class PublicVoiceEvalTests(unittest.TestCase):
    def test_mandarin_cer_ignores_spacing_case_and_punctuation(self):
        self.assertEqual("导航去浦东机场t1", normalize_asr_text("导航 去浦东机场 T1。"))
        self.assertEqual(0.0, character_error_rate("导航去浦东机场 T1。", "导航去浦东机场t1"))
        self.assertGreater(character_error_rate("导航去浦东机场", "导航去虹桥机场"), 0)

    def test_real_audio_rows_flow_through_partial_final_and_duplicate_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_public_voice_rows(
                [_row(), _row(
                    clip_id="dev-001-0002",
                    reference="把温度调到二十二度",
                    hypothesis="把温度调到二十度",
                    critical_terms=["二十二度"],
                )],
                Path(directory) / "shadow.db",
            )
        self.assertEqual(2, result["clips"])
        self.assertEqual(2, result["ingestion"]["accepted"])
        self.assertEqual(2, result["ingestion"]["reasons"]["partial_transcript"])
        self.assertEqual(2, result["ingestion"]["reasons"]["idempotent_duplicate"])
        self.assertTrue(result["gate_pass"])
        self.assertAlmostEqual(2 / 3, result["asr"]["critical_term_recall"])

    def test_empty_final_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_public_voice_rows(
                [_row(hypothesis="")], Path(directory) / "shadow.db"
            )
        self.assertFalse(result["gate_pass"])
        # The original final and its replay are both invalid before they can
        # reach the idempotency ledger, so both must fail closed.
        self.assertEqual(2, result["ingestion"]["rejected"])
        self.assertEqual(0, result["ingestion"]["accepted"])


if __name__ == "__main__":
    unittest.main()
