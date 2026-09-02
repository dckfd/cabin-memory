from __future__ import annotations

import unittest

from .cockpit_asr_diagnostic import run_diagnostics


class CockpitASRDiagnosticTests(unittest.TestCase):
    def test_all_synthetic_chinese_asr_cases_fail_closed_or_answer(self):
        result = run_diagnostics()

        self.assertEqual(12, result["case_count"])
        self.assertEqual(result["case_count"], result["passed"])
        self.assertEqual(0, result["model_calls"])
        self.assertEqual(0, result["generation_tokens"])


if __name__ == "__main__":
    unittest.main()
