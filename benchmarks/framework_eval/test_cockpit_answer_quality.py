from __future__ import annotations

import unittest

from .cockpit_answer_quality import audit_user_facing_answer


class CockpitAnswerQualityTests(unittest.TestCase):
    def test_internal_evidence_artifacts_are_rejected(self):
        result = audit_user_facing_answer("根据 evidence #2，应选择最近的充电站。")
        self.assertFalse(result.passed)
        self.assertIn("internal_evidence_artifact", result.violations)

    def test_field_scoped_abstention_does_not_deny_known_person(self):
        good = audit_user_facing_answer(
            "The history does not identify Emma's school, so it cannot be determined.",
            known_subjects_with_other_history=["Emma"],
        )
        bad = audit_user_facing_answer(
            "There is no information about an individual named Emma.",
            known_subjects_with_other_history=["Emma"],
        )
        self.assertTrue(good.passed)
        self.assertIn("abstention_scope_too_broad:Emma", bad.violations)

    def test_required_final_state_fields_are_complete(self):
        contract = [["浦东机场", "Pudong"], ["T1"], ["取消", "cancel"]]
        bad = audit_user_facing_answer("浦东机场T1", required_term_groups=contract)
        good = audit_user_facing_answer(
            "浦东机场T1；虹桥机场T2已取消。", required_term_groups=contract
        )
        self.assertFalse(bad.passed)
        self.assertTrue(good.passed)

    def test_forbidden_person_attribute_transfer_is_detected(self):
        rule = [{
            "subject": "小雨",
            "attributes": ["喜欢安静", "安静的素食", "露台座位"],
            "window": 72,
        }]
        bad = audit_user_facing_answer(
            "小雨喜欢安静的素食餐厅和露台座位。", forbidden_attributions=rule
        )
        good = audit_user_facing_answer(
            "陈偏好安静、素食和露台；小雨需要无花生。", forbidden_attributions=rule
        )
        self.assertFalse(bad.passed)
        self.assertTrue(good.passed)


if __name__ == "__main__":
    unittest.main()
