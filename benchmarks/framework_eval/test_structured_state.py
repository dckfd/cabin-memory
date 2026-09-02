import unittest

from .answering import _answer_contract, _validate_answer_contract
from .structured_state import parse_events, resolve_state_answer, temporal_context


class StructuredStateTests(unittest.TestCase):
    def test_counts_only_completed_source_events(self):
        context = "\n".join([
            "[s1:001] [source_time=2026-01-01T08:00:00+08:00] [source_role=user] 【甲】导航到星河湾，已到达。",
            "[s2:001] [source_time=2026-01-02T08:00:00+08:00] [source_role=user] 【甲】导航到星河湾，已经到达。",
            "[s3:001] [source_time=2026-01-03T08:00:00+08:00] [source_role=user] 【甲】导航到云杉路。",
        ])
        result = resolve_state_answer("按地点统计去得最多的地方一共几次？", context)
        self.assertEqual(result["value"], "星河湾，共2次。")
        self.assertEqual(len(result["source_ids"]), 2)

    def test_latest_is_time_ordered_and_pair_is_complete(self):
        context = "\n".join([
            "[s2:001] [source_time=2026-01-02T08:00:00+08:00] 【甲】导航到乙，已到达。",
            "[s1:001] [source_time=2026-01-01T08:00:00+08:00] 【甲】导航到甲，已到达。",
        ])
        result = resolve_state_answer("上一次和最后一次导航分别去了哪里？", context)
        self.assertEqual(result["value"], "上一次：甲；最后一次：乙。")

    def test_cancelled_event_is_not_counted(self):
        context = "[s1:001] [source_time=2026-01-01T08:00:00+08:00] 【甲】导航到甲，已取消。"
        self.assertIsNone(resolve_state_answer("按地点统计一共几次？", context))

    def test_inclusive_cutoff_does_not_leak_later_event(self):
        context = "\n".join([
            "[old:001] [source_time=2026-03-01T09:00:00+08:00] 【甲】导航到旧地点，已到达。",
            "[new:001] [source_time=2026-03-04T09:00:00+08:00] 【甲】导航到新地点，已到达。",
        ])
        question = "只看3月1日这次及更早的记录，3月4日不能倒灌，导航去哪里？"
        filtered = temporal_context(question, context, {"question_date": "2026-03-24T10:00:00+08:00", "timezone": "Asia/Shanghai"})
        self.assertIn("旧地点", filtered)
        self.assertNotIn("新地点", filtered)

    def test_two_date_snapshot_keeps_prior_state_transitions(self):
        context = "\n".join([
            "[base:001] [source_time=2026-03-10T09:00:00+08:00] 【甲】临时目的地为甲地。",
            "[change:001] [source_time=2026-03-15T09:00:00+08:00] 【甲】3月15日至18日改成乙地。",
            "[restore:001] [source_time=2026-03-19T09:00:00+08:00] 【甲】恢复为甲地。",
        ])
        question = "按日期分别查3月16日和3月20日的临时目的地。"
        filtered = temporal_context(question, context, {"question_date": "2026-03-24T10:00:00+08:00", "timezone": "Asia/Shanghai"})
        self.assertEqual(filtered, context)

    def test_two_date_effective_state_ledger(self):
        context = "\n".join([
            "[base:001] [source_time=2026-03-10T09:00:00+08:00] [source_role=user] 【甲】平时说临时目的地就导航到甲地。",
            "[change:001] [source_time=2026-03-11T09:00:00+08:00] [source_role=user] 【甲】3月15日至18日临时目的地改成乙地。",
            "[restore:001] [source_time=2026-03-19T09:00:00+08:00] [source_role=user] 【甲】临时安排结束，从今天起临时目的地恢复为甲地。",
        ])
        result = resolve_state_answer(
            "按日期分别查甲的临时目的地：3月16日和3月20日各是哪里？",
            context,
            metadata={"question_date": "2026-03-24T10:00:00+08:00", "timezone": "Asia/Shanghai"},
        )
        self.assertEqual(result["value"], "3月16日是乙地；3月20日是甲地。")
        self.assertEqual(result["reason"], "structured_effective_time_snapshots")

    def test_cutoff_query_extracts_named_field_only(self):
        context = "\n".join([
            "[old:001] [source_time=2026-03-01T09:00:00+08:00] [source_role=user] 【甲】帮我找一家评分4.5分以上，人均消费100-150元的餐馆。",
            "[new:001] [source_time=2026-03-04T09:00:00+08:00] [source_role=user] 【甲】再找一家人均消费500-1000元的餐馆。",
        ])
        result = resolve_state_answer(
            "只看3月1日及更早记录，3月4日不能倒灌，甲当时明确查询的餐馆人均消费是什么？",
            context,
            metadata={"question_date": "2026-03-24T10:00:00+08:00", "timezone": "Asia/Shanghai"},
        )
        self.assertEqual(result["value"], "截至3月1日是100-150元。")
        self.assertEqual(result["source_ids"], ["old:001"])

    def test_cutoff_query_preserves_person_binding(self):
        context = "\n".join([
            "[a:001] [source_time=2026-03-01T09:00:00+08:00] [source_role=user] 【甲】查下周四的航班。",
            "[b:001] [source_time=2026-03-01T09:00:00+08:00] [source_role=user] 【乙】查下周六的航班。",
        ])
        result = resolve_state_answer(
            "只看3月1日及更早记录，甲当时明确查询的飞机日期是什么？",
            context,
            metadata={"question_date": "2026-03-24T10:00:00+08:00", "timezone": "Asia/Shanghai"},
        )
        self.assertEqual(result["value"], "截至3月1日是下周四。")
        self.assertEqual(result["source_ids"], ["a:001"])

    def test_temporal_answer_contracts(self):
        dual = "按日期分别查3月16日和3月20日各是哪里？"
        self.assertIn("每个查询日期", _answer_contract(dual))
        self.assertFalse(_validate_answer_contract(dual, "3月16日是甲地。")[0])
        self.assertTrue(_validate_answer_contract(dual, "3月16日是甲地；3月20日是乙地。")[0])
        cutoff = "只看3月1日及更早记录，3月4日不能倒灌。"
        self.assertFalse(_validate_answer_contract(cutoff, "是甲地。")[0])
        self.assertTrue(_validate_answer_contract(cutoff, "截至3月1日是甲地。")[0])


if __name__ == "__main__":
    unittest.main()
