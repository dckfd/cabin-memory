from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .answering import AnswerConfig, OpenAIAnswerer, answer_retrieval_file
from .datasets.prepare_temporal_cockpit import VARIANTS, rewrite_question
from .schema import ContentPart
from .temporal import humanize_temporal_span, resolve_temporal_query


class TemporalAdapterTests(unittest.TestCase):
    def test_chinese_relative_day_parts_use_query_timezone(self):
        resolved = resolve_temporal_query(
            "我昨天上午让你导航去哪，前天晚上空调开了多少度？",
            {
                "query_time": "2026-08-24T15:30:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        )

        self.assertEqual("Asia/Shanghai", resolved.timezone_name)
        self.assertEqual(2, len(resolved.spans))
        self.assertEqual("2026-08-23T06:00:00+08:00", resolved.spans[0].start.isoformat())
        self.assertEqual("2026-08-23T12:00:00+08:00", resolved.spans[0].end.isoformat())
        self.assertEqual("morning", resolved.spans[0].granularity)
        self.assertEqual("2026-08-22T18:00:00+08:00", resolved.spans[1].start.isoformat())
        self.assertEqual("2026-08-23T00:00:00+08:00", resolved.spans[1].end.isoformat())

    def test_longmemeval_question_date_is_a_compatible_anchor(self):
        resolved = resolve_temporal_query(
            "What did I do yesterday morning?",
            {"question_date": "2023/05/30 (Tue) 23:17"},
        )

        self.assertEqual("question_date", resolved.anchor_source)
        self.assertEqual("2023-05-30T23:17:00+00:00", resolved.anchor.isoformat())
        self.assertEqual("2023-05-29T06:00:00+00:00", resolved.spans[0].start.isoformat())
        self.assertIn("normalized_query_time", resolved.retrieval_text("question"))

    def test_anchor_offset_is_preserved_without_named_timezone(self):
        resolved = resolve_temporal_query(
            "昨天晚上",
            {"query_time": "2026-08-24T00:30:00+08:00"},
        )

        self.assertEqual("+08:00", resolved.timezone_name)
        self.assertEqual("2026-08-23T18:00:00+08:00", resolved.spans[0].start.isoformat())

    def test_relative_expression_is_not_invented_without_anchor(self):
        resolved = resolve_temporal_query("昨天上午导航去了哪里？")

        self.assertFalse(resolved.active)
        self.assertEqual("昨天上午导航去了哪里？", resolved.answer_text("昨天上午导航去了哪里？"))

    def test_explicit_date_does_not_require_anchor_or_prompt_envelope(self):
        resolved = resolve_temporal_query("2026-08-23 导航去了哪里？")

        self.assertTrue(resolved.active)
        self.assertFalse(resolved.relative)
        self.assertEqual("2026-08-23", resolved.spans[0].start.date().isoformat())
        self.assertEqual("question", resolved.answer_text("question"))

    def test_ambiguous_recency_is_an_ordering_operator(self):
        resolved = resolve_temporal_query(
            "上次下雨时我去了哪里？",
            {"query_time": "2026-08-24T15:30:00+08:00"},
        )

        self.assertEqual(("latest",), resolved.operators)
        self.assertEqual((), resolved.spans)

    def test_existing_english_human_date_contract_is_preserved(self):
        resolved = resolve_temporal_query(
            "I went yesterday, last Friday, and last week.",
            {"query_time": "2023-07-15T13:51:00Z"},
        )
        values = {span.raw.casefold(): humanize_temporal_span(span)
                  for span in resolved.spans}

        self.assertEqual("14 July 2023", values["yesterday"])
        self.assertEqual("14 July 2023", values["last friday"])
        self.assertEqual("3 July 2023 to 9 July 2023", values["last week"])

        weeks_ago = resolve_temporal_query(
            "That happened four weeks ago.",
            {"query_time": "2023-07-15T13:51:00Z"},
        )
        self.assertEqual(
            "17 June 2023", humanize_temporal_span(weeks_ago.spans[0])
        )

    def test_quoted_relative_word_is_content_not_a_query_time_constraint(self):
        resolved = resolve_temporal_query(
            'During last Tuesday\'s interaction, which location was requested '
            'for the forecast on "today"?',
            {"query_time": "2024-01-23T15:30:00Z", "timezone": "UTC"},
        )

        self.assertEqual(1, len(resolved.spans))
        self.assertEqual("last Tuesday", resolved.spans[0].raw)
        self.assertEqual(
            "2024-01-16T00:00:00+00:00", resolved.spans[0].start.isoformat()
        )
        self.assertNotIn('"today"=', resolved.answer_text("question"))

        quoted_only = resolve_temporal_query(
            'What did the driver mean by "yesterday" and "last time"?',
            {"query_time": "2024-01-23T15:30:00Z"},
        )
        self.assertFalse(quoted_only.active)

    def test_answer_envelope_is_compact_and_preserves_raw_question(self):
        question = "我前天晚上把空调调到多少度？"
        resolved = resolve_temporal_query(
            question,
            {"query_time": "2026-08-24T15:30:00+08:00", "timezone": "Asia/Shanghai"},
        )
        rendered = resolved.answer_text(question)

        self.assertIn("request_time=2026-08-24T15:30:00+08:00", rendered)
        self.assertIn('"前天晚上"=2026-08-22T18:00:00+08:00/', rendered)
        self.assertTrue(rendered.endswith(question))

    def test_answer_payload_injects_anchor_only_into_temporal_user_message(self):
        answerer = OpenAIAnswerer(AnswerConfig(
            "http://unused", "", "model",
            temporal_query_mode="interval_v1",
            temporal_default_timezone="Asia/Shanghai",
        ))
        payload = answerer.build_payload(
            "我昨天上午让你导航去哪？",
            "[D1] Driver: 导航去虹桥机场",
            question_metadata={
                "query_time": "2026-08-24T15:30:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        )

        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        self.assertNotIn("request_time=", system)
        self.assertIn("Evidence:\n[D1] Driver: 导航去虹桥机场", system)
        self.assertIn("request_time=2026-08-24T15:30:00+08:00", user)
        self.assertIn('"昨天上午"=2026-08-23T06:00:00+08:00/', user)
        self.assertTrue(user.endswith("我昨天上午让你导航去哪？"))

    def test_answer_payload_keeps_ordinary_and_unanchored_queries_identical(self):
        answerer = OpenAIAnswerer(AnswerConfig(
            "http://unused", "", "model",
            temporal_query_mode="interval_v1",
        ))

        ordinary = answerer.build_payload(
            "打开空调", "evidence",
            question_metadata={"query_time": "2026-08-24T15:30:00+08:00"},
        )
        unanchored = answerer.build_payload("昨天导航去了哪里？", "evidence")

        self.assertEqual("打开空调", ordinary["messages"][1]["content"])
        self.assertEqual(
            "昨天导航去了哪里？", unanchored["messages"][1]["content"]
        )

    def test_multimodal_payload_enriches_text_part_without_touching_media(self):
        image = ContentPart(type="image", uri="https://example.test/route.png")
        answerer = OpenAIAnswerer(AnswerConfig(
            "http://unused", "", "model", multimodal=True,
            temporal_query_mode="interval_v1",
        ))

        payload = answerer.build_payload(
            "昨天去了哪里？", "evidence", (image,),
            question_metadata={"query_time": "2026-08-24T15:30:00Z"},
        )
        content = payload["messages"][1]["content"]

        self.assertIn("request_time=", content[0]["text"])
        self.assertEqual(
            "https://example.test/route.png",
            content[1]["image_url"]["url"],
        )

    def test_answer_file_uses_question_metadata_and_records_temporal_cost(self):
        answerer = OpenAIAnswerer(AnswerConfig(
            "http://unused", "", "model",
            temporal_query_mode="interval_v1",
            temporal_default_timezone="Asia/Shanghai",
        ))
        row = {
            "framework": "tencentdb",
            "question": {
                "question_id": "q1",
                "conversation_id": "conv1",
                "text": "我前天晚上把空调调到多少度？",
                "answers": ["22度"],
                "category": "temporal",
                "metadata": {
                    "query_time": "2026-08-24T15:30:00+08:00",
                    "timezone": "Asia/Shanghai",
                },
            },
            "context": "Driver: 把空调调到22度",
            "hits": [],
            "metrics": {"context_chars": 18},
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "retrieval.jsonl"
            target = Path(directory) / "answers.jsonl"
            source.write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with mock.patch.object(
                answerer, "answer", return_value=("22度", {"prompt_tokens": 20})
            ) as answer:
                answer_retrieval_file(source, target, answerer)
            result = json.loads(target.read_text(encoding="utf-8"))

        prepared = answer.call_args.args[0]
        self.assertIn("request_time=2026-08-24T15:30:00+08:00", prepared)
        self.assertTrue(result["answer_temporal"]["injected"])
        self.assertGreater(result["answer_temporal"]["added_chars"], 0)
        self.assertEqual("22度", result["predicted_answer"])

    def test_temporal_cockpit_variants_resolve_to_the_original_source_date(self):
        row = {
            "qa_id": "q1",
            "question": (
                "During the vehicle interaction logged on 2024-01-08, "
                "where did the driver navigate?"
            ),
            "answer": ["home"],
            "evidence": ["D1"],
            "category": "navigation",
            "metadata": {"interaction_date": "2024-01-08"},
        }

        for variant in VARIANTS:
            with self.subTest(variant=variant):
                rewritten = rewrite_question(row, variant)
                resolved = resolve_temporal_query(
                    rewritten["question"], rewritten["metadata"]
                )
                self.assertTrue(resolved.relative)
                self.assertEqual(
                    "2024-01-08", resolved.spans[0].start.date().isoformat()
                )
                self.assertNotIn("2024-01-08", rewritten["question"])
                self.assertEqual(row["answer"], rewritten["answer"])
                self.assertEqual(row["evidence"], rewritten["evidence"])


if __name__ == "__main__":
    unittest.main()
