from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .answering import AnswerConfig, OpenAIAnswerer, answer_retrieval_file
from .cockpit_episode import EpisodeTurn, compile_navigation_episode
from .cockpit_slots import extract_clarification_reply, extract_cockpit_answer


class CockpitSlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = """[1 sources=S02S024T01,S02S024T02,S02S024T03 time=2025-01-24T08:00:00Z]
[S02S024T01] [source_time=2025-01-24T08:00:00Z] Driver: please set my alarm for
[S02S024T02] [source_time=2025-01-24T08:00:05Z] Car Assistant: What should I use for the time?
[S02S024T03] [source_time=2025-01-24T08:00:10Z] Driver: five pm
"""
        self.question = (
            "During yesterday's driver voice interaction that began "
            '"please set my alarm for", what did the driver reply when the '
            'car assistant asked for the "time" detail?'
        )
        self.metadata = {
            "query_time": "2025-01-25T15:30:00Z",
            "timezone": "UTC",
        }

    def test_reads_one_adjacent_grounded_clarification(self):
        candidate = extract_clarification_reply(
            self.question, self.context, self.metadata
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("five pm", candidate.value)
        self.assertEqual(
            ("S02S024T01", "S02S024T02", "S02S024T03"),
            candidate.source_ids,
        )
        self.assertEqual(1.0, candidate.confidence)

    def test_rejects_unanchored_or_non_reply_questions(self):
        self.assertIsNone(extract_clarification_reply(
            "What time is my alarm?", self.context, self.metadata
        ))
        self.assertIsNone(extract_clarification_reply(
            self.question, self.context, {}
        ))

    def test_rejects_wrong_slot_and_ambiguous_matches(self):
        wrong_slot = self.context.replace("for the time?", "for the date?")
        self.assertIsNone(extract_clarification_reply(
            self.question, wrong_slot, self.metadata
        ))

        ambiguous = self.context + self.context.replace(
            "S02S024", "S03S024"
        ).replace("five pm", "six pm")
        self.assertIsNone(extract_clarification_reply(
            self.question, ambiguous, self.metadata
        ))

    def test_does_not_need_gold_answers_or_evidence_ids(self):
        candidate = extract_clarification_reply(
            self.question,
            self.context,
            {"query_time": "2025-01-25T15:30:00Z"},
        )

        self.assertEqual("five pm", candidate.value if candidate else "")

    def test_unified_answer_path_bypasses_model_and_records_provenance(self):
        answerer = OpenAIAnswerer(AnswerConfig(
            "http://unused", "", "weak-edge-model",
            temporal_query_mode="interval_v1",
            deterministic_slot_mode="cockpit_v1",
        ))
        row = self._retrieval_row()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "retrieval.jsonl"
            target = Path(directory) / "answers.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with mock.patch.object(answerer, "answer") as model_answer:
                summary = answer_retrieval_file(source, target, answerer)
            result = json.loads(target.read_text(encoding="utf-8"))

        model_answer.assert_not_called()
        self.assertEqual("five pm", result["predicted_answer"])
        self.assertEqual("deterministic_slot", result["answer_route"]["route"])
        self.assertFalse(result["answer_route"]["model_called"])
        self.assertFalse(result["answer_route"]["uses_gold_or_evidence_ids"])
        self.assertEqual(0, result["usage"]["total_tokens"])
        self.assertGreaterEqual(result["answer_route"]["seconds"], 0.0)
        self.assertEqual(
            {"deterministic_slot": 1, "model": 0},
            summary["answer_routes"],
        )
        self.assertEqual(
            1,
            summary["answer_route_latency"]["deterministic_slot"]["count"],
        )

    def test_unified_answer_path_falls_back_for_unsupported_question(self):
        answerer = OpenAIAnswerer(AnswerConfig(
            "http://unused", "", "weak-edge-model",
            deterministic_slot_mode="clarification_v1",
        ))
        row = self._retrieval_row()
        row["question"]["text"] = "What time is my alarm?"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "retrieval.jsonl"
            target = Path(directory) / "answers.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with mock.patch.object(
                answerer,
                "answer",
                return_value=("five pm", {"total_tokens": 12}),
            ) as model_answer:
                summary = answer_retrieval_file(source, target, answerer)
            result = json.loads(target.read_text(encoding="utf-8"))

        model_answer.assert_called_once()
        self.assertEqual("model", result["answer_route"]["route"])
        self.assertEqual(12, summary["tokens"]["total_tokens"])

    def test_calendar_time_slot_respects_speaker_anchor_and_date(self):
        question = (
            "During yesterday's vehicle interaction, what time did the "
            'driver specify for the "swimming" reminder?'
        )
        context = """[1 sources=S1T01,S1T02,S1T03 time=2025-01-24T08:00:00Z]
[S1T01] [source_time=2025-01-24T08:00:00Z] Driver: Schedule swimming.
[S1T02] [source_time=2025-01-24T08:00:05Z] Car Assistant: What time?
[S1T03] [source_time=2025-01-24T08:00:10Z] Driver: At 3 pm tomorrow.

[2 sources=S2T01 time=2025-01-20T08:00:00Z]
[S2T01] [source_time=2025-01-20T08:00:00Z] Driver: Swimming at 8 am.
"""
        candidate = extract_cockpit_answer(
            question,
            context,
            {"query_time": "2025-01-25T15:30:00Z"},
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("3 pm", candidate.value)
        self.assertEqual("grounded_calendar_time_slot", candidate.reason)
        self.assertEqual(("S1T03",), candidate.source_ids)

    def test_weather_location_slot_tracks_a_standalone_dialogue_value(self):
        question = (
            "During last Friday's vehicle interaction, which location did "
            'the driver specify for the forecast on "this week"?'
        )
        context = """[1 sources=W1T01,W1T02,W1T03 time=2025-01-24T08:00:00Z]
[W1T01] [source_time=2025-01-24T08:00:00Z] Driver: Show this week's forecast.
[W1T02] [source_time=2025-01-24T08:00:05Z] Car Assistant: Which city?
[W1T03] [source_time=2025-01-24T08:00:10Z] Driver: San Mateo, please.
[W1T04] [source_time=2025-01-24T08:00:15Z] Car Assistant: This week in San Mateo will be cloudy.
"""
        candidate = extract_cockpit_answer(
            question,
            context,
            {"query_time": "2025-01-31T15:30:00Z"},
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("San Mateo", candidate.value)
        self.assertEqual("grounded_weather_location_slot", candidate.reason)
        self.assertEqual(("W1T03",), candidate.source_ids)

    def test_cockpit_compiler_rejects_ambiguous_calendar_times(self):
        question = (
            "During yesterday's vehicle interaction, what time did the car "
            'assistant report for the "conference" event?'
        )
        context = """[1 sources=C1T01,C1T02 time=2025-01-24T08:00:00Z]
[C1T01] [source_time=2025-01-24T08:00:00Z] Driver: Which conference?
[C1T02] [source_time=2025-01-24T08:00:05Z] Car Assistant: The conference is either at 1 pm or 5 pm.
"""

        self.assertIsNone(extract_cockpit_answer(
            question,
            context,
            {"query_time": "2025-01-25T15:30:00Z"},
        ))

    def test_navigation_state_tracks_user_reselection_and_address(self):
        question = (
            "During yesterday's vehicle interaction, what destination did "
            'the driver mention in their "shopping center" request?'
        )
        context = """[1 sources=N1T01,N1T02,N1T03,N1T04 time=2025-01-24T08:00:00Z]
[N1T01] [source_time=2025-01-24T08:00:00Z] Driver: find the nearest shopping mall
[N1T02] [source_time=2025-01-24T08:00:05Z] Car Assistant: Midtown Shopping Center is closest. Ravenswood Shopping Center has no traffic.
[N1T03] [source_time=2025-01-24T08:00:10Z] Driver: Let's go to Ravenswood then.
[N1T04] [source_time=2025-01-24T08:00:15Z] Car Assistant: Setting directions to 434 Arastradero Rd.
"""

        candidate = extract_cockpit_answer(
            question,
            context,
            {"query_time": "2025-01-25T15:30:00Z"},
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("Ravenswood Shopping Center", candidate.value)
        self.assertEqual("grounded_navigation_state", candidate.reason)
        self.assertEqual(("N1T02", "N1T03", "N1T04"), candidate.source_ids)

    def test_navigation_state_keeps_home_alias_and_binds_address(self):
        question = (
            "During yesterday's vehicle interaction, which destination did "
            'the car assistant select for the driver\'s "home" request?'
        )
        context = """[1 sources=H1T01,H1T02,H1T03,H1T04 time=2025-01-24T08:00:00Z]
[H1T01] [source_time=2025-01-24T08:00:00Z] Driver: quickest route home
[H1T02] [source_time=2025-01-24T08:00:05Z] Car Assistant: Your home is five miles away.
[H1T03] [source_time=2025-01-24T08:00:10Z] Driver: Give me the address and set the route.
[H1T04] [source_time=2025-01-24T08:00:15Z] Car Assistant: Setting route to 5671 Barringer Street now.
"""

        candidate = extract_cockpit_answer(
            question,
            context,
            {"query_time": "2025-01-25T15:30:00Z"},
        )
        episode = compile_navigation_episode([
            EpisodeTurn("H1T01", "Driver", "quickest route home", sequence=0),
            EpisodeTurn(
                "H1T02", "Car Assistant", "Your home is five miles away.",
                sequence=1,
            ),
            EpisodeTurn(
                "H1T03", "Driver", "Give me the address and set the route.",
                sequence=2,
            ),
            EpisodeTurn(
                "H1T04", "Car Assistant",
                "Setting route to 5671 Barringer Street now.", sequence=3,
            ),
        ], domain="navigation")

        self.assertEqual("home", candidate.value if candidate else "")
        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertEqual("home", episode.destination)
        self.assertEqual("5671 Barringer Street", episode.address)
        self.assertIn("H1T04", episode.source_ids)

    def test_navigation_state_prefers_structured_nlu_slots(self):
        episode = compile_navigation_episode([
            EpisodeTurn(
                "S1", "Driver", "带我去那里", sequence=0,
                metadata={
                    "intent": "navigation.set_destination",
                    "slots": {
                        "destination": "虹桥机场",
                        "address": "申达一路1号",
                        "state": "confirmed",
                    },
                },
            ),
        ], intent="navigation.set_destination", domain="navigation")

        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertEqual("虹桥机场", episode.destination)
        self.assertEqual("申达一路1号", episode.address)
        self.assertEqual(1.0, episode.confidence)

    def test_navigation_state_accepts_canonical_slot_over_noisy_chinese_asr(self):
        episode = compile_navigation_episode([
            EpisodeTurn(
                "ZH1", "驾驶员", "带我去红桥鸡场T2", "2026-08-23T09:00:00+08:00",
                0, metadata={
                    "nlu": {
                        "intent": {
                            "name": "navigation.set_destination",
                            "confidence": 0.99,
                        },
                        "slots": [{
                            "name": "destination",
                            "value": "虹桥机场T2",
                            "confidence": 0.93,
                        }],
                    },
                    "asr": {
                        "text": "带我去红桥鸡场T2",
                        "confidence": 0.72,
                    },
                },
            ),
            EpisodeTurn(
                "ZH2", "车机", "已开始导航", "2026-08-23T09:00:01+08:00", 1,
            ),
        ], intent="navigation.set_destination", domain="navigation")

        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertEqual("虹桥机场T2", episode.destination)
        self.assertEqual("confirmed", episode.state)
        self.assertGreaterEqual(episode.confidence, 0.99)
        self.assertEqual(("ZH1", "ZH2"), episode.source_ids)

        hit = {
            "source_ids": ["ZH1", "ZH2"],
            "metadata": {"typed_cockpit_episode": episode.to_dict()},
        }
        candidate = extract_cockpit_answer(
            "我昨天上午最后让你导航去哪儿了？",
            "",
            {
                "query_time": "2026-08-24T15:30:00+08:00",
                "timezone": "Asia/Shanghai",
            },
            retrieval_hits=[hit],
        )
        self.assertEqual("虹桥机场T2", candidate.value if candidate else "")
        self.assertEqual(
            "grounded_typed_navigation_episode",
            candidate.reason if candidate else "",
        )

    def test_navigation_state_rejects_low_confidence_and_conflicting_nbest(self):
        low_confidence = compile_navigation_episode([
            EpisodeTurn(
                "LOW1", "驾驶员", "带我去那里", sequence=0,
                metadata={
                    "intent": "navigation.set_destination",
                    "slots": [{
                        "name": "destination",
                        "value": "虹桥机场",
                        "confidence": 0.72,
                    }],
                },
            ),
        ], intent="navigation.set_destination", domain="navigation")
        conflicting = compile_navigation_episode([
            EpisodeTurn(
                "NB1", "驾驶员", "带我去虹桥", sequence=0,
                metadata={
                    "intent": "navigation.set_destination",
                    "slots": [{
                        "name": "destination",
                        "value": "虹桥机场",
                        "confidence": 0.91,
                    }, {
                        "name": "destination",
                        "value": "虹桥火车站",
                        "confidence": 0.89,
                    }],
                },
            ),
        ], intent="navigation.set_destination", domain="navigation")

        self.assertIsNone(low_confidence)
        self.assertIsNone(conflicting)

    def test_navigation_state_tracks_chinese_correction_and_cancellation(self):
        corrected_turns = [
            EpisodeTurn(
                "C1", "驾驶员", "去虹桥机场", sequence=0,
                metadata={
                    "slots": {
                        "destination": {
                            "value": "虹桥机场",
                            "confidence": 0.99,
                        },
                        "state": "已确认",
                    },
                },
            ),
            EpisodeTurn(
                "C2", "驾驶员", "不对，改去虹桥火车站", sequence=1,
                metadata={
                    "slots": [{
                        "slot": "destination",
                        "value": "虹桥火车站",
                        "confidence": 0.99,
                    }, {
                        "slot": "state",
                        "value": "已选择",
                        "confidence": 0.99,
                    }],
                },
            ),
        ]
        corrected = compile_navigation_episode(
            corrected_turns,
            intent="navigation.set_destination",
            domain="navigation",
        )
        cancelled = compile_navigation_episode(
            corrected_turns + [EpisodeTurn(
                "C3", "驾驶员", "算了取消导航", sequence=2,
                metadata={"navigation_state": "已取消"},
            )],
            intent="navigation.set_destination",
            domain="navigation",
        )

        self.assertIsNotNone(corrected)
        self.assertEqual(
            "虹桥火车站", corrected.destination if corrected else ""
        )
        self.assertIsNotNone(cancelled)
        self.assertEqual("cancelled", cancelled.state if cancelled else "")
        self.assertEqual(
            "cancel", cancelled.transitions[-1].action if cancelled else ""
        )
        self.assertIsNone(extract_cockpit_answer(
            "我昨天导航去哪儿了？",
            "",
            {"query_time": "2026-08-24T15:30:00+08:00"},
            retrieval_hits=[{
                "source_ids": ["C1", "C2", "C3"],
                "metadata": {
                    "typed_cockpit_episode": cancelled.to_dict(),
                },
            }],
        ))

    def test_navigation_state_ignores_discourse_words_and_distance_phrases(self):
        episode = compile_navigation_episode([
            EpisodeTurn(
                "N3T01", "Driver", "Find me a Chinese restaurant.", sequence=0,
            ),
            EpisodeTurn(
                "N3T02", "Car Assistant",
                "There is a Jing Jing 2 miles away.", sequence=1,
            ),
            EpisodeTurn(
                "N3T03", "Driver", "Please set the GPS to go there.",
                sequence=2,
            ),
            EpisodeTurn(
                "N3T04", "Car Assistant", "Navigation is set.", sequence=3,
            ),
            EpisodeTurn(
                "N3T05", "Car Assistant", "Anytime!", sequence=4,
            ),
        ], domain="navigation")

        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertEqual("Jing Jing", episode.destination)
        self.assertEqual("", episode.address)
        self.assertGreaterEqual(episode.confidence, 0.97)

    def test_navigation_state_keeps_confirmed_poi_after_pleasantries(self):
        episode = compile_navigation_episode([
            EpisodeTurn(
                "N4T01", "Driver", "Where is a hospital within 4 miles?",
                sequence=0,
            ),
            EpisodeTurn(
                "N4T02", "Car Assistant",
                "Palo Alto Medical Foundation is 4 miles away.", sequence=1,
            ),
            EpisodeTurn(
                "N4T03", "Driver", "Yes, set the shortest route.", sequence=2,
            ),
            EpisodeTurn(
                "N4T04", "Car Assistant", "Setting GPS now.", sequence=3,
            ),
            EpisodeTurn(
                "N4T05", "Car Assistant", "Have a safe day.", sequence=4,
            ),
        ], domain="navigation")

        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertEqual("Palo Alto Medical Foundation", episode.destination)
        self.assertGreaterEqual(episode.confidence, 0.97)

    def test_navigation_state_accepts_one_nearest_offer_after_user_confirmation(self):
        episode = compile_navigation_episode([
            EpisodeTurn(
                "N5T01", "Driver", "Take me to the nearest parking garage.",
                sequence=0,
            ),
            EpisodeTurn(
                "N5T02", "Car Assistant",
                "The nearest parking garage is Palo Alto Garage R. Navigate there?",
                sequence=1,
            ),
            EpisodeTurn(
                "N5T03", "Driver", "Sure, pick the quickest route.", sequence=2,
            ),
            EpisodeTurn(
                "N5T04", "Car Assistant", "I sent the info to your screen.",
                sequence=3,
            ),
        ], domain="navigation")

        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertEqual("Palo Alto Garage R", episode.destination)
        self.assertGreaterEqual(episode.confidence, 0.97)

    def test_navigation_anchor_normalizes_spoken_possessive(self):
        question = (
            "During yesterday's vehicle interaction, which destination did "
            'the car assistant select for the driver\'s "friends house" request?'
        )
        context = """[1 sources=N6T01,N6T02,N6T03,N6T04 time=2025-01-24T08:00:00Z]
[N6T01] [source_time=2025-01-24T08:00:00Z] Driver: I need the address to my friend's house.
[N6T02] [source_time=2025-01-24T08:00:05Z] Car Assistant: Jills house is located at 347 Alta Mesa Ave.
[N6T03] [source_time=2025-01-24T08:00:10Z] Driver: Send the fastest route.
[N6T04] [source_time=2025-01-24T08:00:15Z] Car Assistant: Setting navigation details now. There's a collision nearby.
"""

        candidate = extract_cockpit_answer(
            question,
            context,
            {"query_time": "2025-01-25T15:30:00Z"},
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("Jills house", candidate.value)

    def test_navigation_state_falls_back_without_time_anchor(self):
        question = (
            "During yesterday's vehicle interaction, which destination did "
            'the car assistant select for the driver\'s "hospital" request?'
        )
        context = """[N1T01] Driver: Find a hospital.
[N1T02] Car Assistant: Navigating to Stanford Express Care.
"""

        self.assertIsNone(extract_cockpit_answer(question, context, {}))

    def test_navigation_answer_reads_source_bound_typed_episode(self):
        question = (
            "During yesterday's vehicle interaction, what destination did "
            'the driver mention in their "shopping center" request?'
        )
        episode = compile_navigation_episode([
            EpisodeTurn(
                "N2T01", "Driver", "find the nearest shopping mall",
                "2025-01-24T08:00:00Z", 0,
            ),
            EpisodeTurn(
                "N2T02", "Car Assistant",
                "Midtown Shopping Center or Ravenswood Shopping Center?",
                "2025-01-24T08:00:05Z", 1,
            ),
            EpisodeTurn(
                "N2T03", "Driver", "Let's go to Ravenswood then.",
                "2025-01-24T08:00:10Z", 2,
            ),
            EpisodeTurn(
                "N2T04", "Car Assistant",
                "Setting directions to 434 Arastradero Rd.",
                "2025-01-24T08:00:15Z", 3,
            ),
        ], domain="navigation")
        self.assertIsNotNone(episode)
        assert episode is not None
        hit = {
            "source_ids": ["N2T01", "N2T02", "N2T03", "N2T04"],
            "metadata": {"typed_cockpit_episode": episode.to_dict()},
        }

        candidate = extract_cockpit_answer(
            question,
            "",
            {"query_time": "2025-01-25T15:30:00Z"},
            retrieval_hits=[hit],
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("Ravenswood Shopping Center", candidate.value)
        self.assertEqual("grounded_typed_navigation_episode", candidate.reason)

        hit["source_ids"] = ["unrelated-source"]
        self.assertIsNone(extract_cockpit_answer(
            question,
            "",
            {"query_time": "2025-01-25T15:30:00Z"},
            retrieval_hits=[hit],
        ))

    def _retrieval_row(self) -> dict:
        return {
            "framework": "tencentdb",
            "question": {
                "question_id": "q1",
                "conversation_id": "conv1",
                "text": self.question,
                # Deliberately wrong: the fast path must never inspect it.
                "answers": ["not the answer"],
                "category": "alarm",
                "metadata": self.metadata,
            },
            "context": self.context,
            "hits": [],
            "metrics": {"context_chars": len(self.context)},
        }


if __name__ == "__main__":
    unittest.main()
