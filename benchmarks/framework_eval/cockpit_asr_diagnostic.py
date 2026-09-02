from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cockpit_episode import EpisodeTurn, TypedCockpitEpisode, compile_navigation_episode
from .cockpit_slots import extract_cockpit_answer


@dataclass(frozen=True)
class _DiagnosticCase:
    case_id: str
    turns: tuple[EpisodeTurn, ...]
    question: str
    metadata: Mapping[str, Any]
    expected_build: bool
    expected_answer: str | None
    extra_turn_sets: tuple[tuple[EpisodeTurn, ...], ...] = ()
    force_bad_lineage: bool = False


def _slot_turn(
    message_id: str,
    text: str,
    destination: str,
    confidence: float,
    *,
    timestamp: str = "2026-08-23T09:00:00+08:00",
    speaker: str = "驾驶员",
    sequence: int = 0,
    state: str = "已确认",
) -> EpisodeTurn:
    slots = [{
        "name": "destination",
        "value": destination,
        "confidence": confidence,
    }]
    if state:
        slots.append({
            "name": "state", "value": state, "confidence": confidence,
        })
    return EpisodeTurn(
        message_id,
        speaker,
        text,
        timestamp,
        sequence,
        metadata={
            "nlu": {
                "intent": {
                    "name": "navigation.set_destination",
                    "confidence": max(confidence, 0.0),
                },
                "slots": slots,
            },
            "asr": {"text": text, "confidence": max(0.0, confidence - 0.2)},
        },
    )


def _compile(turns: Sequence[EpisodeTurn]) -> TypedCockpitEpisode | None:
    return compile_navigation_episode(
        turns,
        intent="navigation.set_destination",
        domain="navigation",
    )


def _hit(
    episode: TypedCockpitEpisode,
    *,
    bad_lineage: bool = False,
) -> dict[str, Any]:
    return {
        "source_ids": (
            ["unrelated-source"] if bad_lineage else list(episode.source_ids)
        ),
        "metadata": {"typed_cockpit_episode": episode.to_dict()},
    }


def _cases() -> tuple[_DiagnosticCase, ...]:
    yesterday = {
        "query_time": "2026-08-24T15:30:00+08:00",
        "timezone": "Asia/Shanghai",
    }
    day_before = dict(yesterday)
    return (
        _DiagnosticCase(
            "homophone_canonical_t2",
            (
                _slot_turn(
                    "H1", "带我去红桥鸡场T2", "虹桥机场T2", 0.93
                ),
                EpisodeTurn(
                    "H2", "车机", "已开始导航",
                    "2026-08-23T09:00:01+08:00", 1,
                ),
            ),
            "我昨天上午让你导航去哪儿了？",
            yesterday,
            True,
            "虹桥机场T2",
        ),
        _DiagnosticCase(
            "filler_and_punctuation",
            (
                _slot_turn(
                    "F1", "嗯，那个……麻烦导航去静安寺啊", "静安寺", 0.98
                ),
            ),
            "我昨天上午导航到哪里了？",
            yesterday,
            True,
            "静安寺",
        ),
        _DiagnosticCase(
            "correction_supersedes",
            (
                _slot_turn("C1", "去虹桥火车站", "虹桥火车站", 0.99),
                _slot_turn(
                    "C2", "不对，改去虹桥机场", "虹桥机场", 0.99,
                    timestamp="2026-08-23T09:00:05+08:00", sequence=1,
                    state="已选择",
                ),
            ),
            "我昨天上午最后导航到哪儿了？",
            yesterday,
            True,
            "虹桥机场",
        ),
        _DiagnosticCase(
            "passenger_evening_command",
            (
                _slot_turn(
                    "P1", "去外滩", "外滩", 0.99,
                    timestamp="2026-08-22T20:00:00+08:00", speaker="乘客",
                ),
            ),
            "乘客前天晚上让你导航去哪儿了？",
            day_before,
            True,
            "外滩",
        ),
        _DiagnosticCase(
            "spoken_home_alias",
            (_slot_turn("A1", "回加", "家", 0.99),),
            "我昨天上午导航到哪里了？",
            yesterday,
            True,
            "家",
        ),
        _DiagnosticCase(
            "charging_poi_canonical",
            (_slot_turn(
                "E1", "找个充点桩", "虹桥天地充电站", 0.98
            ),),
            "我昨天上午导航到哪儿了？",
            yesterday,
            True,
            "虹桥天地充电站",
        ),
        _DiagnosticCase(
            "low_confidence_rejected",
            (_slot_turn("L1", "去虹桥", "虹桥机场", 0.72),),
            "我昨天上午导航到哪儿了？",
            yesterday,
            False,
            None,
        ),
        _DiagnosticCase(
            "conflicting_nbest_rejected",
            (EpisodeTurn(
                "N1", "驾驶员", "带我去虹桥", "2026-08-23T09:00:00+08:00",
                0, metadata={
                    "intent": "navigation.set_destination",
                    "slots": [{
                        "name": "destination", "value": "虹桥机场",
                        "confidence": 0.91,
                    }, {
                        "name": "destination", "value": "虹桥火车站",
                        "confidence": 0.89,
                    }],
                },
            ),),
            "我昨天上午导航到哪儿了？",
            yesterday,
            False,
            None,
        ),
        _DiagnosticCase(
            "cancelled_destination_not_answered",
            (
                _slot_turn("X1", "去浦东机场", "浦东机场", 0.99),
                EpisodeTurn(
                    "X2", "驾驶员", "算了，取消导航",
                    "2026-08-23T09:00:05+08:00", 1,
                    metadata={"navigation_state": "已取消"},
                ),
            ),
            "我昨天上午导航到哪儿了？",
            yesterday,
            True,
            None,
        ),
        _DiagnosticCase(
            "missing_query_time_fails_closed",
            (_slot_turn("T1", "去人民广场", "人民广场", 0.99),),
            "我昨天上午导航到哪儿了？",
            {},
            True,
            None,
        ),
        _DiagnosticCase(
            "source_lineage_mismatch_fails_closed",
            (_slot_turn("S1", "去世纪公园", "世纪公园", 0.99),),
            "我昨天上午导航到哪儿了？",
            yesterday,
            True,
            None,
            force_bad_lineage=True,
        ),
        _DiagnosticCase(
            "same_interval_ambiguity_fails_closed",
            (_slot_turn("M1", "去静安寺", "静安寺", 0.99),),
            "我昨天上午导航到哪儿了？",
            yesterday,
            True,
            None,
            extra_turn_sets=((_slot_turn(
                "M2", "去人民广场", "人民广场", 0.99,
                timestamp="2026-08-23T10:00:00+08:00",
            ),),),
        ),
    )


def run_diagnostics() -> dict[str, Any]:
    rows = []
    for case in _cases():
        episode = _compile(case.turns)
        episodes = [episode] if episode is not None else []
        episodes.extend(
            candidate
            for turns in case.extra_turn_sets
            for candidate in [_compile(turns)]
            if candidate is not None
        )
        hits = [
            _hit(item, bad_lineage=case.force_bad_lineage)
            for item in episodes
        ]
        answer = extract_cockpit_answer(
            case.question,
            "",
            case.metadata,
            retrieval_hits=hits,
        )
        observed_answer = answer.value if answer is not None else None
        build_ok = (episode is not None) == case.expected_build
        answer_ok = observed_answer == case.expected_answer
        rows.append({
            "case_id": case.case_id,
            "expected_build": case.expected_build,
            "observed_build": episode is not None,
            "expected_answer": case.expected_answer,
            "observed_answer": observed_answer,
            "answer_route": answer.reason if answer is not None else "fallback",
            "model_called": False,
            "pass": build_ok and answer_ok,
        })
    passed = sum(bool(row["pass"]) for row in rows)
    deterministic = sum(
        row["answer_route"] == "grounded_typed_navigation_episode"
        for row in rows
    )
    return {
        "schema_version": 1,
        "suite": "synthetic_chinese_asr_heldout_v1",
        "scope": (
            "adapter-side deterministic construction/query diagnostics; "
            "not a public benchmark or real microphone/ASR evaluation"
        ),
        "case_count": len(rows),
        "passed": passed,
        "pass_rate": passed / len(rows) if rows else 0.0,
        "deterministic_answers": deterministic,
        "model_calls": 0,
        "generation_tokens": 0,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run_diagnostics()
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if summary["passed"] == summary["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
