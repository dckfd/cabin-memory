from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


VERSION = "cockpit-long-memory-20-v1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "challenges"
    / "cockpit_long_memory_20_v1"
)


def _turn(speaker: str, content: str) -> dict:
    role = "assistant" if speaker == "Car Assistant" else "user"
    return {
        "role": role,
        # LongMemEval's adapter does not expose a separate speaker field.  A
        # source-visible prefix preserves driver/passenger attribution without
        # changing that benchmark adapter or its prompts.
        "content": f"[{speaker}] {content}",
        "has_answer": False,
    }


def _session(
    conversation_id: str,
    index: int,
    timestamp: str,
    turns: Iterable[tuple[str, str]],
) -> dict:
    return {
        "session_index": index,
        "source_session_id": f"{conversation_id}-s{index:02d}",
        "date_time": timestamp,
        "messages": [_turn(speaker, content) for speaker, content in turns],
    }


def _question(
    index: int,
    conversation_id: str,
    category: str,
    text: str,
    answer: str,
    question_time: str,
    evidence: Iterable[str] = (),
    *,
    abstention: bool = False,
    abilities: Iterable[str] = (),
) -> dict:
    return {
        "qa_id": f"cockpit-hard#q{index:02d}",
        "sample_id": conversation_id,
        "category": category,
        "question_type": category,
        "question": text,
        "answer": [answer],
        "question_date": question_time,
        # The existing LongMemEval adapter treats this field as evidence IDs.
        # We intentionally store exact message IDs, not just session IDs, so
        # evidence recall remains auditable in the common runner.
        "answer_session_ids": list(evidence),
        "is_abstention": abstention,
        "metadata": {
            "timezone": "Asia/Shanghai",
            "abilities": list(abilities),
            "original_synthetic": True,
        },
    }


def build() -> tuple[list[dict], list[dict]]:
    conversations: list[dict] = []
    questions: list[dict] = []

    cn1 = "cockpit-hard-cn-01"
    cn1_sessions = [
        _session(cn1, 1, "2026-05-01T07:30:00+08:00", [
            ("驾驶员陈", "请记住：工作日早上我宁愿走滨江路，多花五分钟也不要上高架；收费站也要避开。"),
            ("Car Assistant", "记住了：早高峰优先滨江路，避开高架和收费站。"),
        ]),
        _session(cn1, 2, "2026-05-02T18:10:00+08:00", [
            ("副驾小雨", "我对花生严重过敏，和我一起吃饭时一定选无花生的餐厅。"),
            ("Car Assistant", "已记录小雨的花生过敏约束。"),
        ]),
        _session(cn1, 3, "2026-05-03T12:00:00+08:00", [
            ("驾驶员陈", "我喜欢安静的素食餐厅，最好还有露台座位。"),
            ("Car Assistant", "我会优先考虑安静、素食且有露台的餐厅。"),
        ]),
        _session(cn1, 4, "2026-05-04T09:20:00+08:00", [
            ("驾驶员陈", "帮我看一下胎压，先不要改任何车辆设置。"),
            ("Car Assistant", "四个轮胎胎压都在正常范围内。"),
        ]),
        _session(cn1, 5, "2026-05-05T07:25:00+08:00", [
            ("驾驶员陈", "更新一下通勤偏好：现在工作日早上以最快路线为准，可以走高架，但仍然避开收费站。"),
            ("Car Assistant", "通勤偏好已更新：最快路线、高架可用、继续避开收费站。"),
        ]),
        _session(cn1, 6, "2026-05-06T18:30:00+08:00", [
            ("驾驶员陈", "小雨也在车上，按我们的要求找一家晚餐餐厅。"),
            ("Car Assistant", "绿园是素食餐厅但使用花生油；莲坊安静、无花生、有素食菜单和露台。"),
            ("驾驶员陈", "那就选莲坊。"),
            ("Car Assistant", "好的，已选择莲坊。"),
        ]),
        _session(cn1, 7, "2026-05-07T14:00:00+08:00", [
            ("驾驶员陈", "导航到虹桥机场T2。"),
            ("Car Assistant", "准备导航到虹桥机场T2。"),
            ("驾驶员陈", "不对，改去浦东机场T1，虹桥那条路线取消。"),
            ("Car Assistant", "已取消虹桥路线，最终目的地更新为浦东机场T1。"),
        ]),
        _session(cn1, 8, "2026-05-08T08:40:00+08:00", [
            ("驾驶员陈", "把车辆保养预约在5月12日周二上午8点。"),
            ("Car Assistant", "保养已安排在5月12日上午8点。"),
        ]),
        _session(cn1, 9, "2026-05-09T11:15:00+08:00", [
            ("驾驶员陈", "把5月12日8点的保养改到5月15日周五上午10点，原预约取消。"),
            ("Car Assistant", "已取消原预约并改到5月15日上午10点。"),
        ]),
        _session(cn1, 10, "2026-05-09T19:00:00+08:00", [
            ("驾驶员陈", "播放今天的本地新闻。"),
            ("Car Assistant", "正在播放本地新闻。"),
        ]),
    ]
    conversations.append(_conversation(cn1, cn1_sessions))
    questions.extend([
        _question(1, cn1, "knowledge-update",
                  "截至5月10日，驾驶员工作日早上的最新路线偏好是什么？请同时说明高架和收费站约束。",
                  "选择最快路线，可以走高架，但仍要避开收费站。",
                  "2026-05-10T09:00:00+08:00",
                  [f"{cn1}-s01:001", f"{cn1}-s05:001"],
                  abilities=["knowledge_update", "constraint_tracking"]),
        _question(2, cn1, "multi-session",
                  "小雨在车上且要吃晚餐时，综合两个人在不同会话中的要求，应该选择哪家餐厅？为什么？",
                  "选择莲坊，因为它安静、提供素食和露台，并且无花生，满足驾驶员偏好与小雨的过敏约束。",
                  "2026-05-10T18:00:00+08:00",
                  [f"{cn1}-s02:001", f"{cn1}-s03:001", f"{cn1}-s06:002", f"{cn1}-s06:003"],
                  abilities=["multi_session_reasoning", "multi_speaker", "constraint_intersection"]),
        _question(3, cn1, "knowledge-update",
                  "5月7日那次机场导航经过改口后，最终确认的机场和航站楼是什么？",
                  "浦东机场T1；虹桥机场T2已被取消。",
                  "2026-05-10T18:05:00+08:00",
                  [f"{cn1}-s07:001", f"{cn1}-s07:003", f"{cn1}-s07:004"],
                  abilities=["supersession", "final_state"]),
        _question(4, cn1, "temporal-reasoning",
                  "按5月10日时的最新安排，车辆保养最终在什么日期和时间？",
                  "5月15日周五上午10点。",
                  "2026-05-10T18:10:00+08:00",
                  [f"{cn1}-s08:001", f"{cn1}-s09:001", f"{cn1}-s09:002"],
                  abilities=["temporal_reasoning", "reschedule", "latest_state"]),
        _question(5, cn1, "multi-session",
                  "驾驶员最常听的播客节目是什么？",
                  "历史中没有提供驾驶员常听的播客节目，无法确定。",
                  "2026-05-10T18:15:00+08:00", abstention=True,
                  abilities=["abstention"]),
    ])

    cn2 = "cockpit-hard-cn-02"
    cn2_sessions = [
        _session(cn2, 1, "2026-06-01T20:10:00+08:00", [
            ("驾驶员李", "我周一晚去了徐汇滨江超充站，那里充电很方便。"),
            ("Car Assistant", "已记录本次在徐汇滨江超充站充电。"),
        ]),
        _session(cn2, 2, "2026-06-02T08:05:00+08:00", [
            ("驾驶员李", "今天空气质量怎么样？"),
            ("Car Assistant", "今天空气质量良好。"),
        ]),
        _session(cn2, 3, "2026-06-03T19:50:00+08:00", [
            ("驾驶员李", "周三我又去了徐汇滨江超充站。"),
            ("Car Assistant", "已记录第二次在徐汇滨江超充站充电。"),
        ]),
        _session(cn2, 4, "2026-06-05T21:00:00+08:00", [
            ("驾驶员李", "周五我在虹桥枢纽充电站充了电。"),
            ("Car Assistant", "已记录虹桥枢纽充电站。"),
        ]),
        _session(cn2, 5, "2026-06-06T10:30:00+08:00", [
            ("驾驶员李", "我通常喜欢有顶棚和卫生间的充电站；但电量低于15%时是例外，最近的站优先。"),
            ("Car Assistant", "已记录：通常优先顶棚和卫生间，低于15%时改为距离优先。"),
        ]),
        _session(cn2, 6, "2026-06-07T16:20:00+08:00", [
            ("Car Assistant", "当前电量12%。安悦露天充电站距2公里，带顶棚的云汇站距12公里。"),
            ("驾驶员李", "电量太低，按例外规则去最近的安悦。"),
            ("Car Assistant", "已开始导航到安悦露天充电站。"),
        ]),
        _session(cn2, 7, "2026-06-08T09:00:00+08:00", [
            ("驾驶员李", "把康平路88号的爸妈家记作‘老地方’。"),
            ("Car Assistant", "已将‘老地方’关联到康平路88号。"),
        ]),
        _session(cn2, 8, "2026-06-09T17:40:00+08:00", [
            ("驾驶员李", "帮我找附近停车场，但不要开始导航。"),
            ("Car Assistant", "找到了三个停车场，没有启动导航。"),
        ]),
        _session(cn2, 9, "2026-06-10T08:15:00+08:00", [
            ("驾驶员李", "更新一下：以后‘老地方’指公司地下车库；康平路88号只叫‘爸妈家’。"),
            ("Car Assistant", "别名已更新：老地方是公司地下车库，爸妈家是康平路88号。"),
        ]),
        _session(cn2, 10, "2026-06-11T13:00:00+08:00", [
            ("驾驶员李", "我自己开车时空调喜欢21度。"),
            ("Car Assistant", "已记录驾驶员空调偏好为21度。"),
        ]),
        _session(cn2, 11, "2026-06-12T13:05:00+08:00", [
            ("副驾阿岚", "我坐副驾时喜欢24度，别用驾驶员的21度。"),
            ("Car Assistant", "已分别记录阿岚24度和驾驶员21度。"),
        ]),
    ]
    conversations.append(_conversation(cn2, cn2_sessions))
    questions.extend([
        _question(6, cn2, "multi-session",
                  "6月第一周记录的三个充电事件中，驾驶员最常去哪个充电站，共去了几次？",
                  "徐汇滨江超充站，共两次。",
                  "2026-06-13T09:00:00+08:00",
                  [f"{cn2}-s01:001", f"{cn2}-s03:001", f"{cn2}-s04:001"],
                  abilities=["multi_session_reasoning", "frequency_aggregation"]),
        _question(7, cn2, "single-session-preference",
                  "如果现在电量只有10%，充电站选择规则是什么？顶棚和卫生间是否仍然优先？",
                  "应优先选择最近的充电站；低于15%时距离优先，顶棚和卫生间不再是首要条件。",
                  "2026-06-13T09:05:00+08:00",
                  [f"{cn2}-s05:001", f"{cn2}-s06:001", f"{cn2}-s06:002"],
                  abilities=["conditional_preference", "exception_handling"]),
        _question(8, cn2, "knowledge-update",
                  "目前‘老地方’和‘爸妈家’分别指哪里？请使用最新的别名定义。",
                  "老地方指公司地下车库；爸妈家指康平路88号。",
                  "2026-06-13T09:10:00+08:00",
                  [f"{cn2}-s07:001", f"{cn2}-s09:001", f"{cn2}-s09:002"],
                  abilities=["knowledge_update", "alias_resolution"]),
        _question(9, cn2, "multi-session",
                  "比较驾驶员李和副驾阿岚的空调温度偏好，两个人分别是多少度？",
                  "驾驶员李偏好21度，副驾阿岚偏好24度。",
                  "2026-06-13T09:15:00+08:00",
                  [f"{cn2}-s10:001", f"{cn2}-s11:001"],
                  abilities=["multi_speaker", "comparison"]),
        _question(10, cn2, "multi-session",
                  "驾驶员使用哪一家充电网络的会员卡付款？",
                  "历史中没有提到任何充电网络会员卡或付款方式，无法确定。",
                  "2026-06-13T09:20:00+08:00", abstention=True,
                  abilities=["abstention"]),
    ])

    en1 = "cockpit-hard-en-01"
    en1_sessions = [
        _session(en1, 1, "2026-07-01T18:00:00+08:00", [
            ("Driver Maya", "For restaurants, I usually prefer quiet Italian places with outdoor seating."),
            ("Car Assistant", "I will prioritize quiet Italian restaurants with outdoor seating."),
        ]),
        _session(en1, 2, "2026-07-02T08:30:00+08:00", [
            ("Driver Maya", "My doctor currently thinks I am lactose intolerant, so avoid dairy for me."),
            ("Car Assistant", "I will include a dairy-free constraint."),
        ]),
        _session(en1, 3, "2026-07-03T19:10:00+08:00", [
            ("Driver Maya", "Please find somewhere suitable for dinner."),
            ("Car Assistant", "Bella Notte is quiet and Italian but indoors. Garden Table is quiet, outdoors, and has dairy-free pasta."),
            ("Driver Maya", "Use Garden Table while I still need dairy-free food."),
            ("Car Assistant", "Garden Table selected."),
        ]),
        _session(en1, 4, "2026-07-05T09:00:00+08:00", [
            ("Driver Maya", "Read the tire-pressure status only; do not change anything."),
            ("Car Assistant", "All tires are within the recommended range."),
        ]),
        _session(en1, 5, "2026-07-08T12:00:00+08:00", [
            ("Driver Maya", "Update my food profile: the test was wrong and I am not lactose intolerant. Dairy is fine again. Keep my quiet Italian and outdoor-seating preferences."),
            ("Car Assistant", "Updated: dairy is allowed; quiet Italian restaurants with outdoor seating remain preferred."),
        ]),
        _session(en1, 6, "2026-07-09T07:45:00+08:00", [
            ("Driver Maya", "For my commute, avoid both tunnels and toll roads."),
            ("Car Assistant", "Commute routes will avoid tunnels and toll roads."),
        ]),
        _session(en1, 7, "2026-07-15T07:40:00+08:00", [
            ("Driver Maya", "Change the commute rule: choose the fastest route even through tunnels, but still avoid toll roads."),
            ("Car Assistant", "Updated: fastest route, tunnels allowed, toll roads still excluded."),
        ]),
        _session(en1, 8, "2026-07-16T20:00:00+08:00", [
            ("Driver Maya", "I visited Oak Street Charging Hub tonight."),
            ("Car Assistant", "The Oak Street charging visit is logged."),
        ]),
        _session(en1, 9, "2026-07-18T20:20:00+08:00", [
            ("Driver Maya", "I went back to Oak Street Charging Hub for another charge."),
            ("Car Assistant", "A second Oak Street visit is logged."),
        ]),
        _session(en1, 10, "2026-07-20T17:30:00+08:00", [
            ("Driver Maya", "I visited Pine Avenue Fast Charge today."),
            ("Car Assistant", "The Pine Avenue visit is logged."),
        ]),
        _session(en1, 11, "2026-07-21T10:00:00+08:00", [
            ("Driver Maya", "Book the tire inspection for July 28 at 9 a.m."),
            ("Car Assistant", "The tire inspection is scheduled for July 28 at 9 a.m."),
        ]),
        _session(en1, 12, "2026-07-24T14:00:00+08:00", [
            ("Driver Maya", "Move the July 28 tire inspection to July 30 at 2 p.m. and cancel the old slot."),
            ("Car Assistant", "The old slot is cancelled; the inspection is now July 30 at 2 p.m."),
        ]),
    ]
    conversations.append(_conversation(en1, en1_sessions))
    questions.extend([
        _question(11, en1, "knowledge-update",
                  "What is Maya's current restaurant profile after the medical update, including whether dairy is allowed?",
                  "Dairy is allowed again, while she still prefers quiet Italian restaurants with outdoor seating.",
                  "2026-07-25T09:00:00+08:00",
                  [f"{en1}-s01:001", f"{en1}-s02:001", f"{en1}-s05:001"],
                  abilities=["knowledge_update", "profile_persistence"]),
        _question(12, en1, "knowledge-update",
                  "What is Maya's latest commute rule, and how did the tunnel constraint change while the toll constraint stayed the same?",
                  "Use the fastest route and allow tunnels, but continue avoiding toll roads; tunnels changed from forbidden to allowed.",
                  "2026-07-25T09:05:00+08:00",
                  [f"{en1}-s06:001", f"{en1}-s07:001"],
                  abilities=["knowledge_update", "constraint_diff"]),
        _question(13, en1, "multi-session",
                  "Across Maya's recorded charging visits, which station was used most often and how many times?",
                  "Oak Street Charging Hub, twice.",
                  "2026-07-25T09:10:00+08:00",
                  [f"{en1}-s08:001", f"{en1}-s09:001", f"{en1}-s10:001"],
                  abilities=["multi_session_reasoning", "frequency_aggregation"]),
        _question(14, en1, "temporal-reasoning",
                  "As of July 25, when is Maya's tire inspection finally scheduled?",
                  "July 30 at 2 p.m.",
                  "2026-07-25T09:15:00+08:00",
                  [f"{en1}-s11:001", f"{en1}-s12:001", f"{en1}-s12:002"],
                  abilities=["temporal_reasoning", "reschedule", "latest_state"]),
        _question(15, en1, "multi-session",
                  "What seat position does Maya prefer for long trips?",
                  "The history never states Maya's preferred seat position, so it cannot be determined.",
                  "2026-07-25T09:20:00+08:00", abstention=True,
                  abilities=["abstention"]),
    ])

    en2 = "cockpit-hard-en-02"
    en2_sessions = [
        _session(en2, 1, "2026-08-01T18:00:00+08:00", [
            ("Driver Noah", "When I drive alone, I prefer jazz at volume 20."),
            ("Car Assistant", "Solo driving preference recorded: jazz at volume 20."),
        ]),
        _session(en2, 2, "2026-08-02T09:00:00+08:00", [
            ("Passenger Emma", "When I am in the car, please use children's audiobooks and never set the volume above 12."),
            ("Car Assistant", "Emma's profile is children's audiobooks with a maximum volume of 12."),
        ]),
        _session(en2, 3, "2026-08-03T08:00:00+08:00", [
            ("Driver Noah", "My normal work destination is Harbor Lab at 5 Dock Road."),
            ("Car Assistant", "Work is set to Harbor Lab, 5 Dock Road."),
        ]),
        _session(en2, 4, "2026-08-12T17:00:00+08:00", [
            ("Driver Noah", "Play the latest sports headlines."),
            ("Car Assistant", "Playing the latest sports headlines."),
        ]),
        _session(en2, 5, "2026-08-31T16:00:00+08:00", [
            ("Driver Noah", "From September 1 through September 14, my temporary project means 'work' should route to Riverside Office at 20 River Street."),
            ("Car Assistant", "For September 1-14, work will resolve to Riverside Office, 20 River Street."),
        ]),
        _session(en2, 6, "2026-09-15T17:00:00+08:00", [
            ("Driver Noah", "The temporary project has finished. Change 'work' back to Harbor Lab at 5 Dock Road."),
            ("Car Assistant", "Work now resolves to Harbor Lab again."),
        ]),
        _session(en2, 7, "2026-09-16T19:00:00+08:00", [
            ("Driver Noah", "I prefer Japanese food, but I never eat raw fish."),
            ("Car Assistant", "Japanese food is preferred, with raw fish excluded."),
        ]),
        _session(en2, 8, "2026-10-01T11:00:00+08:00", [
            ("Driver Noah", "For October I am trying vegan food. Keep the Japanese preference and the no-raw-fish rule."),
            ("Car Assistant", "October profile: vegan Japanese food, no raw fish."),
        ]),
        _session(en2, 9, "2026-10-02T18:30:00+08:00", [
            ("Driver Noah", "Find dinner that fits my current constraints."),
            ("Car Assistant", "Sakura Sushi specializes in raw fish. Miso Garden serves vegan Japanese ramen with no raw fish."),
            ("Driver Noah", "Choose Miso Garden."),
            ("Car Assistant", "Miso Garden selected."),
        ]),
        _session(en2, 10, "2026-10-03T10:00:00+08:00", [
            ("Driver Noah", "We moved. Home is now 90 Cedar Lane instead of 14 Pine Road."),
            ("Car Assistant", "Home address updated to 90 Cedar Lane; 14 Pine Road is no longer current."),
        ]),
        _session(en2, 11, "2026-10-04T15:00:00+08:00", [
            ("Driver Noah", "Show the remaining battery range without starting a route."),
            ("Car Assistant", "Estimated remaining range is 186 kilometers."),
        ]),
    ]
    conversations.append(_conversation(en2, en2_sessions))
    questions.extend([
        _question(16, en2, "multi-session",
                  "Compare the audio policy when Noah drives alone with the policy when Emma is in the car.",
                  "Alone, Noah prefers jazz at volume 20; with Emma, use children's audiobooks and keep volume at or below 12.",
                  "2026-10-05T09:00:00+08:00",
                  [f"{en2}-s01:001", f"{en2}-s02:001"],
                  abilities=["multi_speaker", "conditional_preference", "comparison"]),
        _question(17, en2, "temporal-reasoning",
                  "Where should the alias 'work' resolve on September 10 and on September 16, respectively?",
                  "On September 10 it should resolve to Riverside Office at 20 River Street; on September 16 it should resolve to Harbor Lab at 5 Dock Road.",
                  "2026-10-05T09:05:00+08:00",
                  [f"{en2}-s03:001", f"{en2}-s05:001", f"{en2}-s06:001"],
                  abilities=["temporal_reasoning", "interval_state", "knowledge_update"]),
        _question(18, en2, "single-session-preference",
                  "Which restaurant best matches Noah's current October food constraints, and why?",
                  "Miso Garden, because it offers vegan Japanese food without raw fish.",
                  "2026-10-05T09:10:00+08:00",
                  [f"{en2}-s07:001", f"{en2}-s08:001", f"{en2}-s09:002", f"{en2}-s09:003"],
                  abilities=["constraint_intersection", "personalized_recommendation"]),
        _question(19, en2, "knowledge-update",
                  "What is Noah's latest home address, and which previous address did it replace?",
                  "His latest home address is 90 Cedar Lane, replacing 14 Pine Road.",
                  "2026-10-05T09:15:00+08:00",
                  [f"{en2}-s10:001", f"{en2}-s10:002"],
                  abilities=["knowledge_update", "supersession"]),
        _question(20, en2, "multi-session",
                  "Which school does Emma attend?",
                  "The history does not identify Emma's school, so it cannot be determined.",
                  "2026-10-05T09:20:00+08:00", abstention=True,
                  abilities=["abstention"]),
    ])

    evidence_ids = {
        value for question in questions for value in question["answer_session_ids"]
    }
    for conversation in conversations:
        for session in conversation["sessions"]:
            source_id = session["source_session_id"]
            for index, message in enumerate(session["messages"], 1):
                message["has_answer"] = f"{source_id}:{index:03d}" in evidence_ids
    return conversations, questions


def _conversation(conversation_id: str, sessions: list[dict]) -> dict:
    return {
        "sample_id": conversation_id,
        "source_question_id": None,
        "session_count": len(sessions),
        "message_count": sum(len(session["messages"]) for session in sessions),
        "sessions": sessions,
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the original 20-question cockpit long-memory challenge."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    conversations, questions = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "conversations.jsonl", conversations)
    _write_jsonl(args.output_dir / "questions.jsonl", questions)
    manifest = {
        "name": "Cockpit Long Memory Challenge 20",
        "version": VERSION,
        "created": "2026-08-26",
        "language": ["zh-CN", "en"],
        "conversation_count": len(conversations),
        "session_count": sum(row["session_count"] for row in conversations),
        "message_count": sum(row["message_count"] for row in conversations),
        "question_count": len(questions),
        "original_synthetic": True,
        "copied_source_utterances": False,
        "design_inspirations": [
            {
                "name": "CarMem",
                "url": "https://github.com/johanneskirmayr/CarMem",
                "use": "in-car preference, maintenance, and retrieval taxonomy",
            },
            {
                "name": "LongMemEval",
                "url": "https://github.com/xiaowu0162/LongMemEval",
                "use": "multi-session, update, temporal, preference, and abstention taxonomy",
            },
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selection = {
        "schema_version": 1,
        "protocol_id": VERSION,
        "dataset_id": "longmemeval",
        "selection": "all_original_hard_cockpit_questions",
        "seed": "curated-v1",
        "question_ids": [row["qa_id"] for row in questions],
        "conversation_ids": [row["sample_id"] for row in conversations],
        "counts": {
            "questions": len(questions),
            "conversations": len(conversations),
            "sessions": manifest["session_count"],
            "messages": manifest["message_count"],
        },
        "leakage_controls": {
            "answers_read_for_selection": False,
            "evidence_ids_read_for_selection": False,
        },
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
