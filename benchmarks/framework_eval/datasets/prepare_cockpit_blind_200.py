#!/usr/bin/env python3
"""Build a sealed 200-question cockpit-memory holdout.

The corpus is deterministic but parameterized across identities, entities,
dates, values, language, and distractors.  It is generated before the
response-contract validator is implemented and must not be regenerated after
the first end-to-end score.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "challenges" / "cockpit_blind_200_v1"
SEED = 2026082701

ZH_NAMES = ["林舟", "周宁", "顾言", "沈清", "苏禾", "陆遥", "江岚", "程野"]
ZH_PASSENGERS = ["安安", "小满", "乐乐", "思思", "可可", "晨晨"]
EN_NAMES = ["Avery", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Quinn", "Parker"]
EN_PASSENGERS = ["Nora", "Eli", "Iris", "Leo", "Zoe", "Milo"]
ZH_STATIONS = ["云桥超充站", "星湾充电站", "河畔快充站", "松林补能站", "港城超充站"]
EN_STATIONS = ["Maple Charging Hub", "Cedar Fast Charge", "Riverside Charging Hub", "Harbor Charge Point", "Pine Charging Hub"]
ZH_GARAGES = ["公司东区地库", "研发园二号车库", "滨江办公楼地库", "总部北门车库"]
EN_GARAGES = ["East Office Garage", "Research Park Garage 2", "Riverside Office Garage", "North HQ Garage"]


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def add_session(sessions: list[dict], sample: str, when: str, messages: list[tuple[str, str, bool]]) -> tuple[str, list[str]]:
    index = len(sessions) + 1
    source_id = f"{sample}-s{index:02d}"
    sessions.append({
        "session_index": index,
        "source_session_id": source_id,
        "date_time": when,
        "messages": [
            {"role": role, "content": content, "has_answer": has_answer}
            for role, content, has_answer in messages
        ],
    })
    return source_id, [f"{source_id}:{message_index:03d}" for message_index in range(1, len(messages) + 1)]


def question(
    *, pack: int, slot: int, sample: str, category: str, text: str, answer: str,
    question_date: str, evidence: list[str], ability: str, abstention: bool = False,
) -> dict:
    return {
        "qa_id": f"cockpit-blind#p{pack:02d}-q{slot:02d}",
        "sample_id": sample,
        "category": category,
        "question_type": category,
        "question": text,
        "answer": [answer],
        "question_date": question_date,
        "answer_session_ids": evidence,
        "is_abstention": abstention,
        "metadata": {
            "timezone": "Asia/Shanghai",
            "ability": ability,
            "holdout": True,
            "seed": SEED,
            "pack": pack,
        },
    }


def build_pack(pack: int, rng: random.Random) -> tuple[dict, list[dict]]:
    zh = pack % 2 == 1
    sample = f"cockpit-blind-{'zh' if zh else 'en'}-{pack:02d}"
    driver = (ZH_NAMES if zh else EN_NAMES)[pack % 8]
    passenger = (ZH_PASSENGERS if zh else EN_PASSENGERS)[pack % 6]
    station_a = (ZH_STATIONS if zh else EN_STATIONS)[pack % 5]
    station_b = (ZH_STATIONS if zh else EN_STATIONS)[(pack + 2) % 5]
    garage = (ZH_GARAGES if zh else EN_GARAGES)[pack % 4]
    parent_address = f"桂平路{70 + pack}号" if zh else f"{120 + pack} Lake Avenue"
    temp_office = f"江湾项目部{pack}号楼" if zh else f"River Project Office {pack}"
    normal_office = f"港湾实验室{pack}号" if zh else f"Harbor Lab {pack}"
    driver_temp = 20 + pack % 3
    passenger_temp = 23 + pack % 3
    old_day = 26 + pack % 2
    new_day = 29 + pack % 2
    low_threshold = 12 + pack % 5
    low_now = low_threshold - 3
    sessions: list[dict] = []
    questions: list[dict] = []

    # Distractors deliberately share names and generic update language.
    add_session(sessions, sample, "2026-08-01T09:00:00+08:00", [
        ("user", f"[{driver}] " + ("把后备箱整理提醒设在周末。" if zh else "Remind me this weekend to organize the trunk."), False),
        ("assistant", "提醒已记录。" if zh else "Reminder recorded.", False),
    ])

    # 1. Three-event aggregation: A, B, A.
    agg_ids: list[str] = []
    for day, station in [(2, station_a), (4, station_b), (6, station_a)]:
        _, ids = add_session(sessions, sample, f"2026-08-{day:02d}T18:00:00+08:00", [
            ("user", f"[{driver}] " + (f"今天在{station}完成了充电。" if zh else f"I charged at {station} today."), True),
            ("assistant", (f"已记录在{station}的充电事件。" if zh else f"Recorded the charging visit at {station}."), False),
        ])
        agg_ids.append(ids[0])
    questions.append(question(
        pack=pack, slot=1, sample=sample, category="multi-session", ability="aggregation-frequency",
        text=("这三个充电事件中，最常去的是哪个站，一共几次？" if zh else "Across these three recorded charging visits, which station was used most often, and how many times?"),
        answer=(f"{station_a}，共2次。" if zh else f"{station_a}, twice."),
        question_date="2026-08-08T09:00:00+08:00", evidence=agg_ids,
    ))

    # 2. Latest route policy with one changed and one retained constraint.
    _, route_old = add_session(sessions, sample, "2026-08-07T07:20:00+08:00", [
        ("user", f"[{driver}] " + ("工作日早上避开高架和收费道路。" if zh else "On weekday mornings, avoid highways and toll roads."), True),
        ("assistant", "已记录早高峰路线规则。" if zh else "The morning route rule is recorded.", False),
    ])
    _, route_new = add_session(sessions, sample, "2026-08-09T07:15:00+08:00", [
        ("user", f"[{driver}] " + ("更新通勤规则：选最快路线，高架可以走，但收费道路仍要避开。" if zh else "Update my commute: take the fastest route, highways are allowed now, but still avoid toll roads."), True),
        ("assistant", "规则已更新。" if zh else "The commute rule has been updated.", False),
    ])
    questions.append(question(
        pack=pack, slot=2, sample=sample, category="knowledge-update", ability="latest-final-update",
        text=("截至8月10日，最新通勤规则是什么？高架和收费道路分别怎么处理？" if zh else "As of August 10, what is the latest commute rule for highways and toll roads?"),
        answer=("选择最快路线，允许走高架，但继续避开收费道路。" if zh else "Use the fastest route and allow highways, but continue avoiding toll roads."),
        question_date="2026-08-10T08:00:00+08:00", evidence=[route_old[0], route_new[0]],
    ))

    # 3. Two-date alias validity with an explicit restoration boundary.
    _, alias_base = add_session(sessions, sample, "2026-08-11T10:00:00+08:00", [
        ("user", f"[{driver}] " + (f"平时‘工作地’指{normal_office}。" if zh else f"Normally, the alias 'work site' resolves to {normal_office}."), True),
        ("assistant", "基础别名已记录。" if zh else "The baseline alias is recorded.", False),
    ])
    _, alias_temp = add_session(sessions, sample, "2026-08-12T10:00:00+08:00", [
        ("user", f"[{driver}] " + (f"8月15日至8月21日，‘工作地’临时改为{temp_office}。" if zh else f"From August 15 through August 21, temporarily resolve 'work site' to {temp_office}."), True),
        ("assistant", "临时有效期已记录。" if zh else "The temporary validity interval is recorded.", False),
    ])
    _, alias_restore = add_session(sessions, sample, "2026-08-22T08:00:00+08:00", [
        ("user", f"[{driver}] " + (f"临时项目结束，从今天起‘工作地’恢复为{normal_office}。" if zh else f"The temporary project is over; starting today, restore 'work site' to {normal_office}."), True),
        ("assistant", "别名已恢复。" if zh else "The alias has been restored.", False),
    ])
    questions.append(question(
        pack=pack, slot=3, sample=sample, category="temporal-reasoning", ability="two-date-validity",
        text=("8月18日和8月23日，‘工作地’分别应该解析到哪里？" if zh else "Where should 'work site' resolve on August 18 and on August 23, respectively?"),
        answer=(f"8月18日是{temp_office}；8月23日是{normal_office}。" if zh else f"On August 18 it is {temp_office}; on August 23 it is {normal_office}."),
        question_date="2026-08-24T09:00:00+08:00", evidence=[alias_base[0], alias_temp[0], alias_restore[0]],
    ))

    # 4. Cross-session owner/value comparison.
    _, owner_a = add_session(sessions, sample, "2026-08-13T12:00:00+08:00", [
        ("user", f"[{driver}] " + (f"我开车时空调偏好{driver_temp}度。" if zh else f"When I drive, I prefer the cabin at {driver_temp} degrees."), True),
        ("assistant", "驾驶员温度已记录。" if zh else "The driver's temperature is recorded.", False),
    ])
    _, owner_b = add_session(sessions, sample, "2026-08-14T12:00:00+08:00", [
        ("user", f"[{passenger}] " + (f"我坐副驾时喜欢{passenger_temp}度。" if zh else f"When I am the passenger, I prefer {passenger_temp} degrees."), True),
        ("assistant", "乘客温度已记录。" if zh else "The passenger's temperature is recorded.", False),
    ])
    questions.append(question(
        pack=pack, slot=4, sample=sample, category="multi-session", ability="multi-person-cross-session",
        text=(f"比较驾驶员{driver}和副驾{passenger}在不同会话里的空调偏好，两个人分别是多少度？" if zh else f"Compare {driver}'s and passenger {passenger}'s climate preferences from different sessions. What temperature does each prefer?"),
        answer=(f"驾驶员{driver}偏好{driver_temp}度，副驾{passenger}偏好{passenger_temp}度。" if zh else f"{driver} prefers {driver_temp} degrees and {passenger} prefers {passenger_temp} degrees."),
        question_date="2026-08-24T09:10:00+08:00", evidence=[owner_a[0], owner_b[0]],
    ))

    # 5. Appointment correction and cancellation.
    _, appt_old = add_session(sessions, sample, "2026-08-16T09:00:00+08:00", [
        ("user", f"[{driver}] " + (f"轮胎检查先预约在8月{old_day}日上午9点。" if zh else f"Book the tire inspection for August {old_day} at 9 a.m."), True),
        ("assistant", "原预约已登记。" if zh else "The original appointment is booked.", False),
    ])
    _, appt_new = add_session(sessions, sample, "2026-08-17T11:00:00+08:00", [
        ("user", f"[{driver}] " + (f"取消8月{old_day}日的检查，最终改到8月{new_day}日下午3点。" if zh else f"Cancel the August {old_day} inspection and finally move it to August {new_day} at 3 p.m."), True),
        ("assistant", "旧预约已取消，新时间已确认。" if zh else "The old slot is cancelled and the new time is confirmed.", True),
    ])
    questions.append(question(
        pack=pack, slot=5, sample=sample, category="knowledge-update", ability="update-cancel-negation",
        text=("改期以后，轮胎检查最终安排在什么时候？" if zh else "After the correction, when is the tire inspection finally scheduled?"),
        answer=(f"最终为8月{new_day}日下午3点；8月{old_day}日上午9点的旧预约已取消。" if zh else f"It is finally August {new_day} at 3 p.m.; the August {old_day} 9 a.m. slot was cancelled."),
        question_date="2026-08-25T10:00:00+08:00", evidence=[appt_old[0], appt_new[0], appt_new[1]],
    ))

    # 6. Conditional priority replacement.
    _, priority_ids = add_session(sessions, sample, "2026-08-18T08:00:00+08:00", [
        ("user", f"[{driver}] " + (f"通常充电站优先有顶棚和卫生间；但电量低于{low_threshold}%时，改为距离优先。" if zh else f"Normally prioritize charging stations with a canopy and restroom; when battery is below {low_threshold}%, distance becomes the priority instead."), True),
        ("assistant", "条件优先规则已记录。" if zh else "The conditional priority rule is recorded.", False),
    ])
    questions.append(question(
        pack=pack, slot=6, sample=sample, category="single-session-preference", ability="conditional-priority",
        text=(f"现在电量只有{low_now}%，选充电站时什么优先？顶棚和卫生间是否仍然是首要条件？" if zh else f"The battery is at {low_now}%. What is primary when choosing a charger, and are canopy and restroom still primary?"),
        answer=("距离优先；顶棚和卫生间不再是首要条件。" if zh else "Distance is primary; canopy and restroom are no longer primary under this condition."),
        question_date="2026-08-25T10:10:00+08:00", evidence=[priority_ids[0]],
    ))

    # 7. Field-scoped abstention. The person has history but not this field.
    absent_kind = pack % 4
    if zh:
        absent_questions = [
            f"副驾{passenger}就读哪所学校？",
            f"驾驶员{driver}最常听的播客节目是什么？",
            f"驾驶员{driver}用哪家充电网络的会员卡付款？",
            f"驾驶员{driver}长途出行喜欢什么座椅位置？",
        ]
        absent_answers = [
            f"历史没有说明{passenger}就读的学校，无法确定。",
            f"历史没有说明驾驶员{driver}常听的播客，无法确定。",
            f"历史没有说明驾驶员{driver}的充电会员卡或付款网络，无法确定。",
            f"历史没有说明驾驶员{driver}的长途座椅位置，无法确定。",
        ]
    else:
        absent_questions = [
            f"Which school does {passenger} attend?",
            f"Which podcast does {driver} listen to most often?",
            f"Which charging network membership card does {driver} use for payment?",
            f"What seat position does {driver} prefer on long trips?",
        ]
        absent_answers = [
            f"The history does not identify {passenger}'s school, so it cannot be determined.",
            f"The history does not identify {driver}'s usual podcast, so it cannot be determined.",
            f"The history does not identify a charging membership card or payment network for {driver}.",
            f"The history does not identify {driver}'s preferred long-trip seat position.",
        ]
    questions.append(question(
        pack=pack, slot=7, sample=sample, category="multi-session", ability="insufficient-evidence-abstention",
        text=absent_questions[absent_kind], answer=absent_answers[absent_kind],
        question_date="2026-08-25T10:20:00+08:00", evidence=[], abstention=True,
    ))

    # 8. Two aliases updated together; every target must be returned.
    _, aliases_old = add_session(sessions, sample, "2026-08-19T08:00:00+08:00", [
        ("user", f"[{driver}] " + (f"先把{parent_address}叫作‘常去的地方’。" if zh else f"For now, call {parent_address} 'usual place'."), True),
        ("assistant", "旧别名已记录。" if zh else "The old alias is recorded.", False),
    ])
    _, aliases_new = add_session(sessions, sample, "2026-08-20T08:00:00+08:00", [
        ("user", f"[{driver}] " + (f"更新别名：以后‘常去的地方’指{garage}；{parent_address}只叫‘家人住址’。" if zh else f"Update the aliases: from now on, 'usual place' means {garage}, while '{parent_address}' is only 'family home'."), True),
        ("assistant", (f"已更新：常去的地方是{garage}，家人住址是{parent_address}。" if zh else f"Updated: 'usual place' is {garage}, and 'family home' is {parent_address}."), True),
    ])
    questions.append(question(
        pack=pack, slot=8, sample=sample, category="knowledge-update", ability="multi-target-final-state",
        text=("目前‘常去的地方’和‘家人住址’分别指哪里？" if zh else "What do 'usual place' and 'family home' currently resolve to, respectively?"),
        answer=(f"常去的地方是{garage}；家人住址是{parent_address}。" if zh else f"'Usual place' is {garage}; 'family home' is {parent_address}."),
        question_date="2026-08-25T10:30:00+08:00", evidence=[aliases_old[0], aliases_new[0], aliases_new[1]],
    ))

    # Shuffle only distractor wording selection, never event order.
    if rng.random() < 0.5:
        add_session(sessions, sample, "2026-08-21T16:00:00+08:00", [
            ("user", f"[{driver}] " + ("今天不调整音量。" if zh else "Do not change the audio volume today."), False),
            ("assistant", "未调整音量。" if zh else "The volume was not changed.", False),
        ])

    conversation = {
        "sample_id": sample,
        "source_question_id": None,
        "session_count": len(sessions),
        "message_count": sum(len(item["messages"]) for item in sessions),
        "sessions": sessions,
    }
    return conversation, questions


def validate(conversations: list[dict], questions: list[dict]) -> None:
    assert len(conversations) == 25
    assert len(questions) == 200
    assert len({row["qa_id"] for row in questions}) == 200
    # Repeated query shapes are intentional: the answer must depend on each
    # conversation's facts rather than memorizing a surface-form template.
    assert len({(row["sample_id"], row["question"]) for row in questions}) == 200
    assert sum(bool(row["is_abstention"]) for row in questions) == 25
    abilities = Counter(row["metadata"]["ability"] for row in questions)
    assert abilities == {
        "aggregation-frequency": 25,
        "latest-final-update": 25,
        "two-date-validity": 25,
        "multi-person-cross-session": 25,
        "update-cancel-negation": 25,
        "conditional-priority": 25,
        "insufficient-evidence-abstention": 25,
        "multi-target-final-state": 25,
    }
    ids = {
        f"{session['source_session_id']}:{index:03d}"
        for conversation in conversations
        for session in conversation["sessions"]
        for index, _ in enumerate(session["messages"], 1)
    }
    assert all(evidence in ids for row in questions for evidence in row["answer_session_ids"])


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite sealed dataset: {OUTPUT}")
    rng = random.Random(SEED)
    conversations: list[dict] = []
    questions: list[dict] = []
    for pack in range(1, 26):
        conversation, pack_questions = build_pack(pack, rng)
        conversations.append(conversation)
        questions.extend(pack_questions)
    validate(conversations, questions)
    OUTPUT.mkdir(parents=True)
    dump_jsonl(OUTPUT / "conversations.jsonl", conversations)
    dump_jsonl(OUTPUT / "questions.jsonl", questions)
    distribution = Counter(row["metadata"]["ability"] for row in questions)
    manifest = {
        "dataset_id": "cockpit-blind-200-v1",
        "seed": SEED,
        "conversation_count": len(conversations),
        "question_count": len(questions),
        "abstention_count": sum(bool(row["is_abstention"]) for row in questions),
        "languages": {"zh": 13 * 8, "en": 12 * 8},
        "ability_distribution": dict(sorted(distribution.items())),
        "protocol": "seal before validator implementation; exactly one end-to-end score; no post-score fixes",
    }
    dump(OUTPUT / "manifest.json", manifest)
    (OUTPUT / "README.md").write_text(
        "# Cockpit Blind 200 v1\n\n"
        "A deterministic, sealed text-transcript holdout for cockpit long-memory evaluation. "
        "ASR quality is intentionally out of scope. The set contains 25 packs and eight abilities per pack. "
        "Do not regenerate or inspect individual answers after scoring begins.\n",
        encoding="utf-8",
    )
    checksums = []
    for name in ("README.md", "conversations.jsonl", "manifest.json", "questions.jsonl"):
        digest = hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
        checksums.append(f"{digest}  {name}")
    (OUTPUT / "BLIND_SEAL.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
