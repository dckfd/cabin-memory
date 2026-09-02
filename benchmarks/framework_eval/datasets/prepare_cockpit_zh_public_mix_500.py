#!/usr/bin/env python3
"""Build a sealed Chinese cockpit-memory holdout from pinned public corpora.

The public corpora provide new Chinese utterances, entities, dialogue-state
changes and user preferences.  This compiler adds only the long-memory event
relations needed by the benchmark (time, update, cancellation, ownership and
abstention).  Every public value keeps source-level lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SEED = 2026082901
DATASET_ID = "cockpit-zh-public-mix-500-v4"
PERMISSIVE_SUBSET_ID = "cockpit-zh-public-crosswoz-250-v4"
QUESTION_PREAMBLE = ""
TZ = "+08:00"

CROSSWOZ_COMMIT = "df82c9fdff91b9b130f2d6b89110d3870ba6260e"
DURECDIAL_COMMIT = "1309cc072afd0b832e899e49c62906e700ff6acf"
RISAWOZ_COMMIT = "1e2d3be2b25a0540519db1664a172d85ddd62b29"

ABILITIES = (
    "aggregation-frequency",
    "latest-final-update",
    "two-date-validity",
    "multi-person-cross-session",
    "final-cancellation",
    "conditional-priority",
    "insufficient-evidence-abstention",
    "multi-target-final-state",
    "cutoff-state",
    "correction-retained-constraint",
)

SURFACE_STYLES = (
    "clean",
    "no-punctuation",
    "colloquial-filler",
    "elliptical-reference",
    "homophone-text-noise",
)

CATEGORIES = {
    "aggregation-frequency": "multi-session",
    "latest-final-update": "knowledge-update",
    "two-date-validity": "temporal-reasoning",
    "multi-person-cross-session": "multi-session",
    "final-cancellation": "knowledge-update",
    "conditional-priority": "single-session-preference",
    "insufficient-evidence-abstention": "multi-session",
    "multi-target-final-state": "knowledge-update",
    "cutoff-state": "temporal-reasoning",
    "correction-retained-constraint": "knowledge-update",
}

POSITIVE_PROFILE_KEYS = (
    "喜欢 的 电影",
    "喜欢 的 音乐",
    "接受 的 电影",
    "接受 的 音乐",
    "同意 的 电影",
    "同意 的 音乐",
)

CHANGE_SLOTS = {
    "名称", "推荐菜", "评分", "人均消费", "价格", "酒店类型",
    "酒店设施-健身房", "门票", "游玩时间", "日期", "城市", "目的地",
    "出发地", "坐席", "舱位档次", "价位", "菜系", "最适合人群",
    "年代", "类型", "车系", "系列", "主演", "车型", "片名",
}

CROSS_LOCATION_DOMAINS = {"餐馆", "景点", "酒店"}
RISAWOZ_LOCATION_DOMAINS = {"餐厅", "旅游景点", "酒店", "医院"}
RISAWOZ_MEDIA_DOMAINS = {"电影", "电视剧"}
LOCATION_SLOTS = {"名称", "地址"}
MEDIA_SLOTS = {"名称", "片名"}


@dataclass(frozen=True)
class SourceRef:
    ref_id: str
    dataset: str
    split: str
    record_id: str
    turn_id: str
    field: str
    value: str
    excerpt: str
    license_id: str
    license_class: str

    def payload(self) -> dict:
        return {
            "ref_id": self.ref_id,
            "dataset": self.dataset,
            "split": self.split,
            "record_id": self.record_id,
            "turn_id": self.turn_id,
            "field": self.field,
            "value": self.value,
            "source_excerpt": self.excerpt,
            "license": self.license_id,
            "license_class": self.license_class,
        }


@dataclass(frozen=True)
class SourceFact:
    value: str
    domain: str
    slot: str
    utterance: str
    response: str
    ref: SourceRef


@dataclass(frozen=True)
class ChangeAnchor:
    dataset: str
    record_id: str
    old: SourceFact
    new: SourceFact


@dataclass(frozen=True)
class PersonFact:
    name: str
    preference: str
    preference_kind: str
    ref: SourceRef


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(*parts: object) -> str:
    raw = "\u241f".join(str(part) for part in parts)
    return hashlib.sha256(f"{SEED}\u241e{raw}".encode("utf-8")).hexdigest()


def normalized(text: object) -> str:
    return "".join(re.findall(r"[0-9a-z\u3400-\u9fff]+", str(text).lower()))


def clean_value(value: object) -> str:
    text = str(value).strip()
    if re.search(r"[\u3400-\u9fff]", text):
        text = re.sub(r"\s+", "", text)
    else:
        text = re.sub(r"\s+", " ", text)
    text = text.replace("-–", "-").replace("--", "-")
    return text.strip(" ，。！？；：、")


def valid_value(value: str) -> bool:
    if not value or value.lower() in {"none", "null", "unknown"}:
        return False
    if value in {"无", "是", "否", "不限", "都可以"}:
        return False
    if "id=" in value or len(value) > 48:
        return False
    return bool(re.search(r"[0-9a-z\u3400-\u9fff]", value.lower()))


def load_exclusions(
    roots: Sequence[Path],
) -> tuple[set[tuple[str, str]], set[str], set[tuple[str, str]], list[dict]]:
    """Load only source identities from earlier sealed sets.

    Excluding prior anchor dialogues and exact source references makes a later
    fact graph independent of earlier benchmark versions without consulting
    their questions or gold answers. Normalized values are also collected so
    callers can request the stricter policy when the public corpus has enough
    distinct values.
    """
    anchor_records: set[tuple[str, str]] = set()
    ref_ids: set[str] = set()
    values: set[tuple[str, str]] = set()
    receipts: list[dict] = []
    for root in roots:
        refs_path = root / "source_refs.jsonl"
        conversations_path = root / "conversations.jsonl"
        if not refs_path.is_file() or not conversations_path.is_file():
            raise SystemExit(f"missing exclusion source identities under: {root}")
        for line in conversations_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metadata = json.loads(line)["metadata"]
            anchor_records.add((
                str(metadata["public_anchor_dataset"]),
                str(metadata["public_anchor_record_id"]),
            ))
        row_count = 0
        for line in refs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = str(row["dataset"])
            ref_ids.add(str(row["ref_id"]))
            value = normalized(row.get("value") or "")
            if value:
                values.add((dataset, value))
            row_count += 1
        receipts.append({
            "dataset_root": str(root.resolve()),
            "source_refs_sha256": sha256(refs_path),
            "conversations_sha256": sha256(conversations_path),
            "source_ref_count": row_count,
        })
    return anchor_records, ref_ids, values, receipts


def load_anchor_exclusions(
    roots: Sequence[Path],
) -> tuple[set[tuple[str, str]], list[dict]]:
    """Exclude prior anchor dialogues without exhausting reusable public facts.

    This is the weaker, explicitly audited isolation lane for a public corpus
    that no longer contains enough unused entity facts to build another full
    benchmark.  It still guarantees new anchor dialogues; callers must exclude
    prior question files separately and must not claim source-fact disjointness.
    """
    anchor_records: set[tuple[str, str]] = set()
    receipts: list[dict] = []
    for root in roots:
        conversations_path = root / "conversations.jsonl"
        if not conversations_path.is_file():
            raise SystemExit(f"missing anchor exclusion conversations under: {root}")
        row_count = 0
        for line in conversations_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metadata = json.loads(line)["metadata"]
            anchor_records.add((
                str(metadata["public_anchor_dataset"]),
                str(metadata["public_anchor_record_id"]),
            ))
            row_count += 1
        receipts.append({
            "dataset_root": str(root.resolve()),
            "conversations_sha256": sha256(conversations_path),
            "conversation_row_count": row_count,
            "policy": "anchor-dialogue-only; public entity facts may repeat",
        })
    return anchor_records, receipts


def source_fact_is_new(
    ref: SourceRef, value: str,
    excluded_ref_ids: set[str],
    excluded_values: set[tuple[str, str]],
) -> bool:
    return (
        ref.ref_id not in excluded_ref_ids
        and (ref.dataset, normalized(value)) not in excluded_values
    )


def load_prior_questions(paths: Sequence[Path]) -> tuple[set[str], list[dict]]:
    questions: set[str] = set()
    receipts: list[dict] = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing prior question file: {path}")
        row_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            question = normalized(json.loads(line).get("question") or "")
            if question:
                questions.add(question)
            row_count += 1
        receipts.append({
            "questions_path": str(path.resolve()),
            "questions_sha256": sha256(path),
            "question_row_count": row_count,
        })
    return questions, receipts


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: Iterable[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def zip_json(path: Path, member: str) -> object:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(member))


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_ref(
    *, dataset: str, split: str, record_id: str, turn_id: str, act_id: int,
    domain: str, slot: str, value: str, excerpt: str,
) -> SourceRef:
    if dataset == "CrossWOZ":
        license_id = "Apache-2.0 (repository declaration)"
        license_class = "permissive-public-source"
    elif dataset == "RiSAWOZ":
        license_id = "CC-BY-NC-4.0"
        license_class = "noncommercial-public-source"
    else:
        license_id = "CC-BY-NC-SA-4.0"
        license_class = "noncommercial-sharealike-public-source"
    key = f"{dataset}:{split}:{record_id}:{turn_id}:{act_id}:{domain}:{slot}:{value}"
    return SourceRef(
        ref_id=f"src-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}",
        dataset=dataset,
        split=split,
        record_id=str(record_id),
        turn_id=str(turn_id),
        field=f"{domain}.{slot}" if domain else slot,
        value=value,
        excerpt=excerpt,
        license_id=license_id,
        license_class=license_class,
    )


def record_unambiguous_turn_changes(
    *, dataset: str, record_id: str,
    facts_by_key: dict[tuple[str, str], list[SourceFact]],
    seen: dict[tuple[str, str], SourceFact], changes: list[ChangeAnchor],
) -> None:
    """Treat a turn as a state observation only when its field has one value.

    Public task-oriented corpora can annotate multiple alternatives in one
    utterance with the same domain and slot.  Such co-mentions are not temporal
    updates and must never become an old/new event pair.
    """
    for key in sorted(facts_by_key):
        by_value: dict[str, SourceFact] = {}
        for fact in facts_by_key[key]:
            by_value.setdefault(normalized(fact.value), fact)
        if len(by_value) != 1:
            continue
        fact = next(iter(by_value.values()))
        previous = seen.get(key)
        if (
            previous
            and previous.value != fact.value
            and previous.ref.turn_id != fact.ref.turn_id
        ):
            changes.append(ChangeAnchor(dataset, record_id, previous, fact))
        seen[key] = fact


def extract_crosswoz(path: Path) -> tuple[list[ChangeAnchor], list[SourceFact]]:
    raw = zip_json(path, "test.json")
    assert isinstance(raw, dict)
    anchors: list[ChangeAnchor] = []
    entities: list[SourceFact] = []
    for record_id in sorted(raw, key=lambda value: int(value)):
        row = raw[record_id]
        messages = row["messages"]
        seen: dict[tuple[str, str], SourceFact] = {}
        changes: list[ChangeAnchor] = []
        for message_index, message in enumerate(messages):
            utterance = str(message.get("content") or "").strip()
            role = str(message.get("role") or "")
            turn_change_facts: dict[tuple[str, str], list[SourceFact]] = {}
            response = ""
            if role == "usr" and message_index + 1 < len(messages):
                following = messages[message_index + 1]
                if following.get("role") == "sys":
                    response = str(following.get("content") or "").strip()
            for act_index, action in enumerate(message.get("dialog_act") or []):
                if len(action) < 4:
                    continue
                intent, domain, slot, raw_value = map(str, action[:4])
                value = clean_value(raw_value)
                if not valid_value(value) or normalized(value) not in normalized(utterance):
                    continue
                ref = source_ref(
                    dataset="CrossWOZ", split="test", record_id=str(record_id),
                    turn_id=f"message-{message_index}", act_id=act_index,
                    domain=domain, slot=slot, value=value, excerpt=utterance,
                )
                fact = SourceFact(value, domain, slot, utterance, response, ref)
                if domain in CROSS_LOCATION_DOMAINS and slot in LOCATION_SLOTS:
                    entities.append(fact)
                if role != "usr" or intent != "Inform" or slot not in CHANGE_SLOTS:
                    continue
                key = (domain, slot)
                turn_change_facts.setdefault(key, []).append(fact)
            record_unambiguous_turn_changes(
                dataset="CrossWOZ", record_id=str(record_id),
                facts_by_key=turn_change_facts, seen=seen, changes=changes,
            )
        if changes:
            anchors.append(min(changes, key=lambda item: rank(
                item.record_id, item.old.domain, item.old.slot, item.old.value, item.new.value,
            )))
    return anchors, entities


def extract_risawoz(path: Path) -> tuple[list[ChangeAnchor], list[SourceFact], list[SourceFact]]:
    raw = zip_json(path, "source-data/all_test600_new.json")
    assert isinstance(raw, list)
    anchors: list[ChangeAnchor] = []
    entities: list[SourceFact] = []
    media: list[SourceFact] = []
    for row in sorted(raw, key=lambda item: str(item["dialogue_id"])):
        record_id = str(row["dialogue_id"])
        seen: dict[tuple[str, str], SourceFact] = {}
        changes: list[ChangeAnchor] = []
        for turn in row["dialogue"]:
            turn_id = str(turn["turn_id"])
            user_utterance = str(turn.get("user_utterance") or "").strip()
            system_utterance = str(turn.get("system_utterance") or "").strip()
            turn_change_facts: dict[tuple[str, str], list[SourceFact]] = {}
            for role, utterance, actions, response in (
                ("user", user_utterance, turn.get("user_actions") or [], system_utterance),
                ("system", system_utterance, turn.get("system_actions") or [], ""),
            ):
                for act_index, action in enumerate(actions):
                    if len(action) < 4:
                        continue
                    intent, domain, slot, raw_value = map(str, action[:4])
                    value = clean_value(raw_value)
                    if not valid_value(value) or normalized(value) not in normalized(utterance):
                        continue
                    ref = source_ref(
                        dataset="RiSAWOZ", split="test", record_id=record_id,
                        turn_id=f"turn-{turn_id}-{role}", act_id=act_index,
                        domain=domain, slot=slot, value=value, excerpt=utterance,
                    )
                    fact = SourceFact(value, domain, slot, utterance, response, ref)
                    if domain in RISAWOZ_LOCATION_DOMAINS and slot in LOCATION_SLOTS:
                        entities.append(fact)
                    if domain in RISAWOZ_MEDIA_DOMAINS and slot in MEDIA_SLOTS:
                        media.append(fact)
                    if role != "user" or intent != "Inform" or slot not in CHANGE_SLOTS:
                        continue
                    key = (domain, slot)
                    turn_change_facts.setdefault(key, []).append(fact)
            record_unambiguous_turn_changes(
                dataset="RiSAWOZ", record_id=record_id,
                facts_by_key=turn_change_facts, seen=seen, changes=changes,
            )
        if changes:
            anchors.append(min(changes, key=lambda item: rank(
                item.record_id, item.old.domain, item.old.slot, item.old.value, item.new.value,
            )))
    return anchors, entities, media


def extract_durecdial(path: Path) -> list[PersonFact]:
    with zipfile.ZipFile(path) as archive:
        lines = archive.read("data/zh_test.txt").decode("utf-8").splitlines()
    by_name: dict[str, PersonFact] = {}
    for line_index, line in enumerate(lines, 1):
        row = json.loads(line)
        profile = row.get("user_profile") or {}
        name = clean_value(profile.get("姓名") or "")
        if not name or not re.fullmatch(r"[\u3400-\u9fff]{2,5}", name):
            continue
        chosen: tuple[str, str, int] | None = None
        for key in POSITIVE_PROFILE_KEYS:
            raw_value = profile.get(key)
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value_index, candidate in enumerate(values):
                value = clean_value(candidate or "")
                if valid_value(value):
                    chosen = (key, value, value_index)
                    break
            if chosen:
                break
        if not chosen:
            continue
        key, value, value_index = chosen
        ref = source_ref(
            dataset="DuRecDial", split="zh_test", record_id=f"line-{line_index}",
            turn_id="user_profile", act_id=value_index, domain="用户画像", slot=key,
            value=value, excerpt=json.dumps({"姓名": name, key: value}, ensure_ascii=False),
        )
        fact = PersonFact(name, value, key.replace(" ", ""), ref)
        current = by_name.get(name)
        if current is None or rank(ref.ref_id) < rank(current.ref.ref_id):
            by_name[name] = fact
    return sorted(by_name.values(), key=lambda item: rank(item.name, item.preference))


def choose_unique_facts(facts: Sequence[SourceFact], count: int, label: str) -> list[SourceFact]:
    selected: list[SourceFact] = []
    seen: set[str] = set()
    for fact in sorted(facts, key=lambda item: rank(item.ref.ref_id, item.value)):
        key = normalized(fact.value)
        if not key or key in seen:
            continue
        selected.append(fact)
        seen.add(key)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"not enough unique {label}: wanted {count}, found {len(selected)}")


def choose_unique_anchors(anchors: Sequence[ChangeAnchor], count: int, label: str) -> list[ChangeAnchor]:
    selected = sorted(anchors, key=lambda item: rank(
        item.dataset, item.record_id, item.old.ref.ref_id, item.new.ref.ref_id,
    ))[:count]
    if len(selected) != count or len({item.record_id for item in selected}) != count:
        raise RuntimeError(f"not enough unique {label} change dialogues")
    return selected


def synthetic_names() -> list[str]:
    surnames = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华")
    given = ("宁", "澄", "遥")
    names = [surname + suffix for surname in surnames for suffix in given]
    assert len(names) >= 75 and len(set(names)) == len(names)
    return names[:75]


def surface(text: str, style: str) -> str:
    if style == "clean":
        return text
    if style == "no-punctuation":
        return re.sub(r"[，。！？；：、,“”‘’（）()《》？]", "", text)
    if style == "colloquial-filler":
        return "嗯，那个，" + text
    if style == "elliptical-reference":
        return "前面那些记录，" + re.sub(r"^(请|帮我|麻烦你)", "", text)
    if style == "homophone-text-noise":
        return text.replace("目的地", "目的的").replace("座舱", "坐舱").replace("门禁", "门进")
    raise ValueError(style)


def authored_message(content: str, *, style: str, refs: Sequence[SourceRef] = ()) -> dict:
    return {
        "role": "user",
        "content": surface(content, style),
        "has_answer": True,
        "metadata": {
            "origin": "benchmark-authored-relation",
            "transformation": "public values embedded in a cockpit long-memory event",
            "source_ref_ids": [ref.ref_id for ref in refs],
            "surface_style": style,
        },
    }


def assistant_message(content: str, *, refs: Sequence[SourceRef] = (), public: bool = False) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "has_answer": False,
        "metadata": {
            "origin": "public-context" if public else "benchmark-authored-acknowledgement",
            "source_ref_ids": [ref.ref_id for ref in refs],
        },
    }


def add_session(
    sessions: list[dict], sample_id: str, when: str, messages: Sequence[dict],
    lineage: list[dict], used_refs: dict[str, SourceRef],
) -> list[str]:
    session_index = len(sessions) + 1
    source_session_id = f"{sample_id}-s{session_index:02d}"
    session = {
        "session_index": session_index,
        "source_session_id": source_session_id,
        "date_time": when,
        "messages": list(messages),
    }
    sessions.append(session)
    message_ids: list[str] = []
    for message_index, message in enumerate(messages, 1):
        message_id = f"{source_session_id}:{message_index:03d}"
        message_ids.append(message_id)
        metadata = message.get("metadata") or {}
        ref_ids = list(metadata.get("source_ref_ids") or [])
        lineage.append({
            "sample_id": sample_id,
            "session_id": source_session_id,
            "message_id": message_id,
            "role": message["role"],
            "origin": metadata.get("origin", "unknown"),
            "surface_style": metadata.get("surface_style"),
            "source_ref_ids": ref_ids,
        })
        for ref_id in ref_ids:
            if ref_id not in used_refs:
                raise AssertionError(f"source reference not registered before session add: {ref_id}")
    return message_ids


def public_anchor_message(person: str, fact: SourceFact, style: str) -> dict:
    return {
        "role": "user",
        "content": surface(f"【{person}】{fact.utterance}", style),
        "has_answer": True,
        "metadata": {
            "origin": "public-utterance-with-speaker-prefix",
            "transformation": "speaker prefix plus declared STT text style",
            "source_ref_ids": [fact.ref.ref_id],
            "surface_style": style,
        },
    }


def make_question(
    *, pack: int, slot: int, sample_id: str, ability: str, text: str, answer: str,
    question_date: str, evidence: Sequence[str], style: str, license_class: str,
    source_datasets: Sequence[str], chain_spec: dict, required_values: Sequence[str],
    abstention: bool = False,
) -> dict:
    version = DATASET_ID.rsplit("-", 1)[-1]
    return {
        "qa_id": f"cockpit-zh-public-mix-{version}#p{pack:02d}-q{slot:02d}",
        "sample_id": sample_id,
        "category": CATEGORIES[ability],
        "question_type": CATEGORIES[ability],
        "question": surface(f"{QUESTION_PREAMBLE}{text}", style),
        "answer": [answer],
        "question_date": question_date,
        "answer_session_ids": list(evidence),
        "is_abstention": abstention,
        "metadata": {
            "timezone": "Asia/Shanghai",
            "ability": ability,
            "holdout": True,
            "public_source_derived": True,
            "seed": SEED,
            "pack": pack,
            "surface_style": style,
            "license_class": license_class,
            "source_datasets": sorted(set(source_datasets)),
            "chain_spec": chain_spec,
            "required_evidence_values": list(required_values),
        },
    }


def register(used_refs: dict[str, SourceRef], *refs: SourceRef) -> None:
    for ref in refs:
        previous = used_refs.get(ref.ref_id)
        if previous is not None and previous != ref:
            raise AssertionError(f"source reference collision: {ref.ref_id}")
        used_refs[ref.ref_id] = ref


def build_pack(
    *, pack: int, anchor: ChangeAnchor, entities: Sequence[SourceFact],
    media: Sequence[SourceFact], people: Sequence[PersonFact], style: str, license_class: str,
    lineage: list[dict], used_refs: dict[str, SourceRef],
) -> tuple[dict, list[dict]]:
    assert len(entities) == 6 and len(people) == 3
    sample_id = f"cockpit-zh-public-mix-p{pack:02d}"
    primary = people[0].name
    e0, e1, e2, e3, e4, e5 = entities
    register(
        used_refs, anchor.old.ref, anchor.new.ref,
        *(item.ref for item in entities), *(item.ref for item in media), *(item.ref for item in people),
    )
    sessions: list[dict] = []
    questions: list[dict] = []

    # Authentic public dialogue-state change, isolated as two dated sessions.
    old_ids = add_session(sessions, sample_id, f"2026-03-01T09:00:00{TZ}", [
        public_anchor_message(primary, anchor.old, style),
        assistant_message(anchor.old.response or "本轮查询已记录。", refs=[anchor.old.ref], public=True),
    ], lineage, used_refs)
    new_ids = add_session(sessions, sample_id, f"2026-03-04T09:00:00{TZ}", [
        public_anchor_message(primary, anchor.new, style),
        assistant_message(anchor.new.response or "新的查询条件已记录。", refs=[anchor.new.ref], public=True),
    ], lineage, used_refs)

    # Five completed destination events: A, B, A, B, A.
    aggregation_values = [e0, e1, e0, e1, e0]
    aggregation_ids: list[str] = []
    for day, fact in zip(range(5, 10), aggregation_values):
        ids = add_session(sessions, sample_id, f"2026-03-{day:02d}T18:00:00{TZ}", [
            authored_message(f"【{primary}】今天车机导航到{fact.value}，已经到达。", style=style, refs=[fact.ref]),
            assistant_message(f"已记录到达{fact.value}。", refs=[fact.ref]),
        ], lineage, used_refs)
        aggregation_ids.append(ids[0])

    # Baseline, temporary validity window, and explicit restoration.
    base_ids = add_session(sessions, sample_id, f"2026-03-10T10:00:00{TZ}", [
        authored_message(f"【{primary}】平时说临时目的地就导航到{e2.value}。", style=style, refs=[e2.ref]),
        assistant_message("基础目的地口令已记录。", refs=[e2.ref]),
    ], lineage, used_refs)
    temporary_ids = add_session(sessions, sample_id, f"2026-03-11T10:00:00{TZ}", [
        authored_message(f"【{primary}】3月15日到18日临时目的地改成{e3.value}。", style=style, refs=[e3.ref]),
        assistant_message("临时有效期已记录。", refs=[e3.ref]),
    ], lineage, used_refs)

    # Three owners from independent public-profile or permissive public-entity facts.
    person_ids: list[str] = []
    for day, person in zip(range(12, 15), people):
        if person.preference_kind == "常用目的地":
            preference_text = f"我的常用车机目的地记成{person.preference}，只归到我名下。"
        else:
            preference_text = f"停车休息时我喜欢的车载媒体内容是{person.preference}，只归到我名下。"
        ids = add_session(sessions, sample_id, f"2026-03-{day:02d}T12:00:00{TZ}", [
            authored_message(
                f"【{person.name}】{preference_text}",
                style=style, refs=[person.ref],
            ),
            assistant_message(f"已记录{person.name}的独立偏好。", refs=[person.ref]),
        ], lineage, used_refs)
        person_ids.append(ids[0])

    threshold = 13 + pack % 5
    current_battery = threshold - 3
    priority_ids = add_session(sessions, sample_id, f"2026-03-15T08:00:00{TZ}", [
        authored_message(
            f"【{primary}】平时选补能点按休息设施、评分、距离排序；电量低于{threshold}%时改成距离、评分、休息设施。",
            style=style,
        ),
        assistant_message("补能点的条件优先级已记录。"),
    ], lineage, used_refs)

    old_day = 3 + pack % 4
    new_day = old_day + 3
    appointment_old_ids = add_session(sessions, sample_id, f"2026-03-16T09:00:00{TZ}", [
        authored_message(
            f"【{primary}】先约4月{old_day}日上午10点去{e4.value}做车辆检查。",
            style=style, refs=[e4.ref],
        ),
        assistant_message("首个车辆检查预约已记录。", refs=[e4.ref]),
    ], lineage, used_refs)
    appointment_new_ids = add_session(sessions, sample_id, f"2026-03-17T09:00:00{TZ}", [
        authored_message(
            f"【{primary}】取消4月{old_day}日上午的安排，改约4月{new_day}日下午3点去{e5.value}。",
            style=style, refs=[e4.ref, e5.ref],
        ),
        assistant_message("旧预约取消，新预约已记录。", refs=[e4.ref, e5.ref]),
    ], lineage, used_refs)
    appointment_cancel_ids = add_session(sessions, sample_id, f"2026-03-18T09:00:00{TZ}", [
        authored_message(
            f"【{primary}】4月{new_day}日下午去{e5.value}的检查也最终取消，取消后先不再约。",
            style=style, refs=[e5.ref],
        ),
        assistant_message("最终取消状态已记录，当前没有替代预约。", refs=[e5.ref]),
    ], lineage, used_refs)

    restore_ids = add_session(sessions, sample_id, f"2026-03-19T08:00:00{TZ}", [
        authored_message(f"【{primary}】临时安排结束，从今天起临时目的地恢复为{e2.value}。", style=style, refs=[e2.ref]),
        assistant_message("目的地口令已恢复。", refs=[e2.ref]),
    ], lineage, used_refs)

    aliases_old_ids = add_session(sessions, sample_id, f"2026-03-20T08:00:00{TZ}", [
        authored_message(
            f"【{primary}】车机口令先这样：午饭地点是{e0.value}，过夜地点是{e2.value}，会合地点是{e4.value}。",
            style=style, refs=[e0.ref, e2.ref, e4.ref],
        ),
        assistant_message("三项目的地口令已记录。", refs=[e0.ref, e2.ref, e4.ref]),
    ], lineage, used_refs)
    aliases_new_ids = add_session(sessions, sample_id, f"2026-03-21T08:00:00{TZ}", [
        authored_message(
            f"【{primary}】更新口令：午饭地点改成{e1.value}，过夜地点仍是{e2.value}，会合地点改成{e5.value}。",
            style=style, refs=[e1.ref, e2.ref, e5.ref],
        ),
        assistant_message("口令表已按字段更新。", refs=[e1.ref, e2.ref, e5.ref]),
    ], lineage, used_refs)

    volume = 4 + pack % 3
    if media:
        assert len(media) == 2
        content_old, content_new = media
        content_label = "停车休息时的车载媒体内容"
        old_content_text = f"停车休息时的车载内容先播放{content_old.value}"
        new_content_text = f"不是{content_old.value}，停车休息时的车载内容改为{content_new.value}"
    else:
        content_old, content_new = e3, e5
        content_label = "夜间导航目的地"
        old_content_text = f"夜间导航目的地先设为{content_old.value}"
        new_content_text = f"夜间导航目的地不是{content_old.value}，改为{content_new.value}"
    content_old_ids = add_session(sessions, sample_id, f"2026-03-22T21:00:00{TZ}", [
        authored_message(
            f"【{primary}】{old_content_text}，导航播报音量最多{volume}格，路线继续避开收费道路。",
            style=style, refs=[content_old.ref],
        ),
        assistant_message("夜间方案与两项约束已记录。", refs=[content_old.ref]),
    ], lineage, used_refs)
    content_new_ids = add_session(sessions, sample_id, f"2026-03-23T21:00:00{TZ}", [
        authored_message(
            f"【{primary}】纠正一下，{new_content_text}；导航播报音量上限和避开收费道路都不变。",
            style=style, refs=[content_old.ref, content_new.ref],
        ),
        assistant_message("夜间方案已纠正，既有约束继续有效。", refs=[content_old.ref, content_new.ref]),
    ], lineage, used_refs)

    pack_sources = [anchor.dataset]
    pack_sources.extend(ref.dataset for person in people for ref in [person.ref])
    qdate = f"2026-03-24T10:00:00{TZ}"
    questions.append(make_question(
        pack=pack, slot=1, sample_id=sample_id, ability="aggregation-frequency",
        text=f"把{primary}这五次已完成导航按地点统计，去得最多的是哪里，一共几次？",
        answer=f"{e0.value}，共3次。", question_date=qdate, evidence=aggregation_ids,
        style=style, license_class=license_class, source_datasets=[e0.ref.dataset, e1.ref.dataset],
        chain_spec={"operation": "mode", "events": [item.value for item in aggregation_values], "winner": e0.value, "count": 3},
        required_values=[e0.value, e1.value],
    ))
    questions.append(make_question(
        pack=pack, slot=2, sample_id=sample_id, ability="latest-final-update",
        text=f"回看{primary}的这段{anchor.old.domain}检索，最后一次明确查询的{anchor.old.slot}是什么，前一次是什么？",
        answer=f"最后一次是{anchor.new.value}；前一次是{anchor.old.value}。", question_date=qdate,
        evidence=[old_ids[0], new_ids[0]], style=style, license_class=license_class,
        source_datasets=[anchor.dataset],
        chain_spec={"operation": "latest", "field": f"{anchor.old.domain}.{anchor.old.slot}", "updates": [anchor.old.value, anchor.new.value]},
        required_values=[anchor.old.value, anchor.new.value],
    ))
    questions.append(make_question(
        pack=pack, slot=3, sample_id=sample_id, ability="two-date-validity",
        text=f"按日期分别查{primary}的临时目的地：3月16日和3月20日各是哪里？",
        answer=f"3月16日是{e3.value}；3月20日是{e2.value}。", question_date=qdate,
        evidence=[base_ids[0], temporary_ids[0], restore_ids[0]], style=style,
        license_class=license_class, source_datasets=[e2.ref.dataset, e3.ref.dataset],
        chain_spec={"operation": "interval-state", "base": e2.value, "temporary": e3.value, "temporary_range": ["2026-03-15", "2026-03-18"], "queries": {"2026-03-16": e3.value, "2026-03-20": e2.value}},
        required_values=[e2.value, e3.value],
    ))
    questions.append(make_question(
        pack=pack, slot=4, sample_id=sample_id, ability="multi-person-cross-session",
        text=(
            f"综合三次独立会话，{people[0].name}、{people[1].name}、{people[2].name}各自的"
            + ("常用车机目的地是什么？" if people[0].preference_kind == "常用目的地" else "停车休息媒体偏好是什么？")
        ),
        answer="；".join(f"{person.name}是{person.preference}" for person in people) + "。",
        question_date=qdate, evidence=person_ids, style=style, license_class=license_class,
        source_datasets=[person.ref.dataset for person in people],
        chain_spec={"operation": "owner-binding", "bindings": {person.name: person.preference for person in people}},
        required_values=[person.name for person in people] + [person.preference for person in people],
    ))
    questions.append(make_question(
        pack=pack, slot=5, sample_id=sample_id, ability="final-cancellation",
        text=f"把{primary}的车辆检查预约链还原到最后：现在有有效预约吗，最后取消哪一项，之后是否重约？",
        answer=f"当前没有有效预约；最后取消的是4月{new_day}日下午3点去{e5.value}的检查，之后没有重约。",
        question_date=qdate, evidence=[appointment_old_ids[0], appointment_new_ids[0], appointment_cancel_ids[0]],
        style=style, license_class=license_class, source_datasets=[e4.ref.dataset, e5.ref.dataset],
        chain_spec={"operation": "final-cancellation", "initial": f"4月{old_day}日上午10点@{e4.value}", "rescheduled": f"4月{new_day}日下午3点@{e5.value}", "final": None, "replacement_after_cancel": False},
        required_values=[str(new_day), e5.value, "取消后先不再约"],
    ))
    questions.append(make_question(
        pack=pack, slot=6, sample_id=sample_id, ability="conditional-priority",
        text=f"{primary}现在电量{current_battery}%，选补能点时距离、评分、休息设施按什么顺序？",
        answer="距离第一，评分第二，休息设施第三。", question_date=qdate, evidence=[priority_ids[0]],
        style=style, license_class=license_class, source_datasets=[],
        chain_spec={"operation": "conditional-priority", "threshold": threshold, "current": current_battery, "normal_order": ["休息设施", "评分", "距离"], "active_order": ["距离", "评分", "休息设施"]},
        required_values=[str(threshold), "距离", "评分", "休息设施"],
    ))
    questions.append(make_question(
        pack=pack, slot=7, sample_id=sample_id, ability="insufficient-evidence-abstention",
        text=f"查询{people[1].name}的公司门禁卡准确编号；没有直接编号就不要根据已有偏好猜。",
        answer=f"现有记录没有{people[1].name}的公司门禁卡编号，无法确定。", question_date=qdate,
        evidence=[], style=style, license_class=license_class, source_datasets=[people[1].ref.dataset],
        chain_spec={"operation": "field-scoped-abstention", "owner": people[1].name, "requested_field": "公司门禁卡编号", "available_field": people[1].preference_kind},
        required_values=[], abstention=True,
    ))
    questions.append(make_question(
        pack=pack, slot=8, sample_id=sample_id, ability="multi-target-final-state",
        text=f"给出{primary}当前三项目的地口令：午饭地点、过夜地点、会合地点分别指哪里？",
        answer=f"午饭地点是{e1.value}；过夜地点是{e2.value}；会合地点是{e5.value}。",
        question_date=qdate, evidence=[aliases_old_ids[0], aliases_new_ids[0]], style=style,
        license_class=license_class, source_datasets=[e0.ref.dataset],
        chain_spec={"operation": "field-wise-latest", "initial": {"午饭地点": e0.value, "过夜地点": e2.value, "会合地点": e4.value}, "final": {"午饭地点": e1.value, "过夜地点": e2.value, "会合地点": e5.value}},
        required_values=[e0.value, e1.value, e2.value, e4.value, e5.value],
    ))
    questions.append(make_question(
        pack=pack, slot=9, sample_id=sample_id, ability="cutoff-state",
        text=f"只看3月1日这次及更早的记录，3月4日那次查询不能倒灌，{primary}当时明确查询的{anchor.old.domain}{anchor.old.slot}是什么？",
        answer=f"截至3月1日是{anchor.old.value}。", question_date=qdate,
        evidence=[old_ids[0], new_ids[0]], style=style, license_class=license_class,
        source_datasets=[anchor.dataset],
        chain_spec={"operation": "cutoff", "cutoff": "2026-03-01", "events": [{"date": "2026-03-01", "value": anchor.old.value}, {"date": "2026-03-04", "value": anchor.new.value}], "answer": anchor.old.value},
        required_values=[anchor.old.value, anchor.new.value],
    ))
    questions.append(make_question(
        pack=pack, slot=10, sample_id=sample_id, ability="correction-retained-constraint",
        text=f"核对{primary}夜间车载设置的终版：当前{content_label}、播报音量上限、收费道路规则，以及旧值是否还有效？",
        answer=f"当前{content_label}是{content_new.value}，导航播报音量最多{volume}格，继续避开收费道路；旧值{content_old.value}已失效。",
        question_date=qdate, evidence=[content_old_ids[0], content_new_ids[0]], style=style,
        license_class=license_class, source_datasets=[content_old.ref.dataset, content_new.ref.dataset],
        chain_spec={"operation": "correction-with-retention", "old_content": content_old.value, "new_content": content_new.value, "retained": {"音量上限": volume, "收费道路": "避开"}, "old_valid": False},
        required_values=[content_old.value, content_new.value, str(volume), "避开收费道路"],
    ))

    conversation = {
        "sample_id": sample_id,
        "source_question_id": None,
        "session_count": len(sessions),
        "message_count": sum(len(session["messages"]) for session in sessions),
        "sessions": sessions,
        "metadata": {
            "public_anchor_dataset": anchor.dataset,
            "public_anchor_record_id": anchor.record_id,
            "surface_style": style,
            "license_class": license_class,
        },
    }
    return conversation, questions


def validate(conversations: Sequence[dict], questions: Sequence[dict], lineage: Sequence[dict]) -> None:
    assert len(conversations) == 50
    assert len(questions) == 500
    assert len(lineage) == sum(row["message_count"] for row in conversations)
    assert len({row["sample_id"] for row in conversations}) == 50
    assert len({row["qa_id"] for row in questions}) == 500
    assert len({row["question"] for row in questions}) == 500
    assert len({normalized(row["question"]) for row in questions}) == 500
    assert Counter(row["metadata"]["ability"] for row in questions) == {ability: 50 for ability in ABILITIES}
    assert Counter(row["metadata"]["surface_style"] for row in questions) == {style: 100 for style in SURFACE_STYLES}
    assert sum(bool(row["is_abstention"]) for row in questions) == 50
    assert all(row["session_count"] == 21 and row["message_count"] == 42 for row in conversations)
    assert all(re.search(r"[\u3400-\u9fff]", row["question"]) for row in questions)
    evidence_ids = {
        f"{session['source_session_id']}:{message_index:03d}"
        for conversation in conversations
        for session in conversation["sessions"]
        for message_index, _ in enumerate(session["messages"], 1)
    }
    assert all(evidence in evidence_ids for row in questions for evidence in row["answer_session_ids"])
    by_sample = {row["sample_id"]: row for row in conversations}
    for row in questions:
        conversation = by_sample[row["sample_id"]]
        own_prefix = conversation["sample_id"] + "-s"
        assert all(evidence.startswith(own_prefix) for evidence in row["answer_session_ids"])
        assert row["answer"] and normalized(row["answer"][0])
        assert bool(row["answer_session_ids"]) != bool(row["is_abstention"])


def source_registry(source_root: Path) -> dict:
    cross = source_root / "CrossWOZ"
    durec = source_root / "DuRecDial"
    risa = source_root / "RiSAWOZ"
    assert git_head(cross) == CROSSWOZ_COMMIT
    assert git_head(durec) == DURECDIAL_COMMIT
    assert git_head(risa) == RISAWOZ_COMMIT
    return {
        "schema_version": 1,
        "snapshot_date": "2026-08-29",
        "sources": [
            {
                "dataset": "CrossWOZ",
                "official_repository": "https://github.com/thu-coai/CrossWOZ",
                "commit": CROSSWOZ_COMMIT,
                "artifact": "data/crosswoz/test.json.zip",
                "artifact_sha256": sha256(cross / "data/crosswoz/test.json.zip"),
                "license_file_sha256": sha256(cross / "LICENSE"),
                "declared_data_or_repository_license": "Apache-2.0",
                "license_class": "permissive-public-source",
                "included_packs": 25,
            },
            {
                "dataset": "RiSAWOZ",
                "official_repository": "https://github.com/terryqj0107/RiSAWOZ",
                "commit": RISAWOZ_COMMIT,
                "artifact": "RiSAWOZ-data/source-data.zip:source-data/all_test600_new.json",
                "artifact_sha256": sha256(risa / "RiSAWOZ-data/source-data.zip"),
                "license_file_sha256": sha256(risa / "LICENSE"),
                "declared_dataset_license": "CC-BY-NC-4.0",
                "license_class": "noncommercial-public-source",
                "included_packs": 25,
            },
            {
                "dataset": "DuRecDial 2.0",
                "official_repository": "https://github.com/liuzeming01/DuRecDial",
                "commit": DURECDIAL_COMMIT,
                "artifact": "data.zip:data/zh_test.txt",
                "artifact_sha256": sha256(durec / "data.zip"),
                "license_file_sha256": sha256(durec / "LICENSE"),
                "declared_dataset_license": "CC-BY-NC-SA-4.0",
                "license_class": "noncommercial-sharealike-public-source",
                "included_profile_facts": 75,
            },
        ],
        "excluded": [
            {
                "dataset": "CATSLU",
                "official_page": "https://sites.google.com/view/catslu/home/",
                "reason": "public download exists, but no explicit dataset redistribution/commercial license was verified",
            },
            {
                "dataset": "AISHELL-5",
                "official_page": "https://openslr.org/159/",
                "reason": "ASR/diarization corpus without long-memory facts or QA gold; ASR quality is out of scope",
            },
        ],
    }


def manifest_for(conversations: Sequence[dict], questions: Sequence[dict], dataset_id: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "schema_version": 1,
        "seed": SEED,
        "language": "zh",
        "input_modality": "ASR text transcript; audio and ASR accuracy are out of scope",
        "conversation_count": len(conversations),
        "question_count": len(questions),
        "abstention_count": sum(bool(row["is_abstention"]) for row in questions),
        "session_count": sum(row["session_count"] for row in conversations),
        "message_count": sum(row["message_count"] for row in conversations),
        "ability_distribution": dict(sorted(Counter(row["metadata"]["ability"] for row in questions).items())),
        "surface_style_distribution": dict(sorted(Counter(row["metadata"]["surface_style"] for row in questions).items())),
        "anchor_source_distribution": dict(sorted(Counter(row["metadata"]["public_anchor_dataset"] for row in conversations).items())),
        "license_class_distribution": dict(sorted(Counter(row["metadata"]["license_class"] for row in questions).items())),
        "source_split_policy": "one unique public anchor dialogue per pack; public test records are transformed into novel long-memory relations",
        "holdout_policy": "sealed before any end-to-end answer or judge run; no post-score repair in place",
        "contamination_note": "public source utterances may have appeared in model pretraining; answers depend on new cross-session compositions and cannot be copied from source QA",
    }


def write_readme(path: Path, full: bool) -> None:
    version = DATASET_ID.rsplit("-", 1)[-1]
    if full:
        body = f"""# Cockpit ZH Public Mix 500 {version}

A deterministic Chinese text-transcript holdout for cockpit long-memory QA. It contains 50 entirely new memory packs and 500 questions, balanced across ten abilities and five STT-like text styles. Public utterances and values come from pinned CrossWOZ, RiSAWOZ and DuRecDial snapshots; temporal, ownership, cancellation and evidence-contract relations are benchmark-authored and fully traceable.

This full track is **research/non-commercial only** because RiSAWOZ is CC-BY-NC-4.0 and DuRecDial 2.0 is CC-BY-NC-SA-4.0. Use `permissive-source-250/` for the CrossWOZ-only track, subject to your own license review.

The dataset was sealed before any answer-model or Judge call. Do not tune on these 500 questions or inspect gold answers during evaluation.
"""
    else:
        body = f"""# Cockpit ZH CrossWOZ 250 {version}

The first 25 packs (250 questions) of the public-mix benchmark, derived only from the pinned CrossWOZ repository plus benchmark-authored long-memory relations. It excludes RiSAWOZ and DuRecDial content. This is a permissive-source evaluation lane, not legal advice or a substitute for deployment license review.
"""
    path.write_text(body, encoding="utf-8")


def write_attribution(path: Path) -> None:
    path.write_text("""# Source attribution and use constraints

- CrossWOZ: Qi Zhu et al., *CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset*, pinned from https://github.com/thu-coai/CrossWOZ at commit `df82c9f`; repository license declaration Apache-2.0.
- RiSAWOZ: Jun Quan et al., *RiSAWOZ*, pinned from https://github.com/terryqj0107/RiSAWOZ at commit `1e2d3be`; dataset license CC-BY-NC-4.0.
- DuRecDial 2.0: Zeming Liu et al., *DuRecDial 2.0*, pinned from https://github.com/liuzeming01/DuRecDial at commit `1309cc0`; dataset license CC-BY-NC-SA-4.0.

The full 500-question mix inherits non-commercial and share-alike constraints. CATSLU was not copied because an explicit dataset redistribution/commercial license was not verified. This record is an engineering provenance audit, not legal advice.
""", encoding="utf-8")


def write_protocol(path: Path) -> None:
    path.write_text("""# Sealed evaluation protocol

1. Verify `BLIND_SEAL.sha256` before ingestion and after scoring.
2. Ingest each `sample_id` into an isolated memory namespace.
3. Run every question exactly once with the frozen answer model/configuration; no retries except transport failures recorded as failures.
4. Run one independent Judge pass only after answers are immutable.
5. Perform deterministic answer-contract checks and stratified human review.
6. Never repair this directory after seeing scores. Any fix creates a new RC and a new holdout.
7. Report the full 500 research track and the 250 permissive-source track separately.
""", encoding="utf-8")


def seal_directory(root: Path) -> None:
    rows = []
    own_seal = root / "BLIND_SEAL.sha256"
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != own_seal):
        rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "BLIND_SEAL.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    global SEED, DATASET_ID, PERMISSIVE_SUBSET_ID, QUESTION_PREAMBLE

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--permissive-subset-id", default=PERMISSIVE_SUBSET_ID)
    parser.add_argument("--question-preamble", default=QUESTION_PREAMBLE)
    parser.add_argument(
        "--exclude-dataset-root", type=Path, action="append", default=[],
        help="exclude anchor records and exact source refs from an earlier sealed dataset",
    )
    parser.add_argument(
        "--exclude-anchor-dataset-root", type=Path, action="append", default=[],
        help=(
            "exclude only anchor dialogues from an earlier sealed dataset; "
            "public entity facts may repeat and prior questions must be excluded separately"
        ),
    )
    parser.add_argument(
        "--exclude-normalized-values", action="store_true",
        help="also exclude repeated public values (requires a sufficiently large corpus)",
    )
    parser.add_argument(
        "--exclude-question-file", type=Path, action="append", default=[],
        help="reject output if any normalized question occurs in this JSONL file",
    )
    args = parser.parse_args()
    SEED = args.seed
    DATASET_ID = args.dataset_id
    PERMISSIVE_SUBSET_ID = args.permissive_subset_id
    QUESTION_PREAMBLE = args.question_preamble
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {args.output_dir}")
    if args.audit_report.exists():
        raise SystemExit(f"refusing to overwrite existing audit report: {args.audit_report}")

    registry = source_registry(args.source_root)
    cross_zip = args.source_root / "CrossWOZ/data/crosswoz/test.json.zip"
    durec_zip = args.source_root / "DuRecDial/data.zip"
    risa_zip = args.source_root / "RiSAWOZ/RiSAWOZ-data/source-data.zip"
    cross_anchors_all, cross_entities_all = extract_crosswoz(cross_zip)
    risa_anchors_all, risa_entities_all, risa_media_all = extract_risawoz(risa_zip)
    people_all = extract_durecdial(durec_zip)

    (
        excluded_anchor_records,
        excluded_ref_ids,
        excluded_values,
        exclusion_receipts,
    ) = load_exclusions(args.exclude_dataset_root)
    anchor_only_records, anchor_only_exclusion_receipts = load_anchor_exclusions(
        args.exclude_anchor_dataset_root
    )
    excluded_anchor_records.update(anchor_only_records)
    enforced_excluded_values = excluded_values if args.exclude_normalized_values else set()
    cross_anchors_all = [
        item for item in cross_anchors_all
        if (item.dataset, item.record_id) not in excluded_anchor_records
        and source_fact_is_new(item.old.ref, item.old.value, excluded_ref_ids, enforced_excluded_values)
        and source_fact_is_new(item.new.ref, item.new.value, excluded_ref_ids, enforced_excluded_values)
    ]
    risa_anchors_all = [
        item for item in risa_anchors_all
        if (item.dataset, item.record_id) not in excluded_anchor_records
        and source_fact_is_new(item.old.ref, item.old.value, excluded_ref_ids, enforced_excluded_values)
        and source_fact_is_new(item.new.ref, item.new.value, excluded_ref_ids, enforced_excluded_values)
    ]
    cross_entities_all = [
        item for item in cross_entities_all
        if source_fact_is_new(item.ref, item.value, excluded_ref_ids, enforced_excluded_values)
    ]
    risa_entities_all = [
        item for item in risa_entities_all
        if source_fact_is_new(item.ref, item.value, excluded_ref_ids, enforced_excluded_values)
    ]
    risa_media_all = [
        item for item in risa_media_all
        if source_fact_is_new(item.ref, item.value, excluded_ref_ids, enforced_excluded_values)
    ]
    people_all = [
        item for item in people_all
        if source_fact_is_new(item.ref, item.preference, excluded_ref_ids, enforced_excluded_values)
    ]

    cross_anchors = choose_unique_anchors(cross_anchors_all, 25, "CrossWOZ")
    risa_anchors = choose_unique_anchors(risa_anchors_all, 25, "RiSAWOZ")
    cross_entities = choose_unique_facts(cross_entities_all, 150, "CrossWOZ entities")
    risa_entities = choose_unique_facts(risa_entities_all, 150, "RiSAWOZ entities")
    risa_media = choose_unique_facts(risa_media_all, 50, "RiSAWOZ media titles")
    durec_people = people_all[:75]
    if len(durec_people) != 75 or len({item.name for item in durec_people}) != 75:
        raise RuntimeError("not enough unique DuRecDial profile owners")

    rng = random.Random(SEED)
    names = synthetic_names()
    rng.shuffle(names)
    conversations: list[dict] = []
    questions: list[dict] = []
    lineage: list[dict] = []
    used_refs: dict[str, SourceRef] = {}
    for pack in range(1, 51):
        style = SURFACE_STYLES[(pack - 1) % len(SURFACE_STYLES)]
        if pack <= 25:
            offset = pack - 1
            anchor = cross_anchors[offset]
            entities = cross_entities[offset * 6:(offset + 1) * 6]
            person_names = names[offset * 3:(offset + 1) * 3]
            people = [
                PersonFact(person_names[index], entities[index * 2].value, "常用目的地", entities[index * 2].ref)
                for index in range(3)
            ]
            media: list[SourceFact] = []
            license_class = "permissive-public-source"
        else:
            offset = pack - 26
            anchor = risa_anchors[offset]
            entities = risa_entities[offset * 6:(offset + 1) * 6]
            media = risa_media[offset * 2:(offset + 1) * 2]
            people = durec_people[offset * 3:(offset + 1) * 3]
            license_class = "noncommercial-mixed-public-source"
        conversation, pack_questions = build_pack(
            pack=pack, anchor=anchor, entities=entities, media=media, people=people, style=style,
            license_class=license_class, lineage=lineage, used_refs=used_refs,
        )
        conversations.append(conversation)
        questions.extend(pack_questions)

    validate(conversations, questions, lineage)
    prior_question_files = [
        root / "questions.jsonl" for root in args.exclude_dataset_root
    ] + args.exclude_question_file
    prior_questions, prior_question_receipts = load_prior_questions(
        prior_question_files
    )
    normalized_questions = {normalized(row["question"]) for row in questions}
    question_overlap = normalized_questions & prior_questions
    if question_overlap:
        raise RuntimeError(
            f"refusing overlapping holdout questions: {len(question_overlap)}"
        )
    args.output_dir.mkdir(parents=True)
    jsonl_dump(args.output_dir / "conversations.jsonl", conversations)
    jsonl_dump(args.output_dir / "questions.jsonl", questions)
    jsonl_dump(args.output_dir / "source_lineage.jsonl", lineage)
    jsonl_dump(args.output_dir / "source_refs.jsonl", [used_refs[key].payload() for key in sorted(used_refs)])
    json_dump(args.output_dir / "source_registry.json", registry)
    json_dump(args.output_dir / "manifest.json", manifest_for(conversations, questions, DATASET_ID))
    write_readme(args.output_dir / "README.md", full=True)
    write_attribution(args.output_dir / "SOURCE_ATTRIBUTION.md")
    write_protocol(args.output_dir / "BLIND_PROTOCOL.md")

    subset_dir = args.output_dir / "permissive-source-250"
    subset_dir.mkdir()
    subset_conversations = conversations[:25]
    subset_samples = {row["sample_id"] for row in subset_conversations}
    subset_questions = [row for row in questions if row["sample_id"] in subset_samples]
    subset_lineage = [row for row in lineage if row["sample_id"] in subset_samples]
    subset_ref_ids = {ref_id for row in subset_lineage for ref_id in row["source_ref_ids"]}
    subset_refs = [used_refs[key].payload() for key in sorted(subset_ref_ids)]
    jsonl_dump(subset_dir / "conversations.jsonl", subset_conversations)
    jsonl_dump(subset_dir / "questions.jsonl", subset_questions)
    jsonl_dump(subset_dir / "source_lineage.jsonl", subset_lineage)
    jsonl_dump(subset_dir / "source_refs.jsonl", subset_refs)
    json_dump(subset_dir / "source_registry.json", {
        **registry,
        "sources": [item for item in registry["sources"] if item["dataset"] == "CrossWOZ"],
    })
    json_dump(subset_dir / "manifest.json", manifest_for(
        subset_conversations, subset_questions, PERMISSIVE_SUBSET_ID,
    ))
    write_readme(subset_dir / "README.md", full=False)
    write_protocol(subset_dir / "BLIND_PROTOCOL.md")
    seal_directory(subset_dir)
    seal_directory(args.output_dir)

    audit = {
        "dataset_id": DATASET_ID,
        "compiler_seed": SEED,
        "exclusion_receipts": exclusion_receipts,
        "anchor_only_exclusion_receipts": anchor_only_exclusion_receipts,
        "excluded_anchor_record_count": len(excluded_anchor_records),
        "anchor_only_excluded_record_count": len(anchor_only_records),
        "excluded_source_ref_count": len(excluded_ref_ids),
        "excluded_normalized_value_count": len(excluded_values),
        "normalized_value_exclusion_enforced": args.exclude_normalized_values,
        "question_preamble": QUESTION_PREAMBLE,
        "prior_question_receipts": prior_question_receipts,
        "normalized_prior_question_count": len(prior_questions),
        "normalized_question_overlap_count": len(question_overlap),
        "source_registry": registry,
        "candidate_counts": {
            "crosswoz_change_dialogues": len(cross_anchors_all),
            "risawoz_change_dialogues": len(risa_anchors_all),
            "crosswoz_entity_facts": len(cross_entities_all),
            "risawoz_entity_facts": len(risa_entities_all),
            "risawoz_media_facts": len(risa_media_all),
            "durecdial_unique_profile_owners": len(people_all),
        },
        "selected": {
            "crosswoz_anchor_dialogues": [item.record_id for item in cross_anchors],
            "risawoz_anchor_dialogues": [item.record_id for item in risa_anchors],
            "durecdial_profile_owners": 75,
            "source_refs": len(used_refs),
        },
        "output_manifest": manifest_for(conversations, questions, DATASET_ID),
        "seal_sha256": sha256(args.output_dir / "BLIND_SEAL.sha256"),
        "answer_or_judge_calls": 0,
    }
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.audit_report, audit)
    print(json.dumps(audit["output_manifest"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
