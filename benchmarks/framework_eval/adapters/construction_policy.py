from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ..schema import Session


_ACK_ONLY = re.compile(
    r"^(?:hi|hello|hey|thanks?|thank you|okay|ok|sure|great|goodbye|bye|"
    r"yes|no|好的|谢谢|你好|再见|嗯|行)[.!?，。！\s]*$",
    re.IGNORECASE,
)
_PROFILE = re.compile(
    r"\b(?:prefer|preference|favorite|favourite|usually|always|never|"
    r"remember that|call me|my home|home address|my work|work address|"
    r"default route|avoid tolls?)\b|"
    r"(?:偏好|喜欢|最爱|通常|总是|从不|记住|称呼我|我家|家庭地址|公司地址|"
    r"默认路线|避开收费)",
    re.IGNORECASE,
)
_EVENT_MUTATION = re.compile(
    r"\b(?:schedule|scheduled|reschedule|appointment|meeting|remind|reminder|"
    r"set (?:an? )?alarm|create|add|remove|delete|cancel|update|change|book|"
    r"reserve|send (?:an? )?(?:email|message)|call [a-z]|navigate to|"
    r"directions to|route (?:me )?to|take me to)\b|"
    r"(?:安排|预约|会议|提醒|闹钟|创建|添加|删除|取消|更新|修改|预订|发邮件|"
    r"发消息|打电话|导航到|路线到|带我去)",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"\b(?:actually|instead|I meant|correction|change that|not .+ but|"
    r"make that|rather than)\b|"
    r"(?:其实|改成|我是说|纠正|不是.+而是|换成)",
    re.IGNORECASE,
)
_TOOL_OUTCOME = re.compile(
    r"\b(?:has been|have been|successfully|is now|are now|scheduled|"
    r"added|removed|cancelled|canceled|updated|route selected|navigation started|"
    r"failed|unable to|could not)\b|"
    r"(?:已成功|已经|设置为|已安排|已添加|已删除|已取消|已更新|开始导航|执行失败|"
    r"无法执行)",
    re.IGNORECASE,
)
_RECORD = re.compile(
    r"\b(?:address|contact|phone number|destination|location|route|calendar|"
    r"weather|forecast|temperature|battery|charging|parking)\b|"
    r"(?:地址|联系人|电话号码|目的地|位置|路线|日历|天气|预报|温度|电量|充电|停车)",
    re.IGNORECASE,
)
_READ_ONLY_INTENT = re.compile(
    r"(?:query|get|check|show|find|search|play|weather|forecast|news|qa|"
    r"recommend|lookup|read)",
    re.IGNORECASE,
)
_MUTATING_INTENT = re.compile(
    r"(?:create|set|add|remove|delete|cancel|update|change|send|call|schedule|"
    r"navigate|route|book|reserve)",
    re.IGNORECASE,
)
_DURABLE_INTENT = re.compile(
    r"(?:calendar|appointment|meeting|remind|reminder|alarm|email|message|"
    r"schedule|booking|reservation|reserve|contact)",
    re.IGNORECASE,
)
_TRANSIENT_CONTROL_INTENT = re.compile(
    r"(?:play|pause|volume|audio|music|radio|podcast|light|window|seat|"
    r"climate|temperature|fan|vehicle|iot|transport|navigation|route)",
    re.IGNORECASE,
)
_CANCEL_OR_DELETE = re.compile(
    r"\b(?:cancel|delete|remove|erase|forget|stop|clear)\b|"
    r"(?:取消|删除|移除|清除|停止)",
    re.IGNORECASE,
)
_UPDATE = re.compile(
    r"\b(?:update|change|reschedule|move|replace|instead|make that)\b|"
    r"(?:更新|修改|改成|换成|重新安排)",
    re.IGNORECASE,
)
_FAILED_OUTCOME = re.compile(
    r"\b(?:failed|unable to|could not|cannot|did not|wasn't|was not)\b|"
    r"(?:失败|无法|不能|未能)",
    re.IGNORECASE,
)
_QUESTION_LIKE = re.compile(
    r"^\s*(?:what|when|where|who|why|how|which|is|are|am|was|were|"
    r"do|does|did|can|could|would|will|have|has|show|find|check|tell|"
    r"请问|什么|何时|什么时候|哪里|哪儿|谁|为什么|怎么|是否|有没有|"
    r"查询|查看)",
    re.IGNORECASE,
)
_DURABLE_SELF_FACT = re.compile(
    r"\b(?:my name is|call me|i (?:live|work|study) (?:at|in|for)|"
    r"i (?:own|have) (?:a|an|two|three|\d+)\s+"
    r"(?:car|house|home|dog|cat|pet|child|children)|"
    r"i am (?:a|an)\s+[a-z][a-z -]{2,})\b|"
    r"(?:我叫|称呼我|我住在|我在.+工作|我有.+(?:车|房|狗|猫|宠物|孩子))",
    re.IGNORECASE,
)
_AUTOBIOGRAPHICAL_EVENT = re.compile(
    r"\bi (?:went|visited|attended|joined|started|finished|graduated|"
    r"bought|purchased|moved|travelled|traveled|met|saw|won|lost|learned|"
    r"adopted|booked|reserved|scheduled)\b|"
    r"(?:我(?:去过|去了|参观了|参加了|加入了|开始了|完成了|毕业了|买了|"
    r"搬到|遇到|看了|赢了|学会了|领养了|预订了|安排了))",
    re.IGNORECASE,
)
_FUTURE_TIME = re.compile(
    r"\b(?:tomorrow|tonight|next\s+(?:day|week|month|year|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday)|later|upcoming)\b|"
    r"(?:明天|今晚|下周|下个月|明年|稍后|即将)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConstructionDecision:
    memory_type: str
    extract_l1: bool
    reason: str
    intent: str
    domain: str
    substantive_user_messages: int
    source_characters: int
    write_action: str
    scene: str
    lifecycle: str
    temporal_scope: str
    confidence: float
    episode_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata_text(session: Session, *keys: str) -> str:
    values = []
    for key in keys:
        value = session.metadata.get(key)
        if value is not None:
            values.append(str(value))
    for message in session.messages:
        for key in keys:
            value = message.metadata.get(key)
            if value is not None:
                values.append(str(value))
    return " ".join(values)


def _safe_label(value: str, *, fallback: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+", "_", str(value or "").casefold()
    ).strip("_")
    return normalized[:80] or fallback


def _scene_label(intent: str, domain: str) -> str:
    raw = str(domain or "").strip()
    if not raw:
        raw = re.split(r"[_:./-]", str(intent or "").strip(), maxsplit=1)[0]
    label = _safe_label(raw, fallback="general")
    aliases = {
        "audio": "media",
        "music": "media",
        "play": "media",
        "maps": "navigation",
        "route": "navigation",
        "transport": "navigation",
        "car": "vehicle",
        "iot": "vehicle",
    }
    return aliases.get(label, label)


def _episode_key(memory_type: str, scene: str, intent: str) -> str:
    intent_label = _safe_label(intent, fallback=memory_type)
    return f"{scene}:{intent_label}"


def _structured_decision(
    *,
    memory_type: str,
    extract_l1: bool,
    reason: str,
    intent: str,
    domain: str,
    substantive_user_messages: int,
    source_characters: int,
    write_action: str,
    lifecycle: str,
    temporal_scope: str,
    confidence: float,
) -> ConstructionDecision:
    scene = _scene_label(intent, domain)
    return ConstructionDecision(
        memory_type=memory_type,
        extract_l1=extract_l1,
        reason=reason,
        intent=intent,
        domain=domain,
        substantive_user_messages=substantive_user_messages,
        source_characters=source_characters,
        write_action=write_action,
        scene=scene,
        lifecycle=lifecycle,
        temporal_scope=temporal_scope,
        confidence=round(min(1.0, max(0.0, confidence)), 3),
        episode_key=_episode_key(memory_type, scene, intent),
    )


def classify_session(session: Session) -> ConstructionDecision:
    """Route a session without an LLM and without benchmark QA labels."""
    user_messages = [
        message.render_text().strip()
        for message in session.messages
        if str(message.role).casefold() == "user" and message.render_text().strip()
    ]
    substantive = [value for value in user_messages if not _ACK_ONLY.fullmatch(value)]
    text = "\n".join(
        message.render_text() for message in session.messages
        if message.render_text().strip()
    )
    user_text = "\n".join(substantive)
    assistant_text = "\n".join(
        message.render_text() for message in session.messages
        if str(message.role).casefold() in {"assistant", "tool"}
    )
    intent = _metadata_text(
        session, "intent", "source_intent", "scenario", "source_scenario"
    ).strip()
    domain = _metadata_text(
        session, "domain", "source_domain", "source_scenario"
    ).strip()
    source_characters = len(text)

    if not substantive:
        kind, extract, reason = "transient", False, "ack_or_empty"
    elif _PROFILE.search(user_text) or re.search(
        r"(?:preference|profile|persona)", intent, re.IGNORECASE
    ):
        kind, extract, reason = "profile", True, "stable_preference_or_identity"
    elif _CORRECTION.search(user_text):
        kind, extract, reason = "event", True, "user_correction"
    elif (
        _MUTATING_INTENT.search(intent)
        and not _READ_ONLY_INTENT.fullmatch(intent.strip())
    ) or _EVENT_MUTATION.search(user_text):
        kind, extract, reason = "event", True, "state_changing_intent"
    elif _TOOL_OUTCOME.search(assistant_text) and (
        _EVENT_MUTATION.search(text) or _MUTATING_INTENT.search(intent)
    ):
        kind, extract, reason = "event", True, "durable_tool_outcome"
    elif _RECORD.search(text) or domain:
        # LeanMem-style record: keep exact L0 text and source pointer. It does
        # not need a lossy LLM summary; retrieval can promote it on demand.
        kind, extract, reason = "record", False, "source_grounded_record"
    else:
        kind, extract, reason = "transient", False, "one_shot_or_low_value"
    structured = {
        "transient": ("no_op", "rejected", "turn", 0.98),
        "record": ("retain", "accepted", "source_time", 0.75),
        "profile": ("update", "accepted", "persistent", 0.90),
        "event": (
            "update" if reason == "user_correction" else "add",
            "confirmed" if reason == "durable_tool_outcome" else "pending",
            "future" if _FUTURE_TIME.search(text) else "event_time",
            0.85,
        ),
    }[kind]
    return _structured_decision(
        memory_type=kind,
        extract_l1=extract,
        reason=reason,
        intent=intent,
        domain=domain,
        substantive_user_messages=len(substantive),
        source_characters=source_characters,
        write_action=structured[0],
        lifecycle=structured[1],
        temporal_scope=structured[2],
        confidence=structured[3],
    )


def classify_session_v2(session: Session) -> ConstructionDecision:
    """Scene-aware write gate for task dialogue and long-term chat.

    Raw L0 is always retained by the TencentDB adapter.  This policy controls
    only promotion into the lossy L1 extractor.  It deliberately consumes no
    benchmark question, answer, category, or evidence labels and makes no LLM
    call.  The action/state fields form an auditable adapter-side transaction
    record inspired by temporal memory ledgers.
    """
    user_messages = [
        message.render_text().strip()
        for message in session.messages
        if str(message.role).casefold() == "user" and message.render_text().strip()
    ]
    substantive = [value for value in user_messages if not _ACK_ONLY.fullmatch(value)]
    text = "\n".join(
        message.render_text() for message in session.messages
        if message.render_text().strip()
    )
    user_text = "\n".join(substantive)
    assistant_text = "\n".join(
        message.render_text() for message in session.messages
        if str(message.role).casefold() in {"assistant", "tool"}
    )
    intent = _metadata_text(
        session, "intent", "source_intent", "scenario", "source_scenario"
    ).strip()
    domain = _metadata_text(
        session, "domain", "source_domain", "source_scenario"
    ).strip()
    source_characters = len(text)
    mutation = bool(
        (_MUTATING_INTENT.search(intent) and not _READ_ONLY_INTENT.fullmatch(
            intent.strip()
        ))
        or _EVENT_MUTATION.search(user_text)
    )
    durable_intent = bool(_DURABLE_INTENT.search(intent))
    confirmed_outcome = bool(_TOOL_OUTCOME.search(assistant_text))
    failed_outcome = bool(_FAILED_OUTCOME.search(assistant_text))
    question_like = bool(
        _QUESTION_LIKE.search(user_text) or "?" in user_text or "？" in user_text
    )

    if not substantive:
        values = (
            "transient", False, "ack_or_empty", "no_op", "rejected", "turn", 0.99
        )
    elif (
        _PROFILE.search(user_text)
        or re.search(r"(?:preference|profile|persona)", intent, re.IGNORECASE)
        or (_DURABLE_SELF_FACT.search(user_text) and not question_like)
    ):
        values = (
            "profile", True, "stable_preference_or_identity", "update",
            "accepted", "persistent", 0.94,
        )
    elif _CORRECTION.search(user_text) or _UPDATE.search(user_text):
        values = (
            "event", True, "user_correction", "update",
            "confirmed" if confirmed_outcome else "pending",
            "future" if _FUTURE_TIME.search(text) else "event_time", 0.95,
        )
    elif _CANCEL_OR_DELETE.search(user_text) and mutation:
        values = (
            "event", True, "event_expiry_request", "expire",
            "failed" if failed_outcome else (
                "confirmed" if confirmed_outcome else "pending"
            ),
            "event_time", 0.93,
        )
    elif confirmed_outcome and mutation:
        values = (
            "event", True, "durable_tool_outcome",
            "expire" if _CANCEL_OR_DELETE.search(text) else (
                "update" if _UPDATE.search(text) else "add"
            ),
            "failed" if failed_outcome else "confirmed",
            "future" if _FUTURE_TIME.search(text) else "event_time", 0.96,
        )
    elif mutation and durable_intent:
        # Calendar, reminders, messages, and reservations remain useful even
        # when the capture ends before a tool confirmation.  Marking them
        # pending prevents an unconfirmed request from becoming a fact.
        values = (
            "event", True, "durable_pending_request", "pending", "pending",
            "future" if _FUTURE_TIME.search(text) else "event_time", 0.82,
        )
    elif _AUTOBIOGRAPHICAL_EVENT.search(user_text) and not question_like:
        values = (
            "event", True, "autobiographical_event", "add", "accepted",
            "event_time", 0.88,
        )
    elif mutation and _TRANSIENT_CONTROL_INTENT.search(
        f"{intent} {domain}"
    ):
        values = (
            "record", False, "transient_control_retained_in_l0", "retain",
            "accepted", "turn", 0.90,
        )
    elif _RECORD.search(text) or domain:
        values = (
            "record", False, "source_grounded_record", "retain", "accepted",
            "source_time", 0.82,
        )
    else:
        values = (
            "transient", False, "one_shot_or_low_value", "no_op", "rejected",
            "turn", 0.80,
        )
    kind, extract, reason, action, lifecycle, temporal_scope, confidence = values
    return _structured_decision(
        memory_type=kind,
        extract_l1=extract,
        reason=reason,
        intent=intent,
        domain=domain,
        substantive_user_messages=len(substantive),
        source_characters=source_characters,
        write_action=action,
        lifecycle=lifecycle,
        temporal_scope=temporal_scope,
        confidence=confidence,
    )


def all_sessions_policy(session: Session) -> ConstructionDecision:
    text = "".join(message.render_text() for message in session.messages)
    return _structured_decision(
        memory_type="all",
        extract_l1=True,
        reason="full_construction_baseline",
        intent=_metadata_text(session, "intent", "source_intent").strip(),
        domain=_metadata_text(
            session, "domain", "source_domain", "source_scenario"
        ).strip(),
        substantive_user_messages=sum(
            1 for message in session.messages
            if str(message.role).casefold() == "user"
            and message.render_text().strip()
        ),
        source_characters=len(text),
        write_action="retain",
        lifecycle="accepted",
        temporal_scope="source_time",
        confidence=1.0,
    )
