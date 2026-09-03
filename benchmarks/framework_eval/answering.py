from __future__ import annotations

import concurrent.futures
import json
import os
import time
import urllib.request
import re
from dataclasses import dataclass
from pathlib import Path

from .cockpit_slots import (
    SlotAnswerCandidate,
    extract_clarification_reply,
    extract_cockpit_answer,
)
from .schema import ContentPart
from .temporal import TemporalQuery, resolve_temporal_query
from .structured_state import temporal_context


@dataclass(frozen=True)
class AnswerConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 128
    timeout: int = 180
    temperature: float = 0.0
    retries: int = 2
    multimodal: bool = False
    max_media_parts: int = 8
    system_template: str = ""
    temporal_query_mode: str = "disabled"
    temporal_default_timezone: str = "UTC"
    deterministic_slot_mode: str = "disabled"

    def __post_init__(self) -> None:
        if self.temporal_query_mode not in {"disabled", "interval_v1"}:
            raise ValueError(
                "temporal_query_mode must be disabled or interval_v1"
            )
        if self.deterministic_slot_mode not in {
            "disabled", "clarification_v1", "cockpit_v1",
        }:
            raise ValueError(
                "deterministic_slot_mode must be disabled, clarification_v1, "
                "or cockpit_v1"
            )

    @classmethod
    def from_env(cls) -> "AnswerConfig":
        return cls(
            base_url=os.environ["MEMEVAL_ANSWER_BASE_URL"].rstrip("/"),
            api_key=os.environ["MEMEVAL_ANSWER_API_KEY"],
            model=os.environ["MEMEVAL_ANSWER_MODEL"],
            max_tokens=int(os.getenv("MEMEVAL_ANSWER_MAX_TOKENS", "128")),
            timeout=int(os.getenv("MEMEVAL_ANSWER_TIMEOUT", "180")),
            multimodal=_as_bool(os.getenv("MEMEVAL_ANSWER_MULTIMODAL", "false")),
            max_media_parts=int(os.getenv("MEMEVAL_ANSWER_MAX_MEDIA_PARTS", "8")),
            system_template=os.getenv("MEMEVAL_ANSWER_SYSTEM_TEMPLATE", ""),
            temporal_query_mode=os.getenv(
                "MEMEVAL_ANSWER_TEMPORAL_QUERY_MODE",
                os.getenv("TDAI_EVAL_TEMPORAL_QUERY_MODE", "disabled"),
            ).strip().lower(),
            temporal_default_timezone=os.getenv(
                "MEMEVAL_ANSWER_TEMPORAL_DEFAULT_TIMEZONE",
                os.getenv("TDAI_EVAL_TEMPORAL_DEFAULT_TIMEZONE", "UTC"),
            ).strip() or "UTC",
            deterministic_slot_mode=os.getenv(
                "MEMEVAL_ANSWER_DETERMINISTIC_SLOT_MODE", "disabled"
            ).strip().lower(),
        )

    @classmethod
    def from_json(cls, path: Path, *, model_override: str | None = None) -> "AnswerConfig":
        raw = _load_config_data(path.resolve())
        section = _resolve_env(raw.get("answer_model") or {})
        prompts = _resolve_env(raw.get("prompts") or {})
        return cls(
            base_url=str(section["base_url"]).rstrip("/"),
            api_key=str(section.get("api_key") or ""),
            model=model_override or str(section["model"]),
            max_tokens=int(section.get("max_tokens", 128)),
            timeout=int(section.get("timeout_seconds", 180)),
            temperature=float(section.get("temperature", 0)),
            retries=int(section.get("retries", 2)),
            multimodal=_as_bool(section.get("multimodal", False)),
            max_media_parts=int(section.get("max_media_parts", 8)),
            system_template=str(prompts.get("answer_system") or ""),
            temporal_query_mode=str(
                section.get("temporal_query_mode")
                or os.getenv(
                    "MEMEVAL_ANSWER_TEMPORAL_QUERY_MODE",
                    os.getenv("TDAI_EVAL_TEMPORAL_QUERY_MODE", "disabled"),
                )
            ).strip().lower(),
            temporal_default_timezone=str(
                section.get("temporal_default_timezone")
                or os.getenv(
                    "MEMEVAL_ANSWER_TEMPORAL_DEFAULT_TIMEZONE",
                    os.getenv("TDAI_EVAL_TEMPORAL_DEFAULT_TIMEZONE", "UTC"),
                )
            ).strip() or "UTC",
            deterministic_slot_mode=str(
                section.get("deterministic_slot_mode")
                or os.getenv(
                    "MEMEVAL_ANSWER_DETERMINISTIC_SLOT_MODE", "disabled"
                )
            ).strip().lower(),
        )


class OpenAIAnswerer:
    def __init__(self, config: AnswerConfig) -> None:
        self.config = config

    def answer(self, question: str, context: str,
               parts: tuple[ContentPart, ...] = (),
               *, question_metadata: dict | None = None) -> tuple[str, dict]:
        body = self.build_payload(
            question, context, parts, question_metadata=question_metadata
        )
        answer, usage = self._complete(body)
        valid, reason = _validate_answer_contract(question, answer)
        if valid:
            return answer, usage
        # One bounded repair turn is cheaper and safer than accepting a
        # structurally incomplete two-date/cutoff answer.  Evidence is not
        # changed and no desired value is supplied to the model.
        body["messages"].extend([
            {"role": "assistant", "content": answer},
            {"role": "user", "content": f"上一答复违反回答契约（{reason}）。请仅依据同一证据重新给出满足契约的最短完整答案。"},
        ])
        repaired, repair_usage = self._complete(body)
        return repaired, _merge_usage(usage, repair_usage)

    def _complete(self, body: dict) -> tuple[str, dict]:
        request = urllib.request.Request(
            self.config.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.config.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    result = json.loads(response.read())
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        else:  # pragma: no cover - loop either breaks or raises
            raise RuntimeError(str(last_error))
        return str(result["choices"][0]["message"].get("content") or "").strip(), dict(result.get("usage", {}))

    def deterministic_answer(
        self,
        question: str,
        context: str,
        metadata: dict | None = None,
        retrieval_hits: list[dict] | None = None,
    ) -> SlotAnswerCandidate | None:
        """Return only a lossless, fully grounded answer candidate.

        Any unsupported shape or ambiguity deliberately falls through to the
        configured answer model. This keeps the fast path safe for mixed
        benchmarks and for weaker edge models.
        """
        if self.config.deterministic_slot_mode == "disabled":
            return None
        extractor = (
            extract_cockpit_answer
            if self.config.deterministic_slot_mode == "cockpit_v1"
            else extract_clarification_reply
        )
        if extractor is extract_cockpit_answer:
            return extractor(
                question,
                context,
                metadata,
                default_timezone=self.config.temporal_default_timezone,
                retrieval_hits=retrieval_hits or [],
            )
        return extractor(
            question,
            context,
            metadata,
            default_timezone=self.config.temporal_default_timezone,
        )

    def prepare_question(
        self,
        question: str,
        metadata: dict | None = None,
    ) -> tuple[str, dict]:
        """Add a deterministic time anchor without changing answer prompts."""
        started = time.monotonic()
        if self.config.temporal_query_mode == "disabled":
            temporal = TemporalQuery(
                None, self.config.temporal_default_timezone, ""
            )
        else:
            temporal = resolve_temporal_query(
                question,
                metadata,
                default_timezone=self.config.temporal_default_timezone,
            )
        prepared = temporal.answer_text(question)
        contract = _answer_contract(question)
        if contract and "[回答契约]" not in prepared:
            prepared = f"{prepared}\n\n[回答契约] {contract}"
        result = temporal.metadata()
        result.update({
            "mode": self.config.temporal_query_mode,
            "injected": prepared != question,
            "added_chars": len(prepared) - len(question),
            "normalization_seconds": time.monotonic() - started,
        })
        return prepared, result

    def build_payload(self, question: str, context: str,
                      parts: tuple[ContentPart, ...] = (),
                      *, question_metadata: dict | None = None) -> dict:
        """Build a request without performing I/O, enabling contract tests."""
        question, _temporal = self.prepare_question(question, question_metadata)
        default_template = (
            "Answer using only the supplied evidence. Return only the smallest complete answer, "
            "with no citations or explanation. Preserve exact names, quantities, relations, and "
            "temporal granularity. When evidence includes a message or session timestamp, resolve "
            "relative time expressions only when the conversion is unambiguous. If evidence is "
            "insufficient, say so.\n\nEvidence:\n{context}"
        )
        template = self.config.system_template or default_template
        try:
            system = template.format(context=context, question_date="")
        except (KeyError, ValueError):
            # A malformed user template must not silently drop evidence.
            system = default_template.format(context=context)
        user_content: str | list[dict] = question
        if self.config.multimodal:
            media = [part for part in parts if part.type == "image" and part.uri]
            if media:
                user_content = [{"type": "text", "text": question}]
                user_content.extend(
                    {"type": "image_url", "image_url": {"url": part.uri}}
                    for part in media[:self.config.max_media_parts]
                )
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "enable_thinking": False,
            # DeepSeek V4 ignores the legacy enable_thinking extension for
            # short answers unless the OpenAI-compatible thinking object is
            # also present. Without this field reasoning tokens can consume
            # the entire completion budget and leave message.content empty.
            "thinking": {"type": "disabled"},
        }


def _answer_contract(question: str) -> str:
    """Compile structural obligations without supplying answer values."""
    dates = list(dict.fromkeys(re.findall(r"\d{1,2}月\d{1,2}日", question)))
    if len(dates) >= 2 and re.search(r"分别|各是|各自", question):
        return "必须分别回答每个查询日期，答案中逐一保留对应日期；缺少任一日期的可靠证据就明确说明该日期证据不足。"
    if dates and re.search(r"截至|不晚于|及更早|之前的记录|只看", question):
        return f"只允许使用截止边界及更早证据；答案以“截至{dates[0]}”开头，不得用边界之后的信息补全。"
    people = _contract_people(question)
    if people and re.search(r"(?:各自|分别).*(?:偏好|常用|习惯|默认|目的[地的])", question):
        return (
            f"必须按问题顺序分别回答{ '、'.join(people) }，每个人只能绑定其本人证据；"
            "长期偏好不得被一次性播放、导航或临时设置覆盖；缺少任一人物证据就明确指出该人物证据不足。"
        )
    if (
        re.search(r"(?:最后一次|最近一次|最新一次)", question)
        and re.search(r"(?:前一次|上一次)", question)
    ):
        return (
            "必须按来源事件时间排序而非检索排名排序；先明确标注“最后一次”，再标注“前一次”，"
            "且只输出问题指定的字段；不足两个不同来源事件就回答证据不足。"
        )
    return ""


def _validate_answer_contract(question: str, answer: str) -> tuple[bool, str]:
    if not str(answer or "").strip():
        return False, "空回答"
    dates = list(dict.fromkeys(re.findall(r"\d{1,2}月\d{1,2}日", question)))
    if len(dates) >= 2 and re.search(r"分别|各是|各自", question):
        missing = [date for date in dates[:2] if date not in answer]
        if missing:
            return False, "缺少日期：" + "、".join(missing)
    if dates and re.search(r"截至|不晚于|及更早|之前的记录|只看", question):
        if not answer.startswith(f"截至{dates[0]}"):
            return False, "未按截止边界作答"
    people = _contract_people(question)
    if people and re.search(r"(?:各自|分别).*(?:偏好|常用|习惯|默认|目的[地的])", question):
        missing_people = [person for person in people if person not in answer]
        if missing_people:
            return False, "缺少人物：" + "、".join(missing_people)
        positions = [answer.index(person) for person in people]
        if positions != sorted(positions):
            return False, "人物输出顺序与问题不一致"
    if (
        re.search(r"(?:最后一次|最近一次|最新一次)", question)
        and re.search(r"(?:前一次|上一次)", question)
    ):
        final_marker = answer.find("最后一次")
        previous_markers = [
            position for marker in ("前一次", "上一次")
            for position in [answer.find(marker)] if position >= 0
        ]
        if final_marker < 0 or not previous_markers:
            return False, "缺少最后一次/前一次标签"
        if final_marker > min(previous_markers):
            return False, "最后一次/前一次输出顺序错误"
    return True, ""


def _contract_people(question: str) -> tuple[str, ...]:
    """Extract an explicitly separated person list without guessing names."""
    match = re.search(
        r"([^：:，,。；;？?]{1,80}?(?:、|和|与|及)[^：:，,。；;？?]{1,40})各自",
        str(question or ""),
    )
    if not match:
        return ()
    values = [
        value.strip()
        for value in re.split(r"、|和|与|及", match.group(1))
        if value.strip()
    ]
    return tuple(values) if 2 <= len(values) <= 8 else ()


def _merge_usage(first: dict, second: dict) -> dict:
    keys = set(first) | set(second)
    return {
        key: ((first.get(key, 0) or 0) + (second.get(key, 0) or 0))
        if isinstance(first.get(key, 0), (int, float)) and isinstance(second.get(key, 0), (int, float))
        else second.get(key, first.get(key))
        for key in keys
    }


def answer_retrieval_file(input_path: Path, output_path: Path, answerer: OpenAIAnswerer,
                          *, resume: bool = False, concurrency: int = 1) -> dict:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as existing:
            completed_ids = {str(json.loads(line).get("qa_id")) for line in existing if line.strip()}
    completed = 0
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    routes = {"deterministic_slot": 0, "model": 0}
    route_timings: dict[str, list[float]] = {
        "deterministic_slot": [], "model": [],
    }
    started = time.monotonic()
    mode = "a" if resume else "w"
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pending = [
        row for row in rows
        if str(row["question"]["question_id"]) not in completed_ids
    ]

    def generate(row: dict) -> tuple[dict, dict]:
        answer_started = time.monotonic()
        question = row["question"]
        qa_id = str(question["question_id"])
        prepared_question, temporal = answerer.prepare_question(
            str(question.get("text", "")),
            dict(question.get("metadata") or {}),
        )
        context = temporal_context(
            str(question.get("text", "")),
            str(row.get("context") or ""),
            dict(question.get("metadata") or {}),
        )
        candidate = answerer.deterministic_answer(
            str(question.get("text", "")),
            context,
            dict(question.get("metadata") or {}),
            list(row.get("hits") or []),
        )
        if candidate is not None:
            answer = candidate.value
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            answer_route = {
                "mode": answerer.config.deterministic_slot_mode,
                "route": "deterministic_slot",
                "model_called": False,
                "confidence": candidate.confidence,
                "reason": candidate.reason,
                "slot_label": candidate.slot_label,
                "command_anchor": candidate.command_anchor,
                "source_ids": list(candidate.source_ids),
                "uses_gold_or_evidence_ids": False,
            }
        else:
            answer, usage = answerer.answer(
                prepared_question,
                context,
                _parts_from_retrieval_row(row),
            )
            answer_route = {
                "mode": answerer.config.deterministic_slot_mode,
                "route": "model",
                "model_called": True,
                "reason": "no_unique_grounded_slot_candidate",
                "uses_gold_or_evidence_ids": False,
            }
        answer_route["seconds"] = time.monotonic() - answer_started
        return ({
            "schema_version": 1,
            "framework": row["framework"],
            "qa_id": qa_id,
            "conversation_id": question["conversation_id"],
            "category": question.get("category", ""),
            "gold_answers": question["answers"],
            "predicted_answer": answer,
            "retrieval_metrics": row["metrics"],
            "answer_temporal": temporal,
            "answer_route": answer_route,
            "usage": usage,
        }, usage)

    with output_path.open(mode, encoding="utf-8") as target:
        if concurrency == 1:
            generated = map(generate, pending)
            executor = None
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="memeval-answer",
            )
            # Executor.map preserves input order, making the artifact stable
            # across reruns while still allowing requests to overlap.
            generated = executor.map(generate, pending)
        try:
            for result, usage in generated:
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
                target.flush()
                for key in totals:
                    totals[key] += int(usage.get(key, 0))
                route = str(result["answer_route"]["route"])
                routes[route] = routes.get(route, 0) + 1
                route_timings.setdefault(route, []).append(float(
                    result["answer_route"].get("seconds", 0.0)
                ))
                completed += 1
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    return {
        "questions": completed,
        "skipped": len(rows) - len(pending),
        "concurrency": concurrency,
        "tokens": totals,
        "answer_routes": routes,
        "answer_route_latency": {
            route: _latency_summary(values)
            for route, values in route_timings.items()
        },
        "seconds": time.monotonic() - started,
        "output": str(output_path),
    }


def _resolve_env(value):
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", value)
    return os.environ.get(match.group(1), match.group(2) or "") if match else value


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _latency_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_seconds": None, "p50_seconds": None,
                "p95_seconds": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_seconds": sum(ordered) / len(ordered),
        "p50_seconds": ordered[int(0.50 * (len(ordered) - 1))],
        "p95_seconds": ordered[int(0.95 * (len(ordered) - 1))],
    }


def _parts_from_retrieval_row(row: dict) -> tuple[ContentPart, ...]:
    raw_parts = list((row.get("question") or {}).get("parts") or [])
    for hit in row.get("hits") or []:
        raw_parts.extend(hit.get("parts") or [])
    unique: list[ContentPart] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_parts:
        part = ContentPart.from_dict(raw)
        key = (part.type, part.uri, part.text)
        if key not in seen:
            unique.append(part)
            seen.add(key)
    return tuple(unique)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = (_deep_merge(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict) else value)
    return result


def _load_config_data(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    return _deep_merge(_load_config_data((path.parent / str(parent)).resolve()), raw) if parent else raw
