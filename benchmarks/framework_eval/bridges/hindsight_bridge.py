from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = os.getenv("MEMEVAL_HINDSIGHT_URL", "http://127.0.0.1:18888").rstrip("/")
RUN_ID = os.getenv("MEMEVAL_HINDSIGHT_RUN_ID", "locomo-half-v2")
RUN_ROOT = Path(os.getenv(
    "MEMEVAL_HINDSIGHT_RUN_ROOT",
    "benchmarks/framework_eval_runs/locomo-half-v2/hindsight",
)).resolve()
HTTP_TIMEOUT = int(os.getenv("MEMEVAL_HINDSIGHT_HTTP_TIMEOUT", "1800"))
RECALL_BUDGET = os.getenv("MEMEVAL_HINDSIGHT_RECALL_BUDGET", "high")
RECALL_MAX_TOKENS = int(os.getenv("MEMEVAL_HINDSIGHT_RECALL_MAX_TOKENS", "4096"))


def _safe_id(value: str) -> str:
    prefix = "".join(char if char.isalnum() or char == "-" else "-" for char in value)
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{prefix[:48]}-{digest}"


def _bank_id(conversation_id: str) -> str:
    return _safe_id(f"memeval-{RUN_ID}-{conversation_id}")


def _conversation_root(conversation_id: str) -> Path:
    return RUN_ROOT / "lineage" / _safe_id(conversation_id)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "memeval-hindsight/1"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"Hindsight HTTP {exc.code} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hindsight request failed {path}: {exc}") from exc
    return json.loads(raw) if raw else {}


def _iso_timestamp(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    for pattern in ("%I:%M %p on %d %B, %Y", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = datetime.strptime(value, pattern)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.isoformat()
    except ValueError:
        return None


def _session_item(conversation_id: str, session: dict) -> tuple[dict, dict]:
    messages = list(session.get("messages") or [])
    # Match Hindsight's official LoCoMo runner: one JSON-encoded document per
    # session, preserving speaker attribution and the session event timestamp.
    content = json.dumps([
        {
            "speaker": str(message.get("speaker") or message.get("role") or "unknown"),
            "text": str(message.get("content") or ""),
        }
        for message in messages
    ], ensure_ascii=False)
    speakers = list(dict.fromkeys(
        str(message.get("speaker") or "") for message in messages
        if str(message.get("speaker") or "")
    ))
    session_id = str(session["session_id"])
    source_ids = [
        str(message.get("message_id")) for message in messages
        if str(message.get("message_id") or "")
    ]
    timestamp = _iso_timestamp(str(session.get("timestamp") or ""))
    item = {
        "content": content,
        "context": (
            f"Conversation between {' and '.join(speakers[:2])} "
            f"({session_id} of {conversation_id})"
        ),
        "document_id": session_id,
        "metadata": {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "source_ids": json.dumps(source_ids, ensure_ascii=False),
            "source_timestamp": str(session.get("timestamp") or ""),
        },
    }
    if timestamp:
        item["timestamp"] = timestamp
    lineage = {
        "document_id": session_id,
        "session_id": session_id,
        "source_ids": source_ids,
        "timestamp": str(session.get("timestamp") or ""),
    }
    return item, lineage


def _load_lineage(conversation_id: str) -> dict[str, dict]:
    path = _conversation_root(conversation_id) / "documents.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["document_id"]): row for row in rows}


def _prepare(payload: dict) -> dict:
    _request("GET", "/health")
    conversation_id = str(payload["conversation_id"])
    root = _conversation_root(conversation_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pending-items.json").write_text("[]\n", encoding="utf-8")
    manifest = {
        "framework": "hindsight",
        "track": "unified-retrieval",
        "official_strategy": "one JSON document per LoCoMo session",
        "official_entrypoint": (
            "third_party/memory_frameworks/hindsight/"
            "hindsight-dev/benchmarks/locomo/locomo_benchmark.py"
        ),
        "conversation_id": conversation_id,
        "bank_id": _bank_id(conversation_id),
        "server": BASE_URL,
        "readiness_policy": "wait for all bank operations to complete",
        "recall_budget": RECALL_BUDGET,
        "recall_max_tokens": RECALL_MAX_TOKENS,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True, "bank_id": manifest["bank_id"], "store": str(root)}


def _ingest(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    root = _conversation_root(conversation_id)
    root.mkdir(parents=True, exist_ok=True)
    item, lineage = _session_item(conversation_id, payload["session"])
    pending_path = root / "pending-items.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8")) if pending_path.exists() else []
    pending.append({"item": item, "lineage": lineage})
    pending_path.write_text(
        json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True, "queued": 1, "document_id": item["document_id"]}


def _existing_document_ids(conversation_id: str) -> set[str]:
    """List retained documents so a failed synchronous batch can be resumed.

    Hindsight commits each document as it completes. A later provider failure
    can therefore leave a valid prefix in the bank even though the batch HTTP
    request itself failed. Query all pages and submit only missing document IDs
    on the next finalize call.
    """
    bank = urllib.parse.quote(_bank_id(conversation_id), safe="")
    found: set[str] = set()
    offset = 0
    page_size = 100
    while True:
        query = urllib.parse.urlencode({"limit": page_size, "offset": offset})
        result = _request("GET", f"/v1/default/banks/{bank}/documents?{query}")
        rows = list(result.get("items") or [])
        found.update(str(row["id"]) for row in rows if str(row.get("id") or ""))
        offset += len(rows)
        total = int(result.get("total") or len(rows))
        if not rows or offset >= total:
            return found


def _missing_pending_rows(pending: list[dict], existing_ids: set[str]) -> list[dict]:
    return [
        row for row in pending
        if str(row.get("item", {}).get("document_id") or "") not in existing_ids
    ]


def _operation_groups(operations: list[dict]) -> tuple[list[dict], list[dict]]:
    terminal_success = {"completed"}
    terminal_failure = {"failed", "error", "cancelled", "canceled"}
    failed = [
        row for row in operations
        if str(row.get("status") or "").lower() in terminal_failure
    ]
    active = [
        row for row in operations
        if str(row.get("status") or "").lower()
        not in terminal_success | terminal_failure
    ]
    return active, failed


def _operations(conversation_id: str) -> list[dict]:
    bank = urllib.parse.quote(_bank_id(conversation_id), safe="")
    rows: list[dict] = []
    offset = 0
    page_size = 100
    while True:
        query = urllib.parse.urlencode({"limit": page_size, "offset": offset})
        result = _request("GET", f"/v1/default/banks/{bank}/operations?{query}")
        page = list(result.get("operations") or [])
        rows.extend(page)
        offset += len(page)
        total = int(result.get("total") or len(page))
        if not page or offset >= total:
            return rows


def _wait_until_ready(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    timeout = max(0.0, float(payload.get("timeout") or 0.0))
    poll_seconds = max(
        0.1, float(os.getenv("MEMEVAL_HINDSIGHT_READY_POLL_SECONDS", "5"))
    )
    started = time.monotonic()
    while True:
        operations = _operations(conversation_id)
        active, failed = _operation_groups(operations)
        if failed:
            sample = failed[0]
            raise RuntimeError(
                "Hindsight operation failed: "
                f"id={sample.get('id')} task={sample.get('task_type')} "
                f"error={sample.get('error_message') or sample.get('status')}"
            )
        if not active:
            return {
                "ok": True,
                "operations": len(operations),
                "wait_seconds": time.monotonic() - started,
            }
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            summary = [
                {
                    "id": row.get("id"),
                    "task_type": row.get("task_type"),
                    "status": row.get("status"),
                    "progress": row.get("progress"),
                }
                for row in active[:5]
            ]
            raise TimeoutError(
                f"Hindsight did not become ready within {timeout}s: {summary}"
            )
        time.sleep(min(poll_seconds, max(0.1, timeout - elapsed)))


def _finalize(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    root = _conversation_root(conversation_id)
    pending_path = root / "pending-items.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    existing_ids = _existing_document_ids(conversation_id)
    missing = _missing_pending_rows(pending, existing_ids)
    items = [row["item"] for row in missing]
    if items:
        result = _request(
            "POST",
            f"/v1/default/banks/{urllib.parse.quote(_bank_id(conversation_id), safe='')}/memories",
            {"items": items, "async": False},
        )
        if not result.get("success"):
            raise RuntimeError(f"Hindsight retain did not succeed: {result}")
    else:
        result = {"success": True, "skipped_existing": len(pending)}
    lineage = [row["lineage"] for row in pending]
    (root / "documents.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "retain-result.json").write_text(
        json.dumps({
            "pending_document_count": len(pending),
            "existing_document_count": len(existing_ids),
            "submitted_document_count": len(items),
            "response": result,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "ok": True,
        "ingested": len(items),
        "reused": len(pending) - len(items),
        "bank_id": _bank_id(conversation_id),
    }


def _result_score(row: dict) -> float | None:
    scores = row.get("scores")
    if isinstance(scores, dict) and scores.get("final") is not None:
        return float(scores["final"])
    return None


def _source_ids(row: dict, lineage: dict[str, dict]) -> list[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    encoded = metadata.get("source_ids")
    if isinstance(encoded, str):
        try:
            values = json.loads(encoded)
            if isinstance(values, list):
                return [str(value) for value in values]
        except json.JSONDecodeError:
            pass
    document_id = str(row.get("document_id") or "")
    return [str(value) for value in lineage.get(document_id, {}).get("source_ids", [])]


def _search(payload: dict) -> dict:
    question = payload["question"]
    conversation_id = str(question["conversation_id"])
    body = {
        "query": str(question["text"]),
        # Omit ``types`` exactly as Hindsight's official LoCoMo runner does;
        # the server default is world + experience (observations excluded).
        "budget": RECALL_BUDGET,
        "max_tokens": RECALL_MAX_TOKENS,
        "trace": False,
    }
    result = _request(
        "POST",
        f"/v1/default/banks/{urllib.parse.quote(_bank_id(conversation_id), safe='')}/memories/recall",
        body,
    )
    lineage = _load_lineage(conversation_id)
    hits = []
    for row in list(result.get("results") or [])[:max(1, int(payload.get("limit", 8)))]:
        document_id = str(row.get("document_id") or "")
        document = lineage.get(document_id, {})
        source_ids = _source_ids(row, lineage)
        if document_id and document_id not in source_ids:
            source_ids.append(document_id)
        hits.append({
            "content": str(row.get("text") or ""),
            "score": _result_score(row),
            "source_ids": source_ids,
            "metadata": {
                "framework_memory_id": str(row.get("id") or ""),
                "memory_type": row.get("type"),
                "document_id": document_id,
                "session_id": document.get("session_id", document_id),
                "timestamp": (
                    row.get("occurred_start")
                    or document.get("timestamp")
                    or row.get("mentioned_at")
                    or ""
                ),
                "occurred_end": row.get("occurred_end"),
                "entities": row.get("entities") or [],
                "scores": row.get("scores") or {},
                "lineage_granularity": "session",
            },
        })
    return {"hits": hits}


def dispatch(request: dict) -> dict:
    operation = request.get("operation")
    payload = request.get("payload") or {}
    if operation == "prepare":
        return _prepare(payload)
    if operation == "ingest_session":
        return _ingest(payload)
    if operation == "finalize":
        return _finalize(payload)
    if operation == "search":
        return _search(payload)
    if operation == "wait_until_ready":
        return _wait_until_ready(payload)
    if operation == "close":
        return {"ok": True}
    raise ValueError(f"unsupported operation: {operation}")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        response = dispatch(request)
    except Exception as exc:
        response = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
