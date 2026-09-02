from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
STORE_ROOT = Path(
    os.environ.get(
        "MEMEVAL_STORE_ROOT",
        ROOT / "benchmarks/framework_eval_runs/locomo-half-v2/memos/stores",
    )
).resolve()
RUNTIME_ROOT = Path(
    os.environ.get(
        "MEMOS_RUNTIME_ROOT",
        ROOT / "benchmarks/framework_eval_runtimes/memos",
    )
).resolve()
CONFIG_PATH = Path(
    os.environ.get(
        "MEMOS_EVAL_CONFIG",
        ROOT / "benchmarks/production/locomo_config_qwen38.json",
    )
).resolve()
MEMOS_COMMIT = "8d310a7a4be6bbb9c04823a88f2ebaca6ae20baf"


def _safe_name(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "conversation"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _conversation_dir(conversation_id: str) -> Path:
    return STORE_ROOT / _safe_name(conversation_id)


def _manifest_path(conversation_id: str) -> Path:
    return _conversation_dir(conversation_id) / "framework-eval-manifest.json"


@contextlib.contextmanager
def _conversation_lock(conversation_id: str):
    lock_root = STORE_ROOT / "_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"{_safe_name(conversation_id)}.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_manifest(conversation_id: str) -> dict[str, Any]:
    path = _manifest_path(conversation_id)
    if not path.exists():
        raise RuntimeError(f"MemOS conversation was not prepared: {conversation_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", value)
    if not match:
        return value
    return os.environ.get(match.group(1), match.group(2) or "")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    if parent:
        raw = _deep_merge(_load_config((path.parent / str(parent)).resolve()), raw)
    return _resolve_env(raw)


def _speaker_specs(manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    return (
        {"key": "speaker_a", "name": str(manifest["speaker_a"])},
        {"key": "speaker_b", "name": str(manifest["speaker_b"])},
    )


def _perspective_messages(session: dict[str, Any], owner: str) -> list[dict[str, str]]:
    timestamp = str(session.get("timestamp") or "")
    result = []
    for message in session.get("messages") or []:
        speaker = str(message.get("speaker") or message.get("role") or "unknown")
        message_timestamp = str(message.get("timestamp") or timestamp)
        source_id = str(message.get("message_id") or "")
        content = str(message.get("content") or "")
        # This mirrors MemOS's official LoCoMo ingestion: each participant owns
        # one cube, their messages are `user`, and the peer's are `assistant`.
        # Time and source id are placed in-band so extracted relative events
        # remain resolvable by the downstream answer model.
        rendered = f"[{message_timestamp}; source={source_id}] {speaker}: {content}"
        result.append({
            "role": "user" if speaker == owner else "assistant",
            "content": rendered,
        })
    return result


def _memory(conversation_id: str, speaker_key: str):
    # Desktop proxy tools sometimes export the non-standard `socks://` scheme,
    # which causes httpx clients to fail at construction. HTTP(S)_PROXY remains
    # available; remove only the ambiguous all-proxy inside this child process.
    for proxy_key in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(proxy_key, "").startswith("socks://"):
            os.environ.pop(proxy_key, None)
    os.environ.setdefault("MEMOS_BASE_PATH", str(RUNTIME_ROOT / "state"))
    os.environ.setdefault("MOS_EMBEDDER_TIMEOUT", "120")

    from memos.configs.memory import MemoryConfigFactory
    from memos.memories.factory import MemoryFactory

    cfg = _load_config()
    llm = cfg["memory_model"]
    embedding = cfg["embedding"]
    expected_model = os.getenv("MEMOS_LLM_MODEL", "qwen3.8-max")
    model = str(os.getenv("MEMOS_LLM_MODEL") or llm.get("model") or "")
    if model != expected_model:
        raise RuntimeError(f"MemOS memory model must be {expected_model}, got {model or '<empty>'}")
    vector_dimension = int(embedding.get("dimensions") or 1024)
    qdrant_path = _conversation_dir(conversation_id) / speaker_key / "qdrant"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    config = MemoryConfigFactory(
        backend="general_text",
        config={
            "extractor_llm": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": model,
                    "api_key": str(os.getenv("MEMOS_LLM_API_KEY") or llm.get("api_key") or ""),
                    "api_base": str(
                        os.getenv("MEMOS_LLM_BASE_URL") or llm.get("base_url") or ""
                    ),
                    "temperature": 0.0,
                    "max_tokens": int(os.getenv("MEMOS_LLM_MAX_TOKENS", "4096")),
                    "top_p": 1.0,
                    "remove_think_prefix": True,
                    # Structured extraction needs the JSON answer, not a hidden
                    # reasoning trace. TokenPlan/qwen3.8 accepts this standard
                    # Qwen control and it avoids spending most completion tokens
                    # on reasoning before the schema is emitted.
                    "extra_body": {"enable_thinking": False},
                },
            },
            "vector_db": {
                "backend": "qdrant",
                "config": {
                    "path": str(qdrant_path),
                    "collection_name": "locomo_textual_memory",
                    "distance_metric": "cosine",
                    "vector_dimension": vector_dimension,
                },
            },
            "embedder": {
                "backend": "universal_api",
                "config": {
                    "provider": "openai",
                    "model_name_or_path": str(embedding["model"]),
                    "embedding_dims": vector_dimension,
                    "api_key": str(
                        os.getenv("MEMOS_EMBEDDING_API_KEY")
                        or embedding.get("api_key")
                        or ""
                    ),
                    "base_url": str(
                        os.getenv("MEMOS_EMBEDDING_BASE_URL")
                        or embedding.get("base_url")
                        or ""
                    ),
                    "max_tokens": 8192,
                },
            },
        },
    )
    return MemoryFactory.from_config(config)


def _prepare(payload: dict[str, Any]) -> dict[str, Any]:
    conversation_id = str(payload["conversation_id"])
    metadata = payload.get("metadata") or {}
    speaker_a = str(metadata.get("speaker_a") or "speaker_a")
    speaker_b = str(metadata.get("speaker_b") or "speaker_b")
    with _conversation_lock(conversation_id):
        path = _manifest_path(conversation_id)
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (manifest.get("speaker_a"), manifest.get("speaker_b")) != (speaker_a, speaker_b):
                raise RuntimeError("existing MemOS store has incompatible speakers")
        else:
            cfg = _load_config()
            manifest = {
                "framework": "memos",
                "framework_version": "2.0.27",
                "framework_commit": MEMOS_COMMIT,
                "backend": "general_text+embedded_qdrant+universal_api",
                "conversation_id": conversation_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "memory_model": str(cfg["memory_model"].get("model") or ""),
                "memory_model_thinking": False,
                "embedding_model": str(cfg["embedding"].get("model") or ""),
                "vector_dimension": int(cfg["embedding"].get("dimensions") or 1024),
                "completed_perspective_sessions": [],
                "memory_counts": {},
                "ready": False,
                "prepared_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(path, manifest)
    return {"ok": True, "store": str(_conversation_dir(conversation_id))}


def _ingest(payload: dict[str, Any]) -> dict[str, Any]:
    conversation_id = str(payload["conversation_id"])
    session = payload["session"]
    session_id = str(session["session_id"])
    source_ids = [
        str(message.get("message_id") or "")
        for message in (session.get("messages") or [])
        if str(message.get("message_id") or "")
    ]
    added = 0
    with _conversation_lock(conversation_id):
        manifest = _read_manifest(conversation_id)
        completed = set(manifest.get("completed_perspective_sessions") or [])
        for spec in _speaker_specs(manifest):
            completion_key = f"{session_id}:{spec['key']}"
            if completion_key in completed:
                continue
            memory = _memory(conversation_id, spec["key"])
            messages = _perspective_messages(session, spec["name"])
            extracted = memory.extract(messages)
            enriched = []
            for item in extracted:
                value = item.model_dump()
                metadata = dict(value.get("metadata") or {})
                metadata.update({
                    "user_id": spec["name"],
                    "session_id": session_id,
                    "source": "conversation",
                    "info": {
                        "source_ids": source_ids,
                        "timestamp": str(session.get("timestamp") or ""),
                        "perspective": spec["key"],
                    },
                })
                value["metadata"] = metadata
                enriched.append(value)
            if enriched:
                memory.add(enriched)
            added += len(enriched)
            completed.add(completion_key)
            counts = dict(manifest.get("memory_counts") or {})
            counts[spec["key"]] = int(counts.get(spec["key"], 0)) + len(enriched)
            manifest["memory_counts"] = counts
            manifest["completed_perspective_sessions"] = sorted(completed)
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(_manifest_path(conversation_id), manifest)
    return {"ok": True, "ingested": added, "session_id": session_id}


def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
    conversation_id = str(payload["conversation_id"])
    with _conversation_lock(conversation_id):
        manifest = _read_manifest(conversation_id)
        manifest["ready"] = True
        manifest["finalized_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(_manifest_path(conversation_id), manifest)
    return {"ok": True}


def _search(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload["question"]
    conversation_id = str(question["conversation_id"])
    query = str(question["text"])
    limit = max(1, int(payload.get("limit", 8)))
    manifest = _read_manifest(conversation_id)
    candidates = []
    for spec in _speaker_specs(manifest):
        memory = _memory(conversation_id, spec["key"])
        query_vector = memory.embedder.embed([query])[0]
        for result in memory.vector_db.search(query_vector, limit):
            item = dict(result.payload or {})
            metadata = dict(item.get("metadata") or {})
            info = dict(metadata.get("info") or {})
            candidates.append({
                "content": f"[Memory owner: {spec['name']}] {item.get('memory', '')}",
                "score": float(result.score) if result.score is not None else None,
                "source_ids": [str(value) for value in info.get("source_ids") or []],
                "metadata": {
                    "speaker": spec["name"],
                    "perspective": spec["key"],
                    "session_id": metadata.get("session_id"),
                    "timestamp": info.get("timestamp", ""),
                    "memory_id": item.get("id", ""),
                    "lineage_scope": "source-session",
                },
            })
    candidates.sort(key=lambda row: row.get("score") or float("-inf"), reverse=True)
    seen = set()
    hits = []
    for candidate in candidates:
        key = candidate["content"].casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        hits.append(candidate)
        if len(hits) >= limit:
            break
    return {"hits": hits}


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
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
    if operation == "close":
        return {"ok": True}
    raise ValueError(f"unsupported operation: {operation}")


def main() -> int:
    protocol_stdout = sys.stdout
    try:
        # MemOS and its providers may emit diagnostics during import. The JSON
        # process protocol reserves stdout for exactly one response object.
        sys.stdout = sys.stderr
        request = json.loads(sys.stdin.read())
        response = dispatch(request)
    except Exception as exc:
        response = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.stdout = protocol_stdout
    protocol_stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
