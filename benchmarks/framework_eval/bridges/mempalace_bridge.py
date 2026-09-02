from __future__ import annotations

import hashlib
import json
import os
import sys
import contextlib
import fcntl
from datetime import datetime, timezone
from pathlib import Path

# Some desktop proxy tools export the ambiguous ``socks://`` scheme. HTTPX
# requires the concrete SOCKS version. Normalize only inside this isolated
# runtime; never mutate the user's shell configuration.
for _proxy_key in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(_proxy_key, "").startswith("socks://"):
        os.environ[_proxy_key] = "socks5://" + os.environ[_proxy_key][8:]

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from mempalace.config import DEFAULT_COLLECTION_NAME
import mempalace.palace as palace_module
from mempalace.palace import get_collection
from mempalace.searcher import search_memories


STORE_ROOT = Path(os.environ.get(
    "MEMEVAL_STORE_ROOT",
    "benchmarks/framework_eval_runs/stores/mempalace",
)).resolve()
MODEL_CACHE = Path(os.environ.get(
    "MEMEVAL_MODEL_CACHE",
    "benchmarks/framework_eval_runtimes/mempalace/models/all-MiniLM-L6-v2",
)).resolve()
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
# Chroma otherwise writes to ~/.cache. Keep every runtime artifact inside the
# project so the benchmark is portable and does not mutate the user's profile.
ONNXMiniLM_L6_V2.DOWNLOAD_PATH = str(MODEL_CACHE)


@contextlib.contextmanager
def _workspace_lock(palace_path: str):
    """MemPalace defaults locks to ~/.mempalace; isolate them per eval run."""
    lock_root = STORE_ROOT / "_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(palace_path).encode()).hexdigest()[:24]
    with (lock_root / f"{key}.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


palace_module.mine_palace_lock = _workspace_lock


def _palace(conversation_id: str) -> Path:
    digest = hashlib.sha256(conversation_id.encode()).hexdigest()[:16]
    return STORE_ROOT / f"{conversation_id.replace('/', '_')}-{digest}"


def _prepare(payload: dict) -> dict:
    path = _palace(str(payload["conversation_id"]))
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "framework": "mempalace",
        "conversation_id": payload["conversation_id"],
        "backend": "chroma",
        "embedding_model": os.getenv("MEMPALACE_EMBEDDING_MODEL", "minilm"),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    (path / "framework-eval-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True, "store": str(path)}


def _ingest(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    session = payload["session"]
    path = _palace(conversation_id)
    path.mkdir(parents=True, exist_ok=True)
    collection = get_collection(
        str(path), collection_name=DEFAULT_COLLECTION_NAME, create=True, backend="chroma"
    )
    documents, ids, metadatas = [], [], []
    for index, message in enumerate(session.get("messages") or []):
        source_id = str(message.get("message_id") or f"{session['session_id']}:{index}")
        speaker = str(message.get("speaker") or message.get("role") or "unknown")
        content = str(message.get("content") or "")
        documents.append(f"{speaker}: {content}")
        ids.append(source_id)
        metadatas.append({
            "wing": conversation_id,
            "room": str(session["session_id"]),
            "source_file": source_id,
            "chunk_index": index,
            "speaker": speaker,
            "session_id": str(session["session_id"]),
            "authored_at": str(message.get("timestamp") or session.get("timestamp") or ""),
            "filed_at": datetime.now(timezone.utc).isoformat(),
        })
    if documents:
        collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    return {"ok": True, "ingested": len(documents), "store": str(path)}


def _search(payload: dict) -> dict:
    question = payload["question"]
    conversation_id = str(question["conversation_id"])
    result = search_memories(
        query=str(question["text"]),
        palace_path=str(_palace(conversation_id)),
        n_results=max(1, int(payload.get("limit", 8))),
        candidate_strategy=os.getenv("MEMPALACE_CANDIDATE_STRATEGY", "vector"),
    )
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    hits = []
    for row in result.get("results") or []:
        source_id = str(row.get("source_path") or row.get("source_file") or "")
        session_id = str(row.get("room") or "")
        hits.append({
            "content": str(row.get("text") or ""),
            "score": row.get("similarity"),
            "source_ids": [value for value in (source_id, session_id) if value],
            "metadata": {
                key: value for key, value in row.items()
                if key not in {"text", "_sort_key", "_source_file_full"}
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
    if operation == "search":
        return _search(payload)
    if operation in {"finalize", "close"}:
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
