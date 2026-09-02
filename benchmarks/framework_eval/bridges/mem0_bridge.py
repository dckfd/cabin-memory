from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
STORE_ROOT = Path(
    os.environ.get(
        "MEMEVAL_STORE_ROOT",
        ROOT / "benchmarks/framework_eval_runs/stores/mem0",
    )
).resolve()
MODEL_ROOT = Path(
    os.environ.get(
        "MEM0_ONNX_MODEL_ROOT",
        ROOT
        / "benchmarks/framework_eval_runtimes/mempalace/models/all-MiniLM-L6-v2/onnx",
    )
).resolve()

# Provider endpoints used by the benchmark are directly reachable, while a
# desktop SOCKS proxy can make long-running extraction calls stall.  Keep this
# process deterministic and direct by default; deployments that require an
# environment proxy can opt in explicitly.
if os.getenv("MEM0_LLM_USE_ENV_PROXY", "false").lower() not in {"1", "true", "yes"}:
    for _proxy_key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        os.environ.pop(_proxy_key, None)
else:
    for _proxy_key in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(_proxy_key, "").startswith("socks://"):
            os.environ[_proxy_key] = "socks5://" + os.environ[_proxy_key][8:]

# Keep Mem0 telemetry and every cache/store artifact inside the benchmark run.
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_DIR", str(STORE_ROOT / "_mem0-global"))

from mem0 import Memory  # noqa: E402
from mem0.embeddings.base import EmbeddingBase  # noqa: E402
from mem0.utils.factory import EmbedderFactory  # noqa: E402


LOGGER = logging.getLogger("framework_eval.mem0_bridge")


class OnnxMiniLMEmbedding(EmbeddingBase):
    """Local all-MiniLM-L6-v2 embedder backed by the existing ONNX artifact.

    Mem0's normal local HuggingFace provider loads the same model through
    sentence-transformers/PyTorch.  This process bridge swaps only that runtime
    implementation for ONNX so the benchmark does not download a 500+ MB
    PyTorch wheel or spend provider tokens on embeddings.
    """

    dimensions = 384
    max_length = 256

    def __init__(self, config=None):
        super().__init__(config)
        required = ("model.onnx", "tokenizer.json")
        missing = [name for name in required if not (MODEL_ROOT / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"ONNX MiniLM model is incomplete at {MODEL_ROOT}: {', '.join(missing)}"
            )
        self.config.model = "all-MiniLM-L6-v2-onnx"
        self.config.embedding_dims = self.dimensions

    @cached_property
    def tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer.from_file(str(MODEL_ROOT / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=self.max_length)
        return tokenizer

    @cached_property
    def model(self) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("MEM0_ONNX_THREADS", "1"))
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(MODEL_ROOT / "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )

    def embed(self, text, memory_action=None):
        return self.embed_batch([str(text)], memory_action)[0]

    def embed_batch(self, texts, memory_action="add"):
        values = [str(value).replace("\n", " ") for value in texts]
        if not values:
            return []
        encoded = self.tokenizer.encode_batch(values)
        input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.asarray(
            [item.attention_mask for item in encoded], dtype=np.int64
        )
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        hidden = self.model.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0]
        expanded_mask = np.expand_dims(attention_mask, -1)
        pooled = np.sum(hidden * expanded_mask, axis=1) / np.clip(
            np.sum(expanded_mask, axis=1), 1e-9, None
        )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norms, 1e-12, None)
        return normalized.astype(np.float32).tolist()


# ``huggingface`` remains the public Mem0 provider in the serialized config.
# The replacement is process-local and preserves its local MiniLM semantics.
EmbedderFactory.provider_to_class["huggingface"] = (
    "benchmarks.framework_eval.bridges.mem0_bridge.OnnxMiniLMEmbedding"
)


def _safe_id(value: str) -> str:
    readable = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{readable[:64]}-{digest}"


def _store(conversation_id: str) -> Path:
    return STORE_ROOT / _safe_id(conversation_id)


@contextlib.contextmanager
def _store_lock(conversation_id: str):
    path = _store(conversation_id)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".bridge.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _llm_settings() -> tuple[str, str, str]:
    model = os.getenv("MEM0_LLM_MODEL", "qwen3.8-max")
    base_url = os.getenv("MEM0_LLM_BASE_URL") or os.getenv("TDAI_MEMORY_LLM_BASE_URL", "")
    api_key = os.getenv("MEM0_LLM_API_KEY") or os.getenv("TDAI_MEMORY_LLM_API_KEY", "")
    if not base_url:
        raise RuntimeError("MEM0_LLM_BASE_URL is required")
    if not api_key:
        raise RuntimeError("MEM0_LLM_API_KEY is required")
    return model, base_url, api_key


def _memory(conversation_id: str) -> Memory:
    model, base_url, api_key = _llm_settings()
    path = _store(conversation_id)
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "v1.1",
        "history_db_path": str(path / "history.sqlite3"),
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": "mem0",
                "path": str(path / "faiss"),
                "distance_strategy": "cosine",
                "embedding_model_dims": OnnxMiniLMEmbedding.dimensions,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "all-MiniLM-L6-v2",
                "embedding_dims": OnnxMiniLMEmbedding.dimensions,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "api_key": api_key,
                "openai_base_url": base_url,
                "temperature": 0.1,
                "top_p": 0.1,
                "max_tokens": int(os.getenv("MEM0_LLM_MAX_TOKENS", "4096")),
                "is_reasoning_model": False,
            },
        },
        "custom_instructions": (
            "Extract durable factual memories, events, preferences, relationships, "
            "plans, and outcomes with explicit actor names. Preserve source IDs. "
            "When a statement uses relative time, resolve it against the supplied "
            "session-time anchor when unambiguous and retain both the absolute date "
            "and original temporal wording. Keep distinct facts separate and do not "
            "invent details. These rules are domain- and dataset-independent."
        ),
    }
    return Memory.from_config(config)


def _read_state(conversation_id: str) -> dict[str, Any]:
    path = _store(conversation_id) / "bridge-state.json"
    if not path.exists():
        return {"completed_sessions": {}, "memory_events": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _prepare(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    path = _store(conversation_id)
    path.mkdir(parents=True, exist_ok=True)
    model, base_url, _ = _llm_settings()
    manifest = {
        "schema_version": 1,
        "framework": "mem0-oss",
        "framework_version": "2.0.17",
        "framework_commit": "4debc58a83377b18be81ae1e5969a300736b2fac",
        "conversation_id": conversation_id,
        "official_api": ["Memory.from_config", "Memory.add", "Memory.search"],
        "bridge": "benchmarks.framework_eval.bridges.mem0_bridge",
        "bridge_transformations": [
            "canonical role/speaker/source/time fields rendered into message text",
            "Mem0 search results converted to canonical MemoryHit rows",
            "existing all-MiniLM-L6-v2 ONNX runtime replaces PyTorch loading",
        ],
        "llm": {"provider": "openai-compatible", "model": model, "base_url": base_url},
        "embedding": {
            "provider_declared_to_mem0": "huggingface",
            "model": "all-MiniLM-L6-v2",
            "runtime": "onnxruntime-cpu",
            "dimensions": OnnxMiniLMEmbedding.dimensions,
            "local": True,
        },
        "vector_store": {"provider": "faiss", "distance": "cosine"},
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(path / "framework-eval-manifest.json", manifest)
    return {"ok": True, "store": str(path)}


def _message_for_mem0(message: dict, session: dict) -> dict[str, str]:
    source_id = str(message.get("message_id") or "")
    authored_at = str(message.get("timestamp") or session.get("timestamp") or "")
    speaker = str(message.get("speaker") or message.get("role") or "unknown")
    role = str(message.get("role") or "user")
    if role not in {"user", "assistant"}:
        role = "user"
    content = str(message.get("content") or "")
    anchor = f"[source_id={source_id}; session_time={authored_at}]"
    return {"role": role, "content": f"{anchor} {speaker}: {content}"}


def _ingest(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    session = dict(payload["session"])
    session_id = str(session["session_id"])
    messages = [
        _message_for_mem0(message, session)
        for message in (session.get("messages") or [])
        if str(message.get("content") or "").strip()
    ]
    source_ids = [
        str(message.get("message_id") or "")
        for message in (session.get("messages") or [])
        if str(message.get("message_id") or "")
    ]
    with _store_lock(conversation_id):
        state = _read_state(conversation_id)
        completed = state.setdefault("completed_sessions", {})
        if session_id in completed:
            return {
                "ok": True,
                "skipped": True,
                "session_id": session_id,
                "memory_events": completed[session_id].get("memory_events", 0),
            }
        memory = _memory(conversation_id)
        result = memory.add(
            messages,
            user_id=conversation_id,
            metadata={
                "conversation_id": conversation_id,
                "session_id": session_id,
                "authored_at": str(session.get("timestamp") or ""),
                "source_ids": source_ids,
            },
            infer=True,
        )
        rows = list(result.get("results") or [])
        completed[session_id] = {
            "message_count": len(messages),
            "memory_events": len(rows),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        state["memory_events"] = int(state.get("memory_events", 0)) + len(rows)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(_store(conversation_id) / "bridge-state.json", state)
        return {
            "ok": True,
            "session_id": session_id,
            "ingested": len(messages),
            "memory_events": len(rows),
            "store": str(_store(conversation_id)),
        }


def _finalize(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    with _store_lock(conversation_id):
        state = _read_state(conversation_id)
        memory = _memory(conversation_id)
        rows = memory.get_all(filters={"user_id": conversation_id}, top_k=10000)
        state["finalized_at"] = datetime.now(timezone.utc).isoformat()
        state["stored_memory_count"] = len(rows.get("results") or [])
        _write_json_atomic(_store(conversation_id) / "bridge-state.json", state)
    return {
        "ok": True,
        "stored_memory_count": state["stored_memory_count"],
        "completed_sessions": len(state.get("completed_sessions", {})),
    }


def _search(payload: dict) -> dict:
    question = dict(payload["question"])
    conversation_id = str(question["conversation_id"])
    with _store_lock(conversation_id):
        result = _memory(conversation_id).search(
            str(question["text"]),
            top_k=max(1, int(payload.get("limit", 8))),
            filters={"user_id": conversation_id},
            threshold=float(os.getenv("MEM0_SEARCH_THRESHOLD", "0.1")),
        )
    hits = []
    for row in result.get("results") or []:
        metadata = dict(row.get("metadata") or {})
        source_ids = metadata.get("source_ids") or []
        if isinstance(source_ids, str):
            try:
                source_ids = json.loads(source_ids)
            except json.JSONDecodeError:
                source_ids = [source_ids]
        hits.append(
            {
                "content": str(row.get("memory") or ""),
                "score": row.get("score"),
                "source_ids": [str(value) for value in source_ids if str(value)],
                "metadata": {
                    **metadata,
                    "mem0_memory_id": str(row.get("id") or ""),
                    "framework": "mem0-oss",
                },
            }
        )
    return {"hits": hits}


def _update(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    message = dict(payload["message"])
    with _store_lock(conversation_id):
        result = _memory(conversation_id).update(
            str(payload["memory_id"]),
            text=str(message.get("content") or ""),
            metadata={
                "source_ids": [str(message.get("message_id") or "")],
                "authored_at": str(message.get("timestamp") or ""),
            },
        )
    return {"ok": True, "result": result}


def _delete(payload: dict) -> dict:
    conversation_id = str(payload["conversation_id"])
    with _store_lock(conversation_id):
        result = _memory(conversation_id).delete(str(payload["memory_id"]))
    return {"ok": True, "result": result}


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
    if operation == "update":
        return _update(payload)
    if operation == "delete":
        return _delete(payload)
    if operation == "close":
        return {"ok": True}
    raise ValueError(f"unsupported operation: {operation}")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        response = dispatch(request)
    except Exception as exc:
        LOGGER.exception("Mem0 bridge operation failed")
        response = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
