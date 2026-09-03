from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, unquote

from .adaptive_policy import (
    AdaptiveDecision,
    decide_l0_fast_path,
    decide_l0_fast_path_v2,
    score_l0_candidate,
)
from .base import MemoryAdapter
from .construction_policy import (
    all_sessions_policy,
    classify_session,
    classify_session_v2,
)
from ..cockpit_episode import (
    EpisodeTurn,
    compile_navigation_episode,
    episode_from_dict,
)
from ..cockpit_slots import extract_cockpit_answer
from ..schema import ContentPart, Conversation, MemoryHit, Question, Session
from ..structured_state import resolve_state_answer
from ..temporal import (
    TemporalQuery,
    humanize_temporal_span,
    parse_temporal_timestamp,
    resolve_temporal_query,
)
from ..typed_episode_index import (
    SQLiteTypedEpisodeIndex,
    TypedEpisodeIndexError,
    TypedEpisodeRecord,
    TypedEpisodeScope,
)


@dataclass(frozen=True)
class _V3Perspective:
    """One human participant's isolated view of a multi-party dialogue."""

    speaker: str
    team_id: str
    agent_id: str
    user_id: str
    task_id: str

    def isolation(self) -> tuple[str, str, str, str]:
        return self.team_id, self.agent_id, self.user_id, self.task_id


@dataclass(frozen=True)
class _V3L0Message:
    """One canonical raw message loaded from the v3 conversation store."""

    backend_id: str
    session_id: str
    role: str
    content: str
    source_ids: tuple[str, ...]
    source_timestamp: str
    backend_recorded_at: str


@dataclass(frozen=True)
class _V3L0History:
    """Chronological L0 sessions plus constant-time anchor lookups."""

    sessions: dict[str, tuple[_V3L0Message, ...]]
    by_backend_id: dict[str, tuple[str, int]]
    by_source_id: dict[str, tuple[str, int]]
    source_session_provenance: bool


class TencentDBHTTPAdapter(MemoryAdapter):
    adapter_id = "tencentdb_http"
    capabilities = frozenset({"ingest", "search", "wait_until_ready"})

    def __init__(self, base_url: str, *, api_key: str = "", timeout: int = 600,
                 adapter_id: str = "tencentdb") -> None:
        self.adapter_id = adapter_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # The v2 productized Gateway uses strict v3 tenancy fields. Keep the
        # legacy v1/v2 contract as the default for existing local tests, while
        # allowing the same adapter to evaluate the official v2 service with
        # TDAI_HTTP_API_VERSION=v3.
        self.api_version = os.getenv("TDAI_HTTP_API_VERSION", "v2").lower()
        self.team_id = os.getenv("TDAI_EVAL_TEAM_ID", "default")
        self.agent_id = os.getenv("TDAI_EVAL_AGENT_ID", "default")
        self.user_id = os.getenv("TDAI_EVAL_USER_ID", "default")
        self.task_id = os.getenv("TDAI_EVAL_TASK_ID", "")
        self.service_id = os.getenv("TDAI_EVAL_SERVICE_ID", "default")
        # v3 metadata CRUD has an additional end-user auth layer. Keep this
        # credential in the environment; never persist it in an isolation map.
        self.user_key = os.getenv("TDAI_EVAL_USER_KEY", "")
        user_key_file = os.getenv("TDAI_EVAL_USER_KEY_FILE", "").strip()
        if not self.user_key and user_key_file:
            self.user_key = Path(user_key_file).read_text(encoding="utf-8").strip()
        mapping_path = os.getenv("TDAI_EVAL_ISOLATION_MAP", "")
        self.mapping_path = Path(mapping_path).resolve() if mapping_path else None
        self.perspective_mode = os.getenv(
            "TDAI_EVAL_PERSPECTIVE_MODE", "auto"
        ).strip().lower()
        if self.perspective_mode not in {"auto", "multi", "single"}:
            raise ValueError(
                "TDAI_EVAL_PERSPECTIVE_MODE must be auto, multi, or single"
            )
        requested_layers = {
            item.strip().upper()
            for item in os.getenv(
                "TDAI_EVAL_MEMORY_LAYERS", "L0,L1"
            ).split(",")
            if item.strip()
        }
        invalid_layers = requested_layers - {"L0", "L1", "L2", "L3"}
        if invalid_layers or not requested_layers:
            raise ValueError(
                "TDAI_EVAL_MEMORY_LAYERS must contain one or more of L0,L1,L2,L3"
            )
        self.memory_layers = frozenset(requested_layers)
        self.retrieval_policy = os.getenv(
            "TDAI_EVAL_RETRIEVAL_POLICY", "fixed"
        ).strip().lower()
        if self.retrieval_policy not in {"fixed", "adaptive"}:
            raise ValueError(
                "TDAI_EVAL_RETRIEVAL_POLICY must be fixed or adaptive"
            )
        if self.retrieval_policy == "adaptive" and "L0" not in self.memory_layers:
            raise ValueError("adaptive retrieval requires the L0 memory layer")
        self.adaptive_fast_l0_k = max(
            1, int(os.getenv("TDAI_EVAL_ADAPTIVE_FAST_L0_K", "1"))
        )
        self.adaptive_fallback_l0_k = max(
            self.adaptive_fast_l0_k,
            int(os.getenv("TDAI_EVAL_ADAPTIVE_FALLBACK_L0_K", "3")),
        )
        self.adaptive_fallback_l1_k = max(
            0, int(os.getenv("TDAI_EVAL_ADAPTIVE_FALLBACK_L1_K", "2"))
        )
        self.adaptive_fallback_l2_k = max(
            0, int(os.getenv("TDAI_EVAL_ADAPTIVE_FALLBACK_L2_K", "0"))
        )
        self.adaptive_fallback_l3_k = max(
            0, int(os.getenv("TDAI_EVAL_ADAPTIVE_FALLBACK_L3_K", "0"))
        )
        self.adaptive_min_coverage = min(1.0, max(
            0.0, float(os.getenv(
                "TDAI_EVAL_ADAPTIVE_MIN_COVERAGE", "0.45"
            )),
        ))
        self.adaptive_min_score_margin = max(
            0.0, float(os.getenv(
                "TDAI_EVAL_ADAPTIVE_MIN_SCORE_MARGIN", "0.15"
            )),
        )
        self.adaptive_policy_version = os.getenv(
            "TDAI_EVAL_ADAPTIVE_POLICY_VERSION", "v1"
        ).strip().lower()
        if self.adaptive_policy_version not in {"v1", "v2"}:
            raise ValueError(
                "TDAI_EVAL_ADAPTIVE_POLICY_VERSION must be v1 or v2"
            )
        self.adaptive_slot_rerank = os.getenv(
            "TDAI_EVAL_ADAPTIVE_SLOT_RERANK",
            "true" if self.adaptive_policy_version == "v2" else "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.adaptive_fast_context_chars = max(
            256, int(os.getenv(
                "TDAI_EVAL_ADAPTIVE_FAST_CONTEXT_CHARS", "900"
            )),
        )
        self.adaptive_fallback_context_chars = max(
            self.adaptive_fast_context_chars,
            int(os.getenv(
                "TDAI_EVAL_ADAPTIVE_FALLBACK_CONTEXT_CHARS", "2200"
            )),
        )
        self.adaptive_context_line_overflow_chars = max(
            0,
            int(os.getenv(
                "TDAI_EVAL_ADAPTIVE_CONTEXT_LINE_OVERFLOW_CHARS", "0"
            )),
        )
        # L0 is written synchronously, while L1/L2/L3 may still be building.
        # In an adaptive route, waiting for every layer before looking at L0
        # defeats the fast path.  Opt-in lazy readiness defers those waits
        # until the route actually needs a higher layer.
        self.adaptive_lazy_layer_readiness = os.getenv(
            "TDAI_EVAL_ADAPTIVE_LAZY_LAYER_READINESS", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.adaptive_layer_wait_budget_seconds = max(
            0.0,
            float(os.getenv(
                "TDAI_EVAL_ADAPTIVE_LAYER_WAIT_BUDGET_SECONDS", "0"
            )),
        )
        self.l1_write_policy = os.getenv(
            "TDAI_EVAL_L1_WRITE_POLICY", "all"
        ).strip().lower()
        if self.l1_write_policy not in {
            "all", "cockpit_selective_v1", "cockpit_episode_v2",
        }:
            raise ValueError(
                "TDAI_EVAL_L1_WRITE_POLICY must be all, "
                "cockpit_selective_v1, or cockpit_episode_v2"
            )
        self.l1_batch_mode = os.getenv(
            "TDAI_EVAL_L1_BATCH_MODE", "session"
        ).strip().lower()
        if self.l1_batch_mode not in {"session", "conversation"}:
            raise ValueError(
                "TDAI_EVAL_L1_BATCH_MODE must be session or conversation"
            )
        if (
            self.l1_batch_mode == "conversation"
            and self.l1_write_policy not in {
                "cockpit_selective_v1", "cockpit_episode_v2",
            }
        ):
            raise ValueError(
                "conversation L1 batching requires a selective construction policy"
            )
        self.l1_batch_max_messages = max(
            1, int(os.getenv("TDAI_EVAL_L1_BATCH_MAX_MESSAGES", "128"))
        )
        self.l0_batch_mode = os.getenv(
            "TDAI_EVAL_L0_BATCH_MODE", "session"
        ).strip().lower()
        if self.l0_batch_mode not in {"session", "conversation"}:
            raise ValueError(
                "TDAI_EVAL_L0_BATCH_MODE must be session or conversation"
            )
        # MemoryCore's public v3 contract rejects arrays above 100 items.
        # Clamp locally so a profile cannot create a partially flushed run.
        self.l0_batch_max_messages = min(100, max(
            1, int(os.getenv("TDAI_EVAL_L0_BATCH_MAX_MESSAGES", "100"))
        ))
        self.l0_flush_concurrency = max(
            1, int(os.getenv("TDAI_EVAL_L0_FLUSH_CONCURRENCY", "1"))
        )
        self.l1_compact_selected_sessions = os.getenv(
            "TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.l1_compact_selected_sessions and self.l1_batch_mode != "conversation":
            raise ValueError(
                "selected-session L1 compaction requires conversation batching"
            )
        self.l1_typed_episode_headers = os.getenv(
            "TDAI_EVAL_L1_TYPED_EPISODE_HEADERS",
            "true" if self.l1_write_policy == "cockpit_episode_v2" else "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.l1_typed_episode_headers and not self.l1_compact_selected_sessions:
            raise ValueError(
                "typed L1 episode headers require selected-session compaction"
            )
        self.typed_cockpit_episodes = os.getenv(
            "TDAI_EVAL_TYPED_COCKPIT_EPISODES",
            "true" if self.l1_write_policy == "cockpit_episode_v2" else "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.typed_episode_short_circuit = os.getenv(
            "TDAI_EVAL_TYPED_EPISODE_SHORT_CIRCUIT", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.typed_episode_short_circuit and not self.typed_cockpit_episodes:
            raise ValueError(
                "typed episode short circuit requires typed cockpit episodes"
            )
        self.typed_slot_min_confidence = min(1.0, max(
            0.0,
            float(os.getenv(
                "TDAI_EVAL_TYPED_SLOT_MIN_CONFIDENCE", "0.85"
            )),
        ))
        self.typed_slot_min_margin = min(1.0, max(
            0.0,
            float(os.getenv("TDAI_EVAL_TYPED_SLOT_MIN_MARGIN", "0.08")),
        ))
        self.l1_zero_output_retries = max(
            0, int(os.getenv("TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES", "0"))
        )
        if self.l1_zero_output_retries and self.l1_batch_mode != "conversation":
            raise ValueError(
                "L1 zero-output repair requires conversation batching so an "
                "empty result can be attributed to one isolated profile scope"
            )
        self.l1_zero_output_terminal = os.getenv(
            "TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL", "fail"
        ).strip().lower()
        if self.l1_zero_output_terminal not in {"fail", "l0_only"}:
            raise ValueError(
                "TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL must be fail or l0_only"
            )
        self.l1_zero_output_wait_for_repair = os.getenv(
            "TDAI_EVAL_L1_ZERO_OUTPUT_WAIT_FOR_REPAIR", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        trace_path = os.getenv("TDAI_EVAL_CONSTRUCTION_TRACE", "").strip()
        self.construction_trace_path = (
            Path(trace_path).resolve() if trace_path else None
        )
        episode_index_path = os.getenv(
            "TDAI_EVAL_TYPED_EPISODE_INDEX_PATH", ""
        ).strip()
        self.typed_episode_index_path = (
            Path(episode_index_path).resolve() if episode_index_path else None
        )
        self.typed_episode_index_required = os.getenv(
            "TDAI_EVAL_TYPED_EPISODE_INDEX_REQUIRED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.typed_episode_index_ttl_seconds = max(
            0.0,
            float(os.getenv(
                "TDAI_EVAL_TYPED_EPISODE_INDEX_TTL_SECONDS", "0"
            )),
        )
        self.typed_episode_index_busy_timeout_ms = max(
            1,
            int(os.getenv(
                "TDAI_EVAL_TYPED_EPISODE_INDEX_BUSY_TIMEOUT_MS", "5000"
            )),
        )
        if self.typed_episode_index_required and self.typed_episode_index_path is None:
            raise ValueError(
                "required typed episode index needs "
                "TDAI_EVAL_TYPED_EPISODE_INDEX_PATH"
            )
        if self.typed_episode_index_path and not self.typed_cockpit_episodes:
            raise ValueError(
                "typed episode index requires typed cockpit episodes"
            )
        self._typed_episode_index: SQLiteTypedEpisodeIndex | None = None
        self._typed_episode_index_failures = 0
        self._typed_episode_index_writes = 0
        self._typed_episode_index_reads = 0
        self._typed_episode_index_migrations = 0
        self._typed_episode_index_commits = 0
        self._typed_episode_index_last_error = ""
        if self.typed_episode_index_path is not None:
            try:
                self._typed_episode_index = SQLiteTypedEpisodeIndex(
                    self.typed_episode_index_path,
                    busy_timeout_ms=self.typed_episode_index_busy_timeout_ms,
                )
            except TypedEpisodeIndexError as exc:
                self._typed_episode_index_failures += 1
                self._typed_episode_index_last_error = type(exc).__name__
                if self.typed_episode_index_required:
                    raise RuntimeError(
                        "required typed episode index is unavailable"
                    ) from exc
        self.l23_schedule = os.getenv(
            "TDAI_EVAL_L23_SCHEDULE", "disabled"
        ).strip().lower()
        if self.l23_schedule not in {"disabled", "buffered_dirty_event"}:
            raise ValueError(
                "TDAI_EVAL_L23_SCHEDULE must be disabled or "
                "buffered_dirty_event"
            )
        self.l23_readiness_mode = os.getenv(
            "TDAI_EVAL_L23_READINESS_MODE", "required"
        ).strip().lower()
        if self.l23_readiness_mode not in {"required", "dirty_only"}:
            raise ValueError(
                "TDAI_EVAL_L23_READINESS_MODE must be required or dirty_only"
            )
        self._construction_counts: Counter[str] = Counter()
        self._construction_action_counts: Counter[str] = Counter()
        self._construction_lifecycle_counts: Counter[str] = Counter()
        self._construction_episode_counts: Counter[str] = Counter()
        self._construction_typed_episode_counts: Counter[str] = Counter()
        self._construction_sessions = 0
        self._construction_extracted = 0
        self._construction_suppressed = 0
        self._construction_messages = 0
        self._construction_source_messages = 0
        self._construction_characters = 0
        self._construction_selected_characters = 0
        self._construction_compacted_sessions = 0
        self._construction_trace_loaded = False
        self._construction_trace_valid = False
        self._construction_trace_scopes: set[tuple[str, str]] = set()
        self._construction_dirty_scopes: set[tuple[str, str]] = set()
        self._construction_trace_source_sessions: dict[
            tuple[str, str], set[str]
        ] = {}
        self._construction_trace_transport_sessions: dict[
            tuple[str, str], set[str]
        ] = {}
        self._construction_source_session_by_id: dict[
            tuple[str, str], dict[str, str]
        ] = {}
        self._construction_trace_episodes_by_source_id: dict[
            tuple[str, str], dict[str, dict]
        ] = {}
        self._construction_trace_typed_episode_rows: dict[
            tuple[str, str], tuple[dict, ...]
        ] = {}
        self._construction_trace_sessions_complete = False
        self._pending_typed_episode_commits: dict[
            str, dict[tuple[str, str, str, str, str], dict]
        ] = {}
        self._l0_pending_requests: dict[str, list[dict]] = {}
        self._l0_conversation_batches: dict[tuple[str, ...], dict] = {}
        self._l1_conversation_batches: dict[tuple[str, ...], dict] = {}
        self._construction_l0_batches = 0
        self._construction_l0_batched_sessions = 0
        self._construction_l0_concurrent_requests = 0
        self._construction_l1_batches = 0
        self._construction_l1_repair_attempts = 0
        self._construction_l1_repaired_scopes = 0
        self._construction_l1_empty_scopes: set[tuple[str, str]] = set()
        self.l2_results = max(0, int(os.getenv("TDAI_EVAL_L2_RESULTS", "2")))
        self.l3_results = max(0, int(os.getenv("TDAI_EVAL_L3_RESULTS", "2")))
        self.l2_max_chars = max(
            256, int(os.getenv("TDAI_EVAL_L2_MAX_CHARS", "3000"))
        )
        self.l3_max_chars = max(
            256, int(os.getenv("TDAI_EVAL_L3_MAX_CHARS", "2500"))
        )
        # A semantic L0 hit often lands on one side of a dialogue pair: the
        # question but not its answer, or vice versa. Optionally expand an L0
        # anchor with chronological neighbours from the *same backend
        # session*. This is adapter-side retrieval post-processing; it does
        # not depend on benchmark evidence labels or alter MemoryCore.
        radius = max(0, int(os.getenv("TDAI_EVAL_L0_WINDOW_RADIUS", "0")))
        self.l0_window_before = max(
            0, int(os.getenv("TDAI_EVAL_L0_WINDOW_BEFORE", str(radius)))
        )
        self.l0_window_after = max(
            0, int(os.getenv("TDAI_EVAL_L0_WINDOW_AFTER", str(radius)))
        )
        self.l0_window_max_messages = max(
            0, int(os.getenv("TDAI_EVAL_L0_WINDOW_MAX_MESSAGES", "0"))
        )
        # Retrieve a wider backend candidate pool before applying the common
        # top-k. A minimum L0 quota can then prevent high-level summaries from
        # crowding all raw dialogue anchors out of the final context.
        self.candidate_multiplier = max(
            1, int(os.getenv("TDAI_EVAL_CANDIDATE_MULTIPLIER", "1"))
        )
        self.l0_min_results = max(
            0, int(os.getenv("TDAI_EVAL_L0_MIN_RESULTS", "0"))
        )
        self.l0_min_fraction = min(1.0, max(
            0.0, float(os.getenv("TDAI_EVAL_L0_MIN_FRACTION", "0"))
        ))
        self.l0_session_bm25_results = max(
            0, int(os.getenv("TDAI_EVAL_L0_SESSION_BM25_RESULTS", "0"))
        )
        self.l0_session_bm25_weight = max(
            0.0, float(os.getenv("TDAI_EVAL_L0_SESSION_BM25_WEIGHT", "2"))
        )
        self.l0_explicit_date_boost = os.getenv(
            "TDAI_EVAL_L0_EXPLICIT_DATE_BOOST", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.l0_explicit_date_results = max(
            0, int(os.getenv("TDAI_EVAL_L0_EXPLICIT_DATE_RESULTS", "0"))
        )
        self.temporal_query_mode = os.getenv(
            "TDAI_EVAL_TEMPORAL_QUERY_MODE", "disabled"
        ).strip().lower()
        if self.temporal_query_mode not in {"disabled", "interval_v1"}:
            raise ValueError(
                "TDAI_EVAL_TEMPORAL_QUERY_MODE must be disabled or interval_v1"
            )
        self.temporal_query_results = max(
            0, int(os.getenv("TDAI_EVAL_TEMPORAL_QUERY_RESULTS", "2"))
        )
        self.temporal_default_timezone = os.getenv(
            "TDAI_EVAL_TEMPORAL_DEFAULT_TIMEZONE", "UTC"
        ).strip() or "UTC"
        self.temporal_event_time_match = os.getenv(
            "TDAI_EVAL_TEMPORAL_EVENT_TIME_MATCH", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.temporal_short_circuit = os.getenv(
            "TDAI_EVAL_TEMPORAL_SHORT_CIRCUIT", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.structured_chain_retrieval = os.getenv(
            "TDAI_EVAL_STRUCTURED_CHAIN_RETRIEVAL", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.l0_diversify_sessions = os.getenv(
            "TDAI_EVAL_L0_DIVERSIFY_SESSIONS", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.l0_diverse_results = max(
            0, int(os.getenv("TDAI_EVAL_L0_DIVERSE_RESULTS", "0"))
        )
        if self.l0_diversify_sessions and self.l0_diverse_results == 0:
            self.l0_diverse_results = self.l0_min_results
        self.l0_first = os.getenv(
            "TDAI_EVAL_L0_FIRST", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.l0_last = os.getenv(
            "TDAI_EVAL_L0_LAST", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.l0_first and self.l0_last:
            raise ValueError(
                "TDAI_EVAL_L0_FIRST and TDAI_EVAL_L0_LAST are mutually exclusive"
            )
        self.l0_mark_anchors = os.getenv(
            "TDAI_EVAL_L0_MARK_ANCHORS", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.l0_resolve_relative_time = os.getenv(
            "TDAI_EVAL_L0_RESOLVE_RELATIVE_TIME", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.l0_humanize_time = os.getenv(
            "TDAI_EVAL_L0_HUMANIZE_TIME", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.l0_focus_anchors = max(
            0, int(os.getenv("TDAI_EVAL_L0_FOCUS_ANCHORS", "0"))
        )
        self.l0_reasoning_focus_anchors = max(
            self.l0_focus_anchors,
            int(os.getenv(
                "TDAI_EVAL_L0_REASONING_FOCUS_ANCHORS",
                str(self.l0_focus_anchors),
            )),
        )
        self.l0_focus_modes = frozenset(
            item.strip().lower()
            for item in os.getenv("TDAI_EVAL_L0_FOCUS_MODE", "all").split(",")
            if item.strip()
        )
        invalid_focus_modes = self.l0_focus_modes - {
            "all", "aggregate", "when", "emotion", "inference",
        }
        if invalid_focus_modes:
            raise ValueError(
                "TDAI_EVAL_L0_FOCUS_MODE supports all, aggregate, when, "
                "emotion, inference"
            )
        self.l0_focus_before = max(
            0, int(os.getenv("TDAI_EVAL_L0_FOCUS_BEFORE", "1"))
        )
        self.l0_focus_after = max(
            0, int(os.getenv("TDAI_EVAL_L0_FOCUS_AFTER", "1"))
        )
        self.l0_focus_response_radius = max(
            0, int(os.getenv("TDAI_EVAL_L0_FOCUS_RESPONSE_RADIUS", "2"))
        )
        self.l0_focus_max_chars = max(
            512, int(os.getenv("TDAI_EVAL_L0_FOCUS_MAX_CHARS", "6000"))
        )
        # Optional construction-time, source-grounded sidecar.  It is built
        # once from conversations without benchmark questions and provides a
        # compact exact-fact view (including secondary clauses that L1 may
        # intentionally filter).  Keeping it in the adapter avoids changing
        # MemoryCore's extraction or retrieval prompts.
        ledger_path = os.getenv("TDAI_EVAL_LEDGER_PATH", "").strip()
        self.ledger_path = Path(ledger_path).resolve() if ledger_path else None
        self.ledger_fact_results = max(
            0, int(os.getenv("TDAI_EVAL_LEDGER_FACT_RESULTS", "0"))
        )
        self.ledger_include_rollups = os.getenv(
            "TDAI_EVAL_LEDGER_INCLUDE_ROLLUPS", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.ledger_max_chars = max(
            512, int(os.getenv("TDAI_EVAL_LEDGER_MAX_CHARS", "3000"))
        )
        self.ingest_resolve_relative_time = os.getenv(
            "TDAI_EVAL_INGEST_RESOLVE_RELATIVE_TIME", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._perspectives: dict[str, tuple[_V3Perspective, ...]] = {}
        self._profile_cache: dict[str, tuple[dict, ...]] = {}
        self._l0_history_cache: dict[str, _V3L0History] = {}
        self._ledger_cache: dict[str, tuple[dict, ...]] = {}
        self._adaptive_ready_deadlines: dict[str, float] = {}
        self.isolation_map = {}
        if mapping_path:
            with open(mapping_path, encoding="utf-8") as handle:
                mapping = json.load(handle)
            # Run-level tenancy fields live at the top of the isolation
            # manifest; only agent/task IDs vary by conversation. Honor those
            # defaults unless the caller explicitly overrode them via env.
            if "TDAI_EVAL_TEAM_ID" not in os.environ:
                self.team_id = str(mapping.get("team_id", self.team_id))
            if "TDAI_EVAL_USER_ID" not in os.environ:
                self.user_id = str(mapping.get("user_id", self.user_id))
            if "TDAI_EVAL_SERVICE_ID" not in os.environ:
                self.service_id = str(mapping.get("service_id", self.service_id))
            self.isolation_map = mapping.get("conversations", {})

    def _construction_decision(self, session: Session):
        if self.l1_write_policy == "all":
            return all_sessions_policy(session)
        if self.l1_write_policy == "cockpit_episode_v2":
            return classify_session_v2(session)
        return classify_session(session)

    def _isolation(self, conversation_id: str) -> tuple[str, str, str, str]:
        row = self.isolation_map.get(str(conversation_id), {})
        return (
            str(row.get("team_id", self.team_id)),
            str(row.get("agent_id", self.agent_id)),
            str(row.get("user_id", self.user_id)),
            str(row.get("task_id", self.task_id)),
        )

    def _manifest_perspectives(
        self, conversation_id: str
    ) -> tuple[_V3Perspective, ...]:
        row = self.isolation_map.get(str(conversation_id), {})
        raw = row.get("perspectives") or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"perspectives for {conversation_id} must be an object keyed by speaker"
            )
        perspectives: list[_V3Perspective] = []
        for speaker, scope in raw.items():
            if not isinstance(scope, dict):
                raise ValueError(
                    f"perspective {speaker!r} for {conversation_id} must be an object"
                )
            team_id, agent_id, user_id, base_task_id = self._isolation(conversation_id)
            task_id = str(scope.get("task_id", base_task_id))
            if not task_id:
                raise ValueError(
                    f"perspective {speaker!r} for {conversation_id} has no task_id"
                )
            perspectives.append(_V3Perspective(
                speaker=str(speaker),
                team_id=str(scope.get("team_id", team_id)),
                agent_id=str(scope.get("agent_id", agent_id)),
                user_id=str(scope.get("user_id", user_id)),
                task_id=task_id,
            ))
        task_ids = [perspective.task_id for perspective in perspectives]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                f"perspectives for {conversation_id} must use distinct task_ids"
            )
        agent_ids = [perspective.agent_id for perspective in perspectives]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError(
                f"perspectives for {conversation_id} must use distinct agent_ids"
            )
        return tuple(perspectives)

    def _perspective_isolations(
        self, conversation_id: str
    ) -> tuple[_V3Perspective, ...]:
        key = str(conversation_id)
        if key in self._perspectives:
            return self._perspectives[key]
        if self.perspective_mode == "single":
            team_id, agent_id, user_id, task_id = self._isolation(key)
            default = (_V3Perspective("", team_id, agent_id, user_id, task_id),)
            self._perspectives[key] = default
            return default
        configured = self._manifest_perspectives(key)
        if configured:
            self._perspectives[key] = configured
            return configured
        team_id, agent_id, user_id, task_id = self._isolation(key)
        default = (_V3Perspective("", team_id, agent_id, user_id, task_id),)
        self._perspectives[key] = default
        return default

    def _create_perspective_agent(
        self,
        conversation_id: str,
        speaker: str,
        *,
        team_id: str,
        user_id: str,
    ) -> str:
        marker = hashlib.sha256(
            f"{conversation_id}\x1f{speaker}\x1fagent".encode("utf-8")
        ).hexdigest()[:10]
        body = {
            "team_id": team_id,
            "name": f"framework-eval {conversation_id} {speaker} {marker}",
            "description": (
                "Isolated full-memory perspective agent; "
                f"conversation={conversation_id}; owner={speaker}; marker={marker}"
            ),
            "owner_user_id": user_id,
            "visibility": "restricted",
        }
        endpoint = (
            "/v3/meta/agent/create" if self.user_key else "/v2/agent/create"
        )
        response = self._post(endpoint, body)
        if response.get("code") not in (None, 0):
            raise RuntimeError(
                f"failed to create TencentDB perspective agent for {speaker}: {response}"
            )
        agent_id = str((response.get("data") or {}).get("agent_id", ""))
        if not agent_id:
            raise RuntimeError(
                f"TencentDB agent/create returned no agent_id for {speaker}: {response}"
            )
        return agent_id

    def _create_perspective_task(
        self,
        conversation_id: str,
        speaker: str,
        *,
        team_id: str,
        agent_id: str,
        user_id: str,
    ) -> str:
        marker = hashlib.sha256(
            f"{conversation_id}\x1f{speaker}".encode("utf-8")
        ).hexdigest()[:10]
        common = {
            "team_id": team_id,
            "creator_user_id": user_id,
            "title": f"framework-eval {conversation_id} {speaker} {marker}",
            "description": (
                "Framework-eval isolated human-memory perspective; "
                f"conversation={conversation_id}; owner={speaker}; marker={marker}"
            ),
            "source_type": "other",
        }
        if self.user_key:
            endpoint = "/v3/meta/task/create"
            body = {**common, "linked_agents": [{
                "agent_id": agent_id,
                "role_in_task": "memory-perspective",
            }]}
        else:
            # The compatibility management route uses the same backing task
            # store and Bearer/service authentication but does not require an
            # end-user key. v3 remains the data-plane protocol for ingest and
            # search after this one provisioning call.
            endpoint = "/v2/task/create"
            body = {**common, "agent_ids": [agent_id]}
        response = self._post(endpoint, body)
        if response.get("code") not in (None, 0):
            raise RuntimeError(
                f"failed to create TencentDB perspective task for {speaker}: {response}"
            )
        task_id = str((response.get("data") or {}).get("task_id", ""))
        if not task_id:
            raise RuntimeError(
                f"TencentDB task/create returned no task_id for {speaker}: {response}"
            )
        return task_id

    def _persist_perspectives(
        self,
        conversation_id: str,
        perspectives: tuple[_V3Perspective, ...],
    ) -> None:
        """Persist generated task IDs so --skip-ingest reuses the same store."""
        if self.mapping_path is None:
            return
        with self.mapping_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        conversations = manifest.setdefault("conversations", {})
        row = conversations.setdefault(str(conversation_id), {})
        row["perspectives"] = {
            perspective.speaker: {
                "team_id": perspective.team_id,
                "agent_id": perspective.agent_id,
                "user_id": perspective.user_id,
                "task_id": perspective.task_id,
            }
            for perspective in perspectives
        }
        temporary = self.mapping_path.with_name(self.mapping_path.name + ".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.mapping_path)
        self.isolation_map = conversations

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.api_version == "v3":
            headers["x-tdai-service-id"] = self.service_id
        if self.user_key:
            headers["x-tdai-user-key"] = self.user_key
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"TencentDB request {path} failed: HTTP {exc.code}: {detail}"
            ) from exc

    def prepare(self, conversation: Conversation) -> None:
        if self.api_version != "v3" or self.perspective_mode == "single":
            return None

        conversation_id = str(conversation.conversation_id)
        configured = self._manifest_perspectives(conversation_id)
        metadata_speakers = [
            str(conversation.metadata.get(key) or "").strip()
            for key in ("speaker_a", "speaker_b")
        ]
        speakers = list(dict.fromkeys(item for item in metadata_speakers if item))
        if len(speakers) < 2 and self.perspective_mode == "multi":
            speakers = list(dict.fromkeys(
                str(message.speaker or "").strip()
                for session in conversation.sessions
                for message in session.messages
                if str(message.speaker or "").strip()
            ))
        # Auto mode is deliberately conservative. Named speaker_a/speaker_b
        # identifies person-to-person benchmarks such as LoCoMo; ordinary
        # user-to-Agent datasets keep their original single namespace.
        if len(speakers) < 2:
            return None

        configured_by_speaker = {
            perspective.speaker: perspective for perspective in configured
        }
        team_id, base_agent_id, user_id, base_task_id = self._isolation(
            conversation_id
        )
        perspectives: list[_V3Perspective] = []
        created = False
        for index, speaker in enumerate(speakers):
            if speaker in configured_by_speaker:
                perspectives.append(configured_by_speaker[speaker])
                continue
            use_base_agent = index == 0 and bool(base_agent_id)
            perspective_agent_id = base_agent_id if use_base_agent else ""
            if not perspective_agent_id:
                perspective_agent_id = self._create_perspective_agent(
                    conversation_id,
                    speaker,
                    team_id=team_id,
                    user_id=user_id,
                )
                created = True
            task_id = (
                base_task_id
                if index == 0 and use_base_agent and base_task_id
                else ""
            )
            if not task_id:
                task_id = self._create_perspective_task(
                    conversation_id,
                    speaker,
                    team_id=team_id,
                    agent_id=perspective_agent_id,
                    user_id=user_id,
                )
                created = True
            perspectives.append(_V3Perspective(
                speaker=speaker,
                team_id=team_id,
                agent_id=perspective_agent_id,
                user_id=user_id,
                task_id=task_id,
            ))

        result = tuple(perspectives)
        task_ids = [perspective.task_id for perspective in result]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                f"human perspectives for {conversation_id} must use distinct task_ids"
            )
        agent_ids = [perspective.agent_id for perspective in result]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError(
                f"human perspectives for {conversation_id} must use distinct agent_ids"
            )
        self._perspectives[conversation_id] = result
        # Persist not only newly created IDs but also the first speaker's base
        # task assignment. This makes a later --skip-ingest retrieval process
        # reconstruct exactly the same pair of namespaces.
        if created or not configured:
            self._persist_perspectives(conversation_id, result)
        return None

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        if self.api_version == "v3":
            decision = self._construction_decision(session)
            for perspective in self._perspective_isolations(conversation_id):
                team_id, agent_id, user_id, task_id = perspective.isolation()
                service_session_id = _scoped_v3_session_id(
                    session.session_id,
                    conversation_id=conversation_id,
                    team_id=team_id,
                    agent_id=agent_id,
                    user_id=user_id,
                    task_id=task_id,
                )
                messages = []
                for message in session.messages:
                    if not message.content:
                        continue
                    timestamp = _iso_timestamp(message.timestamp)
                    source_role = _perspective_role(message, perspective.speaker)
                    content = _format_v3_content(
                        message, source_role=source_role
                    )
                    if self.ingest_resolve_relative_time:
                        content = _annotate_v3_relative_time(content, timestamp)
                    # The public v3 endpoint currently notifies L1 only when a
                    # role=user round is present. For the adapter experiment,
                    # demote non-selected user rounds to assistant while retaining
                    # the original speaker/role in the reversible content
                    # envelope. The public v3 L0 schema accepts only user and
                    # assistant; an assistant-only batch writes exact L0 but
                    # contributes zero user rounds to the L1 notifier.
                    pipeline_role = _v3_transport_role(
                        source_role,
                        extract_l1=decision.extract_l1,
                    )
                    messages.append({
                        "role": pipeline_role,
                        # The envelope is benchmark provenance, not a QA hint:
                        # every source turn gets the same source id, source
                        # time, and original display speaker.
                        "content": content,
                        **({"timestamp": timestamp} if timestamp else {}),
                    })
                transport_session_id = service_session_id
                batch_selected = (
                    self.l1_batch_mode == "conversation"
                    and decision.extract_l1
                    and any(message["role"] == "user" for message in messages)
                )
                batch_l0 = (
                    self.l0_batch_mode == "conversation"
                    and not batch_selected
                    and not any(
                        message["role"] == "user" for message in messages
                    )
                )
                buffer_l0 = (
                    self.l0_flush_concurrency > 1
                    and not batch_l0
                    and not batch_selected
                    and not any(
                        message["role"] == "user" for message in messages
                    )
                )
                payload = {
                    "team_id": team_id,
                    "agent_id": agent_id,
                    "user_id": user_id,
                    **({"task_id": task_id} if task_id else {}),
                }
                if batch_selected:
                    # Fragmented cockpit commands are independent source
                    # sessions but share one long-lived memory subject. Buffer
                    # selected sessions until finalize so a single transport
                    # cursor can extract them in one or a few large L1 calls;
                    # L2's timer is then coalesced on that same cursor.
                    transport_session_id = _scoped_v3_session_id(
                        "selective-l1-conversation-batch",
                        conversation_id=conversation_id,
                        team_id=team_id,
                        agent_id=agent_id,
                        user_id=user_id,
                        task_id=task_id,
                    )
                    batch_key = (
                        str(conversation_id), team_id, agent_id, user_id, task_id
                    )
                    batch = self._l1_conversation_batches.setdefault(
                        batch_key,
                        {
                            "payload": payload,
                            "session_id": transport_session_id,
                            "messages": [],
                        },
                    )
                    if self.l1_compact_selected_sessions:
                        pipeline_messages = [_compact_v3_l1_session(
                            messages,
                            decision=(
                                decision if self.l1_typed_episode_headers else None
                            ),
                        )]
                        self._construction_compacted_sessions += 1
                    else:
                        pipeline_messages = messages
                    batch["messages"].extend(pipeline_messages)
                elif batch_l0:
                    # Clean/transient source sessions never notify L1 because
                    # every pipeline role is assistant. They can therefore
                    # share a transport cursor without changing extraction.
                    # Durable trace source IDs restore each original session
                    # boundary during retrieval.
                    transport_session_id = _scoped_v3_session_id(
                        "lossless-l0-conversation-batch",
                        conversation_id=conversation_id,
                        team_id=team_id,
                        agent_id=agent_id,
                        user_id=user_id,
                        task_id=task_id,
                    )
                    batch_key = (
                        str(conversation_id), team_id, agent_id, user_id, task_id
                    )
                    batch = self._l0_conversation_batches.setdefault(
                        batch_key,
                        {
                            "payload": payload,
                            "session_id": transport_session_id,
                            "messages": [],
                        },
                    )
                    pipeline_messages = messages
                    batch["messages"].extend(pipeline_messages)
                    self._construction_l0_batched_sessions += 1
                elif buffer_l0:
                    pipeline_messages = messages
                    self._l0_pending_requests.setdefault(
                        str(conversation_id), []
                    ).append({
                        **payload,
                        "session_id": service_session_id,
                        "messages": pipeline_messages,
                    })
                    self._construction_l0_concurrent_requests += 1
                else:
                    pipeline_messages = messages
                    self._post("/v3/conversation/add", {
                        **payload,
                        # MemoryCore's asynchronous cursor is keyed by this
                        # transport session before the L0 query applies tenant
                        # isolation. Task scoping prevents cross-run mixing.
                        "session_id": service_session_id,
                        "messages": pipeline_messages,
                    })
                self._record_construction_decision(
                    conversation_id,
                    session,
                    perspective=perspective,
                    decision=decision,
                    pipeline_messages=pipeline_messages,
                    transport_session_id=transport_session_id,
                    l1_compacted=(
                        batch_selected and self.l1_compact_selected_sessions
                    ),
                    l0_batched=batch_l0,
                    l0_buffered=buffer_l0,
                    source_deferred=(
                        batch_selected or batch_l0 or buffer_l0
                    ),
                )
            return
        event_id = f"framework-eval:{conversation_id}:{session.session_id}"
        self._post("/capture", {
            "event_id": event_id,
            "session_key": conversation_id,
            "session_id": session.session_id,
            "messages": [message.to_dict() for message in session.messages],
        })

    def finalize(self, conversation_id: str) -> None:
        if self.api_version != "v3":
            self._post("/session/end", {"session_key": conversation_id})
            return
        pending_l0 = self._l0_pending_requests.pop(
            str(conversation_id), []
        )
        if pending_l0:
            with ThreadPoolExecutor(
                max_workers=min(self.l0_flush_concurrency, len(pending_l0))
            ) as executor:
                futures = [
                    executor.submit(self._post, "/v3/conversation/add", payload)
                    for payload in pending_l0
                ]
                # Resolve every future so request failures invalidate this
                # fresh namespace instead of silently producing partial L0.
                for future in futures:
                    future.result()
        matching_l0 = [
            key for key in self._l0_conversation_batches
            if key[0] == str(conversation_id)
        ]
        for key in matching_l0:
            batch = self._l0_conversation_batches.pop(key)
            messages = list(batch["messages"])
            for offset in range(0, len(messages), self.l0_batch_max_messages):
                chunk = messages[offset:offset + self.l0_batch_max_messages]
                if not chunk:
                    continue
                self._post("/v3/conversation/add", {
                    **batch["payload"],
                    "session_id": batch["session_id"],
                    "messages": chunk,
                })
                self._construction_l0_batches += 1
        matching = [
            key for key in self._l1_conversation_batches
            if key[0] == str(conversation_id)
        ]
        for key in matching:
            batch = self._l1_conversation_batches.pop(key)
            messages = list(batch["messages"])
            for offset in range(0, len(messages), self.l1_batch_max_messages):
                chunk = messages[offset:offset + self.l1_batch_max_messages]
                if not chunk:
                    continue
                self._post("/v3/conversation/add", {
                    **batch["payload"],
                    "session_id": batch["session_id"],
                    "messages": chunk,
                })
                self._construction_l1_batches += 1
        # A typed episode becomes readable only after every deferred L0/L1
        # transport write above has been acknowledged. If a flush raises, the
        # pending index rows remain inactive and the trace has no commit marker.
        self._commit_deferred_typed_episodes(str(conversation_id))

    def _record_construction_decision(
        self,
        conversation_id: str,
        session: Session,
        *,
        perspective: _V3Perspective,
        decision,
        pipeline_messages: list[dict],
        transport_session_id: str,
        l1_compacted: bool = False,
        l0_batched: bool = False,
        l0_buffered: bool = False,
        source_deferred: bool = False,
    ) -> None:
        typed_episode = None
        if self.typed_cockpit_episodes:
            typed_episode = compile_navigation_episode([
                EpisodeTurn(
                    message_id=str(message.message_id or ""),
                    speaker=(
                        str(message.speaker or "").strip()
                        or (
                            "Driver"
                            if str(message.role).casefold() == "user"
                            else "Car Assistant"
                            if str(message.role).casefold() in {
                                "assistant", "tool",
                            }
                            else str(message.role or "")
                        )
                    ),
                    text=message.render_text(),
                    timestamp=str(message.timestamp or session.timestamp or ""),
                    sequence=index,
                    metadata=dict(message.metadata or {}),
                )
                for index, message in enumerate(session.messages)
                if message.render_text().strip()
            ],
                intent=decision.intent,
                domain=decision.domain,
                structured_slot_min_confidence=(
                    self.typed_slot_min_confidence
                ),
                structured_slot_min_margin=self.typed_slot_min_margin,
            )
        self._construction_sessions += 1
        self._construction_counts[decision.memory_type] += 1
        self._construction_action_counts[decision.write_action] += 1
        self._construction_lifecycle_counts[decision.lifecycle] += 1
        self._construction_episode_counts[decision.episode_key] += 1
        self._construction_messages += len(pipeline_messages)
        source_message_count = sum(
            1 for message in session.messages if message.content
        )
        self._construction_source_messages += source_message_count
        self._construction_characters += decision.source_characters
        if decision.extract_l1:
            self._construction_extracted += 1
            self._construction_selected_characters += decision.source_characters
        else:
            self._construction_suppressed += 1
        scope = (str(conversation_id), perspective.agent_id)
        self._construction_trace_scopes.add(scope)
        source_ids = [
            str(message.message_id)
            for message in session.messages
            if str(message.message_id or "")
        ]
        source_mapping = self._construction_source_session_by_id.setdefault(
            scope, {}
        )
        source_mapping.update({
            source_id: str(session.session_id) for source_id in source_ids
        })
        typed_episode_record: TypedEpisodeRecord | None = None
        if typed_episode is not None:
            self._construction_typed_episode_counts[typed_episode.scene] += 1
            episode_payload = typed_episode.to_dict()
            episode_mapping = self._construction_trace_episodes_by_source_id.setdefault(
                scope, {}
            )
            for source_id in typed_episode.source_ids:
                if source_id:
                    episode_mapping[source_id] = episode_payload
            typed_episode_record = self._upsert_typed_episode_index(
                conversation_id=str(conversation_id),
                perspective=perspective,
                session_id=str(session.session_id),
                source_ids=source_ids,
                episode=episode_payload,
                active=not source_deferred,
            )
            if source_deferred:
                pending_key = (
                    perspective.team_id,
                    perspective.agent_id,
                    perspective.user_id,
                    perspective.task_id,
                    str(session.session_id),
                )
                self._pending_typed_episode_commits.setdefault(
                    str(conversation_id), {}
                )[pending_key] = {
                    "perspective": perspective,
                    "session_id": str(session.session_id),
                    "source_ids": list(source_ids),
                    "episode": dict(episode_payload),
                    "record_revision": (
                        typed_episode_record.revision
                        if typed_episode_record is not None else 1
                    ),
                }
            self._remember_typed_episode_record(
                scope,
                session_id=str(session.session_id),
                source_ids=source_ids,
                episode=episode_payload,
                record_revision=(
                    typed_episode_record.revision
                    if typed_episode_record is not None else 1
                ),
            )
        if decision.extract_l1:
            self._construction_dirty_scopes.add(scope)
            self._construction_trace_transport_sessions.setdefault(
                scope, set()
            ).add(transport_session_id)
        if self.construction_trace_path is None:
            return
        if self.construction_trace_path.is_dir():
            raise ValueError(
                "TDAI_EVAL_CONSTRUCTION_TRACE must name a JSONL file during "
                "ingestion; a directory is supported by readiness-only runs"
            )
        self.construction_trace_path.parent.mkdir(parents=True, exist_ok=True)
        original_user_rounds = sum(
            1 for message in session.messages
            if str(message.role).casefold() == "user"
        )
        pipeline_user_rounds = sum(
            1 for message in pipeline_messages if message["role"] == "user"
        )
        payload = {
            "schema_version": (
                2 if self.l1_write_policy == "cockpit_episode_v2" else 1
            ),
            "conversation_id": conversation_id,
            "session_id": session.session_id,
            "perspective_owner": perspective.speaker,
            "agent_id": perspective.agent_id,
            "team_id": perspective.team_id,
            "user_id": perspective.user_id,
            "task_id": perspective.task_id,
            "policy": self.l1_write_policy,
            "l1_batch_mode": self.l1_batch_mode,
            "transport_session_id": transport_session_id,
            "source_ids": source_ids,
            "decision": decision.to_dict(),
            "original_user_rounds": original_user_rounds,
            "pipeline_user_rounds": pipeline_user_rounds,
            "l0_messages_written": len(pipeline_messages),
            "source_message_count": source_message_count,
            "l1_compacted": l1_compacted,
            "l0_batched": l0_batched,
            "l0_buffered": l0_buffered,
            "source_committed": not source_deferred,
            "typed_episode_header": bool(
                l1_compacted and self.l1_typed_episode_headers
            ),
            **({
                "typed_cockpit_episode": typed_episode.to_dict(),
                "typed_episode_record_revision": (
                    typed_episode_record.revision
                    if typed_episode_record is not None else 1
                ),
            } if typed_episode is not None else {}),
            "l1_trigger_suppressed": (
                not decision.extract_l1 and original_user_rounds > 0
            ),
        }
        with self.construction_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _commit_deferred_typed_episodes(self, conversation_id: str) -> None:
        pending = self._pending_typed_episode_commits.get(conversation_id)
        if not pending:
            return
        markers: list[dict] = []
        for pending_key in sorted(pending):
            item = pending[pending_key]
            perspective = item["perspective"]
            committed = self._upsert_typed_episode_index(
                conversation_id=conversation_id,
                perspective=perspective,
                session_id=str(item["session_id"]),
                source_ids=list(item["source_ids"]),
                episode=dict(item["episode"]),
                initial_revision=int(item["record_revision"]),
                active=True,
            )
            revision = (
                committed.revision
                if committed is not None
                else int(item["record_revision"])
            )
            self._remember_typed_episode_record(
                (conversation_id, perspective.agent_id),
                session_id=str(item["session_id"]),
                source_ids=list(item["source_ids"]),
                episode=dict(item["episode"]),
                record_revision=revision,
            )
            markers.append({
                "schema_version": 3,
                "record_type": "typed_episode_source_commit",
                "conversation_id": conversation_id,
                "session_id": str(item["session_id"]),
                "perspective_owner": perspective.speaker,
                "team_id": perspective.team_id,
                "agent_id": perspective.agent_id,
                "user_id": perspective.user_id,
                "task_id": perspective.task_id,
                "typed_episode_record_revision": revision,
            })
        if self.construction_trace_path is not None:
            with self.construction_trace_path.open(
                "a", encoding="utf-8"
            ) as handle:
                for marker in markers:
                    handle.write(json.dumps(marker, ensure_ascii=False) + "\n")
        self._typed_episode_index_commits += len(markers)
        self._pending_typed_episode_commits.pop(conversation_id, None)

    def construction_metrics(self) -> dict:
        return {
            "policy": self.l1_write_policy,
            "l1_batch_mode": self.l1_batch_mode,
            "l0_batch_mode": self.l0_batch_mode,
            "l0_batch_max_messages": self.l0_batch_max_messages,
            "l0_flush_concurrency": self.l0_flush_concurrency,
            "l0_batches_written": self._construction_l0_batches,
            "l0_batched_sessions": self._construction_l0_batched_sessions,
            "l0_concurrent_requests": (
                self._construction_l0_concurrent_requests
            ),
            "l0_transport_requests_saved": max(
                0,
                self._construction_l0_batched_sessions
                - self._construction_l0_batches,
            ),
            "l1_batches_written": self._construction_l1_batches,
            "l1_zero_output_retries": self.l1_zero_output_retries,
            "l1_zero_output_terminal": self.l1_zero_output_terminal,
            "l1_zero_output_wait_for_repair": (
                self.l1_zero_output_wait_for_repair
            ),
            "l1_repair_attempts": self._construction_l1_repair_attempts,
            "l1_repaired_scopes": self._construction_l1_repaired_scopes,
            "l1_empty_l0_only_scopes": len(
                self._construction_l1_empty_scopes
            ),
            "scope_sessions": self._construction_sessions,
            "selected_for_l1": self._construction_extracted,
            "suppressed_from_l1": self._construction_suppressed,
            "selection_rate": (
                self._construction_extracted / self._construction_sessions
                if self._construction_sessions else 0.0
            ),
            "memory_types": dict(sorted(self._construction_counts.items())),
            "write_actions": dict(sorted(
                self._construction_action_counts.items()
            )),
            "lifecycle_states": dict(sorted(
                self._construction_lifecycle_counts.items()
            )),
            "episode_keys": dict(sorted(
                self._construction_episode_counts.items()
            )),
            "l0_messages_written": self._construction_messages,
            "source_messages_input": self._construction_source_messages,
            "selected_sessions_compacted": self._construction_compacted_sessions,
            "source_characters": self._construction_characters,
            "selected_source_characters": self._construction_selected_characters,
            "selected_character_rate": (
                self._construction_selected_characters
                / self._construction_characters
                if self._construction_characters else 0.0
            ),
            "typed_episode_headers": self.l1_typed_episode_headers,
            "typed_cockpit_episodes_enabled": self.typed_cockpit_episodes,
            "typed_slot_min_confidence": self.typed_slot_min_confidence,
            "typed_slot_min_margin": self.typed_slot_min_margin,
            "typed_cockpit_episodes": sum(
                self._construction_typed_episode_counts.values()
            ),
            "typed_cockpit_episode_scenes": dict(sorted(
                self._construction_typed_episode_counts.items()
            )),
            "typed_episode_index_configured": (
                self.typed_episode_index_path is not None
            ),
            "typed_episode_index_available": (
                self._typed_episode_index is not None
            ),
            "typed_episode_index_required": self.typed_episode_index_required,
            "typed_episode_index_ttl_seconds": (
                self.typed_episode_index_ttl_seconds
            ),
            "typed_episode_index_writes": self._typed_episode_index_writes,
            "typed_episode_index_reads": self._typed_episode_index_reads,
            "typed_episode_index_migrations": (
                self._typed_episode_index_migrations
            ),
            "typed_episode_index_commits": self._typed_episode_index_commits,
            "typed_episode_index_pending": sum(
                len(items)
                for items in self._pending_typed_episode_commits.values()
            ),
            "typed_episode_index_failures": self._typed_episode_index_failures,
            "typed_episode_index_last_error": (
                self._typed_episode_index_last_error or None
            ),
            "l23_schedule": self.l23_schedule,
            "l23_readiness_mode": self.l23_readiness_mode,
            "dirty_profile_scopes": len(self._construction_dirty_scopes),
            "trace": (
                str(self.construction_trace_path)
                if self.construction_trace_path else None
            ),
        }

    def _v3_atomic_count(self, perspective: _V3Perspective) -> int:
        body = {
            "team_id": perspective.team_id,
            "agent_id": perspective.agent_id,
            "user_id": perspective.user_id,
        }
        if perspective.task_id:
            body["task_id"] = perspective.task_id
        response = self._post("/v3/atomic/count", body)
        if response.get("code") not in (None, 0):
            raise RuntimeError(
                "TencentDB atomic/count failed during construction audit: "
                f"{response}"
            )
        return int((response.get("data") or {}).get("total", 0))

    def _selected_l1_source_ids(
        self, conversation: Conversation, perspective: _V3Perspective
    ) -> tuple[str, ...]:
        source_ids: list[str] = []
        for session in conversation.sessions:
            decision = self._construction_decision(session)
            if not decision.extract_l1:
                continue
            messages = [
                message for message in session.messages if message.content
            ]
            if not any(
                _perspective_role(message, perspective.speaker) == "user"
                for message in messages
            ):
                continue
            source_ids.extend(
                str(message.message_id)
                for message in messages if str(message.message_id or "")
            )
        return tuple(dict.fromkeys(source_ids))

    def _v3_l0_source_copy_count(
        self,
        conversation: Conversation,
        perspective: _V3Perspective,
    ) -> int:
        """Infer already executed replays from durable source envelopes."""
        source_ids = self._selected_l1_source_ids(conversation, perspective)
        if not source_ids:
            return 0
        wanted = set(source_ids)
        counts: Counter[str] = Counter()
        common = {
            "team_id": perspective.team_id,
            "agent_id": perspective.agent_id,
            "user_id": perspective.user_id,
        }
        if perspective.task_id:
            common["task_id"] = perspective.task_id
        offset = 0
        page_size = 100
        while True:
            response = self._post("/v3/conversation/query", {
                **common, "limit": page_size, "offset": offset,
            })
            if response.get("code") not in (None, 0):
                raise RuntimeError(
                    "TencentDB conversation/query failed during L1 replay "
                    f"audit: {response}"
                )
            data = response.get("data") or {}
            items = data.get("messages") or []
            for row in items:
                for source_id in _v3_source_ids(
                    str(row.get("content") or "")
                ):
                    if source_id in wanted:
                        counts[source_id] += 1
            offset += len(items)
            if not items or offset >= int(data.get("total") or 0):
                break
        return min((counts[source_id] for source_id in source_ids), default=0)

    def _replay_selected_l1_batch(
        self, conversation: Conversation, perspective: _V3Perspective
    ) -> dict:
        """Replay only persist-worthy rows after a proven empty L1 result.

        Reusing the original transport session leaves its advanced cursor
        intact: the replayed rows receive new ingestion timestamps and become
        the next bounded L1 batch.  Source envelopes stay unchanged, allowing
        normal source-id de-duplication to hide the redundant L0 transport
        rows from answer context.
        """
        payload = {
            "team_id": perspective.team_id,
            "agent_id": perspective.agent_id,
            "user_id": perspective.user_id,
            **({"task_id": perspective.task_id} if perspective.task_id else {}),
        }
        transport_session_id = _scoped_v3_session_id(
            "selective-l1-conversation-batch",
            conversation_id=conversation.conversation_id,
            team_id=perspective.team_id,
            agent_id=perspective.agent_id,
            user_id=perspective.user_id,
            task_id=perspective.task_id,
        )
        replay_messages: list[dict] = []
        selected_sessions = 0
        for session in conversation.sessions:
            decision = self._construction_decision(session)
            if not decision.extract_l1:
                continue
            messages = []
            for message in session.messages:
                if not message.content:
                    continue
                timestamp = _iso_timestamp(message.timestamp)
                source_role = _perspective_role(message, perspective.speaker)
                content = _format_v3_content(
                    message, source_role=source_role
                )
                if self.ingest_resolve_relative_time:
                    content = _annotate_v3_relative_time(content, timestamp)
                messages.append({
                    "role": _v3_transport_role(
                        source_role, extract_l1=True
                    ),
                    "content": content,
                    **({"timestamp": timestamp} if timestamp else {}),
                })
            if not any(message["role"] == "user" for message in messages):
                continue
            selected_sessions += 1
            if self.l1_compact_selected_sessions:
                replay_messages.append(_compact_v3_l1_session(
                    messages,
                    decision=(decision if self.l1_typed_episode_headers else None),
                ))
            else:
                replay_messages.extend(messages)

        batches = 0
        for offset in range(0, len(replay_messages), self.l1_batch_max_messages):
            chunk = replay_messages[offset:offset + self.l1_batch_max_messages]
            if not chunk:
                continue
            self._post("/v3/conversation/add", {
                **payload,
                "session_id": transport_session_id,
                "messages": chunk,
            })
            batches += 1
        if not batches:
            raise RuntimeError(
                "construction trace marked a dirty scope but no selected L1 "
                f"rows could be rebuilt for {conversation.conversation_id}"
            )
        self._l0_history_cache.pop(str(conversation.conversation_id), None)
        self._profile_cache.pop(str(conversation.conversation_id), None)
        return {
            "selected_sessions": selected_sessions,
            "replayed_messages": len(replay_messages),
            "replayed_batches": batches,
        }

    def ensure_construction(
        self, conversation: Conversation, *, timeout: float | None
    ) -> dict:
        """Audit empty L1 output and apply the configured terminal policy."""
        if self.api_version != "v3" or self.l1_zero_output_retries <= 0:
            return {"status": "not_required", "repair_attempts": 0}
        conversation_id = str(conversation.conversation_id)
        self._load_construction_trace_scopes()
        if not self._construction_trace_valid:
            raise RuntimeError(
                "L1 zero-output audit requires a valid construction trace"
            )
        dirty = [
            perspective
            for perspective in self._perspective_isolations(conversation_id)
            if (conversation_id, perspective.agent_id)
            in self._construction_dirty_scopes
        ]
        if not dirty:
            return {
                "status": "clean_scope",
                "dirty_scopes": 0,
                "repair_attempts": 0,
            }

        counts = {
            perspective.agent_id: self._v3_atomic_count(perspective)
            for perspective in dirty
        }
        missing = [
            perspective for perspective in dirty
            if counts[perspective.agent_id] <= 0
        ]
        initially_missing = len(missing)
        if not missing:
            return {
                "status": "verified",
                "dirty_scopes": len(dirty),
                "atomic_counts": counts,
                "repair_attempts": 0,
            }

        deadline = time.monotonic() + (
            max(0.0, timeout) if timeout is not None else float(self.timeout)
        )
        # First allow the original timer-triggered L1 task to finish.  The
        # configured L1 settle window must exceed MemoryCore's L1 idle delay.
        self._wait_until_ready_layers(
            conversation_id,
            timeout=max(0.0, deadline - time.monotonic()),
            layers=frozenset({"L1"}),
        )
        counts.update({
            perspective.agent_id: self._v3_atomic_count(perspective)
            for perspective in missing
        })
        missing = [
            perspective for perspective in missing
            if counts[perspective.agent_id] <= 0
        ]
        if not missing:
            return {
                "status": "verified_after_wait",
                "dirty_scopes": len(dirty),
                "atomic_counts": counts,
                "repair_attempts": 0,
            }
        replay_totals = Counter()
        observed_copies = {
            perspective.agent_id: self._v3_l0_source_copy_count(
                conversation, perspective
            )
            for perspective in missing
        }
        attempts_by_agent = {
            perspective.agent_id: max(
                0, observed_copies[perspective.agent_id] - 1
            )
            for perspective in missing
        }
        attempts_this_process = 0
        while missing:
            replayable = [
                perspective for perspective in missing
                if attempts_by_agent[perspective.agent_id]
                < self.l1_zero_output_retries
            ]
            if not replayable:
                break
            for perspective in replayable:
                replay = self._replay_selected_l1_batch(
                    conversation, perspective
                )
                replay_totals.update(replay)
                attempts_by_agent[perspective.agent_id] += 1
                attempts_this_process += 1
                self._construction_l1_repair_attempts += 1
            if not self.l1_zero_output_wait_for_repair:
                for perspective in missing:
                    self._construction_l1_empty_scopes.add((
                        conversation_id, perspective.agent_id
                    ))
                return {
                    "status": "l0_only_repair_enqueued",
                    "dirty_scopes": len(dirty),
                    "atomic_counts": counts,
                    "empty_agents": [
                        perspective.agent_id for perspective in missing
                    ],
                    "repair_attempts": sum(attempts_by_agent.values()),
                    "repair_attempts_this_process": attempts_this_process,
                    "observed_source_copies": observed_copies,
                    **dict(replay_totals),
                }
            self._wait_until_ready_layers(
                conversation_id,
                timeout=max(0.0, deadline - time.monotonic()),
                layers=frozenset({"L1"}),
            )
            counts.update({
                perspective.agent_id: self._v3_atomic_count(perspective)
                for perspective in replayable
            })
            missing = [
                perspective for perspective in missing
                if counts[perspective.agent_id] <= 0
            ]

        if missing:
            failed_agents = ",".join(
                perspective.agent_id for perspective in missing
            )
            if self.l1_zero_output_terminal == "l0_only":
                for perspective in missing:
                    self._construction_l1_empty_scopes.add((
                        conversation_id, perspective.agent_id
                    ))
                return {
                    "status": "l0_only_after_empty_l1",
                    "dirty_scopes": len(dirty),
                    "atomic_counts": counts,
                    "empty_agents": [
                        perspective.agent_id for perspective in missing
                    ],
                    "repair_attempts": sum(attempts_by_agent.values()),
                    "repair_attempts_this_process": attempts_this_process,
                    "observed_source_copies": observed_copies,
                    **dict(replay_totals),
                }
            raise RuntimeError(
                "TencentDB L1 remained empty after bounded adapter replay; "
                f"conversation={conversation_id}, agents={failed_agents}, "
                f"attempts={sum(attempts_by_agent.values())}"
            )
        self._construction_l1_repaired_scopes += initially_missing
        return {
            "status": "repaired",
            "dirty_scopes": len(dirty),
            "atomic_counts": counts,
            "repair_attempts": sum(attempts_by_agent.values()),
            "repair_attempts_this_process": attempts_this_process,
            "observed_source_copies": observed_copies,
            **dict(replay_totals),
        }

    def _load_construction_trace_scopes(self) -> None:
        """Load selected L1 scopes without consulting questions or answers.

        Ingest and retrieval commonly run in separate processes.  A trace file
        (or a directory containing per-shard JSONL traces) is therefore the
        durable source of truth for deciding which agent-scoped L2/L3 profiles
        are expected.  Any missing or malformed trace fails closed: readiness
        falls back to requiring the profile instead of silently skipping it.
        """
        if self._construction_trace_loaded:
            return
        self._construction_trace_loaded = True
        path = self.construction_trace_path
        if path is None:
            return
        if path.is_file():
            files = (path,)
        elif path.is_dir():
            files = tuple(sorted(path.glob("*.jsonl")))
        else:
            return
        if not files:
            return
        scopes: set[tuple[str, str]] = set()
        dirty_scopes: set[tuple[str, str]] = set()
        source_sessions: dict[tuple[str, str], set[str]] = {}
        transport_sessions: dict[tuple[str, str], set[str]] = {}
        source_session_by_id: dict[tuple[str, str], dict[str, str]] = {}
        typed_episodes_by_source_id: dict[
            tuple[str, str], dict[str, dict]
        ] = {}
        typed_episode_rows: dict[tuple[str, str], dict[str, dict]] = {}
        typed_episode_commits: dict[tuple[str, str, str], dict] = {}
        sessions_complete = True
        try:
            for trace_file in files:
                with trace_file.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            raise ValueError(
                                f"{trace_file}:{line_number} is not an object"
                            )
                        record_type = str(
                            payload.get("record_type") or ""
                        ).strip()
                        if record_type:
                            if record_type != "typed_episode_source_commit":
                                raise ValueError(
                                    f"{trace_file}:{line_number} has unknown "
                                    f"record_type {record_type}"
                                )
                            marker_conversation = str(
                                payload.get("conversation_id") or ""
                            )
                            marker_agent = str(payload.get("agent_id") or "")
                            marker_session = str(
                                payload.get("session_id") or ""
                            )
                            marker_revision = payload.get(
                                "typed_episode_record_revision"
                            )
                            if (
                                not marker_conversation
                                or not marker_agent
                                or not marker_session
                                or isinstance(marker_revision, bool)
                                or not isinstance(marker_revision, int)
                                or marker_revision < 1
                            ):
                                raise ValueError(
                                    f"{trace_file}:{line_number} has invalid "
                                    "typed episode commit marker"
                                )
                            marker = {
                                "record_revision": marker_revision,
                                "team_id": str(payload.get("team_id") or ""),
                                "user_id": str(payload.get("user_id") or ""),
                                "task_id": str(payload.get("task_id") or ""),
                            }
                            marker_key = (
                                marker_conversation,
                                marker_agent,
                                marker_session,
                            )
                            previous_marker = typed_episode_commits.get(marker_key)
                            if previous_marker is None:
                                typed_episode_commits[marker_key] = marker
                            else:
                                if any(
                                    str(previous_marker.get(key) or "")
                                    != str(marker.get(key) or "")
                                    for key in (
                                        "team_id", "user_id", "task_id",
                                    )
                                ):
                                    raise ValueError(
                                        f"{trace_file}:{line_number} has "
                                        "conflicting typed episode commit "
                                        "markers"
                                    )
                                if marker_revision > int(
                                    previous_marker["record_revision"]
                                ):
                                    typed_episode_commits[marker_key] = marker
                            continue
                        conversation_id = payload.get("conversation_id")
                        agent_id = payload.get("agent_id")
                        decision = payload.get("decision")
                        if (
                            conversation_id is None
                            or not str(agent_id or "")
                            or not isinstance(decision, dict)
                            or not isinstance(decision.get("extract_l1"), bool)
                        ):
                            raise ValueError(
                                f"{trace_file}:{line_number} has no valid scope decision"
                            )
                        scope = (str(conversation_id), str(agent_id))
                        scopes.add(scope)
                        if decision["extract_l1"]:
                            dirty_scopes.add(scope)
                        source_session_id = payload.get("session_id")
                        source_ids = payload.get("source_ids") or []
                        if not isinstance(source_ids, list) or any(
                            not isinstance(value, str) for value in source_ids
                        ):
                            raise ValueError(
                                f"{trace_file}:{line_number} has invalid source_ids"
                            )
                        raw_episode = payload.get("typed_cockpit_episode")
                        if raw_episode is not None:
                            if not isinstance(raw_episode, dict):
                                raise ValueError(
                                    f"{trace_file}:{line_number} has invalid "
                                    "typed_cockpit_episode"
                                )
                            episode = episode_from_dict(raw_episode)
                            if episode is None or not episode.source_ids:
                                raise ValueError(
                                    f"{trace_file}:{line_number} has invalid "
                                    "typed cockpit episode fields"
                                )
                            if not set(episode.source_ids).issubset(set(source_ids)):
                                raise ValueError(
                                    f"{trace_file}:{line_number} has typed episode "
                                    "sources outside the source session"
                                )
                            canonical_episode = episode.to_dict()
                            record_key = str(
                                source_session_id
                                or "|".join(source_ids)
                            )
                            record_revision = payload.get(
                                "typed_episode_record_revision", 1
                            )
                            if (
                                isinstance(record_revision, bool)
                                or not isinstance(record_revision, int)
                                or record_revision < 1
                            ):
                                raise ValueError(
                                    f"{trace_file}:{line_number} has invalid "
                                    "typed episode record revision"
                                )
                            source_committed = payload.get(
                                "source_committed", True
                            )
                            if not isinstance(source_committed, bool):
                                raise ValueError(
                                    f"{trace_file}:{line_number} has invalid "
                                    "source_committed flag"
                                )
                            record = {
                                "session_id": str(source_session_id or ""),
                                "source_ids": list(source_ids),
                                "episode": canonical_episode,
                                "record_revision": record_revision,
                                "team_id": str(payload.get("team_id") or ""),
                                "user_id": str(payload.get("user_id") or ""),
                                "task_id": str(payload.get("task_id") or ""),
                                "source_committed": source_committed,
                            }
                            scope_records = typed_episode_rows.setdefault(
                                scope, {}
                            )
                            previous_record = scope_records.get(record_key)
                            if previous_record is None:
                                scope_records[record_key] = record
                            elif previous_record != record:
                                previous_revision = int(
                                    previous_record.get("record_revision") or 1
                                )
                                if record_revision > previous_revision:
                                    scope_records[record_key] = record
                                else:
                                    raise ValueError(
                                        f"{trace_file}:{line_number} maps source "
                                        f"session {record_key} to conflicting "
                                        "typed episode revisions"
                                    )
                        if source_session_id is not None:
                            mapping = source_session_by_id.setdefault(scope, {})
                            for source_id in source_ids:
                                previous = mapping.setdefault(
                                    source_id, str(source_session_id)
                                )
                                if previous != str(source_session_id):
                                    raise ValueError(
                                        f"{trace_file}:{line_number} maps source "
                                        f"{source_id} to multiple sessions"
                                    )
                        pipeline_user_rounds = payload.get("pipeline_user_rounds")
                        if source_session_id is None or (
                            pipeline_user_rounds is not None
                            and (
                                isinstance(pipeline_user_rounds, bool)
                                or not isinstance(pipeline_user_rounds, int)
                            )
                        ):
                            sessions_complete = False
                        elif (
                            decision["extract_l1"]
                            and (
                                pipeline_user_rounds is None
                                or pipeline_user_rounds > 0
                            )
                        ):
                            transport_session_id = str(
                                payload.get("transport_session_id") or ""
                            ).strip()
                            if transport_session_id:
                                transport_sessions.setdefault(scope, set()).add(
                                    transport_session_id
                                )
                            else:
                                source_sessions.setdefault(scope, set()).add(
                                    str(source_session_id)
                                )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return
        if not scopes:
            return
        for scope, records in tuple(typed_episode_rows.items()):
            committed_records: dict[str, dict] = {}
            for record_key, record in records.items():
                if bool(record.get("source_committed", True)):
                    committed_records[record_key] = record
                    continue
                marker = typed_episode_commits.get((
                    scope[0], scope[1], record_key,
                ))
                if marker is None:
                    # The adapter may have crashed before its batched L0 write
                    # completed. Never turn this trace row into readable state.
                    continue
                if any(
                    str(record.get(key) or "")
                    and str(marker.get(key) or "")
                    != str(record.get(key) or "")
                    for key in ("team_id", "user_id", "task_id")
                ):
                    return
                marker_revision = int(marker.get("record_revision") or 0)
                if marker_revision < int(record.get("record_revision") or 1):
                    return
                committed_record = dict(record)
                committed_record["source_committed"] = True
                committed_record["record_revision"] = marker_revision
                committed_records[record_key] = committed_record
            typed_episode_rows[scope] = committed_records
        for scope, records in typed_episode_rows.items():
            episode_mapping = typed_episodes_by_source_id.setdefault(scope, {})
            for record in records.values():
                canonical_episode = dict(record.get("episode") or {})
                episode = episode_from_dict(canonical_episode)
                if episode is None:
                    return
                for source_id in episode.source_ids:
                    previous = episode_mapping.setdefault(
                        source_id, canonical_episode
                    )
                    if previous != canonical_episode:
                        return
        self._construction_trace_scopes.update(scopes)
        self._construction_dirty_scopes.update(dirty_scopes)
        self._construction_trace_source_sessions.update(source_sessions)
        self._construction_trace_transport_sessions.update(transport_sessions)
        for scope, mapping in source_session_by_id.items():
            self._construction_source_session_by_id.setdefault(
                scope, {}
            ).update(mapping)
        for scope, mapping in typed_episodes_by_source_id.items():
            self._construction_trace_episodes_by_source_id.setdefault(
                scope, {}
            ).update(mapping)
        for scope, mapping in typed_episode_rows.items():
            for key in sorted(mapping):
                record = mapping[key]
                self._remember_typed_episode_record(
                    scope,
                    session_id=str(record.get("session_id") or ""),
                    source_ids=[
                        str(item) for item in record.get("source_ids") or []
                    ],
                    episode=dict(record.get("episode") or {}),
                    record_revision=int(
                        record.get("record_revision") or 1
                    ),
                )
                self._migrate_trace_episode_to_index(scope, record)
        self._construction_trace_sessions_complete = sessions_complete
        self._construction_trace_valid = True

    @staticmethod
    def _typed_episode_scope(
        conversation_id: str, perspective: _V3Perspective
    ) -> TypedEpisodeScope:
        return TypedEpisodeScope(
            conversation_id=str(conversation_id),
            team_id=perspective.team_id,
            agent_id=perspective.agent_id,
            user_id=perspective.user_id,
            task_id=perspective.task_id,
        )

    def _typed_episode_index_failure(
        self, exc: TypedEpisodeIndexError
    ) -> None:
        self._typed_episode_index_failures += 1
        self._typed_episode_index_last_error = type(exc).__name__
        if self.typed_episode_index_required:
            raise RuntimeError(
                "required typed episode index operation failed"
            ) from exc

    def _upsert_typed_episode_index(
        self,
        *,
        conversation_id: str,
        perspective: _V3Perspective,
        session_id: str,
        source_ids: list[str],
        episode: dict,
        initial_revision: int = 1,
        active: bool = True,
    ) -> TypedEpisodeRecord | None:
        if self._typed_episode_index is None:
            return None
        try:
            record = self._typed_episode_index.upsert(
                self._typed_episode_scope(conversation_id, perspective),
                session_id=session_id,
                source_ids=source_ids,
                episode=episode,
                ttl_seconds=self.typed_episode_index_ttl_seconds,
                initial_revision=initial_revision,
                active=active,
            )
        except TypedEpisodeIndexError as exc:
            self._typed_episode_index_failure(exc)
            return None
        self._typed_episode_index_writes += 1
        return record

    def _migrate_trace_episode_to_index(
        self, scope: tuple[str, str], record: dict
    ) -> None:
        """Backfill an old JSONL trace only into its exact current namespace."""
        if self._typed_episode_index is None:
            return
        conversation_id, agent_id = scope
        perspective = next((
            item for item in self._perspective_isolations(conversation_id)
            if item.agent_id == agent_id
        ), None)
        if perspective is None:
            return
        declared = {
            "team_id": str(record.get("team_id") or ""),
            "user_id": str(record.get("user_id") or ""),
            "task_id": str(record.get("task_id") or ""),
        }
        actual = {
            "team_id": perspective.team_id,
            "user_id": perspective.user_id,
            "task_id": perspective.task_id,
        }
        if any(
            declared[key] and declared[key] != actual[key]
            for key in declared
        ):
            # A trace from another tenant/run must never be silently rebound.
            return
        migrated = self._upsert_typed_episode_index(
            conversation_id=conversation_id,
            perspective=perspective,
            session_id=str(record.get("session_id") or ""),
            source_ids=[str(item) for item in record.get("source_ids") or []],
            episode=dict(record.get("episode") or {}),
            initial_revision=int(record.get("record_revision") or 1),
        )
        if migrated is not None:
            self._typed_episode_index_migrations += 1

    def _active_typed_episode_index_records(
        self, conversation_id: str, perspective: _V3Perspective
    ) -> tuple[TypedEpisodeRecord, ...] | None:
        if self._typed_episode_index is None:
            return None
        try:
            records = self._typed_episode_index.list_active(
                self._typed_episode_scope(conversation_id, perspective),
                scene="navigation",
            )
        except TypedEpisodeIndexError as exc:
            self._typed_episode_index_failure(exc)
            return None
        self._typed_episode_index_reads += 1
        return records

    def _remember_typed_episode_record(
        self,
        scope: tuple[str, str],
        *,
        session_id: str,
        source_ids: list[str],
        episode: dict,
        record_revision: int = 1,
    ) -> None:
        """Keep the direct index current while a trace is still growing."""
        record = {
            "session_id": session_id,
            "source_ids": list(source_ids),
            "episode": dict(episode),
            "record_revision": max(1, int(record_revision)),
        }
        record_key = session_id or "|".join(source_ids)
        existing = {
            str(item.get("session_id") or "")
            or "|".join(str(value) for value in item.get("source_ids") or [])
            : item
            for item in self._construction_trace_typed_episode_rows.get(
                scope, ()
            )
        }
        previous = existing.get(record_key)
        if previous is None:
            existing[record_key] = record
        elif previous != record:
            previous_revision = int(previous.get("record_revision") or 1)
            if record["record_revision"] > previous_revision:
                existing[record_key] = record
            else:
                raise ValueError(
                    f"source session {record_key} maps to conflicting typed "
                    "episode revisions"
                )
        self._construction_trace_typed_episode_rows[scope] = tuple(
            existing[key] for key in sorted(existing)
        )

    def _pipeline_scope_sessions(
        self, conversation_id: str
    ) -> frozenset[str] | None:
        """Resolve transport session IDs owned by one evaluated conversation.

        The status endpoint is process-global, while benchmark namespaces share
        one MemoryCore worker.  A complete construction trace lets readiness
        ignore tasks from other conversations and store groups.  Missing trace
        provenance deliberately returns ``None`` so callers retain the strict
        historical all-queue wait.
        """
        self._load_construction_trace_scopes()
        if (
            not self._construction_trace_valid
            or not self._construction_trace_sessions_complete
        ):
            return None
        key = str(conversation_id)
        sessions: set[str] = set()
        for perspective in self._perspective_isolations(key):
            scope = (key, perspective.agent_id)
            sessions.update(
                self._construction_trace_transport_sessions.get(scope, set())
            )
            for source_session_id in self._construction_trace_source_sessions.get(
                scope, set()
            ):
                sessions.add(_scoped_v3_session_id(
                    source_session_id,
                    conversation_id=key,
                    team_id=perspective.team_id,
                    agent_id=perspective.agent_id,
                    user_id=perspective.user_id,
                    task_id=perspective.task_id,
                ))
        return frozenset(sessions)

    @staticmethod
    def _pipeline_layers_idle(
        data: dict,
        tracked_layers: tuple[str, ...],
        scope_sessions: frozenset[str] | None,
    ) -> bool:
        """Return whether relevant layer work is idle.

        New MemoryCore status responses expose queued/running session IDs.  If
        either the trace or those fields is unavailable, fall back to global
        counts rather than risking retrieval from a partially built store.
        """
        if scope_sessions is None:
            return all(
                int((data.get(layer) or {}).get("queued", 0)) == 0
                and int((data.get(layer) or {}).get("running", 0)) == 0
                for layer in tracked_layers
            )
        if not scope_sessions:
            return True
        for layer in tracked_layers:
            status = data.get(layer) or {}
            if (
                "queued_sessions" not in status
                or "running_sessions" not in status
            ):
                if (
                    int(status.get("queued", 0)) != 0
                    or int(status.get("running", 0)) != 0
                ):
                    return False
                continue
            active = tuple(status.get("queued_sessions") or ()) + tuple(
                status.get("running_sessions") or ()
            )
            for value in active:
                task_scope = str(value).rsplit("|session:", 1)[-1]
                if task_scope in scope_sessions:
                    return False
        return True

    def _profile_scope_dirty(
        self, conversation_id: str, perspective: _V3Perspective
    ) -> bool | None:
        """Return True/False for a proven trace scope, or None if unknown."""
        if self.l23_readiness_mode != "dirty_only":
            return None
        if (str(conversation_id), perspective.agent_id) in (
            self._construction_l1_empty_scopes
        ):
            return False
        self._load_construction_trace_scopes()
        if not self._construction_trace_valid:
            return None
        scope = (str(conversation_id), perspective.agent_id)
        if scope not in self._construction_trace_scopes:
            return None
        return scope in self._construction_dirty_scopes

    def _conversation_has_dirty_profile_scope(
        self, conversation_id: str
    ) -> bool | None:
        states = [
            self._profile_scope_dirty(conversation_id, perspective)
            for perspective in self._perspective_isolations(conversation_id)
        ]
        if any(state is None for state in states):
            return None
        return any(states)

    def wait_until_ready(self, conversation_id: str, *, timeout: float) -> None:
        """Wait eagerly, or defer higher layers for an adaptive L0 route."""
        if (
            self.api_version == "v3"
            and self.retrieval_policy == "adaptive"
            and self.adaptive_lazy_layer_readiness
        ):
            self._adaptive_ready_deadlines[str(conversation_id)] = (
                time.monotonic() + max(0.0, timeout)
            )
            return
        self._wait_until_ready_layers(
            conversation_id, timeout=timeout, layers=self.memory_layers
        )

    def _wait_until_ready_layers(
        self,
        conversation_id: str,
        *,
        timeout: float,
        layers: frozenset[str],
    ) -> None:
        """Wait until only the requested memory layers are stable."""
        if self.api_version != "v3":
            return
        tracked_layers = tuple(
            layer.lower()
            for layer in ("L1", "L2", "L3")
            if layer in layers
        )
        if not tracked_layers:
            return
        deadline = time.monotonic() + max(0.0, timeout)
        # A zero queue snapshot alone is insufficient: L1 tail drains and L2
        # scheduling live behind timers that are not exposed by pipeline/status.
        # The caller must set this longer than both configured timer delays.
        settle_seconds = max(
            0.0, float(os.getenv("TDAI_EVAL_READY_SETTLE_SECONDS", "10"))
        )
        profile_layers = bool({"L2", "L3"} & layers)
        if profile_layers:
            dirty_state = self._conversation_has_dirty_profile_scope(
                conversation_id
            )
            if dirty_state is False:
                settle_seconds = max(0.0, float(os.getenv(
                    "TDAI_EVAL_CLEAN_READY_SETTLE_SECONDS", "10"
                )))
        else:
            settle_seconds = max(0.0, float(os.getenv(
                "TDAI_EVAL_L1_READY_SETTLE_SECONDS", str(settle_seconds)
            )))
        idle_since: float | None = None
        l1_idle_since: float | None = None
        scope_sessions = self._pipeline_scope_sessions(conversation_id)
        while time.monotonic() < deadline:
            status = self._post("/v2/pipeline/status", {})
            if status.get("code") not in (None, 0):
                raise RuntimeError(f"TencentDB pipeline/status failed: {status}")
            data = status.get("data") or {}
            worker = data.get("worker") or {}
            dead_letters = max(
                int(worker.get("deadLetterCount", 0)),
                int(worker.get("tasksDeadLettered", 0)),
            )
            if dead_letters:
                raise RuntimeError(
                    "TencentDB memory construction reached the dead-letter "
                    f"queue ({dead_letters} task(s)); refusing partial retrieval"
                )
            idle = self._pipeline_layers_idle(
                data, tracked_layers, scope_sessions
            )
            now = time.monotonic()
            if profile_layers:
                # L2 timers are armed after the *last* relevant L1 completes.
                # Anchor the settle window there instead of restarting a full
                # delay after L2/L3 drain, which previously added five idle
                # minutes to every successful async retrieval.
                l1_idle = self._pipeline_layers_idle(
                    data, ("l1",), scope_sessions
                )
                if l1_idle:
                    if l1_idle_since is None:
                        l1_idle_since = now
                else:
                    l1_idle_since = None
                timers_settled = (
                    l1_idle_since is not None
                    and now - l1_idle_since >= settle_seconds
                )
                if (
                    idle
                    and timers_settled
                    and self._profile_layers_ready(conversation_id, layers)
                ):
                    self._profile_cache.pop(str(conversation_id), None)
                    return
                time.sleep(2.0)
                continue
            if idle:
                if idle_since is None:
                    idle_since = now
                if now - idle_since >= settle_seconds:
                    if self._profile_layers_ready(conversation_id, layers):
                        self._profile_cache.pop(str(conversation_id), None)
                        return
            else:
                idle_since = None
            time.sleep(2.0)
        layers = ",".join(layer.upper() for layer in tracked_layers)
        raise TimeoutError(
            f"TencentDB v3 {layers} pipeline/profiles did not become ready for "
            f"{conversation_id}"
        )

    def _profile_layers_ready(
        self,
        conversation_id: str,
        layers: frozenset[str] | None = None,
    ) -> bool:
        requested_layers = self.memory_layers if layers is None else layers
        if not ({"L2", "L3"} & requested_layers):
            return True
        for perspective in self._perspective_isolations(conversation_id):
            # A valid construction trace can prove that no L1 event was
            # emitted for this agent.  Such a clean scope has no L2/L3 work to
            # await.  Unknown or invalid trace state deliberately falls
            # through to the strict historical behavior.
            if self._profile_scope_dirty(conversation_id, perspective) is False:
                continue
            common = {
                "team_id": perspective.team_id,
                "agent_id": perspective.agent_id,
                "user_id": perspective.user_id,
            }
            if "L2" in requested_layers:
                response = self._post("/v3/scenario/count", common)
                if response.get("code") not in (None, 0):
                    raise RuntimeError(
                        f"TencentDB scenario/count failed for {perspective.speaker}: "
                        f"{response}"
                    )
                if int((response.get("data") or {}).get("total", 0)) <= 0:
                    return False
            if "L3" in requested_layers:
                response = self._post("/v3/core/count", common)
                if response.get("code") not in (None, 0):
                    raise RuntimeError(
                        f"TencentDB core/count failed for {perspective.speaker}: "
                        f"{response}"
                    )
                if int((response.get("data") or {}).get("total", 0)) <= 0:
                    return False
        return True

    def _load_v3_profiles(self, conversation_id: str) -> tuple[dict, ...]:
        key = str(conversation_id)
        cached = self._profile_cache.get(key)
        if cached is not None:
            return cached
        rows: list[dict] = []
        for perspective in self._perspective_isolations(key):
            if self._profile_scope_dirty(key, perspective) is False:
                continue
            common = {
                "team_id": perspective.team_id,
                "agent_id": perspective.agent_id,
                "user_id": perspective.user_id,
            }
            if "L2" in self.memory_layers:
                listing = self._post("/v3/scenario/ls", common)
                if listing.get("code") not in (None, 0):
                    raise RuntimeError(
                        f"TencentDB scenario/ls failed for {perspective.speaker}: "
                        f"{listing}"
                    )
                for entry in (listing.get("data") or {}).get("entries", []):
                    path = str(entry.get("path", ""))
                    if not path or path.endswith("/"):
                        continue
                    response = self._post(
                        "/v3/scenario/read", {**common, "path": path}
                    )
                    if response.get("code") not in (None, 0):
                        raise RuntimeError(
                            f"TencentDB scenario/read failed for {path}: {response}"
                        )
                    data = response.get("data") or {}
                    content = str(data.get("content") or "").strip()
                    if content:
                        rows.append({
                            "content": content,
                            "score": None,
                            "source_ids": [],
                            "metadata": {
                                "level": "L2",
                                "path": path,
                                "summary": str(entry.get("summary") or ""),
                                "version": data.get("version", entry.get("version")),
                                "perspective_owner": perspective.speaker,
                                "agent_id": perspective.agent_id,
                            },
                        })
            if "L3" in self.memory_layers:
                response = self._post("/v3/core/read", common)
                if response.get("code") not in (None, 0):
                    raise RuntimeError(
                        f"TencentDB core/read failed for {perspective.speaker}: "
                        f"{response}"
                    )
                data = response.get("data") or {}
                content = str(data.get("content") or "").strip()
                if content:
                    rows.append({
                        "content": content,
                        "score": None,
                        "source_ids": [],
                        "metadata": {
                            "level": "L3",
                            "version": data.get("version"),
                            "perspective_owner": perspective.speaker,
                            "agent_id": perspective.agent_id,
                        },
                    })
        result = tuple(rows)
        self._profile_cache[key] = result
        return result

    def _load_ledger_facts(self, conversation_id: str) -> tuple[dict, ...]:
        key = str(conversation_id)
        cached = self._ledger_cache.get(key)
        if cached is not None:
            return cached
        if self.ledger_path is None or self.ledger_fact_results <= 0:
            self._ledger_cache[key] = ()
            return ()
        with self.ledger_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        row = (payload.get("conversations") or {}).get(key) or {}
        facts = tuple(
            fact for fact in (row.get("facts") or [])
            if isinstance(fact, dict)
            and str(fact.get("statement") or "").strip()
            and (
                self.ledger_include_rollups
                or not str(fact.get("fact_kind") or "").endswith("rollup")
            )
        )
        self._ledger_cache[key] = facts
        return facts

    def _v3_ledger_row(self, conversation_id: str, query: str) -> dict | None:
        facts = self._load_ledger_facts(conversation_id)
        if not facts:
            return None
        speakers = tuple(
            perspective.speaker
            for perspective in self._perspective_isolations(conversation_id)
            if perspective.speaker
        )
        ranked = _rank_v3_ledger_facts(facts, query, speakers)
        if not ranked:
            return None
        rendered = ["[source-grounded lossless memory ledger]"]
        source_ids: list[str] = []
        fact_count = 0
        for _score, fact in ranked[:self.ledger_fact_results]:
            ids = list(dict.fromkeys(
                str(value) for value in (fact.get("source_ids") or [])
                if str(value)
            ))
            subject = str(fact.get("subject") or "").strip()
            category = str(fact.get("category") or "other").strip()
            statement = " ".join(str(fact.get("statement") or "").split())
            event_date = " ".join(
                str(fact.get("event_date") or "").split()
            )
            source_dates = list(dict.fromkeys(
                " ".join(str(value).split())
                for value in (fact.get("source_dates") or [])
                if " ".join(str(value).split())
            ))
            values = list(dict.fromkeys(
                " ".join(str(value).split())
                for value in (fact.get("values") or [])
                if " ".join(str(value).split())
            ))
            details = [
                f"sources={','.join(ids)}" if ids else "sources=unknown",
                f"subject={subject}" if subject else "",
                f"category={category}",
                (
                    f"kind={fact.get('memory_kind') or fact.get('relation_type')}"
                    if fact.get("memory_kind") or fact.get("relation_type") else ""
                ),
                (
                    f"confidence={float(fact.get('confidence')):.2f}"
                    if fact.get("confidence") is not None else ""
                ),
                f"date={event_date}" if event_date else "",
                (
                    f"source_date={','.join(source_dates)}"
                    if source_dates else ""
                ),
            ]
            line = f"- [{' ; '.join(item for item in details if item)}] {statement}"
            missing_values = [
                value for value in values if value.casefold() not in statement.casefold()
            ]
            if missing_values:
                line += f" [exact_values: {', '.join(missing_values)}]"
            current_size = sum(len(item) + 1 for item in rendered)
            if current_size + len(line) > self.ledger_max_chars:
                continue
            rendered.append(line)
            source_ids.extend(ids)
            fact_count += 1
        if fact_count == 0:
            return None
        return {
            "content": "\n".join(rendered),
            "score": None,
            "source_ids": list(dict.fromkeys(source_ids)),
            "metadata": {
                "level": "L1X",
                "role": "lossless-ledger",
                "retrieval_strategy": "adapter_construction_ledger",
                "fact_count": fact_count,
            },
        }

    def _load_v3_l0_history(self, conversation_id: str) -> _V3L0History:
        """Load one canonical perspective's L0 history through the public API.

        Every perspective created by this adapter receives the same raw source
        sessions with different user/assistant roles. Loading a single scope is
        therefore enough for adjacency expansion and avoids duplicate traffic.
        The cache also makes the cost independent of the number of questions.
        """
        key = str(conversation_id)
        cached = self._l0_history_cache.get(key)
        if cached is not None:
            return cached
        perspectives = self._perspective_isolations(key)
        if not perspectives:
            empty = _V3L0History({}, {}, {}, False)
            self._l0_history_cache[key] = empty
            return empty
        perspective = perspectives[0]
        self._load_construction_trace_scopes()
        source_session_by_id = self._construction_source_session_by_id.get(
            (key, perspective.agent_id), {}
        )
        common = {
            "team_id": perspective.team_id,
            "agent_id": perspective.agent_id,
            "user_id": perspective.user_id,
        }
        if perspective.task_id:
            common["task_id"] = perspective.task_id
        raw_messages: list[_V3L0Message] = []
        offset = 0
        page_size = 100
        while True:
            response = self._post(
                "/v3/conversation/query",
                {**common, "limit": page_size, "offset": offset},
            )
            if response.get("code") not in (None, 0):
                raise RuntimeError(
                    f"TencentDB conversation/query failed for {key}: {response}"
                )
            data = response.get("data") or {}
            items = data.get("messages") or []
            for row in items:
                content = str(row.get("content") or "")
                source_ids = tuple(_v3_source_ids(content))
                source_session_id = next((
                    source_session_by_id[source_id]
                    for source_id in source_ids
                    if source_id in source_session_by_id
                ), str(row.get("session_id") or ""))
                raw_messages.append(_V3L0Message(
                    backend_id=str(row.get("id") or ""),
                    # Conversation-batched selected sessions intentionally
                    # share one backend cursor. Trace provenance restores the
                    # original source-session boundaries for L0 windows.
                    session_id=source_session_id,
                    role=str(row.get("role") or ""),
                    content=content,
                    source_ids=source_ids,
                    source_timestamp=_v3_source_timestamp(content),
                    # conversation/query exposes ingestion time here; unlike
                    # source time, it is unique per message and preserves the
                    # original within-session order.
                    backend_recorded_at=str(row.get("timestamp") or ""),
                ))
            offset += len(items)
            total = int(data.get("total") or 0)
            if not items or offset >= total:
                break

        grouped: dict[str, list[_V3L0Message]] = {}
        for message in raw_messages:
            if message.session_id:
                grouped.setdefault(message.session_id, []).append(message)
        sessions: dict[str, tuple[_V3L0Message, ...]] = {}
        by_backend_id: dict[str, tuple[str, int]] = {}
        by_source_id: dict[str, tuple[str, int]] = {}
        for session_id, messages in grouped.items():
            messages.sort(key=lambda item: (
                item.backend_recorded_at,
                item.backend_id,
            ))
            # A bounded L1 recovery can replay the exact source envelope with
            # a new backend ID/timestamp.  Keep the earliest canonical copy in
            # adjacency windows while retaining every backend ID as an alias,
            # so a semantic hit on either transport row resolves identically.
            deduplicated: list[_V3L0Message] = []
            canonical_indices: dict[tuple[str, ...], int] = {}
            backend_indices: dict[str, int] = {}
            for message in messages:
                key = (
                    ("source", *message.source_ids)
                    if message.source_ids
                    else ("backend", message.backend_id)
                )
                index = canonical_indices.get(key)
                if index is None:
                    index = len(deduplicated)
                    canonical_indices[key] = index
                    deduplicated.append(message)
                if message.backend_id:
                    backend_indices[message.backend_id] = index
            session = tuple(deduplicated)
            sessions[session_id] = session
            for backend_id, index in backend_indices.items():
                by_backend_id[backend_id] = (session_id, index)
            for index, message in enumerate(session):
                for source_id in message.source_ids:
                    by_source_id[source_id] = (session_id, index)
        history = _V3L0History(
            sessions,
            by_backend_id,
            by_source_id,
            bool(source_session_by_id) or self.l1_batch_mode == "session",
        )
        self._l0_history_cache[key] = history
        return history

    def _v3_structured_chain_rows(
        self, question: Question, *, limit: int
    ) -> list[dict]:
        """Select complete source evidence for common cockpit chain queries.

        Semantic top-k is a poor ordering primitive for owner-scoped
        preferences and ``latest/previous`` history questions. Project those
        two explicit query shapes over the already-stored canonical L0 event
        stream, then return only the source rows used by the projection. The
        projector is conservative: incomplete ownership/sequence coverage or
        ambiguity returns no rows and leaves the normal retrieval route intact.

        This path consumes neither benchmark evidence IDs nor expected answer
        values, and it never changes memory construction.
        """
        if (
            not self.structured_chain_retrieval
            or "L0" not in self.memory_layers
            or limit <= 0
        ):
            return []
        text = str(question.text or "")
        chain_intent = bool(
            re.search(r"(?:各自|分别).*(?:偏好|常用|习惯|默认|目的[地的])", text)
            or (
                re.search(r"(?:最后一次|最近一次|最新一次)", text)
                and re.search(r"(?:前一次|上一次)", text)
            )
        )
        if not chain_intent:
            return []
        history = self._load_v3_l0_history(question.conversation_id)
        messages = [
            message
            for session in history.sessions.values()
            for message in session
        ]
        # Fail closed on an unexpectedly large stream rather than turning a
        # targeted evidence-completeness route into an unbounded scan.
        if not messages or len(messages) > 1000:
            return []
        context = "\n".join(message.content for message in messages)
        if len(context) > 200_000:
            return []
        resolution = resolve_state_answer(
            text, context, metadata=question.metadata
        )
        if not resolution or resolution.get("reason") not in {
            "structured_owner_scoped_preferences",
            "structured_ordered_query_field",
        }:
            return []
        required = list(dict.fromkeys(
            str(source_id)
            for source_id in (resolution.get("source_ids") or [])
            if str(source_id)
        ))
        if not required or len(required) > limit:
            return []
        rows: list[dict] = []
        for source_id in required:
            location = history.by_source_id.get(source_id)
            if location is None:
                return []
            session_id, index = location
            message = history.sessions[session_id][index]
            rows.append({
                "content": message.content,
                "score": 2.0,
                "source_ids": list(message.source_ids or (source_id,)),
                "metadata": {
                    "level": "L0",
                    "role": _v3_source_role(message.content) or message.role,
                    "backend_message_id": message.backend_id,
                    "backend_recorded_at": message.backend_recorded_at,
                    "timestamp": message.source_timestamp,
                    "source_session_id": session_id,
                    "retrieval_strategy": "structured_chain_evidence_v1",
                    "structured_chain_reason": resolution["reason"],
                    "structured_chain_complete": True,
                },
            })
        return rows

    def _expand_v3_l0_windows(
        self, conversation_id: str, rows: list[dict], *, query: str = ""
    ) -> list[dict]:
        """Merge overlapping same-session windows around selected L0 hits."""
        if not rows or not (self.l0_window_before or self.l0_window_after):
            return rows
        history = self._load_v3_l0_history(conversation_id)
        if not history.source_session_provenance:
            # Conversation-batched L1 writes intentionally share one backend
            # transport session.  A separate retrieval process therefore
            # needs the construction trace to recover original source-session
            # boundaries.  If that provenance is absent, fail closed and keep
            # raw hits instead of silently joining unrelated cockpit commands.
            guarded: list[dict] = []
            for row in rows:
                copied = {**row, "metadata": dict(row.get("metadata") or {})}
                if copied["metadata"].get("level") == "L0":
                    copied["metadata"].update({
                        "window_expansion_skipped": (
                            "missing_source_session_provenance"
                        ),
                        "window_source_session_provenance": False,
                    })
                guarded.append(copied)
            return guarded
        passthrough: list[dict] = []
        windows: dict[str, dict] = {}
        for row in rows:
            metadata = row.get("metadata") or {}
            if metadata.get("level") != "L0":
                passthrough.append(row)
                continue
            location = history.by_backend_id.get(
                str(metadata.get("backend_message_id") or "")
            )
            if location is None:
                for source_id in row.get("source_ids", []):
                    location = history.by_source_id.get(str(source_id))
                    if location is not None:
                        break
            if location is None:
                passthrough.append(row)
                continue
            session_id, anchor_index = location
            session = history.sessions[session_id]
            # A short cockpit exchange is already within the hard message
            # budget, so retain the complete slot-filling episode. Fixed
            # radii can otherwise drop an initiating entity when semantic
            # search anchors on a later assistant answer. Long sessions still
            # use the bounded anchor window and cap below.
            full_short_session = bool(
                self.l0_window_max_messages
                and len(session) <= self.l0_window_max_messages
            )
            if full_short_session:
                start, end = 0, len(session)
            else:
                start = max(0, anchor_index - self.l0_window_before)
                end = min(
                    len(session), anchor_index + self.l0_window_after + 1
                )
            bucket = windows.setdefault(session_id, {
                "indices": set(),
                "anchors": [],
                "anchor_indices": set(),
                "row": row,
                "full_short_session": full_short_session,
            })
            bucket["indices"].update(range(start, end))
            bucket["anchors"].extend(str(item) for item in row.get("source_ids", []))
            bucket["anchor_indices"].add(anchor_index)
            bucket["full_short_session"] = bool(
                bucket["full_short_session"] or full_short_session
            )
            if float(row.get("score") or 0) > float(
                bucket["row"].get("score") or 0
            ):
                bucket["row"] = row

        for session_id, bucket in windows.items():
            selected_indices = sorted(bucket["indices"])
            candidate_message_count = len(selected_indices)
            if (
                self.l0_window_max_messages
                and len(selected_indices) > self.l0_window_max_messages
            ):
                anchor_indices = tuple(sorted(bucket["anchor_indices"]))

                def window_priority(index: int) -> tuple[int, int, int]:
                    nearest = min(
                        anchor_indices, key=lambda anchor: abs(index - anchor)
                    )
                    # At equal distance prefer the response following an
                    # assistant anchor over an earlier unrelated turn.
                    return abs(index - nearest), 0 if index >= nearest else 1, index

                selected_indices = sorted(
                    sorted(selected_indices, key=window_priority)[
                        :self.l0_window_max_messages
                    ]
                )
            messages = [
                history.sessions[session_id][index]
                for index in selected_indices
            ]
            anchor = bucket["row"]
            metadata = dict(anchor.get("metadata") or {})
            metadata.update({
                "role": "window",
                "window_session_id": session_id,
                "window_before": self.l0_window_before,
                "window_after": self.l0_window_after,
                "window_message_count": len(messages),
                "window_candidate_message_count": candidate_message_count,
                "window_max_messages": self.l0_window_max_messages,
                "window_full_short_session": bool(
                    bucket["full_short_session"]
                ),
                "window_truncated": len(messages) < candidate_message_count,
                "window_anchor_source_ids": list(dict.fromkeys(bucket["anchors"])),
            })
            if self.l0_humanize_time and _is_when_question(query) and messages:
                source = _parse_v3_source_datetime(messages[0].source_timestamp)
                if source is not None:
                    metadata["timestamp"] = _human_date(source)
            source_ids = tuple(dict.fromkeys(
                source_id
                for message in messages
                for source_id in message.source_ids
            ))
            perspectives = self._perspective_isolations(str(conversation_id))
            episode_mapping = (
                self._construction_trace_episodes_by_source_id.get(
                    (str(conversation_id), perspectives[0].agent_id), {}
                )
                if perspectives else {}
            )
            episode_payloads = {
                json.dumps(payload, ensure_ascii=False, sort_keys=True): payload
                for source_id in source_ids
                for payload in [episode_mapping.get(source_id)]
                if payload is not None
            }
            if len(episode_payloads) == 1:
                metadata["typed_cockpit_episode"] = next(
                    iter(episode_payloads.values())
                )
            elif len(episode_payloads) > 1:
                # A single L0 window must never silently pick between two
                # independent source transactions. The answer router will
                # therefore ignore the sidecar and parse/fallback normally.
                metadata["typed_cockpit_episode_ambiguous"] = len(
                    episode_payloads
                )
            anchor_source_ids = set(bucket["anchors"])
            rendered_messages = []
            for message in messages:
                is_anchor = bool(anchor_source_ids.intersection(message.source_ids))
                marker = "[retrieval_anchor] " if self.l0_mark_anchors and is_anchor else ""
                content = message.content
                if self.l0_humanize_time and _is_when_question(query):
                    content = _humanize_v3_source_time(
                        content, message.source_timestamp
                    )
                if (
                    self.l0_resolve_relative_time
                    and _is_temporal_question(query)
                ):
                    content = _annotate_v3_relative_time(
                        content, message.source_timestamp
                    )
                rendered_messages.append(marker + content)
            passthrough.append({
                "content": "\n".join(rendered_messages),
                "score": anchor.get("score"),
                "source_ids": list(source_ids),
                "metadata": metadata,
            })
        passthrough.sort(
            key=lambda row: float(row.get("score") or 0), reverse=True
        )
        return passthrough

    def _v3_l0_focus_row(
        self, conversation_id: str, query: str, rows: list[dict]
    ) -> dict | None:
        """Build a compact query-focused excerpt from selected semantic anchors.

        The backend remains responsible for semantic retrieval.  This block
        merely re-presents its selected raw anchors with immediate dialogue
        neighbours, which is useful when the semantic hit is an assistant's
        question and the factual answer is the next turn.  It uses neither
        benchmark evidence IDs nor an additional model call.
        """
        if self.l0_focus_anchors <= 0:
            return None
        if not _should_focus_question(query, self.l0_focus_modes):
            return None
        history = self._load_v3_l0_history(conversation_id)
        query_tokens = set(_focus_tokens(query))
        query_folded = query.casefold()
        subject_speakers = {
            perspective.speaker.casefold()
            for perspective in self._perspective_isolations(conversation_id)
            if perspective.speaker
            and perspective.speaker.casefold() in query_folded
        }
        candidates: list[tuple[float, float, int, int, str, int]] = []
        seen_anchors: set[tuple[str, int]] = set()
        seen_candidates: set[tuple[str, int]] = set()
        for row_rank, row in enumerate(rows):
            metadata = row.get("metadata") or {}
            if metadata.get("level") != "L0":
                continue
            session_id = str(metadata.get("window_session_id") or "")
            anchor_ids = metadata.get("window_anchor_source_ids") or row.get(
                "source_ids", []
            )
            for source_id in anchor_ids:
                location = history.by_source_id.get(str(source_id))
                if location is None:
                    continue
                anchor_session, anchor_index = location
                if session_id and anchor_session != session_id:
                    continue
                key = (anchor_session, anchor_index)
                if key in seen_anchors:
                    continue
                seen_anchors.add(key)
                session = history.sessions[anchor_session]
                if subject_speakers:
                    radius = self.l0_focus_response_radius
                    nearby_indices = [
                        index for index in range(
                            max(0, anchor_index - radius),
                            min(len(session), anchor_index + radius + 1),
                        )
                        if _v3_message_speaker(
                            session[index].content
                        ).casefold() in subject_speakers
                    ]
                else:
                    nearby_indices = [anchor_index]
                for candidate_index in nearby_indices:
                    candidate_key = (anchor_session, candidate_index)
                    if candidate_key in seen_candidates:
                        continue
                    seen_candidates.add(candidate_key)
                    message = session[candidate_index]
                    speaker = _v3_message_speaker(message.content).casefold()
                    body = _v3_message_body(message.content)
                    message_tokens = set(_focus_tokens(body))
                    lexical = float(len(query_tokens & message_tokens))
                    if speaker and speaker in query_folded:
                        lexical += 2.0
                    # A target-speaker question is often conversational glue;
                    # prefer a nearby declarative response at the same lexical
                    # relevance while retaining the question as a fallback.
                    if "?" in body:
                        lexical -= 0.5
                    candidates.append((
                        lexical,
                        float(row.get("score") or 0),
                        -abs(candidate_index - anchor_index),
                        -row_rank,
                        anchor_session,
                        candidate_index,
                    ))
        if not candidates:
            return None
        candidates.sort(reverse=True)

        selected_groups: list[tuple[str, tuple[int, ...]]] = []
        selected_messages: set[tuple[str, int]] = set()
        focus_anchor_limit = self.l0_focus_anchors
        if _is_emotion_question(query) or _is_inference_question(query):
            focus_anchor_limit = self.l0_reasoning_focus_anchors
        for _lexical, _score, _distance, _row_rank, session_id, anchor_index in (
            candidates[:focus_anchor_limit]
        ):
            session = history.sessions[session_id]
            start = max(0, anchor_index - self.l0_focus_before)
            end = min(len(session), anchor_index + self.l0_focus_after + 1)
            indices = tuple(
                index for index in range(start, end)
                if (session_id, index) not in selected_messages
            )
            if not indices:
                continue
            selected_groups.append((session_id, indices))
            selected_messages.update((session_id, index) for index in indices)

        rendered: list[str] = ["[query-focused raw dialogue excerpts]"]
        source_ids: list[str] = []
        temporal = self.l0_resolve_relative_time and _is_temporal_question(query)
        for session_id, indices in selected_groups:
            messages = history.sessions[session_id]
            block: list[str] = []
            for index in indices:
                message = messages[index]
                content = message.content
                if self.l0_humanize_time and _is_when_question(query):
                    content = _humanize_v3_source_time(
                        content, message.source_timestamp
                    )
                if temporal:
                    content = _annotate_v3_relative_time(
                        content, message.source_timestamp
                    )
                block.append(content)
                source_ids.extend(message.source_ids)
            candidate = "\n".join(block)
            current_size = sum(len(item) + 1 for item in rendered)
            if current_size + len(candidate) > self.l0_focus_max_chars:
                continue
            rendered.append(candidate)
        if len(rendered) == 1:
            return None
        return {
            "content": "\n".join(rendered),
            "score": None,
            "source_ids": list(dict.fromkeys(source_ids)),
            "metadata": {
                "level": "L0",
                "role": "focus",
                "retrieval_strategy": "semantic_anchor_focus",
                "focus_anchor_limit": focus_anchor_limit,
                "focus_before": self.l0_focus_before,
                "focus_after": self.l0_focus_after,
            },
        }

    def _annotate_v3_l0_sessions(
        self, conversation_id: str, rows: list[dict]
    ) -> None:
        """Attach backend session IDs to L0 candidates for diversification."""
        history = self._load_v3_l0_history(conversation_id)
        for row in rows:
            metadata = row.get("metadata") or {}
            if metadata.get("level") != "L0" or metadata.get("session_id"):
                continue
            location = history.by_backend_id.get(
                str(metadata.get("backend_message_id") or "")
            )
            if location is None:
                for source_id in row.get("source_ids", []):
                    location = history.by_source_id.get(str(source_id))
                    if location is not None:
                        break
            if location is not None:
                copied = dict(metadata)
                copied["session_id"] = location[0]
                row["metadata"] = copied

    def _v3_l0_session_bm25_rows(
        self, conversation_id: str, query: str
    ) -> list[dict]:
        """Create lexical L0 candidates from distinct sessions.

        TencentDB's semantic search remains the primary retriever. These
        candidates add a complementary, deterministic signal for exact names,
        objects, and repeated facts, while selecting at most one anchor per
        session so the L0 quota covers a wider set of conversations.
        """
        if self.l0_session_bm25_results <= 0:
            return []
        history = self._load_v3_l0_history(conversation_id)
        ranked = _rank_v3_l0_sessions(history, query)
        if not ranked:
            return []
        perspective = self._perspective_isolations(conversation_id)[0]
        rows: list[dict] = []
        for rank, (relevance, session_id, anchor_index) in enumerate(
            ranked[:self.l0_session_bm25_results], 1
        ):
            message = history.sessions[session_id][anchor_index]
            # TencentDB's hybrid search uses RRF-like scores around 1/60.
            # Reuse that scale so lexical and semantic candidates can be
            # merged without treating raw BM25 magnitudes as probabilities.
            fused_score = self.l0_session_bm25_weight / (60.0 + rank)
            rows.append({
                "content": message.content,
                "score": fused_score,
                "source_ids": list(message.source_ids),
                "metadata": {
                    "level": "L0",
                    "role": message.role,
                    "backend_message_id": message.backend_id,
                    "perspective_owner": perspective.speaker,
                    "agent_id": perspective.agent_id,
                    "task_id": perspective.task_id,
                    "backend_recorded_at": message.backend_recorded_at,
                    **({"timestamp": message.source_timestamp}
                       if message.source_timestamp else {}),
                    "retrieval_strategy": "session_bm25",
                    "session_bm25_rank": rank,
                    "session_bm25_score": relevance,
                    "session_id": session_id,
                },
            })
        return rows

    def _resolve_query_temporal(self, question: Question) -> TemporalQuery:
        if self.temporal_query_mode == "disabled":
            return TemporalQuery(None, self.temporal_default_timezone, "")
        return resolve_temporal_query(
            question.text,
            question.metadata,
            default_timezone=self.temporal_default_timezone,
        )

    def _v3_l0_temporal_rows(
        self,
        conversation_id: str,
        query: str,
        temporal: TemporalQuery,
    ) -> list[dict]:
        """Generate time-constrained L0 candidates before semantic top-k.

        This deliberately runs as a parallel candidate path rather than
        filtering an already truncated ANN result. It matches both the source
        turn time (when the driver spoke) and semantic event spans mentioned
        inside that turn (for example, "tomorrow morning").
        """
        if (
            self.temporal_query_mode == "disabled"
            or self.temporal_query_results <= 0
            or not temporal.spans
        ):
            return []
        history = self._load_v3_l0_history(conversation_id)
        if not history.sessions:
            return []
        query_tokens = set(_focus_tokens(query))
        preferred_dimension = _temporal_query_dimension(query)
        candidates: list[tuple[
            float, float, str, int, _V3L0Message, tuple[str, ...],
            tuple[str, ...], dict
        ]] = []
        for session_id, messages in history.sessions.items():
            best: tuple[
                float, float, str, int, _V3L0Message,
                tuple[str, ...], tuple[str, ...], dict
            ] | None = None
            best_key: tuple[float, float, float, float] | None = None
            for index, message in enumerate(messages):
                dimensions: set[str] = set()
                expressions: set[str] = set()
                source_time = parse_temporal_timestamp(
                    message.source_timestamp,
                    default_timezone=temporal.timezone_name,
                )
                if source_time is not None:
                    if {"cutoff", "as_of"} & set(temporal.operators):
                        cutoff_mode = "cutoff" in temporal.operators
                        boundary = (
                            min(span.end for span in temporal.spans)
                            if cutoff_mode else max(span.end for span in temporal.spans)
                        )
                        if source_time < boundary:
                            dimensions.add("cutoff_predecessor" if cutoff_mode else "as_of_predecessor")
                            expressions.add((min if cutoff_mode else max)(temporal.spans, key=lambda span: span.end).raw)
                    else:
                        for span in temporal.spans:
                            if span.contains(source_time):
                                dimensions.add("mentioned_at")
                                expressions.add(span.raw)

                if self.temporal_event_time_match:
                    event_temporal = resolve_temporal_query(
                        _temporal_payload(message.content),
                        {
                            "query_time": message.source_timestamp,
                            "timezone": temporal.timezone_name,
                        },
                        default_timezone=temporal.timezone_name,
                    )
                    if {"cutoff", "as_of"} & set(temporal.operators):
                        cutoff_mode = "cutoff" in temporal.operators
                        boundary = (
                            min(span.end for span in temporal.spans)
                            if cutoff_mode else max(span.end for span in temporal.spans)
                        )
                        if any(event_span.start < boundary for event_span in event_temporal.spans):
                            dimensions.add("event_time_predecessor")
                            expressions.add((min if cutoff_mode else max)(temporal.spans, key=lambda span: span.end).raw)
                    else:
                        for query_span in temporal.spans:
                            if any(
                                query_span.overlaps(event_span)
                                for event_span in event_temporal.spans
                            ):
                                dimensions.add("event_time")
                                expressions.add(query_span.raw)
                if not dimensions:
                    continue

                content_tokens = set(_focus_tokens(message.content))
                lexical = (
                    len(query_tokens & content_tokens) / len(query_tokens)
                    if query_tokens else 0.0
                )
                dimension_bonus = (
                    0.15 if preferred_dimension in dimensions else 0.0
                )
                state_slot_bonus = _shared_temporal_state_slot_bonus(
                    query, message.content
                )
                relevance = score_l0_candidate(
                    query, message.content, backend_rank=index
                )
                # This score places a hard temporal match ahead of approximate
                # dense candidates; exact-slot reranking still considers
                # lexical/entity coverage before the final Top-1 decision.
                score = (
                    1.25
                    + 0.5 * lexical
                    + 0.4 * relevance.critical_slot_coverage
                    + dimension_bonus
                    + state_slot_bonus
                )
                recency = source_time.timestamp() if source_time else 0.0
                item = (
                    score, recency, session_id, index, message,
                    tuple(sorted(dimensions)), tuple(sorted(expressions)),
                    relevance.metadata(),
                )
                # For a question about a past interaction, prefer the first
                # message satisfying all hard slots. A forward episode window
                # then retains both the initiating command and a later slot
                # answer under the same bounded context budget.
                if preferred_dimension == "mentioned_at":
                    selection_key = (
                        relevance.critical_slot_coverage,
                        -float(index),
                        relevance.score,
                        lexical,
                    )
                else:
                    selection_key = (
                        relevance.critical_slot_coverage,
                        relevance.score,
                        lexical,
                        -float(index),
                    )
                if best is None or selection_key > best_key:
                    best = item
                    best_key = selection_key
            if best is not None:
                candidates.append(best)

        latest_first = (
            "latest" in temporal.operators
            or "cutoff" in temporal.operators
            or "as_of" in temporal.operators
        )
        earliest_first = "earliest" in temporal.operators
        candidates.sort(key=lambda item: (
            -item[0],
            (-item[1] if latest_first else item[1] if earliest_first else 0),
            item[2],
        ))
        perspective = self._perspective_isolations(conversation_id)[0]
        rows: list[dict] = []
        for rank, (
            score, _recency, session_id, _index, message,
            dimensions, expressions, relevance,
        ) in enumerate(candidates[:self.temporal_query_results], 1):
            rows.append({
                "content": message.content,
                "score": score,
                "source_ids": list(message.source_ids),
                "metadata": {
                    "level": "L0",
                    "role": message.role,
                    "backend_message_id": message.backend_id,
                    "perspective_owner": perspective.speaker,
                    "agent_id": perspective.agent_id,
                    "task_id": perspective.task_id,
                    "backend_recorded_at": message.backend_recorded_at,
                    **({"timestamp": message.source_timestamp}
                       if message.source_timestamp else {}),
                    "retrieval_strategy": "temporal_interval_v1",
                    "temporal_candidate_rank": rank,
                    "temporal_match_dimensions": list(dimensions),
                    "temporal_match_expressions": list(expressions),
                    "temporal_anchor_relevance": relevance,
                    "session_id": session_id,
                },
            })
        return rows

    def _v3_typed_episode_direct_rows(
        self,
        question: Question,
        temporal: TemporalQuery,
    ) -> list[dict]:
        """Resolve one grounded typed episode without a MemoryCore read.

        The construction trace is a local, source-bound sidecar produced from
        the same L0 messages.  This path is deliberately strict: it requires a
        trusted temporal span, an active high-confidence episode, and the same
        answer compiler used after retrieval to identify exactly one row.  Any
        ambiguity falls through to the normal temporal/ANN retrieval path.
        """
        if (
            not self.typed_episode_short_circuit
            or not temporal.relative
            or not temporal.spans
        ):
            return []
        perspectives = self._perspective_isolations(question.conversation_id)
        if not perspectives:
            return []
        retrieval_strategy = "typed_episode_trace_v1"
        scope = (
            str(question.conversation_id), perspectives[0].agent_id,
        )
        if self.typed_episode_index_path is not None:
            if self._typed_episode_index is None:
                return []
            index_records = self._active_typed_episode_index_records(
                str(question.conversation_id), perspectives[0]
            )
            if index_records == () and not self._construction_trace_loaded:
                # One-time compatibility migration for runs built before the
                # SQLite index existed. New ingests dual-write directly.
                self._load_construction_trace_scopes()
                index_records = self._active_typed_episode_index_records(
                    str(question.conversation_id), perspectives[0]
                )
            if index_records is None:
                return []
            records = tuple({
                "session_id": record.session_id,
                "source_ids": list(record.source_ids),
                "episode": dict(record.episode),
                "record_revision": record.revision,
            } for record in index_records)
            retrieval_strategy = "typed_episode_sqlite_v1"
        else:
            self._load_construction_trace_scopes()
            if not self._construction_trace_valid:
                return []
            records = self._construction_trace_typed_episode_rows.get(scope, ())
        rows: list[dict] = []
        for record in records:
            raw_episode = record.get("episode") or {}
            episode = episode_from_dict(raw_episode)
            if (
                episode is None
                or episode.confidence < 0.97
                or episode.state not in {"selected", "confirmed"}
            ):
                continue
            mentioned_at = parse_temporal_timestamp(
                episode.mentioned_at,
                default_timezone=temporal.timezone_name,
            )
            if mentioned_at is None or not any(
                span.contains(mentioned_at) for span in temporal.spans
            ):
                continue
            source_ids = tuple(
                str(item) for item in record.get("source_ids") or []
                if str(item)
            )
            if (
                not source_ids
                or not episode.source_ids
                or not set(episode.source_ids).issubset(set(source_ids))
            ):
                continue
            rendered = [
                "[source-grounded typed cockpit episode]",
                f"Driver request: {episode.request_text}",
                f"Resolved destination: {episode.destination}",
            ]
            if episode.address:
                rendered.append(f"Address: {episode.address}")
            rows.append({
                "content": "\n".join(rendered),
                "score": 2.0,
                "source_ids": list(source_ids),
                "metadata": {
                    "level": "L0T",
                    "role": "typed-episode",
                    "retrieval_strategy": retrieval_strategy,
                    "session_id": str(record.get("session_id") or ""),
                    "typed_episode_record_revision": int(
                        record.get("record_revision") or 1
                    ),
                    "timestamp": episode.mentioned_at,
                    "typed_cockpit_episode": raw_episode,
                    "perspective_owner": perspectives[0].speaker,
                    "agent_id": perspectives[0].agent_id,
                    "task_id": perspectives[0].task_id,
                },
            })
        if not rows:
            return []
        candidate = extract_cockpit_answer(
            question.text,
            "",
            question.metadata,
            default_timezone=self.temporal_default_timezone,
            retrieval_hits=rows,
        )
        if candidate is None or candidate.reason != (
            "grounded_typed_navigation_episode"
        ):
            return []
        matched = [
            row for row in rows
            if tuple(
                episode_from_dict(
                    row["metadata"]["typed_cockpit_episode"]
                ).source_ids
            ) == candidate.source_ids
        ]
        return matched if len(matched) == 1 else []

    def _v3_l0_explicit_date_rows(
        self, conversation_id: str, query: str
    ) -> list[dict]:
        """Select source sessions constrained by an explicit ISO date.

        This uses only the query and source timestamps preserved at ingest.
        Starting from the first raw turn lets the normal same-session window
        include the complete interaction, even when semantic search would
        otherwise anchor on a later assistant question or a repeated topic.
        """
        dates = _explicit_source_dates(query)
        if self.l0_explicit_date_results <= 0 or not dates:
            return []
        history = self._load_v3_l0_history(conversation_id)
        if not history.sessions:
            return []
        query_tokens = set(_focus_tokens(query))
        candidates: list[tuple[float, str, _V3L0Message, str]] = []
        for session_id, messages in history.sessions.items():
            if not messages:
                continue
            matched = next((
                date for date in dates
                if any(
                    message.source_timestamp.startswith(date)
                    for message in messages
                )
            ), "")
            if not matched:
                continue
            lexical = float(len(query_tokens & {
                token
                for message in messages
                for token in _focus_tokens(message.content)
            }))
            candidates.append((lexical, session_id, messages[0], matched))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        perspective = self._perspective_isolations(conversation_id)[0]
        rows: list[dict] = []
        for rank, (lexical, session_id, message, matched) in enumerate(
            candidates[:self.l0_explicit_date_results], 1
        ):
            rows.append({
                "content": message.content,
                # Explicit source-time equality is a hard retrieval
                # constraint, so it must outrank approximate semantic scores.
                "score": 1.0 + lexical / 1000.0,
                "source_ids": list(message.source_ids),
                "metadata": {
                    "level": "L0",
                    "role": message.role,
                    "backend_message_id": message.backend_id,
                    "perspective_owner": perspective.speaker,
                    "agent_id": perspective.agent_id,
                    "task_id": perspective.task_id,
                    "backend_recorded_at": message.backend_recorded_at,
                    **({"timestamp": message.source_timestamp}
                       if message.source_timestamp else {}),
                    "retrieval_strategy": "explicit_source_date",
                    "explicit_source_date_candidate": matched,
                    "explicit_source_date_rank": rank,
                    "explicit_source_date_lexical_overlap": lexical,
                    "session_id": session_id,
                },
            })
        return rows

    def _rerank_v3_l0_rows(self, rows: list[dict], query: str) -> None:
        """Apply exact-slot-aware reranking to an already broad L0 pool."""
        if not self.adaptive_slot_rerank or len(rows) < 2:
            return
        for backend_rank, row in enumerate(rows):
            relevance = score_l0_candidate(
                query,
                str(row.get("content") or ""),
                backend_rank=backend_rank,
            )
            metadata = dict(row.get("metadata") or {})
            metadata["adapter_relevance"] = relevance.metadata()
            row["metadata"] = metadata
            row["adapter_relevance_score"] = relevance.score
        rows.sort(key=lambda row: (
            float(row.get("adapter_relevance_score") or 0.0),
            float(row.get("score") or 0.0),
        ), reverse=True)
        for rank, row in enumerate(rows, 1):
            metadata = dict(row.get("metadata") or {})
            metadata["adapter_rerank_position"] = rank
            row["metadata"] = metadata

    def _decide_adaptive_v3_l0(
        self,
        conversation_id: str,
        query: str,
        rows: list[dict],
        *,
        limit: int,
    ) -> tuple[list[dict], list[dict], object]:
        rows.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
        self._rerank_v3_l0_rows(rows, query)
        fast_raw = rows[:min(limit, self.adaptive_fast_l0_k)]
        fast_rows = self._expand_v3_l0_windows(
            conversation_id, fast_raw, query=query
        )
        decision_function = (
            decide_l0_fast_path_v2
            if self.adaptive_policy_version == "v2"
            else decide_l0_fast_path
        )
        ranking_score_key = (
            "adapter_relevance_score" if self.adaptive_slot_rerank else "score"
        )

        def ranking_score(row: dict) -> float | None:
            value = row.get(ranking_score_key)
            if value is None:
                value = row.get("score")
            return float(value) if value is not None else None

        decision = decision_function(
            query,
            "\n".join(str(row.get("content") or "") for row in fast_rows),
            top_score=ranking_score(rows[0]) if rows else None,
            second_score=ranking_score(rows[1]) if len(rows) > 1 else None,
            min_coverage=self.adaptive_min_coverage,
            min_score_margin=self.adaptive_min_score_margin,
        )
        return rows, fast_rows, decision

    def _adaptive_v3_routing_metadata(
        self,
        decision,
        *,
        search_calls: int,
        context_budget: int,
        profile_hits: int,
        profile_levels: set[str],
        readiness_layers: list[str],
        readiness_seconds: float,
        unready_layers: list[str],
        temporal: TemporalQuery,
        temporal_rows: list[dict],
        temporal_normalization_seconds: float,
        temporal_candidate_seconds: float,
        temporal_short_circuit_used: bool,
    ) -> dict:
        return {
            "retrieval_policy": (
                f"adaptive_l0_first_{self.adaptive_policy_version}"
            ),
            "adaptive_policy_version": self.adaptive_policy_version,
            "adaptive_slot_rerank": self.adaptive_slot_rerank,
            "retrieval_route": decision.route,
            "adaptive_fallback": decision.fallback,
            "adaptive_reason": decision.reason,
            "adaptive_decision": decision.metadata(),
            "adaptive_search_calls": search_calls,
            "retrieval_context_budget_chars": context_budget,
            "retrieval_context_line_overflow_chars": (
                self.adaptive_context_line_overflow_chars
            ),
            "adaptive_fast_l0_k": self.adaptive_fast_l0_k,
            "adaptive_fallback_l0_k": self.adaptive_fallback_l0_k,
            "adaptive_fallback_l1_k": self.adaptive_fallback_l1_k,
            "adaptive_fallback_l2_k": self.adaptive_fallback_l2_k,
            "adaptive_fallback_l3_k": self.adaptive_fallback_l3_k,
            "adaptive_profile_hits": profile_hits,
            "adaptive_profile_levels": sorted(profile_levels),
            "adaptive_readiness_layers": list(dict.fromkeys(readiness_layers)),
            "adaptive_readiness_seconds": readiness_seconds,
            "adaptive_unready_layers": list(dict.fromkeys(unready_layers)),
            "l23_schedule": self.l23_schedule,
            "temporal_query_mode": self.temporal_query_mode,
            "temporal_query_active": temporal.active,
            "temporal_query_relative": temporal.relative,
            "temporal_query_anchor": (
                temporal.anchor.isoformat() if temporal.anchor else ""
            ),
            "temporal_query_timezone": temporal.timezone_name,
            "temporal_query_anchor_source": temporal.anchor_source,
            "temporal_query_spans": [span.to_dict() for span in temporal.spans],
            "temporal_query_operators": list(temporal.operators),
            "temporal_candidate_hits": len(temporal_rows),
            "temporal_normalization_seconds": temporal_normalization_seconds,
            "temporal_candidate_seconds": temporal_candidate_seconds,
            "temporal_short_circuit_enabled": self.temporal_short_circuit,
            "temporal_short_circuit_used": temporal_short_circuit_used,
            "typed_episode_short_circuit_enabled": (
                self.typed_episode_short_circuit
            ),
            "typed_episode_short_circuit_used": False,
            "typed_episode_candidate_hits": 0,
            "typed_episode_candidate_seconds": 0.0,
        }

    @staticmethod
    def _memory_hits_with_routing(
        rows: list[dict], routing: dict, *, limit: int
    ) -> list[MemoryHit]:
        routed_rows = []
        for row in rows[:limit]:
            routed = {**row, "metadata": dict(row.get("metadata") or {})}
            routed["metadata"].update(routing)
            routed_rows.append(routed)
        return [MemoryHit(
            str(row.get("content", "")),
            float(row["score"]) if row.get("score") is not None else None,
            tuple(str(item) for item in row.get("source_ids", [])),
            dict(row.get("metadata", {})),
        ) for row in routed_rows]

    def _adaptive_v3_search(
        self, question: Question, *, limit: int,
    ) -> list[MemoryHit]:
        """Run an L0-first search and touch L1 only for low-confidence queries."""
        temporal_started = time.monotonic()
        temporal = self._resolve_query_temporal(question)
        backend_query = temporal.retrieval_text(question.text)
        temporal_normalization_seconds = time.monotonic() - temporal_started
        structured_started = time.monotonic()
        structured_rows = self._v3_structured_chain_rows(
            question, limit=limit
        )
        structured_candidate_seconds = time.monotonic() - structured_started
        if structured_rows:
            decision = AdaptiveDecision(
                route="fast",
                fallback=False,
                reason="grounded_structured_chain",
                lexical_coverage=1.0,
                score_margin=1.0,
                quoted_anchor_match=True,
                complex_query=True,
                critical_slot_coverage=1.0,
                missing_critical_slots=(),
                dialogue_complete=True,
            )
            routing = self._adaptive_v3_routing_metadata(
                decision,
                search_calls=0,
                context_budget=self.adaptive_fast_context_chars,
                profile_hits=0,
                profile_levels=set(),
                readiness_layers=[],
                readiness_seconds=0.0,
                unready_layers=[],
                temporal=temporal,
                temporal_rows=[],
                temporal_normalization_seconds=temporal_normalization_seconds,
                temporal_candidate_seconds=0.0,
                temporal_short_circuit_used=False,
            )
            routing.update({
                "structured_chain_short_circuit_used": True,
                "structured_chain_candidate_hits": len(structured_rows),
                "structured_chain_candidate_seconds": (
                    structured_candidate_seconds
                ),
            })
            return self._memory_hits_with_routing(
                structured_rows, routing, limit=limit
            )
        perspectives = self._perspective_isolations(question.conversation_id)
        typed_episode_started = time.monotonic()
        typed_episode_rows = self._v3_typed_episode_direct_rows(
            question, temporal
        )
        typed_episode_candidate_seconds = (
            time.monotonic() - typed_episode_started
        )
        if typed_episode_rows:
            decision = AdaptiveDecision(
                route="fast",
                fallback=False,
                reason="grounded_typed_episode",
                lexical_coverage=1.0,
                score_margin=1.0,
                quoted_anchor_match=True,
                complex_query=False,
                critical_slot_coverage=1.0,
                missing_critical_slots=(),
                dialogue_complete=True,
            )
            routing = self._adaptive_v3_routing_metadata(
                decision,
                search_calls=0,
                context_budget=self.adaptive_fast_context_chars,
                profile_hits=0,
                profile_levels=set(),
                readiness_layers=[],
                readiness_seconds=0.0,
                unready_layers=[],
                temporal=temporal,
                temporal_rows=[],
                temporal_normalization_seconds=temporal_normalization_seconds,
                temporal_candidate_seconds=0.0,
                temporal_short_circuit_used=False,
            )
            routing.update({
                "typed_episode_short_circuit_used": True,
                "typed_episode_candidate_hits": len(typed_episode_rows),
                "typed_episode_candidate_seconds": (
                    typed_episode_candidate_seconds
                ),
            })
            return self._memory_hits_with_routing(
                typed_episode_rows, routing, limit=limit
            )
        temporal_candidate_started = time.monotonic()
        temporal_rows = self._v3_l0_temporal_rows(
            question.conversation_id, backend_query, temporal
        )
        temporal_candidate_seconds = time.monotonic() - temporal_candidate_started
        if self.temporal_short_circuit and temporal_rows:
            temporal_only = list(temporal_rows)
            if self.l0_explicit_date_boost:
                _boost_v3_explicit_source_dates(temporal_only, backend_query)
            temporal_only, temporal_fast_rows, temporal_decision = (
                self._decide_adaptive_v3_l0(
                    question.conversation_id,
                    backend_query,
                    temporal_only,
                    limit=limit,
                )
            )
            if not temporal_decision.fallback:
                routing = self._adaptive_v3_routing_metadata(
                    temporal_decision,
                    search_calls=0,
                    context_budget=self.adaptive_fast_context_chars,
                    profile_hits=0,
                    profile_levels=set(),
                    readiness_layers=[],
                    readiness_seconds=0.0,
                    unready_layers=[],
                    temporal=temporal,
                    temporal_rows=temporal_rows,
                    temporal_normalization_seconds=(
                        temporal_normalization_seconds
                    ),
                    temporal_candidate_seconds=temporal_candidate_seconds,
                    temporal_short_circuit_used=True,
                )
                return self._memory_hits_with_routing(
                    temporal_fast_rows, routing, limit=limit
                )
        candidate_limit = min(100, max(
            self.adaptive_fallback_l0_k,
            self.adaptive_fallback_l0_k * self.candidate_multiplier,
        ))
        l0_rows: list[dict] = []
        search_calls = 0
        for perspective in perspectives:
            team_id, agent_id, user_id, task_id = perspective.isolation()
            common = {
                "team_id": team_id,
                "agent_id": agent_id,
                "user_id": user_id,
                "query": backend_query,
                "limit": candidate_limit,
            }
            if task_id:
                common["task_id"] = task_id
            response = self._post("/v3/conversation/search", common)
            search_calls += 1
            for row in (response.get("data") or {}).get("messages", []):
                content = str(row.get("content", ""))
                source_time = _v3_source_timestamp(content)
                l0_rows.append({
                    "content": content,
                    "score": row.get("score"),
                    "source_ids": _v3_source_ids(content),
                    "metadata": {
                        "level": "L0",
                        "role": row.get("role", ""),
                        "backend_message_id": str(row.get("id", "")),
                        "perspective_owner": perspective.speaker,
                        "agent_id": perspective.agent_id,
                        "task_id": perspective.task_id,
                        "backend_recorded_at": row.get("timestamp", ""),
                        **({"timestamp": source_time} if source_time else {}),
                    },
                })
        l0_rows.extend(temporal_rows)
        if self.l0_explicit_date_results:
            l0_rows.extend(self._v3_l0_explicit_date_rows(
                question.conversation_id, question.text
            ))
        if self.l0_session_bm25_results:
            l0_rows.extend(self._v3_l0_session_bm25_rows(
                question.conversation_id, backend_query
            ))
        deduplicated: dict[tuple[str, ...], dict] = {}
        for row in l0_rows:
            key = _v3_hit_key(row)
            previous = deduplicated.get(key)
            if previous is None or float(row.get("score") or 0) > float(
                previous.get("score") or 0
            ):
                deduplicated[key] = row
        l0_rows = list(deduplicated.values())
        if self.l0_explicit_date_boost:
            _boost_v3_explicit_source_dates(l0_rows, backend_query)
        l0_rows, fast_rows, decision = self._decide_adaptive_v3_l0(
            question.conversation_id,
            backend_query,
            l0_rows,
            limit=limit,
        )
        selected = fast_rows
        context_budget = self.adaptive_fast_context_chars
        profile_hits = 0
        profile_levels: set[str] = set()
        readiness_layers: list[str] = []
        readiness_seconds = 0.0
        unready_layers: list[str] = []

        if decision.fallback:
            fallback_raw = l0_rows[:min(limit, self.adaptive_fallback_l0_k)]
            selected = self._expand_v3_l0_windows(
                question.conversation_id, fallback_raw, query=backend_query
            )
            remaining = max(0, limit - len(selected))
            if (
                remaining
                and self.adaptive_lazy_layer_readiness
                and self.adaptive_layer_wait_budget_seconds > 0
                and "L1" in self.memory_layers
                and self.adaptive_fallback_l1_k
            ):
                ready_started = time.monotonic()
                self._wait_until_ready_layers(
                    question.conversation_id,
                    timeout=self._adaptive_readiness_timeout(
                        question.conversation_id
                    ),
                    layers=frozenset({"L1"}),
                )
                readiness_seconds += time.monotonic() - ready_started
                readiness_layers.append("L1")
            if (
                remaining
                and "L1" in self.memory_layers
                and self.adaptive_fallback_l1_k
            ):
                l1_rows: list[dict] = []
                for perspective in perspectives:
                    team_id, agent_id, user_id, task_id = perspective.isolation()
                    common = {
                        "team_id": team_id,
                        "agent_id": agent_id,
                        "user_id": user_id,
                        "query": backend_query,
                        "limit": min(
                            100,
                            max(
                                self.adaptive_fallback_l1_k,
                                self.adaptive_fallback_l1_k
                                * self.candidate_multiplier,
                            ),
                        ),
                    }
                    if task_id:
                        common["task_id"] = task_id
                    atomic = self._post("/v3/atomic/search", common)
                    search_calls += 1
                    for row in (atomic.get("data") or {}).get("items", []):
                        l1_rows.append({
                            "content": row.get("content", ""),
                            "score": row.get("score"),
                            "source_ids": [],
                            "metadata": {
                                "level": "L1",
                                "type": row.get("type", ""),
                                "backend_memory_id": str(row.get("id", "")),
                                "perspective_owner": perspective.speaker,
                                "agent_id": perspective.agent_id,
                                "task_id": perspective.task_id,
                            },
                        })
                l1_rows.sort(
                    key=lambda row: float(row.get("score") or 0), reverse=True
                )
                selected.extend(l1_rows[:min(
                    remaining, self.adaptive_fallback_l1_k
                )])
            remaining = max(0, limit - len(selected))
            if (
                remaining
                and {"L2", "L3"} & self.memory_layers
                and (
                    self.adaptive_fallback_l2_k
                    or self.adaptive_fallback_l3_k
                )
            ):
                profiles_ready = True
                if self.adaptive_lazy_layer_readiness:
                    if self.adaptive_layer_wait_budget_seconds > 0:
                        ready_started = time.monotonic()
                        self._wait_until_ready_layers(
                            question.conversation_id,
                            timeout=self._adaptive_readiness_timeout(
                                question.conversation_id
                            ),
                            layers=self.memory_layers,
                        )
                        readiness_seconds += time.monotonic() - ready_started
                        readiness_layers.extend(
                            layer for layer in ("L2", "L3")
                            if layer in self.memory_layers
                        )
                    else:
                        profiles_ready = self._profile_layers_ready(
                            question.conversation_id, self.memory_layers
                        )
                        if not profiles_ready:
                            unready_layers.extend(
                                layer for layer in ("L2", "L3")
                                if layer in self.memory_layers
                            )
                profiles = (
                    self._load_v3_profiles(question.conversation_id)
                    if profiles_ready else ()
                )
                l2_rows = [
                    row for row in profiles
                    if (row.get("metadata") or {}).get("level") == "L2"
                ]
                l3_rows = [
                    row for row in profiles
                    if (row.get("metadata") or {}).get("level") == "L3"
                ]
                if "L2" in self.memory_layers and self.adaptive_fallback_l2_k:
                    ranked_l2 = _rank_v3_profiles(l2_rows, question.text)
                    for relevance, row in ranked_l2[:min(
                        remaining, self.adaptive_fallback_l2_k
                    )]:
                        routed = {**row, "metadata": dict(row["metadata"])}
                        routed["content"] = _profile_excerpt(
                            str(row["content"]), question.text, self.l2_max_chars
                        )
                        routed["metadata"]["profile_relevance_score"] = relevance
                        selected.append(routed)
                        profile_hits += 1
                        profile_levels.add("L2")
                    remaining = max(0, limit - len(selected))
                if (
                    remaining
                    and "L3" in self.memory_layers
                    and self.adaptive_fallback_l3_k
                ):
                    question_folded = question.text.casefold()
                    l3_rows.sort(key=lambda row: (
                        str((row.get("metadata") or {}).get(
                            "perspective_owner", ""
                        )).casefold() not in question_folded,
                        str((row.get("metadata") or {}).get(
                            "perspective_owner", ""
                        )),
                    ))
                    for row in l3_rows[:min(
                        remaining, self.adaptive_fallback_l3_k
                    )]:
                        routed = {**row, "metadata": dict(row["metadata"])}
                        routed["content"] = str(row["content"])[
                            :self.l3_max_chars
                        ]
                        selected.append(routed)
                        profile_hits += 1
                        profile_levels.add("L3")
            context_budget = self.adaptive_fallback_context_chars

        routing = self._adaptive_v3_routing_metadata(
            decision,
            search_calls=search_calls,
            context_budget=context_budget,
            profile_hits=profile_hits,
            profile_levels=profile_levels,
            readiness_layers=readiness_layers,
            readiness_seconds=readiness_seconds,
            unready_layers=unready_layers,
            temporal=temporal,
            temporal_rows=temporal_rows,
            temporal_normalization_seconds=temporal_normalization_seconds,
            temporal_candidate_seconds=temporal_candidate_seconds,
            temporal_short_circuit_used=False,
        )
        return self._memory_hits_with_routing(selected, routing, limit=limit)

    def _adaptive_readiness_timeout(self, conversation_id: str) -> float:
        deadline = self._adaptive_ready_deadlines.get(str(conversation_id))
        remaining = (
            float(self.timeout) if deadline is None
            else max(1.0, deadline - time.monotonic())
        )
        return min(remaining, self.adaptive_layer_wait_budget_seconds)

    def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
        if self.api_version == "v3":
            if self.retrieval_policy == "adaptive":
                return self._adaptive_v3_search(question, limit=limit)
            temporal_started = time.monotonic()
            temporal = self._resolve_query_temporal(question)
            backend_query = temporal.retrieval_text(question.text)
            temporal_normalization_seconds = time.monotonic() - temporal_started
            structured_started = time.monotonic()
            structured_rows = self._v3_structured_chain_rows(
                question, limit=limit
            )
            structured_candidate_seconds = time.monotonic() - structured_started
            if structured_rows:
                metadata = {
                    "structured_chain_short_circuit_used": True,
                    "structured_chain_candidate_hits": len(structured_rows),
                    "structured_chain_candidate_seconds": (
                        structured_candidate_seconds
                    ),
                    "temporal_query_mode": self.temporal_query_mode,
                    "temporal_query_active": temporal.active,
                    "temporal_query_relative": temporal.relative,
                }
                for row in structured_rows:
                    row["metadata"].update(metadata)
                return self._memory_hits_with_routing(
                    structured_rows, {}, limit=limit
                )
            rows: list[dict] = []
            candidate_limit = min(100, max(limit, limit * self.candidate_multiplier))
            for perspective in self._perspective_isolations(
                question.conversation_id
            ):
                team_id, agent_id, user_id, task_id = perspective.isolation()
                common = {
                    "team_id": team_id,
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "query": backend_query,
                    "limit": candidate_limit,
                }
                if task_id:
                    common["task_id"] = task_id
                if "L1" in self.memory_layers:
                    atomic = self._post("/v3/atomic/search", common)
                    for row in (atomic.get("data") or {}).get("items", []):
                        rows.append({
                            "content": row.get("content", ""),
                            "score": row.get("score"),
                            # The current v3 L1 response exposes a memory ID but
                            # no exact source-turn lineage. Keep that backend ID
                            # as metadata rather than inventing a source turn.
                            "source_ids": [],
                            "metadata": {
                                "level": "L1",
                                "type": row.get("type", ""),
                                "backend_memory_id": str(row.get("id", "")),
                                "perspective_owner": perspective.speaker,
                                "agent_id": perspective.agent_id,
                                "task_id": perspective.task_id,
                            },
                        })
                # Each source Session has its own Gateway session_id. Omitting
                # it retrieves the full history inside this perspective task.
                if "L0" in self.memory_layers:
                    conversation = self._post("/v3/conversation/search", common)
                    for row in (conversation.get("data") or {}).get("messages", []):
                        content = str(row.get("content", ""))
                        source_time = _v3_source_timestamp(content)
                        rows.append({
                            "content": content,
                            "score": row.get("score"),
                            "source_ids": _v3_source_ids(content),
                            "metadata": {
                                "level": "L0",
                                "role": row.get("role", ""),
                                "backend_message_id": str(row.get("id", "")),
                                "perspective_owner": perspective.speaker,
                                "agent_id": perspective.agent_id,
                                "task_id": perspective.task_id,
                                # This API field is ingestion/recording time.
                                # Restore actual event time from our envelope.
                                "backend_recorded_at": row.get("timestamp", ""),
                                **({"timestamp": source_time} if source_time else {}),
                            },
                        })
            temporal_candidate_started = time.monotonic()
            temporal_rows = (
                self._v3_l0_temporal_rows(
                    question.conversation_id, backend_query, temporal
                )
                if "L0" in self.memory_layers else []
            )
            temporal_candidate_seconds = (
                time.monotonic() - temporal_candidate_started
            )
            rows.extend(temporal_rows)
            if "L0" in self.memory_layers and self.l0_explicit_date_results:
                rows.extend(self._v3_l0_explicit_date_rows(
                    question.conversation_id, question.text
                ))
            if "L0" in self.memory_layers and self.l0_session_bm25_results:
                rows.extend(self._v3_l0_session_bm25_rows(
                    question.conversation_id, backend_query
                ))
            # Both perspective tasks contain the same L0 source turns. Merge
            # exact duplicates before applying top-k so replicated raw rows do
            # not crowd out complementary L1 memories.
            deduplicated: dict[tuple[str, ...], dict] = {}
            for row in rows:
                key = _v3_hit_key(row)
                previous = deduplicated.get(key)
                if previous is None or float(row.get("score") or 0) > float(
                    previous.get("score") or 0
                ):
                    deduplicated[key] = row
            rows = list(deduplicated.values())
            if self.l0_explicit_date_boost:
                _boost_v3_explicit_source_dates(rows, backend_query)
            rows.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
            profile_rows: list[dict] = []
            if {"L2", "L3"} & self.memory_layers:
                profiles = self._load_v3_profiles(question.conversation_id)
                l2_rows = [
                    row for row in profiles
                    if (row.get("metadata") or {}).get("level") == "L2"
                ]
                l3_rows = [
                    row for row in profiles
                    if (row.get("metadata") or {}).get("level") == "L3"
                ]
                ranked_l2 = _rank_v3_profiles(l2_rows, question.text)
                l2_limit = min(self.l2_results, len(ranked_l2))
                l3_limit = min(self.l3_results, len(l3_rows))
                # Keep at least one dynamic L0/L1 slot when those layers are
                # enabled, even for unusually small top-k smoke tests.
                profile_capacity = limit
                if {"L0", "L1"} & self.memory_layers and rows:
                    profile_capacity = max(0, limit - 1)
                while l2_limit + l3_limit > profile_capacity:
                    if l2_limit >= l3_limit and l2_limit > 0:
                        l2_limit -= 1
                    elif l3_limit > 0:
                        l3_limit -= 1
                    else:
                        break
                for relevance, row in ranked_l2[:l2_limit]:
                    selected = {**row, "metadata": dict(row["metadata"])}
                    selected["content"] = _profile_excerpt(
                        str(row["content"]), question.text, self.l2_max_chars
                    )
                    selected["metadata"]["profile_relevance_score"] = relevance
                    profile_rows.append(selected)
                question_folded = question.text.casefold()
                l3_rows.sort(key=lambda row: (
                    str((row.get("metadata") or {}).get(
                        "perspective_owner", ""
                    )).casefold() not in question_folded,
                    str((row.get("metadata") or {}).get(
                        "perspective_owner", ""
                    )),
                ))
                for row in l3_rows[:l3_limit]:
                    selected = {**row, "metadata": dict(row["metadata"])}
                    selected["content"] = str(row["content"])[:self.l3_max_chars]
                    profile_rows.append(selected)
            ledger_row = self._v3_ledger_row(
                question.conversation_id, question.text
            )
            ledger_rows = [ledger_row] if ledger_row is not None else []
            dynamic_limit = max(
                0, limit - len(profile_rows) - len(ledger_rows)
            )
            if self.l0_diverse_results:
                self._annotate_v3_l0_sessions(question.conversation_id, rows)
            dynamic_rows = _select_v3_dynamic_rows(
                rows,
                dynamic_limit,
                l0_min_results=max(
                    self.l0_min_results,
                    math.ceil(dynamic_limit * self.l0_min_fraction),
                ),
                l0_diverse_results=self.l0_diverse_results,
            )
            dynamic_rows = self._expand_v3_l0_windows(
                question.conversation_id, dynamic_rows, query=backend_query
            )
            focus_row = self._v3_l0_focus_row(
                question.conversation_id, question.text, dynamic_rows
            )
            if self.l0_first:
                dynamic_rows.sort(key=lambda row: (
                    0 if (row.get("metadata") or {}).get("level") == "L0" else 1,
                    -float(row.get("score") or 0),
                ))
            if self.l0_last:
                non_l0 = [
                    row for row in dynamic_rows
                    if (row.get("metadata") or {}).get("level") != "L0"
                ]
                l0_rows = [
                    row for row in dynamic_rows
                    if (row.get("metadata") or {}).get("level") == "L0"
                ]
                rows = non_l0 + ledger_rows + profile_rows + l0_rows
            else:
                rows = dynamic_rows + ledger_rows + profile_rows
            rows = rows[:limit]
            if focus_row is not None:
                if len(rows) < limit:
                    rows.append(focus_row)
                else:
                    replace_index = next((
                        index for index in range(len(rows) - 1, -1, -1)
                        if not rows[index].get("source_ids")
                    ), None)
                    if replace_index is not None:
                        rows.pop(replace_index)
                        rows.append(focus_row)
                    elif rows:
                        # A raw-only configuration may have no source-less L1
                        # or profile hit to trade for the focus block. Preserve
                        # the final hit and append the duplicate excerpt inside
                        # it so the requested hit limit and source recall stay
                        # unchanged.
                        final = {**rows[-1], "metadata": dict(
                            rows[-1].get("metadata") or {}
                        )}
                        final["content"] = (
                            f"{final.get('content', '')}\n\n{focus_row['content']}"
                        )
                        final["source_ids"] = list(dict.fromkeys((
                            *rows[-1].get("source_ids", []),
                            *focus_row.get("source_ids", []),
                        )))
                        final["metadata"]["focus_appended"] = True
                        rows[-1] = final
            temporal_metadata = {
                "temporal_query_mode": self.temporal_query_mode,
                "temporal_query_active": temporal.active,
                "temporal_query_relative": temporal.relative,
                "temporal_query_anchor": (
                    temporal.anchor.isoformat() if temporal.anchor else ""
                ),
                "temporal_query_timezone": temporal.timezone_name,
                "temporal_query_anchor_source": temporal.anchor_source,
                "temporal_query_spans": [
                    span.to_dict() for span in temporal.spans
                ],
                "temporal_query_operators": list(temporal.operators),
                "temporal_candidate_hits": len(temporal_rows),
                "temporal_normalization_seconds": temporal_normalization_seconds,
                "temporal_candidate_seconds": temporal_candidate_seconds,
            }
            for row in rows:
                metadata = dict(row.get("metadata") or {})
                metadata.update(temporal_metadata)
                row["metadata"] = metadata
            return [MemoryHit(
                str(row.get("content", "")),
                float(row["score"]) if row.get("score") is not None else None,
                tuple(str(item) for item in row.get("source_ids", [])),
                dict(row.get("metadata", {})),
            ) for row in rows[:limit]]
        response = self._post("/recall", {
            "session_key": question.conversation_id,
            "query": question.text,
            "limit": limit,
            "include_l0": True,
            "multi_hop": True,
        })
        if isinstance(response.get("hits"), list):
            rows = response["hits"]
        else:
            text = response.get("context") or response.get("prependContext") or ""
            rows = [{"content": text, "metadata": {"raw_response": response}}] if text else []
        return [MemoryHit(
            str(row.get("content", "")),
            float(row["score"]) if row.get("score") is not None else None,
            tuple(str(item) for item in row.get("source_ids", [])),
            dict(row.get("metadata", {})),
            tuple(ContentPart.from_dict(part) for part in row.get("parts", [])),
        ) for row in rows]


def _temporal_payload(content: str) -> str:
    """Remove adapter envelopes before resolving an event's semantic time."""
    value = re.sub(r"\[(?:source_time|source_date)=[^\]]+\]", "", str(content or ""))
    value = re.sub(r"\[resolved_relative_time:[^\]]+\]", "", value)
    return value


def _temporal_query_dimension(query: str) -> str:
    folded = str(query or "").casefold()
    mention_patterns = (
        r"(?:说|问|提到|告诉|聊到|让你|叫你|吩咐|请求|那次对话|那次交互|语音交互)",
        r"\b(?:say|said|ask|asked|tell|told|mention|mentioned|request|"
        r"requested|specify|specified)\b",
        r"\bduring\b[^\n]{0,80}\b(?:interaction|conversation)\b",
    )
    return (
        "mentioned_at"
        if any(re.search(pattern, folded) for pattern in mention_patterns)
        else "event_time"
    )


_COCKPIT_STATE_SLOTS = (
    "临时目的地", "午饭地点", "过夜地点", "会合地点", "夜间导航目的地",
    "导航播报音量", "路线偏好", "空调温度", "座椅温度", "车载内容",
)


def _shared_temporal_state_slot_bonus(query: str, content: str) -> float:
    """Prefer predecessor events that mention the state slot being queried.

    This only affects the temporal candidate lane.  It never extracts a value
    or relies on a benchmark label; it prevents unrelated but newer events
    from evicting the state transitions needed for an as-of snapshot.
    """
    normalized_query = str(query or "").replace("目的的", "目的地")
    normalized_content = str(content or "").replace("目的的", "目的地")
    return 0.4 if any(
        slot in normalized_query and slot in normalized_content
        for slot in _COCKPIT_STATE_SLOTS
    ) else 0.0
def _explicit_source_dates(query: str) -> set[str]:
    return set(re.findall(
        r"(?<!\d)20\d{2}-\d{2}-\d{2}(?!\d)", str(query or "")
    ))


def _boost_v3_explicit_source_dates(rows: list[dict], query: str) -> None:
    """Boost raw turns only when a query contains an exact ISO log date.

    This is intentionally opt-in. It does not globally prefer recent or old
    memories; it only resolves an explicit temporal key already present in the
    query against source provenance preserved by the adapter.
    """
    dates = _explicit_source_dates(query)
    if not dates:
        return
    for row in rows:
        metadata = row.get("metadata") or {}
        if metadata.get("level") != "L0":
            continue
        source_time = str(
            metadata.get("timestamp")
            or _v3_source_timestamp(str(row.get("content") or ""))
        )
        matched = next((value for value in dates if source_time.startswith(value)), "")
        if not matched:
            continue
        row["score"] = float(row.get("score") or 0) + 1.0
        metadata = dict(metadata)
        metadata["explicit_source_date_boost"] = matched
        row["metadata"] = metadata


def _iso_timestamp(value: str) -> str:
    """Convert LoCoMo's human timestamp to the v3 ISO schema when possible."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    return ""


def _format_v3_content(message, *, source_role: str | None = None) -> str:
    """Add lossless provenance that survives the two-role v3 transport.

    The public conversation endpoint accepts only user/assistant, while
    cockpit traces also contain system and tool evidence.  `source_role`
    records the semantic role before transport demotion so L1 extraction can
    require tool confirmation for completed actions.
    """
    message_id = str(message.message_id or "").strip()
    speaker = str(message.speaker or "").strip()
    normalized_role = _normalize_v3_source_role(
        source_role if source_role is not None else message.role
    )
    source_time = (
        _iso_timestamp(message.timestamp)
        or str(message.timestamp or "").strip()
    )
    encoded_time = quote(source_time, safe=":,+-.TZ")
    prefix = " ".join(part for part in (
        f"[{message_id}]" if message_id else "",
        f"[source_time={encoded_time}]" if encoded_time else "",
        f"[source_role={normalized_role}]",
        f"{speaker}:" if speaker else "",
    ) if part)
    return f"{prefix} {message.content}".strip() if prefix else message.content


def _compact_v3_l1_session(messages: list[dict], *, decision=None) -> dict:
    """Represent one short source session as one lossless L1 input row.

    Cockpit datasets often fragment one durable command across several tiny
    user/assistant turns. The source envelopes already retain speaker, time,
    and message IDs, so joining those envelopes avoids repeated LLM calls
    without summarising or dropping any source text. The synthetic row is a
    user round solely to preserve MemoryCore's public L1 trigger contract.
    """
    if not messages:
        raise ValueError("cannot compact an empty L1 session")
    content_rows = [
        str(message.get("content") or "").strip()
        for message in messages
        if str(message.get("content") or "").strip()
    ]
    if decision is not None:
        # This is a typed data envelope, not a replacement extraction prompt.
        # Keep it after the exact source rows so speaker/time/source parsers
        # continue to see the original first line unchanged.
        content_rows.append(
            "[memory_episode "
            f"scene={decision.scene} "
            f"type={decision.memory_type} "
            f"action={decision.write_action} "
            f"state={decision.lifecycle} "
            f"temporal={decision.temporal_scope}]"
        )
    compacted = {"role": "user", "content": "\n".join(content_rows)}
    timestamp = next((
        str(message.get("timestamp"))
        for message in messages if message.get("timestamp")
    ), "")
    if timestamp:
        compacted["timestamp"] = timestamp
    return compacted


_V3_SOURCE_PREFIX = re.compile(
    r"^\[([A-Za-z0-9_.:-]+)\](?:\s|$)", re.MULTILINE
)
_V3_SOURCE_TIME = re.compile(r"(?:^|\s)\[source_time=([^\]]+)\](?:\s|$)")
_V3_SOURCE_ROLE = re.compile(
    r"(?:^|\s)\[source_role=(user|assistant|system|tool)\](?:\s|$)",
    re.IGNORECASE,
)
_V3_MESSAGE_SPEAKER = re.compile(
    r"^\[[^\]]+\](?:\s+\[source_time=[^\]]+\])?"
    r"(?:\s+\[source_role=[^\]]+\])?\s+([^:\n]{1,80}):"
)


def _v3_source_ids(content: str) -> list[str]:
    """Recover all IDs from one or more reversible v3 source envelopes."""
    return list(dict.fromkeys(
        match.group(1)
        for match in _V3_SOURCE_PREFIX.finditer(str(content or "").strip())
    ))


def _v3_source_timestamp(content: str) -> str:
    """Recover original event time instead of using backend recorded_at."""
    match = _V3_SOURCE_TIME.search(str(content or "").strip())
    return unquote(match.group(1)) if match else ""


def _v3_source_role(content: str) -> str:
    """Recover the semantic role before v3 transport normalization."""
    match = _V3_SOURCE_ROLE.search(str(content or "").strip())
    return match.group(1).lower() if match else ""


def _v3_message_speaker(content: str) -> str:
    match = _V3_MESSAGE_SPEAKER.match(str(content or "").strip())
    return match.group(1).strip() if match else ""


def _v3_message_body(content: str) -> str:
    raw = str(content or "").strip()
    match = _V3_MESSAGE_SPEAKER.match(raw)
    body = raw[match.end():].strip() if match else raw
    # Captions remain in the final evidence for multimodal questions, but do
    # not let a generic image caption (for example, a beach in the background)
    # outrank a target speaker's explicit textual statement.
    return re.split(
        r"\n\[(?:Image caption|Image query|images?)\s*[:\]]",
        body,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def _is_when_question(question: str) -> bool:
    folded = " ".join(str(question or "").casefold().split())
    return bool(
        re.search(r"\bwhen\b", folded)
        or re.search(r"(?:何时|什么时候|哪天|哪年|哪月)", folded)
    )


def _is_temporal_question(question: str) -> bool:
    """Identify questions whose answer requires an explicit time value."""
    folded = " ".join(str(question or "").casefold().split())
    if _is_when_question(folded):
        return True
    return bool(re.search(
        r"\b(?:what|which)\s+(?:date|day|month|year|time)\b|"
        r"\bhow\s+(?:long|old|recently|many\s+(?:days|weeks|months|years))\b|"
        r"(?:何时|什么时候|哪天|哪年|哪月|多久|多长时间)",
        folded,
    ))


def _is_emotion_question(question: str) -> bool:
    folded = " ".join(str(question or "").casefold().split())
    return bool(
        re.search(r"\bhow\s+(?:did|does|do|was|were|is|are)\b.*\bfeel", folded)
        or re.search(r"(?:感觉如何|感受如何|什么感受|什么心情)", folded)
    )


def _is_inference_question(question: str) -> bool:
    folded = " ".join(str(question or "").casefold().split())
    return bool(
        re.search(r"^(?:would|will|could|should|might|is|are)\b", folded)
        or re.search(r"\b(?:likely|probably|plausibly)\b", folded)
        or re.search(r"(?:会不会|是否会|可能会|大概率)", folded)
    )


def _should_focus_question(question: str, modes: frozenset[str]) -> bool:
    if not modes or "all" in modes:
        return True
    folded = " ".join(str(question or "").casefold().split())
    if "when" in modes and _is_when_question(folded):
        return True
    if "emotion" in modes and _is_emotion_question(folded):
        return True
    if "inference" in modes and _is_inference_question(folded):
        return True
    if "aggregate" in modes:
        patterns = (
            r"\bhow many\b",
            r"\bin what ways\b",
            r"^what\s+(?:are|were)\b",
            r"^where\s+(?:has|have)\b",
            r"^what\b.*\b[a-z]+s(?:/[a-z]+s)?\s+"
            r"(?:does|do|did|has|have|is|are|was|were)\b",
            r"(?:多少|哪些|哪几|分别|列出)",
        )
        if any(re.search(pattern, folded) for pattern in patterns):
            return True
    return False


def _human_date(value: datetime) -> str:
    """Render an exact date without locale- or platform-specific directives."""
    return f"{value.day} {value.strftime('%B')} {value.year}"


def _parse_v3_source_datetime(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolved_v3_relative_times(
    content: str, source_timestamp: str
) -> list[tuple[str, str]]:
    """Resolve unambiguous relative dates against the source session date.

    This is deliberately a small deterministic calendar normalizer, not a
    benchmark-specific answer rule.  Ambiguous expressions such as "recently"
    and "a few weeks ago" are left untouched.
    """
    normalized = resolve_temporal_query(
        str(content or ""), {"query_time": source_timestamp}
    )
    resolved: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for span in normalized.spans:
        if span.kind != "relative":
            continue
        item = (span.raw.casefold(), humanize_temporal_span(span))
        if item not in seen:
            seen.add(item)
            resolved.append(item)
    return resolved


def _annotate_v3_relative_time(content: str, source_timestamp: str) -> str:
    if "[resolved_relative_time:" in str(content or ""):
        return content
    resolved = _resolved_v3_relative_times(content, source_timestamp)
    source = _parse_v3_source_datetime(source_timestamp)
    if not resolved or source is None:
        return content
    mappings = "; ".join(
        f'"{phrase}" = {value}' for phrase, value in resolved
    )
    return (
        f"{content}\n[resolved_relative_time: {mappings}; "
        f"session_date = {_human_date(source)}]"
    )


def _humanize_v3_source_time(content: str, source_timestamp: str) -> str:
    source = _parse_v3_source_datetime(source_timestamp)
    if source is None:
        return content
    return re.sub(
        r"\[source_time=[^\]]+\]",
        f"[source_date={_human_date(source)}]",
        str(content or ""),
        count=1,
    )


def _perspective_role(message, owner_speaker: str) -> str:
    role = str(message.role or "").strip().lower()
    if role not in {"user", "assistant", "system", "tool"}:
        role = "user"
    if not owner_speaker or role in {"system", "tool"}:
        return role
    speaker = str(message.speaker or "").strip()
    if not speaker:
        return role
    return "user" if speaker == owner_speaker else "assistant"


def _normalize_v3_source_role(value: object) -> str:
    role = str(value or "").strip().lower()
    return role if role in {"user", "assistant", "system", "tool"} else "user"


def _v3_transport_role(source_role: str, *, extract_l1: bool) -> str:
    """Map semantic roles to the public v3 user/assistant wire schema."""
    role = _normalize_v3_source_role(source_role)
    if role in {"system", "tool"}:
        return "assistant"
    if role == "user" and not extract_l1:
        return "assistant"
    return role


def _select_v3_dynamic_rows(
    rows: list[dict],
    limit: int,
    *,
    l0_min_results: int,
    l0_diverse_results: int = 0,
) -> list[dict]:
    """Select a ranked L0/L1 mix while optionally reserving raw anchors."""
    if limit <= 0:
        return []
    required_l0 = min(
        max(0, l0_min_results),
        limit,
        sum(
            1
            for row in rows
            if (row.get("metadata") or {}).get("level") == "L0"
        ),
    )
    selected: list[dict] = []
    selected_objects: set[int] = set()
    if required_l0:
        seen_sessions: set[str] = set()
        diverse_target = min(required_l0, max(0, l0_diverse_results))
        if diverse_target:
            for row in rows:
                metadata = row.get("metadata") or {}
                if metadata.get("level") != "L0":
                    continue
                session_id = str(metadata.get("session_id") or "")
                if not session_id or session_id in seen_sessions:
                    continue
                selected.append(row)
                selected_objects.add(id(row))
                seen_sessions.add(session_id)
                if len(selected) >= diverse_target:
                    break
        if len(selected) < required_l0:
            for row in rows:
                if (row.get("metadata") or {}).get("level") != "L0":
                    continue
                if id(row) in selected_objects:
                    continue
                selected.append(row)
                selected_objects.add(id(row))
                if len(selected) >= required_l0:
                    break
    for row in rows:
        if len(selected) >= limit:
            break
        if id(row) in selected_objects:
            continue
        selected.append(row)
    selected.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
    return selected


_PROFILE_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
_PROFILE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "how", "i", "in", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "their", "them", "they", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would", "you", "your",
}


def _profile_tokens(value: str) -> list[str]:
    return [
        token.casefold()
        for token in _PROFILE_TOKEN.findall(str(value or ""))
        if token.casefold() not in _PROFILE_STOPWORDS
    ]


_FOCUS_TOKEN_ALIASES = {
    "named": "name",
    "names": "name",
    "name": "name",
    "children": "child",
    "child": "child",
    "kids": "child",
    "kid": "child",
    "sons": "child",
    "son": "child",
    "daughters": "child",
    "daughter": "child",
    "pets": "pet",
    "pet": "pet",
    "puppy": "pet",
    "pup": "pet",
    "kitty": "pet",
    "kitten": "pet",
    "cats": "pet",
    "cat": "pet",
    "dogs": "pet",
    "dog": "pet",
    "seen": "attend",
    "saw": "attend",
    "went": "attend",
    "attended": "attend",
    "attending": "attend",
    "feel": "emotion",
    "feels": "emotion",
    "feeling": "emotion",
    "felt": "emotion",
    "appreciate": "emotion",
    "appreciated": "emotion",
    "grateful": "emotion",
    "thankful": "emotion",
    "happy": "emotion",
    "scared": "emotion",
}
_FOCUS_GENERIC = frozenset({"many", "times", "some"})


def _focus_tokens(value: str) -> list[str]:
    """Normalize lightweight morphology for query-focused dialogue ranking."""
    normalized: list[str] = []
    for token in _profile_tokens(value):
        if token in _FOCUS_GENERIC:
            continue
        alias = _FOCUS_TOKEN_ALIASES.get(token)
        if alias:
            normalized.append(alias)
            continue
        if len(token) > 4 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 3 and token.endswith("ied"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("ed"):
            token = token[:-1] if token[-3] == "e" else token[:-2]
        elif len(token) > 3 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        normalized.append(_FOCUS_TOKEN_ALIASES.get(token, token))
    return normalized


_LEDGER_CATEGORIES = frozenset({
    "identity", "residence", "relationship", "family", "pet", "education",
    "career", "skill", "instrument", "book", "music", "media", "event",
    "travel", "hobby", "art", "purchase", "possession", "health",
    "emotion", "preference", "plan", "community", "other",
})
_LEDGER_CONCEPTS = {
    "moved": "residence", "move": "residence", "country": "residence",
    "home": "residence", "live": "residence", "lived": "residence",
    "child": "family", "children": "family", "kid": "family",
    "kids": "family", "daughter": "family", "son": "family",
    "husband": "family", "wife": "family", "parent": "family",
    "parents": "family", "mother": "family", "father": "family",
    "sibling": "family", "siblings": "family", "brother": "family",
    "brothers": "family", "sister": "family", "sisters": "family",
    "pet": "pet", "pets": "pet", "puppy": "pet", "kitten": "pet",
    "dog": "pet", "dogs": "pet", "cat": "pet", "cats": "pet",
    "violin": "instrument", "clarinet": "instrument", "piano": "instrument",
    "guitar": "instrument", "drum": "instrument", "flute": "instrument",
    "cello": "instrument", "instrument": "instrument",
    "book": "book", "books": "book", "novel": "book",
    "novels": "book", "read": "book", "reading": "book",
    "artist": "music", "artists": "music", "band": "music",
    "bands": "music", "concert": "music", "concerts": "music",
    "performer": "music", "performers": "music", "singer": "music",
    "singers": "music", "music": "music",
    "bought": "purchase", "buy": "purchase", "purchase": "purchase",
    "purchased": "purchase", "shopping": "purchase", "item": "purchase",
    "own": "possession", "owned": "possession", "belonging": "possession",
    "career": "career", "job": "career", "profession": "career",
    "school": "education", "college": "education", "education": "education",
    "trip": "travel", "travel": "travel", "roadtrip": "travel",
    "beach": "travel", "camping": "travel", "hike": "travel",
    "while": "event", "during": "event", "activity": "event",
    "activities": "event", "happen": "event", "happened": "event",
    "painting": "art", "paint": "art", "pottery": "art", "art": "art",
    "feel": "emotion", "emotion": "emotion",
    "identity": "identity", "community": "community",
    "plan": "plan", "planning": "plan",
}


def _ledger_tokens(value: str) -> list[str]:
    """Add broad personal-memory concepts without discarding exact tokens."""
    raw_tokens = _profile_tokens(value)
    tokens = list(dict.fromkeys(raw_tokens + _focus_tokens(value)))
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        concept = _LEDGER_CONCEPTS.get(token)
        if concept and concept != token:
            expanded.append(concept)
    return expanded


def _rank_v3_ledger_facts(
    facts: tuple[dict, ...], query: str, speakers: tuple[str, ...],
) -> list[tuple[float, dict]]:
    """Rank source-grounded construction facts using only the current query."""
    folded = str(query or "").casefold()
    target_speakers = {
        speaker.casefold() for speaker in speakers
        if speaker and speaker.casefold() in folded
    }
    relation_query = bool(re.search(
        r"\b(?:recommend|suggest|advice|advise|offer|request|select|confirm|from)\w*\b",
        folded,
    ))
    count_query = bool(re.search(
        r"\b(?:how\s+many|number\s+of|count|times?)\b", folded
    ))
    projection_query = bool(re.search(
        r"\b(?:would|will|likely|unlikely|soon|again|expect)\b", folded
    ))
    event_identity_query = bool(re.search(
        r"\b(?:same|another|distinct|separate|deduplicat\w*)\b", folded
    ))
    candidates = [
        fact for fact in facts
        if (
            str(fact.get("fact_kind") or "").casefold() != "relation_rollup"
            or relation_query
        )
        and (
            str(fact.get("memory_kind") or "").casefold() != "event_count"
            or count_query
        )
        and (
            str(fact.get("memory_kind") or "").casefold() != "causal_projection"
            or projection_query
        )
        and (
            str(fact.get("memory_kind") or "").casefold() != "event_identity"
            or event_identity_query
        )
        and (
            not target_speakers
            or str(fact.get("subject") or "").casefold() in target_speakers
            or any(
                str(participant).casefold() in target_speakers
                for participant in (fact.get("participants") or [])
            )
        )
    ]
    query_tokens = set(_ledger_tokens(query))
    if not candidates or not query_tokens:
        return []
    query_categories = query_tokens & _LEDGER_CATEGORIES
    if re.search(r"\b(?:relationship|marital)\s+status\b", folded):
        query_categories.add("identity")
    documents: list[set[str]] = []
    for fact in candidates:
        rendered = " ".join((
            str(fact.get("subject") or ""),
            str(fact.get("category") or ""),
            str(fact.get("topic") or ""),
            str(fact.get("statement") or ""),
            " ".join(str(value) for value in (fact.get("values") or [])),
            str(fact.get("event_date") or ""),
            " ".join(str(value) for value in (fact.get("participants") or [])),
            str(fact.get("relation_type") or ""),
            str(fact.get("memory_kind") or ""),
        ))
        documents.append(set(_ledger_tokens(rendered)))
    frequencies = Counter(token for document in documents for token in document)
    count = len(documents)
    ranked: list[tuple[float, int, dict]] = []
    for index, (fact, document) in enumerate(zip(candidates, documents)):
        overlap = query_tokens & document
        score = sum(
            math.log(1.0 + (count + 1.0) / (frequencies[token] + 0.5))
            for token in overlap
        )
        category = str(fact.get("category") or "other").casefold()
        if category in query_categories:
            score += 8.0
            if str(fact.get("fact_kind") or "").endswith("rollup"):
                score += 2.0
        subject = str(fact.get("subject") or "").casefold()
        if subject and subject in target_speakers:
            score += 2.0
        topic = str(fact.get("topic") or "").casefold()
        if topic and topic in folded:
            score += 3.0
        for value in fact.get("values") or []:
            exact = str(value).strip().casefold()
            if len(exact) >= 3 and exact in folded:
                score += 4.0
        fact_kind = str(fact.get("fact_kind") or "").casefold()
        relation_type = str(fact.get("relation_type") or "").casefold()
        memory_kind = str(fact.get("memory_kind") or "").casefold()
        if fact_kind == "relation_rollup" and relation_query:
            score += 9.0
            if relation_type and relation_type in document:
                score += 1.0
        if memory_kind == "event_count" and count_query:
            score += 10.0
        if memory_kind == "causal_projection" and projection_query:
            score += 9.0
        if memory_kind == "event_identity" and re.search(
            r"\b(?:same|another|happened|event|trip|visit)\b", folded
        ):
            score += 4.0
        if score > 0:
            ranked.append((score, index, fact))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(score, fact) for score, _index, fact in ranked]


def _rank_v3_l0_sessions(
    history: _V3L0History, query: str
) -> list[tuple[float, str, int]]:
    """Rank distinct raw sessions and choose one query-focused anchor each."""
    query_tokens = list(dict.fromkeys(_profile_tokens(query)))
    if not query_tokens or not history.sessions:
        return []
    session_items = sorted(history.sessions.items())
    documents = [
        _profile_tokens("\n".join(message.content for message in messages))
        for _session_id, messages in session_items
    ]
    document_frequency = Counter(
        token for document in documents for token in set(document)
    )
    count = len(documents)
    average_length = max(
        1.0, sum(len(document) for document in documents) / count
    )
    inverse_document_frequency = {
        token: math.log(
            1 + (count - document_frequency[token] + 0.5)
            / (document_frequency[token] + 0.5)
        )
        for token in query_tokens
    }
    ranked: list[tuple[float, str, int]] = []
    for (session_id, messages), document in zip(session_items, documents):
        frequencies = Counter(document)
        length = max(1, len(document))
        session_score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * length / average_length
            )
            session_score += (
                inverse_document_frequency[token]
                * frequency * 2.5 / denominator
            )
        if session_score <= 0:
            continue
        anchor_scores: list[tuple[float, int]] = []
        for index, message in enumerate(messages):
            message_frequencies = Counter(_profile_tokens(message.content))
            score = sum(
                inverse_document_frequency[token]
                * min(2, message_frequencies[token])
                for token in query_tokens
            )
            anchor_scores.append((score, index))
        anchor_score, anchor_index = max(
            anchor_scores, key=lambda item: (item[0], -item[1])
        )
        ranked.append((session_score + 0.5 * anchor_score, session_id, anchor_index))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked


def _rank_v3_profiles(rows: list[dict], query: str) -> list[tuple[float, dict]]:
    """Rank a small agent-scoped L2 corpus without another model/API call."""
    if not rows:
        return []
    documents: list[list[str]] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        documents.append(_profile_tokens("\n".join((
            str(metadata.get("path", "")),
            str(metadata.get("summary", "")),
            str(row.get("content", "")),
        ))))
    query_tokens = list(dict.fromkeys(_profile_tokens(query)))
    document_frequency = Counter(
        token for document in documents for token in set(document)
    )
    average_length = sum(len(document) for document in documents) / len(documents)
    average_length = max(1.0, average_length)
    count = len(documents)
    folded_query = query.casefold()
    ranked: list[tuple[float, dict]] = []
    for row, document in zip(rows, documents):
        frequencies = Counter(document)
        length = max(1, len(document))
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1 + (count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * length / average_length
            )
            score += inverse_document_frequency * frequency * 2.5 / denominator
        owner = str((row.get("metadata") or {}).get(
            "perspective_owner", ""
        )).casefold()
        if owner and owner in folded_query:
            score += 2.0
        ranked.append((score, row))
    ranked.sort(
        key=lambda item: (
            item[0],
            str((item[1].get("metadata") or {}).get("path", "")),
        ),
        reverse=True,
    )
    return ranked


def _profile_excerpt(content: str, query: str, max_chars: int) -> str:
    body = re.sub(
        r"^-----META-START-----\n[\s\S]*?\n-----META-END-----\s*",
        "",
        str(content or ""),
        count=1,
    ).strip()
    if len(body) <= max_chars:
        return body
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    query_tokens = set(_profile_tokens(query))
    scored = []
    for index, paragraph in enumerate(paragraphs):
        frequencies = Counter(_profile_tokens(paragraph))
        overlap = sum(frequencies[token] for token in query_tokens)
        heading_bonus = 0.25 if index == 0 or paragraph.startswith("#") else 0.0
        scored.append((overlap + heading_bonus, index, paragraph))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected: list[tuple[int, str]] = []
    used = 0
    for _score, index, paragraph in scored:
        remaining = max_chars - used - (2 if selected else 0)
        if remaining <= 0:
            break
        if len(paragraph) > remaining:
            if not selected:
                selected.append((index, paragraph[:remaining]))
            break
        selected.append((index, paragraph))
        used += len(paragraph) + (2 if len(selected) > 1 else 0)
    selected.sort(key=lambda item: item[0])
    excerpt = "\n\n".join(paragraph for _, paragraph in selected)
    return excerpt or body[:max_chars]


def _v3_hit_key(row: dict) -> tuple[str, ...]:
    metadata = row.get("metadata") or {}
    level = str(metadata.get("level", ""))
    source_ids = tuple(str(item) for item in row.get("source_ids", []) if item)
    if source_ids:
        return (level, "source", *source_ids)
    content = " ".join(str(row.get("content", "")).casefold().split())
    return (level, "content", content)


def _scoped_v3_session_id(
    source_session_id: str,
    *,
    conversation_id: str,
    team_id: str,
    agent_id: str,
    user_id: str,
    task_id: str,
) -> str:
    """Return a stable transport session ID unique to an isolation scope.

    This is protocol hygiene rather than a dataset hint: the digest contains
    only tenancy/namespace identifiers and never a question, answer, or
    evidence label. Keeping the readable source ID makes operational logs
    auditable while the digest prevents cross-run checkpoint collisions.
    """
    scope = "\x1f".join((team_id, agent_id, user_id, task_id, conversation_id))
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    source = str(source_session_id or "session").strip() or "session"
    return f"{source}--{digest}"
