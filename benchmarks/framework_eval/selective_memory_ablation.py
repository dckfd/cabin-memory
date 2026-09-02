#!/usr/bin/env python3
"""Execute the paired selective-memory ablation protocol safely and resumably."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .plugins import PluginCatalog


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "benchmarks/framework_eval/protocols/selective-memory-v1/plan.json"
)
DEFAULT_RUN_ROOT = (
    ROOT / "benchmarks/framework_eval_runs/selective-memory-v1-20260820"
)
PROFILES = ROOT / "benchmarks/framework_eval/profiles.json"
DATASETS = ROOT / "benchmarks/framework_eval/datasets.json"
DEFAULT_PROVIDER_ENV = (
    ROOT / "third_party/tencentdb-agent-memory-v2/deploy/global-images/.env"
)
GATE_ARMS = frozenset({
    "full-l0-l1",
    "adaptive-top1-top3-l1",
    "selective-l1-async-l23",
})


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: tuple[str, ...]
    environment: dict[str, str]
    log_path: Path
    marker_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"refusing to replace changed run manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _safe_name(value: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    ).strip("-")[:48] or "item"
    digest = hashlib.sha256(str(value).encode()).hexdigest()[:10]
    return f"{readable}-{digest}"


def _selected_protocol(
    protocol_path: Path,
    *,
    phase: int,
    dataset_ids: Iterable[str] = (),
    arm_ids: Iterable[str] = (),
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    requested_datasets = set(dataset_ids)
    requested_arms = set(arm_ids)
    datasets = [
        row for row in protocol.get("datasets") or []
        if int(row.get("execution_phase", 1)) <= phase
        and (not requested_datasets or str(row["dataset"]) in requested_datasets)
    ]
    arms = [
        row for row in protocol.get("arms") or []
        if not requested_arms or str(row["id"]) in requested_arms
    ]
    found_datasets = {str(row["dataset"]) for row in datasets}
    found_arms = {str(row["id"]) for row in arms}
    if requested_datasets - found_datasets:
        raise ValueError(
            "datasets are absent from the selected phase: "
            + ", ".join(sorted(requested_datasets - found_datasets))
        )
    if requested_arms - found_arms:
        raise ValueError(
            "unknown arms: " + ", ".join(sorted(requested_arms - found_arms))
        )
    if not datasets or not arms:
        raise ValueError("selected protocol must contain datasets and arms")
    return {**protocol, "datasets": datasets, "arms": arms}


def validate_execution_subset(
    frozen_plan: dict[str, Any],
    selected_protocol: dict[str, Any],
    *,
    phase: int,
) -> None:
    """Validate a filtered execution against one immutable full run plan."""
    if str(frozen_plan.get("protocol_id") or "") != str(
        selected_protocol.get("protocol_id") or ""
    ):
        raise RuntimeError("frozen run plan uses a different protocol")
    if int(frozen_plan.get("phase", -1)) != int(phase):
        raise RuntimeError(
            "frozen run plan uses a different phase; choose a fresh run root"
        )
    frozen_datasets = {
        str(row.get("dataset")) for row in frozen_plan.get("datasets") or []
    }
    frozen_arms = {
        str(row.get("id")) for row in frozen_plan.get("arms") or []
    }
    selected_datasets = {
        str(row.get("dataset"))
        for row in selected_protocol.get("datasets") or []
    }
    selected_arms = {
        str(row.get("id")) for row in selected_protocol.get("arms") or []
    }
    if selected_datasets - frozen_datasets:
        raise RuntimeError(
            "execution dataset is absent from frozen run plan: "
            + ", ".join(sorted(selected_datasets - frozen_datasets))
        )
    if selected_arms - frozen_arms:
        raise RuntimeError(
            "execution arm is absent from frozen run plan: "
            + ", ".join(sorted(selected_arms - frozen_arms))
        )


def resolve_arm(arm: dict[str, Any]) -> dict[str, Any]:
    retrieval = dict(arm.get("retrieval") or {})
    layers = [str(value).upper() for value in retrieval.get("layers") or []]
    if not layers or set(layers) - {"L0", "L1", "L2", "L3"}:
        raise ValueError(f"invalid layers for arm {arm.get('id')}")
    if retrieval.get("policy") == "adaptive":
        top_k = sum(
            int(retrieval.get(key, 0))
            for key in (
                "fallback_l0_k", "fallback_l1_k",
                "fallback_l2_k", "fallback_l3_k",
            )
        )
        context_tiers = [
            int(value) for value in retrieval.get("context_tiers_chars") or []
        ]
        if len(context_tiers) != 2:
            raise ValueError(
                f"adaptive arm {arm.get('id')} needs two context tiers"
            )
        max_context_chars = max(context_tiers)
    else:
        top_k = int(retrieval.get("top_k", 0))
        max_context_chars = int(retrieval.get("max_context_chars", 0))
    if top_k <= 0 or max_context_chars <= 0:
        raise ValueError(f"arm {arm.get('id')} has no positive retrieval budget")
    return {
        **arm,
        "retrieval": retrieval,
        "resolved_top_k": top_k,
        "resolved_max_context_chars": max_context_chars,
    }


def resolve_store_groups(arms: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for arm in arms:
        group_id = str(arm["store_group"])
        construction = dict(arm.get("construction") or {})
        current = groups.setdefault(group_id, {
            "id": group_id,
            "l1_policy": construction["l1_policy"],
            "l23_schedule": "disabled",
            "arms": [],
        })
        if current["l1_policy"] != construction["l1_policy"]:
            raise ValueError(f"store group {group_id} mixes L1 policies")
        if construction.get("l23_schedule") == "buffered_dirty_event":
            current["l23_schedule"] = "buffered_dirty_event"
        current["arms"].append(str(arm["id"]))
    return groups


def arm_environment(arm: dict[str, Any]) -> dict[str, str]:
    resolved = resolve_arm(arm)
    retrieval = resolved["retrieval"]
    construction = dict(resolved.get("construction") or {})
    tiers = list(retrieval.get("context_tiers_chars") or [900, 2200])
    l23_enabled = bool({"L2", "L3"} & set(retrieval["layers"]))
    selective = construction["l1_policy"] == "cockpit_selective_v1"
    return {
        "TDAI_HTTP_API_VERSION": "v3",
        "TDAI_EVAL_PERSPECTIVE_MODE": "single",
        "TDAI_EVAL_MEMORY_LAYERS": ",".join(retrieval["layers"]),
        "TDAI_EVAL_L1_WRITE_POLICY": str(construction["l1_policy"]),
        "TDAI_EVAL_L1_BATCH_MODE": (
            "conversation" if selective else "session"
        ),
        "TDAI_EVAL_L1_BATCH_MAX_MESSAGES": "128",
        "TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS": (
            "true" if selective else "false"
        ),
        "TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES": "1" if selective else "0",
        "TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL": (
            "l0_only" if selective else "fail"
        ),
        "TDAI_EVAL_L1_ZERO_OUTPUT_WAIT_FOR_REPAIR": (
            "false" if selective else "true"
        ),
        "TDAI_EVAL_L23_SCHEDULE": str(construction["l23_schedule"]),
        "TDAI_EVAL_L23_READINESS_MODE": (
            "dirty_only" if l23_enabled else "required"
        ),
        "TDAI_EVAL_RETRIEVAL_POLICY": str(retrieval["policy"]),
        "TDAI_EVAL_ADAPTIVE_LAZY_LAYER_READINESS": (
            "true" if l23_enabled and retrieval["policy"] == "adaptive"
            else "false"
        ),
        "TDAI_EVAL_ADAPTIVE_LAYER_WAIT_BUDGET_SECONDS": "0",
        "TDAI_EVAL_ADAPTIVE_FAST_L0_K": str(retrieval.get("fast_l0_k", 1)),
        "TDAI_EVAL_ADAPTIVE_FALLBACK_L0_K": str(
            retrieval.get("fallback_l0_k", 3)
        ),
        "TDAI_EVAL_ADAPTIVE_FALLBACK_L1_K": str(
            retrieval.get("fallback_l1_k", 0)
        ),
        "TDAI_EVAL_ADAPTIVE_FALLBACK_L2_K": str(
            retrieval.get("fallback_l2_k", 0)
        ),
        "TDAI_EVAL_ADAPTIVE_FALLBACK_L3_K": str(
            retrieval.get("fallback_l3_k", 0)
        ),
        "TDAI_EVAL_ADAPTIVE_FAST_CONTEXT_CHARS": str(tiers[0]),
        "TDAI_EVAL_ADAPTIVE_FALLBACK_CONTEXT_CHARS": str(tiers[-1]),
        "TDAI_EVAL_L0_WINDOW_RADIUS": "0",
        "TDAI_EVAL_L0_WINDOW_BEFORE": str(retrieval.get("window_before", 0)),
        "TDAI_EVAL_L0_WINDOW_AFTER": str(retrieval.get("window_after", 0)),
        "TDAI_EVAL_CANDIDATE_MULTIPLIER": "1",
        "TDAI_EVAL_L0_EXPLICIT_DATE_RESULTS": "1",
        "TDAI_EVAL_L0_EXPLICIT_DATE_BOOST": "true",
        "TDAI_EVAL_READY_SETTLE_SECONDS": "210" if l23_enabled else "10",
        "TDAI_EVAL_L1_READY_SETTLE_SECONDS": "10",
        "TDAI_EVAL_CLEAN_READY_SETTLE_SECONDS": "10",
    }


def construction_environment(store: dict[str, Any]) -> dict[str, str]:
    buffered = store["l23_schedule"] == "buffered_dirty_event"
    selective = store["l1_policy"] == "cockpit_selective_v1"
    return {
        "TDAI_HTTP_API_VERSION": "v3",
        "TDAI_EVAL_PERSPECTIVE_MODE": "single",
        "TDAI_EVAL_MEMORY_LAYERS": "L0,L1,L2,L3" if buffered else "L0,L1",
        "TDAI_EVAL_L1_WRITE_POLICY": str(store["l1_policy"]),
        "TDAI_EVAL_L1_BATCH_MODE": (
            "conversation" if selective else "session"
        ),
        "TDAI_EVAL_L1_BATCH_MAX_MESSAGES": "128",
        "TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS": (
            "true" if selective else "false"
        ),
        "TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES": "1" if selective else "0",
        "TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL": (
            "l0_only" if selective else "fail"
        ),
        "TDAI_EVAL_L1_ZERO_OUTPUT_WAIT_FOR_REPAIR": (
            "false" if selective else "true"
        ),
        "TDAI_EVAL_L23_SCHEDULE": str(store["l23_schedule"]),
        "TDAI_EVAL_L23_READINESS_MODE": "dirty_only" if buffered else "required",
        "TDAI_EVAL_RETRIEVAL_POLICY": "fixed",
        "TDAI_EVAL_READY_SETTLE_SECONDS": "210" if buffered else "10",
        "TDAI_EVAL_CLEAN_READY_SETTLE_SECONDS": "10",
    }


def select_smoke_manifest(
    dataset,
    protocol_manifest: dict[str, Any],
    *,
    count: int,
    seed: str,
) -> dict[str, Any]:
    allowed = {str(value) for value in protocol_manifest["question_ids"]}
    questions = [
        question for question in dataset.questions()
        if str(question.question_id) in allowed
    ]
    if not 1 <= count <= len(questions):
        raise ValueError("smoke question count is outside the selected dataset")
    by_category: dict[str, list[Any]] = {}
    by_conversation: dict[str, list[Any]] = {}
    for question in questions:
        by_category.setdefault(
            str(question.category or "uncategorized"), []
        ).append(question)
        by_conversation.setdefault(str(question.conversation_id), []).append(
            question
        )
    for category, values in by_category.items():
        values.sort(key=lambda item: _stable_key(
            seed, f"{category}:{item.question_id}"
        ))
    costs: dict[str, int] = {}
    for conversation_id in by_conversation:
        conversation = dataset.conversation(conversation_id)
        costs[conversation_id] = max(1, len(conversation.sessions))

    # A smoke test must be cheap in *construction* units, not merely in QA
    # rows.  Greedily cover distinct categories per ingested session, then
    # fill any remaining QA slots from already-selected histories.  This uses
    # only source structure/category metadata and never answers/evidence IDs.
    target_categories = min(count, len(by_category))
    selected_conversations: list[str] = []
    covered_categories: set[str] = set()
    available_questions = 0
    remaining = set(by_conversation)
    while (
        available_questions < count
        or len(covered_categories) < target_categories
    ):
        if not remaining:
            break

        def priority(conversation_id: str) -> tuple[float, int, float, int, str]:
            values = by_conversation[conversation_id]
            categories = {
                str(value.category or "uncategorized") for value in values
            }
            new_categories = len(categories - covered_categories)
            category_utility = new_categories / costs[conversation_id]
            question_utility = len(values) / costs[conversation_id]
            return (
                category_utility,
                new_categories,
                question_utility,
                -costs[conversation_id],
                _stable_key(seed, conversation_id),
            )

        chosen = max(remaining, key=priority)
        remaining.remove(chosen)
        selected_conversations.append(chosen)
        values = by_conversation[chosen]
        available_questions += len(values)
        covered_categories.update(
            str(value.category or "uncategorized") for value in values
        )

    candidates = [
        question
        for conversation_id in selected_conversations
        for question in by_conversation[conversation_id]
    ]
    candidates.sort(key=lambda item: _stable_key(seed, str(item.question_id)))
    selected: list[Any] = []
    selected_categories: set[str] = set()
    for question in candidates:
        category = str(question.category or "uncategorized")
        if category not in selected_categories:
            selected.append(question)
            selected_categories.add(category)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected_ids = {str(item.question_id) for item in selected}
        selected.extend(
            question for question in candidates
            if str(question.question_id) not in selected_ids
        )
        selected = selected[:count]

    selected_conversation_ids = sorted({
        str(item.conversation_id) for item in selected
    })
    selected_sources = [
        dataset.conversation(conversation_id)
        for conversation_id in selected_conversation_ids
    ]
    return {
        "schema_version": 1,
        "protocol_id": protocol_manifest["protocol_id"],
        "dataset_id": protocol_manifest["dataset_id"],
        "selection": "category_diverse_min_construction_smoke",
        "seed": seed,
        "question_ids": sorted(str(item.question_id) for item in selected),
        "conversation_ids": selected_conversation_ids,
        "counts": {
            "questions": len(selected),
            "conversations": len(selected_conversation_ids),
            "sessions": sum(
                len(conversation.sessions) for conversation in selected_sources
            ),
            "messages": sum(
                len(session.messages)
                for conversation in selected_sources
                for session in conversation.sessions
            ),
            "source_characters": sum(
                len(message.render_text())
                for conversation in selected_sources
                for session in conversation.sessions
                for message in session.messages
            ),
        },
        "leakage_controls": {
            "answers_read_for_selection": False,
            "evidence_ids_read_for_selection": False,
        },
    }


def balanced_shards(
    conversation_ids: list[str], dataset, count: int,
) -> list[list[str]]:
    bins: list[tuple[int, list[str]]] = [
        (0, []) for _ in range(max(1, min(count, len(conversation_ids))))
    ]
    weighted = []
    for conversation_id in conversation_ids:
        conversation = dataset.conversation(conversation_id)
        weight = max(1, sum(
            len(message.render_text())
            for session in conversation.sessions
            for message in session.messages
        ))
        weighted.append((conversation_id, weight))
    for conversation_id, weight in sorted(
        weighted, key=lambda item: (-item[1], item[0])
    ):
        index = min(range(len(bins)), key=lambda value: (bins[value][0], value))
        total, values = bins[index]
        values.append(conversation_id)
        bins[index] = (total + weight, values)
    return [values for _weight, values in bins if values]


def shard_selection_manifest(
    protocol_manifest: dict[str, Any],
    dataset,
    conversation_ids: Iterable[str],
    *,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Restrict a protocol selection to one conversation shard."""
    allowed = {
        str(value) for value in protocol_manifest.get("question_ids") or []
    }
    requested_conversations = {str(value) for value in conversation_ids}
    selected = [
        question for question in dataset.questions()
        if str(question.conversation_id) in requested_conversations
        and str(question.question_id) in allowed
    ]
    question_ids = sorted(str(question.question_id) for question in selected)
    found_conversations = {
        str(question.conversation_id) for question in selected
    }
    missing_conversations = requested_conversations - found_conversations
    if not question_ids or missing_conversations:
        raise RuntimeError(
            "retrieval shard has no selected questions for conversations: "
            + ", ".join(sorted(missing_conversations or requested_conversations))
        )
    return {
        "schema_version": 1,
        "protocol_id": protocol_manifest.get("protocol_id"),
        "dataset_id": protocol_manifest.get("dataset_id"),
        "selection": "protocol_conversation_shard",
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "question_ids": question_ids,
        "conversation_ids": sorted(requested_conversations),
    }


def merge_retrieval_shards(
    shard_paths: list[Path], expected_question_ids: set[str], output: Path,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for path in shard_paths:
        for row in _read_jsonl(path):
            question_id = str(row["question"]["question_id"])
            if question_id in rows:
                duplicates.add(question_id)
            rows[question_id] = row
    found = set(rows)
    if duplicates or found != expected_question_ids:
        raise RuntimeError(
            "retrieval shard merge mismatch: "
            f"missing={sorted(expected_question_ids - found)[:5]} "
            f"extra={sorted(found - expected_question_ids)[:5]} "
            f"duplicates={sorted(duplicates)[:5]}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(rows[key], ensure_ascii=False) + "\n"
            for key in sorted(rows)
        ),
        encoding="utf-8",
    )
    return {"questions": len(rows), "output": str(output)}


def _run_commands(specs: list[CommandSpec], *, parallel: int, label: str) -> None:
    pending = [spec for spec in specs if not spec.marker_path.exists()]
    skipped = len(specs) - len(pending)
    if skipped:
        print(f"[{label}] resume: skipped {skipped} completed command(s)", flush=True)
    failures: list[str] = []
    while pending:
        batch = pending[:max(1, parallel)]
        pending = pending[max(1, parallel):]
        active: list[tuple[CommandSpec, subprocess.Popen, Any, float]] = []
        for spec in batch:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = spec.log_path.open("a", encoding="utf-8")
            started = time.monotonic()
            process = subprocess.Popen(
                spec.command,
                cwd=ROOT,
                env=spec.environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append((spec, process, handle, started))
        last_update = 0.0
        while active:
            remaining = []
            for spec, process, handle, started in active:
                code = process.poll()
                if code is None:
                    remaining.append((spec, process, handle, started))
                    continue
                handle.close()
                elapsed = time.monotonic() - started
                if code:
                    failures.append(spec.name)
                    print(
                        f"[{label}] FAILED {spec.name} exit={code} "
                        f"log={spec.log_path}",
                        flush=True,
                    )
                else:
                    _write_json(spec.marker_path, {
                        "completed": True,
                        "name": spec.name,
                        "seconds": elapsed,
                        "log": str(spec.log_path),
                    })
                    print(
                        f"[{label}] completed {spec.name} ({elapsed:.1f}s)",
                        flush=True,
                    )
            active = remaining
            now = time.monotonic()
            if active and now - last_update >= 10:
                print(
                    f"[{label}] active={len(active)} pending={len(pending)}",
                    flush=True,
                )
                last_update = now
            if active:
                time.sleep(1)
    if failures:
        raise RuntimeError(f"{label} failed: {', '.join(failures)}")


class SelectiveMemoryAblation:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.protocol_path = args.protocol.resolve()
        self.run_root = args.run_root.resolve()
        self.protocol = _selected_protocol(
            self.protocol_path,
            phase=args.phase,
            dataset_ids=args.dataset,
            arm_ids=args.arm,
        )
        self.arms = [resolve_arm(value) for value in self.protocol["arms"]]
        self.stores = resolve_store_groups(self.arms)
        self.catalog = PluginCatalog(
            root=ROOT,
            framework_profiles=PROFILES,
            dataset_profiles=DATASETS,
        )
        self.datasets: dict[str, Any] = {}
        self.manifests: dict[str, dict[str, Any]] = {}
        for row in self.protocol["datasets"]:
            dataset_id = str(row["dataset"])
            manifest_path = (ROOT / str(row["manifest"])).resolve()
            manifest = _read_json(manifest_path)
            dataset_root = (ROOT / str(manifest["dataset_root"])).resolve()
            self.datasets[dataset_id] = self.catalog.create_dataset(
                dataset_id, root=dataset_root
            )
            self.manifests[dataset_id] = manifest

    def _dataset_root(self, dataset_id: str) -> Path:
        return (ROOT / str(
            self.manifests[dataset_id]["dataset_root"]
        )).resolve()

    def _isolation(self, dataset_id: str, store_group: str) -> Path:
        return (
            self.run_root / dataset_id / "stores" / store_group / "isolation.json"
        )

    def _store_root(self, dataset_id: str, store_group: str) -> Path:
        return self._isolation(dataset_id, store_group).parent

    def _arm_root(self, dataset_id: str, arm_id: str, label: str) -> Path:
        return self.run_root / dataset_id / "arms" / arm_id / label

    def _runtime_env(
        self, dataset_id: str, store_group: str,
    ) -> dict[str, str]:
        isolation_path = self._isolation(dataset_id, store_group)
        if not isolation_path.exists():
            raise RuntimeError(
                f"missing fresh isolation manifest: {isolation_path}; "
                "run --stage provision first"
            )
        isolation = _read_json(isolation_path)
        env = dict(os.environ)
        env.setdefault("TDAI_API_KEY", "local-benchmark-bearer")
        env.update({
            "TDAI_EVAL_ISOLATION_MAP": str(isolation_path),
            "TDAI_EVAL_SERVICE_ID": str(isolation.get("service_id") or "default"),
        })
        user_key_file = str(isolation.get("user_key_file") or "")
        if user_key_file:
            env["TDAI_EVAL_USER_KEY_FILE"] = user_key_file
        return env

    def write_plan(self) -> dict[str, Any]:
        datasets = []
        for row in self.protocol["datasets"]:
            dataset_id = str(row["dataset"])
            manifest = self.manifests[dataset_id]
            datasets.append({
                **row,
                "dataset_root": str(self._dataset_root(dataset_id)),
                "protocol_manifest": str((
                    ROOT / str(row["manifest"])
                ).resolve()),
                "stores": {
                    group_id: str(self._isolation(dataset_id, group_id))
                    for group_id in self.stores
                },
                "selected_questions": len(manifest["question_ids"]),
                "selected_conversations": len(manifest["conversation_ids"]),
            })
        payload = {
            "schema_version": 1,
            "protocol_id": self.protocol["protocol_id"],
            "protocol": str(self.protocol_path),
            "protocol_sha256": hashlib.sha256(
                self.protocol_path.read_bytes()
            ).hexdigest(),
            "phase": self.args.phase,
            "datasets": datasets,
            "arms": self.arms,
            "store_groups": self.stores,
            "execution": {
                "smoke_questions_per_dataset": self.args.smoke_questions,
                "smoke_quality_gate": self.args.quality_gate,
                "ingest_parallel": self.args.ingest_parallel,
                "retrieval_workers": self.args.retrieval_workers,
                "retrieval_parallel": self.args.retrieval_parallel,
                "answer_concurrency": self.args.answer_concurrency,
                "answer_parallel": self.args.answer_parallel,
                "base_url": self.args.base_url,
            },
            "required_memory_core_profiles": {
                "full": {
                    "prompt_mode": "cockpit",
                    "l1_idle_seconds": 5,
                    "l2_delay_after_l1_seconds": 86400,
                    "l2_min_interval_seconds": 86400,
                    "l2_max_interval_seconds": 172800,
                    "worker_concurrency": 4,
                    "profile": str(
                        ROOT / "benchmarks/framework_eval/configs/"
                        "tencentdb_full_l1_deferred_l23.env"
                    ),
                },
                "selective": {
                    "prompt_mode": "cockpit",
                    "construction_model": "qwen3.6-flash",
                    "l1_idle_seconds": 5,
                    "l2_delay_after_l1_seconds": 180,
                    "l2_min_interval_seconds": 900,
                    "l2_max_interval_seconds": 7200,
                    # Native L2/L3 locking is agent-scoped.  One conversation
                    # can enqueue multiple session tasks for the same agent,
                    # so unkeyed parallel workers cause lock retries/drops.
                    "worker_concurrency": 1,
                    "l1_batch_mode": "conversation",
                    "l1_transport_batch_messages": 128,
                    "l1_compact_selected_sessions": True,
                    "l1_max_messages_per_extraction": 40,
                    "profile": str(
                        ROOT / "benchmarks/framework_eval/configs/"
                        "tencentdb_selective_async_l23.env"
                    ),
                },
            },
            "cost_controls": {
                "phase_2_requires_explicit_phase": True,
                "longmemeval_not_in_phase_1": True,
                "answer_and_judge_resume": True,
                "construction_token_source": (
                    "service provider ledger; not fabricated by evaluator"
                ),
                "secrets_persisted": False,
            },
        }
        _write_immutable_json(self.run_root / "run-plan.json", payload)
        return payload

    def load_frozen_plan(self) -> dict[str, Any]:
        path = self.run_root / "run-plan.json"
        if not path.exists():
            if self.args.dataset or self.args.arm:
                raise RuntimeError(
                    "filtered execution requires a frozen full run plan; run "
                    "--stage plan without --dataset/--arm first"
                )
            return self.write_plan()
        frozen = _read_json(path)
        frozen_digest = str(frozen.get("protocol_sha256") or "")
        current_digest = hashlib.sha256(
            self.protocol_path.read_bytes()
        ).hexdigest()
        if frozen_digest and frozen_digest != current_digest:
            raise RuntimeError(
                "frozen run plan uses a changed protocol file; choose a fresh "
                "run root"
            )
        validate_execution_subset(
            frozen, self.protocol, phase=self.args.phase
        )
        return frozen

    def _needs_dataset_serialization(self) -> bool:
        """Return whether async profile work must be isolated by dataset.

        MemoryCore's queue and worker metrics are process-global.  Ingesting a
        second dataset before the first dataset's L2/L3 retrieval completes can
        therefore place unrelated work ahead of the first dataset and turn its
        measured readiness latency into cross-dataset head-of-line blocking.
        """
        return len(self.datasets) > 1 and any(
            store["l23_schedule"] == "buffered_dirty_event"
            for store in self.stores.values()
        )

    def _dataset_runner(self, dataset_id: str) -> "SelectiveMemoryAblation":
        if dataset_id not in self.datasets:
            raise ValueError(f"unknown selected dataset: {dataset_id}")
        child_args = argparse.Namespace(**vars(self.args))
        child_args.dataset = [dataset_id]
        child = SelectiveMemoryAblation(child_args)
        child.load_frozen_plan()
        return child

    def run_end_to_end(self, *, smoke: bool) -> None:
        """Run construction through scoring without cross-dataset overlap."""
        if not self._needs_dataset_serialization():
            self.ingest(smoke_only=smoke)
            self.retrieve(smoke=smoke)
            self.answer(smoke=smoke)
            self.score(smoke=smoke)
            return
        label = "smoke" if smoke else "full"
        for dataset_id in self.datasets:
            print(
                f"[dataset-pipeline-{label}] start={dataset_id}", flush=True
            )
            child = self._dataset_runner(dataset_id)
            child.ingest(smoke_only=smoke)
            child.retrieve(smoke=smoke)
            child.answer(smoke=smoke)
            child.score(smoke=smoke)
            print(
                f"[dataset-pipeline-{label}] complete={dataset_id}", flush=True
            )

    def require_safe_standalone_stage(self, stage: str) -> None:
        """Reject split stages that can recreate cross-dataset queue overlap."""
        if stage in {"ingest", "retrieve"} and self._needs_dataset_serialization():
            raise RuntimeError(
                f"--stage {stage} with buffered L2/L3 requires exactly one "
                "--dataset; use --stage smoke/all for automatic end-to-end "
                "dataset serialization"
            )

    def provision(self) -> None:
        expected = [
            self._isolation(dataset_id, group_id)
            for dataset_id in self.datasets
            for group_id in self.stores
        ]
        if all(path.exists() for path in expected):
            print("[provision] all fresh isolation manifests already exist", flush=True)
            return
        if not self.args.principal_manifest or not self.args.principal_user_key_file:
            raise RuntimeError(
                "provision requires --principal-manifest and "
                "--principal-user-key-file"
            )
        principal = _read_json(self.args.principal_manifest.resolve())
        service_id = str(principal.get("service_id") or "default")
        specs = []
        for dataset_id, manifest in self.manifests.items():
            protocol_manifest_path = next(
                (ROOT / str(row["manifest"])).resolve()
                for row in self.protocol["datasets"]
                if str(row["dataset"]) == dataset_id
            )
            for group_id in self.stores:
                isolation = self._isolation(dataset_id, group_id)
                marker = isolation.with_suffix(".provisioned.json")
                if isolation.exists() and not marker.exists():
                    _write_json(marker, {
                        "completed": True,
                        "name": f"{dataset_id}-{group_id}",
                        "seconds": 0,
                        "recovered_existing_manifest": True,
                    })
                run_id = (
                    f"selmem-{dataset_id}-{group_id}-"
                    f"{_stable_key(self.protocol['seed'], str(self.run_root))[:8]}"
                )
                command = (
                    sys.executable,
                    "-m", "benchmarks.framework_eval.provision_tencentdb",
                    "--base-url", self.args.base_url,
                    "--service-id", service_id,
                    "--run-id", run_id,
                    "--dataset", dataset_id,
                    "--dataset-root", str(self._dataset_root(dataset_id)),
                    "--selection-manifest", str(protocol_manifest_path),
                    "--memory-layers", "L0,L1,L2,L3",
                    "--reuse-principal-manifest",
                    str(self.args.principal_manifest.resolve()),
                    "--existing-user-key-file",
                    str(self.args.principal_user_key_file.resolve()),
                    "--output", str(isolation),
                )
                specs.append(CommandSpec(
                    f"{dataset_id}-{group_id}", command,
                    self._provision_env(),
                    isolation.with_suffix(".provision.log"), marker,
                ))
        _run_commands(specs, parallel=1, label="provision")

    def _provision_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("TDAI_API_KEY", "local-benchmark-bearer")
        return env

    def _smoke_manifest(self, dataset_id: str) -> tuple[Path, dict[str, Any]]:
        path = self.run_root / "manifests" / f"{dataset_id}-smoke.json"
        payload = select_smoke_manifest(
            self.datasets[dataset_id], self.manifests[dataset_id],
            count=self.args.smoke_questions,
            seed=str(self.protocol["seed"]),
        )
        _write_immutable_json(path, payload)
        return path, payload

    def ingest(self, *, smoke_only: bool) -> None:
        specs: list[CommandSpec] = []
        for dataset_id, manifest in self.manifests.items():
            if smoke_only:
                _path, selected = self._smoke_manifest(dataset_id)
                conversation_ids = list(selected["conversation_ids"])
            else:
                conversation_ids = list(manifest["conversation_ids"])
            for group_id, store in self.stores.items():
                store_root = self._store_root(dataset_id, group_id)
                for conversation_id in conversation_ids:
                    safe = _safe_name(conversation_id)
                    trace = store_root / "traces" / f"{safe}.jsonl"
                    marker = store_root / "ingest" / f"{safe}.done.json"
                    if trace.exists() and not marker.exists():
                        raise RuntimeError(
                            f"partial ingestion trace without completion marker: {trace}; "
                            "use a fresh namespace instead of risking duplicate L0 rows"
                        )
                    env = self._runtime_env(dataset_id, group_id)
                    env.update(construction_environment(store))
                    env["TDAI_EVAL_CONSTRUCTION_TRACE"] = str(trace)
                    output = store_root / "ingest" / f"{safe}.jsonl"
                    command = (
                        sys.executable,
                        "-m", "benchmarks.framework_eval.cli", "retrieval",
                        "--adapter", "tencentdb",
                        "--dataset", dataset_id,
                        "--dataset-root", str(self._dataset_root(dataset_id)),
                        "--conversation", conversation_id,
                        "--top-k", "1",
                        "--base-url", self.args.base_url,
                        "--output", str(output),
                        "--ingest-only",
                    )
                    specs.append(CommandSpec(
                        f"{dataset_id}-{group_id}-{conversation_id}", command, env,
                        store_root / "ingest" / f"{safe}.log", marker,
                    ))
        _run_commands(
            specs,
            parallel=self.args.ingest_parallel,
            label="smoke-ingest" if smoke_only else "ingest",
        )

    def retrieve(self, *, smoke: bool) -> None:
        for dataset_id, full_manifest in self.manifests.items():
            if smoke:
                manifest_path, manifest = self._smoke_manifest(dataset_id)
                label = "smoke"
                workers = 1
            else:
                manifest = full_manifest
                manifest_path = next(
                    (ROOT / str(row["manifest"])).resolve()
                    for row in self.protocol["datasets"]
                    if str(row["dataset"]) == dataset_id
                )
                label = "full"
                workers = self.args.retrieval_workers
            selected_ids = {str(value) for value in manifest["question_ids"]}
            shards = balanced_shards(
                list(manifest["conversation_ids"]),
                self.datasets[dataset_id], workers,
            )
            specs: list[CommandSpec] = []
            shard_outputs: dict[str, list[Path]] = {}
            for arm in self.arms:
                arm_id = str(arm["id"])
                group_id = str(arm["store_group"])
                arm_root = self._arm_root(dataset_id, arm_id, label)
                shard_outputs[arm_id] = []
                for index, conversations in enumerate(shards):
                    output = arm_root / "shards" / f"shard-{index:02d}.jsonl"
                    shard_outputs[arm_id].append(output)
                    shard_manifest_path = (
                        arm_root / "shards" /
                        f"shard-{index:02d}.selection.json"
                    )
                    _write_immutable_json(
                        shard_manifest_path,
                        shard_selection_manifest(
                            manifest,
                            self.datasets[dataset_id],
                            conversations,
                            shard_index=index,
                            shard_count=len(shards),
                        ),
                    )
                    env = self._runtime_env(dataset_id, group_id)
                    env.update(arm_environment(arm))
                    env["TDAI_EVAL_CONSTRUCTION_TRACE"] = str(
                        self._store_root(dataset_id, group_id) / "traces"
                    )
                    command = [
                        sys.executable,
                        "-m", "benchmarks.framework_eval.cli", "retrieval",
                        "--adapter", "tencentdb",
                        "--dataset", dataset_id,
                        "--dataset-root", str(self._dataset_root(dataset_id)),
                        "--selection-manifest", str(shard_manifest_path),
                        "--top-k", str(arm["resolved_top_k"]),
                        "--max-context-chars",
                        str(arm["resolved_max_context_chars"]),
                        "--base-url", self.args.base_url,
                        "--output", str(output),
                        "--skip-ingest",
                    ]
                    for conversation_id in conversations:
                        command.extend(["--conversation", conversation_id])
                    if set(arm["retrieval"]["layers"]) & {"L1", "L2", "L3"}:
                        command.extend([
                            "--ready-timeout", str(self.args.ready_timeout)
                        ])
                    specs.append(CommandSpec(
                        f"{dataset_id}-{arm_id}-{label}-{index:02d}",
                        tuple(command), env,
                        output.with_suffix(".log"),
                        output.with_suffix(".done.json"),
                    ))
            _run_commands(
                specs,
                parallel=self.args.retrieval_parallel,
                label=f"retrieve-{dataset_id}-{label}",
            )
            for arm in self.arms:
                arm_id = str(arm["id"])
                output = self._arm_root(
                    dataset_id, arm_id, label
                ) / "retrieval.jsonl"
                merge_retrieval_shards(
                    shard_outputs[arm_id], selected_ids, output
                )

    def _answer_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        required = (
            "MEMEVAL_ANSWER_BASE_URL",
            "MEMEVAL_ANSWER_API_KEY",
            "MEMEVAL_ANSWER_MODEL",
        )
        if not all(env.get(value) for value in required):
            provider_env = self.args.provider_env.resolve()
            if not provider_env.exists():
                raise RuntimeError(
                    "answer provider environment is missing and MEMEVAL_ANSWER_* "
                    f"is incomplete: {provider_env}"
                )
            raw = subprocess.check_output(
                [
                    "bash", "-lc",
                    'set -a; source "$1" >/dev/null 2>&1; env -0',
                    "memeval-provider", str(provider_env),
                ],
                cwd=provider_env.parent,
            )
            provider = {}
            for item in raw.decode(errors="replace").split("\0"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    provider[key] = value
            mapping = {
                "MEMEVAL_ANSWER_BASE_URL": "MEMORY_LLM_BASE_URL",
                "MEMEVAL_ANSWER_API_KEY": "MEMORY_LLM_API_KEY",
                "MEMEVAL_ANSWER_MODEL": "MEMORY_LLM_MODEL",
            }
            for target, source in mapping.items():
                if not env.get(target):
                    env[target] = provider.get(source, "")
        missing = [value for value in required if not env.get(value)]
        if missing:
            raise RuntimeError(
                "missing answer provider settings: " + ", ".join(missing)
            )
        env.setdefault("MEMEVAL_ANSWER_MAX_TOKENS", "64")
        env.setdefault("MEMEVAL_ANSWER_TIMEOUT", "300")
        return env

    def answer(self, *, smoke: bool) -> None:
        label = "smoke" if smoke else "full"
        env = self._answer_environment()
        specs = []
        for dataset_id in self.datasets:
            for arm in self.arms:
                arm_id = str(arm["id"])
                arm_root = self._arm_root(dataset_id, arm_id, label)
                retrieval = arm_root / "retrieval.jsonl"
                if not retrieval.exists():
                    raise RuntimeError(f"missing retrieval before answer: {retrieval}")
                predictions = arm_root / "predictions.jsonl"
                command = (
                    sys.executable,
                    "-m", "benchmarks.framework_eval.cli", "answer",
                    "--input", str(retrieval),
                    "--output", str(predictions),
                    "--concurrency", str(self.args.answer_concurrency),
                    "--resume",
                )
                specs.append(CommandSpec(
                    f"{dataset_id}-{arm_id}-{label}", command, env,
                    arm_root / "answer.log", arm_root / "answer.done.json",
                ))
        _run_commands(
            specs, parallel=self.args.answer_parallel, label=f"answer-{label}"
        )

    def score(self, *, smoke: bool) -> None:
        label = "smoke" if smoke else "full"
        if "longmemeval" in self.datasets:
            env = self._answer_environment()
            env.setdefault(
                "MEMEVAL_JUDGE_API_KEY", env["MEMEVAL_ANSWER_API_KEY"]
            )
        else:
            env = dict(os.environ)
        specs = []
        for dataset_id in self.datasets:
            for arm in self.arms:
                arm_id = str(arm["id"])
                arm_root = self._arm_root(dataset_id, arm_id, label)
                predictions = arm_root / "predictions.jsonl"
                if not predictions.exists():
                    raise RuntimeError(f"missing predictions before score: {predictions}")
                command = [
                    sys.executable,
                    "-m", "benchmarks.framework_eval.cli", "score",
                    "--dataset", dataset_id,
                    "--dataset-root", str(self._dataset_root(dataset_id)),
                    "--input", str(predictions),
                    "--output-dir", str(arm_root / "score"),
                ]
                if dataset_id == "longmemeval":
                    command.extend([
                        "--metrics", "llm",
                        "--judge-model", env["MEMEVAL_ANSWER_MODEL"],
                        "--judge-base-url", env["MEMEVAL_ANSWER_BASE_URL"],
                        "--judge-api-key-env", "MEMEVAL_JUDGE_API_KEY",
                        "--concurrency", str(self.args.judge_concurrency),
                        "--resume",
                    ])
                else:
                    command.extend(["--metrics", "exact", "contains"])
                specs.append(CommandSpec(
                    f"{dataset_id}-{arm_id}-{label}", tuple(command), env,
                    arm_root / "score.log", arm_root / "score.done.json",
                ))
        _run_commands(
            specs,
            parallel=max(1, min(self.args.answer_parallel, 2)),
            label=f"score-{label}",
        )

    def enforce_smoke_gate(self) -> dict[str, Any]:
        results = []
        failures = []
        for dataset_id in self.datasets:
            for arm in self.arms:
                arm_id = str(arm["id"])
                if arm_id not in GATE_ARMS:
                    continue
                summary = _read_json(
                    self._arm_root(dataset_id, arm_id, "smoke")
                    / "score" / "score-summary.json"
                )
                if dataset_id == "longmemeval":
                    score = float(summary.get("accuracy") or 0)
                else:
                    score = float(
                        (summary.get("metrics") or {})
                        .get("contains", {}).get("mean") or 0
                    )
                passed = score >= self.args.quality_gate
                row = {
                    "dataset": dataset_id,
                    "arm": arm_id,
                    "score": score,
                    "threshold": self.args.quality_gate,
                    "pass": passed,
                }
                results.append(row)
                if not passed:
                    failures.append(row)
        payload = {"pass": not failures, "results": results}
        _write_json(self.run_root / "smoke-gate.json", payload)
        if failures:
            detail = ", ".join(
                f"{row['dataset']}/{row['arm']}={row['score']:.3f}"
                for row in failures
            )
            raise RuntimeError(
                "smoke quality gate failed; full answer run is blocked: " + detail
            )
        return payload

    def report(self) -> dict[str, Any]:
        rows = []
        for dataset_id in self.datasets:
            for arm in self.arms:
                arm_id = str(arm["id"])
                arm_root = self._arm_root(dataset_id, arm_id, "full")
                rows.append(summarize_arm(
                    dataset_id=dataset_id,
                    arm=arm,
                    retrieval_path=arm_root / "retrieval.jsonl",
                    predictions_path=arm_root / "predictions.jsonl",
                    score_path=arm_root / "score" / "score-summary.json",
                    trace_dir=self._store_root(
                        dataset_id, str(arm["store_group"])
                    ) / "traces",
                ))
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_dataset.setdefault(row["dataset"], []).append(row)
        for dataset_rows in by_dataset.values():
            baseline = next(
                (row for row in dataset_rows if row["arm"] == "full-l0-l1"),
                None,
            )
            if baseline:
                for row in dataset_rows:
                    row["delta_vs_full"] = _comparison_delta(row, baseline)
        payload = {
            "schema_version": 1,
            "protocol_id": self.protocol["protocol_id"],
            "rows": rows,
            "construction_token_note": (
                "Public v3 pipeline does not expose a per-namespace token ledger; "
                "actual build tokens remain null until provider/service ledger export."
            ),
        }
        _write_json(self.run_root / "comparison.json", payload)
        (self.run_root / "comparison.md").write_text(
            _comparison_markdown(rows), encoding="utf-8"
        )
        return payload

    def status(self) -> dict[str, Any]:
        payload = {
            "run_root": str(self.run_root),
            "datasets": {},
        }
        for dataset_id, manifest in self.manifests.items():
            dataset_status = {
                "expected_questions": len(manifest["question_ids"]),
                "expected_conversations": len(manifest["conversation_ids"]),
                "stores": {},
                "arms": {},
            }
            for group_id in self.stores:
                store_root = self._store_root(dataset_id, group_id)
                dataset_status["stores"][group_id] = {
                    "provisioned": self._isolation(dataset_id, group_id).exists(),
                    "ingested_conversations": len(list(
                        (store_root / "ingest").glob("*.done.json")
                    )) if (store_root / "ingest").exists() else 0,
                    "trace_files": len(list(
                        (store_root / "traces").glob("*.jsonl")
                    )) if (store_root / "traces").exists() else 0,
                }
            for arm in self.arms:
                arm_id = str(arm["id"])
                arm_root = self._arm_root(dataset_id, arm_id, "full")
                dataset_status["arms"][arm_id] = {
                    "retrieval_rows": len(_read_jsonl(
                        arm_root / "retrieval.jsonl"
                    )),
                    "prediction_rows": len(_read_jsonl(
                        arm_root / "predictions.jsonl"
                    )),
                    "scored": (
                        arm_root / "score" / "score-summary.json"
                    ).exists(),
                }
            payload["datasets"][dataset_id] = dataset_status
        _write_json(self.run_root / "status.json", payload)
        return payload


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _trace_summary(trace_dir: Path) -> dict[str, Any]:
    rows = []
    if trace_dir.exists():
        for path in sorted(trace_dir.glob("*.jsonl")):
            rows.extend(_read_jsonl(path))
    unique = {}
    for row in rows:
        key = (
            str(row.get("conversation_id")),
            str(row.get("session_id")),
            str(row.get("agent_id")),
        )
        unique[key] = row
    rows = list(unique.values())
    selected = sum(bool((row.get("decision") or {}).get("extract_l1")) for row in rows)
    return {
        "scope_sessions": len(rows),
        "selected_for_l1": selected,
        "suppressed_from_l1": len(rows) - selected,
        "selection_rate": selected / len(rows) if rows else None,
        "source_characters": sum(
            int((row.get("decision") or {}).get("source_characters", 0))
            for row in rows
        ),
        "actual_prompt_tokens": None,
        "actual_completion_tokens": None,
        "actual_total_tokens": None,
        "token_measurement": "awaiting_service_or_provider_ledger",
    }


def summarize_arm(
    *,
    dataset_id: str,
    arm: dict[str, Any],
    retrieval_path: Path,
    predictions_path: Path,
    score_path: Path,
    trace_dir: Path,
) -> dict[str, Any]:
    retrieval = _read_jsonl(retrieval_path)
    predictions = _read_jsonl(predictions_path)
    score = _read_json(score_path) if score_path.exists() else {}
    contexts = [float(row["metrics"]["context_chars"]) for row in retrieval]
    searches = [float(row["metrics"]["search_seconds"]) for row in retrieval]
    recalls = [
        float(row["metrics"]["evidence_recall"])
        for row in retrieval
        if row["metrics"].get("evidence_recall") is not None
    ]
    routes = [str(row["metrics"].get("retrieval_route") or "fixed")
              for row in retrieval]
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    answer_tokens = {
        key: sum(int((row.get("usage") or {}).get(key, 0)) for row in predictions)
        for key in usage_keys
    }
    if dataset_id == "longmemeval":
        accuracy = score.get("accuracy")
        exact = None
    else:
        metrics = score.get("metrics") or {}
        accuracy = (metrics.get("contains") or {}).get("mean")
        exact = (metrics.get("exact") or {}).get("mean")
    return {
        "dataset": dataset_id,
        "arm": str(arm["id"]),
        "store_group": str(arm["store_group"]),
        "questions": len(retrieval),
        "accuracy": accuracy,
        "exact_accuracy": exact,
        "mean_evidence_recall": statistics.fmean(recalls) if recalls else None,
        "context_chars": {
            "mean": statistics.fmean(contexts) if contexts else None,
            "p95": _percentile(contexts, 0.95),
        },
        "search_seconds": {
            "mean": statistics.fmean(searches) if searches else None,
            "p50": _percentile(searches, 0.50),
            "p95": _percentile(searches, 0.95),
        },
        "routing": {
            "fast": sum(value == "fast" for value in routes),
            "fallback": sum(value == "fallback" for value in routes),
            "fixed": sum(value == "fixed" for value in routes),
        },
        "answer_tokens": answer_tokens,
        "mean_answer_prompt_tokens": (
            answer_tokens["prompt_tokens"] / len(predictions)
            if predictions else None
        ),
        "construction": _trace_summary(trace_dir),
    }


def _comparison_delta(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def relative(current, reference):
        if current is None or reference in (None, 0):
            return None
        return (float(current) - float(reference)) / float(reference)

    accuracy = (
        None if row.get("accuracy") is None or baseline.get("accuracy") is None
        else float(row["accuracy"]) - float(baseline["accuracy"])
    )
    return {
        "accuracy_points": accuracy,
        "context_fraction": relative(
            row["context_chars"]["mean"], baseline["context_chars"]["mean"]
        ),
        "search_latency_fraction": relative(
            row["search_seconds"]["mean"], baseline["search_seconds"]["mean"]
        ),
        "answer_prompt_token_fraction": relative(
            row["mean_answer_prompt_tokens"],
            baseline["mean_answer_prompt_tokens"],
        ),
    }


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%" if percent else f"{float(value):.3f}"


def _comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Selective-memory paired ablation",
        "",
        "| Dataset | Arm | Accuracy | Context chars | Search p95 (s) | "
        "Prompt tokens/q | L1 selection |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['arm']} | "
            f"{_format(row['accuracy'], percent=True)} | "
            f"{_format(row['context_chars']['mean'])} | "
            f"{_format(row['search_seconds']['p95'])} | "
            f"{_format(row['mean_answer_prompt_tokens'])} | "
            f"{_format(row['construction']['selection_rate'], percent=True)} |"
        )
    lines.extend([
        "",
        "> Construction token totals remain N/A until exported from the "
        "service/provider ledger; the evaluator never substitutes a character "
        "estimate for an actual token count.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "plan", "provision", "smoke", "ingest", "retrieve",
            "answer", "score", "report", "status", "all",
        ),
        default="plan",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--base-url", default="http://127.0.0.1:8420")
    parser.add_argument("--principal-manifest", type=Path)
    parser.add_argument("--principal-user-key-file", type=Path)
    parser.add_argument("--provider-env", type=Path, default=DEFAULT_PROVIDER_ENV)
    parser.add_argument("--smoke-questions", type=int, default=4)
    parser.add_argument("--quality-gate", type=float, default=0.80)
    parser.add_argument("--ingest-parallel", type=int, default=2)
    parser.add_argument("--retrieval-workers", type=int, default=4)
    parser.add_argument("--retrieval-parallel", type=int, default=4)
    parser.add_argument("--answer-concurrency", type=int, default=4)
    parser.add_argument("--answer-parallel", type=int, default=1)
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 2 <= args.smoke_questions <= 4:
        raise SystemExit("--smoke-questions must be between 2 and 4")
    if not 0 <= args.quality_gate <= 1:
        raise SystemExit("--quality-gate must be between 0 and 1")
    for name in (
        "ingest_parallel", "retrieval_workers", "retrieval_parallel",
        "answer_concurrency", "answer_parallel", "judge_concurrency",
        "ready_timeout",
    ):
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    runner = SelectiveMemoryAblation(args)
    if args.stage == "plan":
        if args.dataset or args.arm:
            raise SystemExit(
                "--stage plan freezes the full phase; omit --dataset/--arm"
            )
        plan = runner.write_plan()
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    runner.load_frozen_plan()
    if args.stage in {"provision", "all"}:
        runner.provision()
    if args.stage in {"smoke", "all"}:
        runner.run_end_to_end(smoke=True)
        gate = runner.enforce_smoke_gate()
        print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "all":
        runner.run_end_to_end(smoke=False)
        report = runner.report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    runner.require_safe_standalone_stage(args.stage)
    if args.stage == "ingest":
        runner.ingest(smoke_only=False)
    if args.stage == "retrieve":
        runner.retrieve(smoke=False)
    if args.stage == "answer":
        runner.answer(smoke=False)
    if args.stage == "score":
        runner.score(smoke=False)
    if args.stage == "report":
        report = runner.report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.stage == "status":
        print(json.dumps(runner.status(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
