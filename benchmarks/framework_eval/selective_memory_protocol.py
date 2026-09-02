from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .plugins import PluginCatalog


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "benchmarks/framework_eval/configs/selective_memory_v1.json"
)
DEFAULT_OUTPUT = ROOT / "benchmarks/framework_eval/protocols/selective-memory-v1"


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _largest_remainder_quotas(
    counts: Counter[str], target: int,
) -> dict[str, int]:
    if target <= 0 or target > sum(counts.values()):
        raise ValueError("target must be within the available question count")
    exact = {
        category: target * count / sum(counts.values())
        for category, count in counts.items()
    }
    quotas = {category: math.floor(value) for category, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda category: (
            -(exact[category] - quotas[category]), category,
        ),
    )
    for category in order[:remaining]:
        quotas[category] += 1
    return quotas


def select_question_ids(
    questions: list[Any], *, strategy: str, target: int, seed: str,
    minimum: int, maximum: int,
) -> tuple[list[str], list[str]]:
    """Return stable question and conversation IDs without reading answers."""
    if not questions:
        raise ValueError("dataset exposes no questions")
    by_conversation: dict[str, list[Any]] = defaultdict(list)
    for question in questions:
        by_conversation[str(question.conversation_id)].append(question)

    if strategy == "all":
        selected = list(questions)
    elif strategy == "conversation_hash_until":
        selected = []
        ordered = sorted(
            by_conversation,
            key=lambda value: _stable_key(seed, value),
        )
        for conversation_id in ordered:
            candidate = by_conversation[conversation_id]
            if len(selected) >= target:
                break
            if len(selected) + len(candidate) > maximum:
                continue
            selected.extend(candidate)
    elif strategy == "stratified_question_hash":
        by_category: dict[str, list[Any]] = defaultdict(list)
        for question in questions:
            by_category[str(question.category or "uncategorized")].append(question)
        quotas = _largest_remainder_quotas(
            Counter({key: len(value) for key, value in by_category.items()}),
            target,
        )
        selected = []
        for category, values in sorted(by_category.items()):
            ordered = sorted(
                values,
                key=lambda question: _stable_key(
                    seed, f"{category}:{question.question_id}"
                ),
            )
            selected.extend(ordered[:quotas[category]])
    else:
        raise ValueError(f"unsupported selection strategy: {strategy}")

    selected = sorted(selected, key=lambda question: str(question.question_id))
    if not minimum <= len(selected) <= maximum:
        raise ValueError(
            f"selection produced {len(selected)} questions outside "
            f"[{minimum}, {maximum}]"
        )
    conversation_ids = sorted({str(item.conversation_id) for item in selected})
    return [str(item.question_id) for item in selected], conversation_ids


def validate_arms(arms: list[dict[str, Any]]) -> None:
    ids = [str(arm.get("id") or "") for arm in arms]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("arm ids must be non-empty and unique")
    expected = {
        "full-l0-l1", "l0-top1", "adaptive-top1-top3-l1",
        "selective-l1", "selective-l1-async-l23",
    }
    if set(ids) != expected:
        raise ValueError(f"protocol must define exactly these arms: {sorted(expected)}")
    for arm in arms:
        retrieval = dict(arm.get("retrieval") or {})
        construction = dict(arm.get("construction") or {})
        if retrieval.get("policy") not in {"fixed", "adaptive"}:
            raise ValueError(f"invalid retrieval policy in {arm['id']}")
        if construction.get("l1_policy") not in {
            "all", "cockpit_selective_v1",
        }:
            raise ValueError(f"invalid L1 policy in {arm['id']}")


def build_protocol(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_arms(list(config.get("arms") or []))
    catalog = PluginCatalog(
        root=ROOT,
        framework_profiles=ROOT / "benchmarks/framework_eval/profiles.json",
        dataset_profiles=ROOT / "benchmarks/framework_eval/datasets.json",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in config["datasets"]:
        dataset_root = (ROOT / str(spec["root"])).resolve()
        dataset = catalog.create_dataset(str(spec["id"]), root=dataset_root)
        question_ids, conversation_ids = select_question_ids(
            dataset.questions(),
            strategy=str(spec["selection"]),
            target=int(spec["target_questions"]),
            seed=str(config["seed"]),
            minimum=int(spec["min_questions"]),
            maximum=int(spec["max_questions"]),
        )
        question_set = set(question_ids)
        selected_questions = [
            item for item in dataset.questions() if item.question_id in question_set
        ]
        selected_conversations = [
            dataset.conversation(value) for value in conversation_ids
        ]
        category_counts = dict(sorted(Counter(
            str(item.category or "uncategorized")
            for item in selected_questions
        ).items()))
        manifest = {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "dataset_id": spec["id"],
            "dataset_root": str(spec["root"]),
            "selection": spec["selection"],
            "seed": str(config["seed"]),
            "question_ids": question_ids,
            "conversation_ids": conversation_ids,
            "counts": {
                "questions": len(question_ids),
                "conversations": len(selected_conversations),
                "sessions": sum(len(item.sessions) for item in selected_conversations),
                "messages": sum(
                    len(session.messages)
                    for item in selected_conversations
                    for session in item.sessions
                ),
                "source_characters": sum(
                    len(message.render_text())
                    for item in selected_conversations
                    for session in item.sessions
                    for message in session.messages
                ),
            },
            "categories": category_counts,
            "leakage_controls": {
                "answers_read_for_selection": False,
                "evidence_ids_read_for_selection": False,
                "whole_conversations_ingested": True,
            },
        }
        output_path = output_dir / f"{spec['id']}.json"
        rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if (
            not force
            and output_path.exists()
            and output_path.read_text(encoding="utf-8") != rendered
        ):
            raise RuntimeError(
                f"refusing to replace changed protocol manifest: {output_path}"
            )
        output_path.write_text(rendered, encoding="utf-8")
        summaries.append({
            "dataset": spec["id"],
            "execution_phase": int(spec.get("execution_phase", 1)),
            **manifest["counts"],
            "manifest": str(output_path.relative_to(ROOT)),
        })

    plan = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "seed": str(config["seed"]),
        "datasets": summaries,
        "arms": config["arms"],
        "paired_comparison": True,
        "execution_started": False,
    }
    plan_path = output_dir / "plan.json"
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if (
        not force
        and plan_path.exists()
        and plan_path.read_text(encoding="utf-8") != rendered
    ):
        raise RuntimeError(f"refusing to replace changed protocol plan: {plan_path}")
    plan_path.write_text(rendered, encoding="utf-8")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic 300-500 question memory ablation manifests"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force", action="store_true",
        help="Replace manifests after an intentional protocol configuration change",
    )
    args = parser.parse_args()
    plan = build_protocol(
        args.config.resolve(), args.output_dir.resolve(), force=args.force
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
