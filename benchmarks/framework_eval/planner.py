from __future__ import annotations

import json
import shutil
from pathlib import Path


def load_profiles(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in data["profiles"]}


def build_plan(*, root: Path, profiles: dict[str, dict], framework_ids: list[str],
               datasets: list[str], split: str, answer_model: str, judge: str,
               track: str = "unified", top_k: int = 8,
               max_context_chars: int = 20000) -> dict:
    if track not in {"unified", "native"}:
        raise ValueError(f"unknown evaluation track: {track}")
    rows = []
    for framework_id in framework_ids:
        if framework_id not in profiles:
            raise ValueError(f"unknown framework profile: {framework_id}")
        profile = profiles[framework_id]
        for dataset in datasets:
            blockers = []
            if dataset not in profile.get("supported_datasets", []):
                blockers.append(f"dataset-not-supported:{dataset}")
            if profile["status"] not in {"adapter-ready", "gate-ready", "ready"}:
                blockers.append(profile["status"])
            for command in profile.get("prerequisites", []):
                if shutil.which(command) is None:
                    blockers.append(f"missing-command:{command}")
            runtime_python = profile.get("python")
            if runtime_python and not (root / str(runtime_python)).exists():
                blockers.append(f"missing-runtime-python:{runtime_python}")
            missing_entries = [value for value in profile.get("official_entrypoints", [])
                               if not (root / value).exists()]
            blockers.extend(f"missing-entrypoint:{value}" for value in missing_entries)
            rows.append({
                "run_id": f"{dataset}-{split}-{track}-{framework_id}",
                "framework": framework_id,
                "dataset": dataset,
                "split": split,
                "execution": profile["execution"],
                "runtime": profile["runtime"],
                "answer_model": answer_model,
                "judge": judge,
                "track": track,
                "retrieval_budget": (
                    {"top_k": top_k, "max_context_chars": max_context_chars}
                    if track == "unified" else {"policy": "framework-native"}
                ),
                "status": "adapter-ready" if not blockers else "blocked",
                "blockers": blockers,
                "required_services": profile.get("services", []),
                "artifacts": {
                    "retrieval": f"benchmarks/framework_eval_runs/{dataset}-{split}-{framework_id}/retrieval.jsonl",
                    "predictions": f"benchmarks/framework_eval_runs/{dataset}-{split}-{framework_id}/predictions.jsonl",
                    "scores": f"benchmarks/framework_eval_runs/{dataset}-{split}-{framework_id}/score",
                },
            })
    return {
        "schema_version": 1,
        "execution_policy": "plan-only",
        "track": track,
        "fairness": {
            "same_answer_model": answer_model,
            "same_judge": judge,
            "framework_specific_question_tuning": False,
            "failures_remain_in_denominator": True,
            "retrieval_budget": (
                {"top_k": top_k, "max_context_chars": max_context_chars}
                if track == "unified" else "framework-native-reported-separately"
            ),
        },
        "run_count": len(rows),
        "adapter_ready_count": sum(row["status"] == "adapter-ready" for row in rows),
        "runs": rows,
    }


def write_plan(plan: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Memory framework evaluation plan",
        "",
        f"- Policy: {plan['execution_policy']}",
        f"- Track: {plan['track']}",
        f"- Runs: {plan['run_count']}",
        f"- Adapter-ready (not executed): {plan['adapter_ready_count']}",
        "",
        "| Run | Framework | Dataset | Runtime | Status | Blockers |",
        "|---|---|---|---|---|---|",
    ]
    for row in plan["runs"]:
        blockers = ", ".join(row["blockers"]) or "-"
        lines.append(f"| {row['run_id']} | {row['framework']} | {row['dataset']} | {row['runtime']} | {row['status']} | {blockers} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
