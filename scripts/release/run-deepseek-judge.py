#!/usr/bin/env python3
"""Independent binary Judge for a frozen cockpit prediction file."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.framework_eval.judges.base import JudgeConfig
from benchmarks.framework_eval.judges.longmemeval import OpenAIJudgeLLM, parse_yes_no


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir / "judge-results.jsonl"
    summary = args.output_dir / "score-summary.json"
    if output.exists() or summary.exists():
        raise SystemExit("refusing to overwrite existing Judge output")
    api_key = os.environ.get("MEMEVAL_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("MEMEVAL_JUDGE_API_KEY or OPENAI_API_KEY is required")
    rows = [
        json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = JudgeConfig(
        metrics=("llm",), model=args.model, base_url=args.base_url,
        api_key=api_key, concurrency=args.concurrency, resume=False,
        metadata={"timeout": 180},
    )

    def judge(row: dict) -> dict:
        prompt = (
            "你是严格二分类评审器。不要回答问题，只比较模型回答和标准答案。"
            "模型回答完整覆盖标准答案且无矛盾才输出 YES，否则输出 NO。只允许输出 YES 或 NO。\n"
            f"标准答案：{'；'.join(row.get('gold_answers') or ())}\n"
            f"模型回答：{row.get('predicted_answer', '')}\n最终标签："
        )
        started = time.monotonic()
        try:
            raw, usage = OpenAIJudgeLLM().complete(prompt, config)
            label, error = parse_yes_no(raw), None
        except Exception as exc:
            raw, usage, label = "", {}, None
            error = f"{type(exc).__name__}: {exc}"
        return {
            "qa_id": row["qa_id"],
            "category": row.get("category", ""),
            "answer_route": row.get("answer_route", {}).get("route", ""),
            "label": label,
            "judge_response": raw,
            "judge_model": config.model,
            "usage": usage,
            "latency_seconds": time.monotonic() - started,
            "error": error,
        }

    completed: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency, thread_name_prefix="judge"
        ) as pool:
            futures = {pool.submit(judge, row): index for index, row in enumerate(rows)}
            pending: dict[int, dict] = {}
            next_index = 0
            for future in concurrent.futures.as_completed(futures):
                pending[futures[future]] = future.result()
                while next_index in pending:
                    result = pending.pop(next_index)
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.append(result)
                    next_index += 1
    correct = sum(item["label"] is True for item in completed)
    errors = sum(item["label"] is None for item in completed)
    categories = {}
    for category in sorted({item["category"] for item in completed}):
        subset = [item for item in completed if item["category"] == category]
        passed = sum(item["label"] is True for item in subset)
        categories[category] = {
            "count": len(subset), "correct": passed,
            "errors": sum(item["label"] is None for item in subset),
            "accuracy": passed / len(subset),
        }
    report = {
        "expected_count": len(rows), "result_count": len(completed),
        "correct": correct, "errors": errors,
        "accuracy": correct / len(rows) if rows else 0,
        "by_category": categories, "judge_model": config.model,
        "results": str(output),
    }
    summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
