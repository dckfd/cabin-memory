#!/usr/bin/env python3
"""Independent DeepSeek Pro judge for the full-500 run."""
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmarks.framework_eval.judges.base import JudgeConfig
from benchmarks.framework_eval.judges.longmemeval import OpenAIJudgeLLM, parse_yes_no


def main() -> None:
    base = Path(__file__).parent
    rows = [json.loads(line) for line in (base / "predictions-flash.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    output, summary = base / "judge-results.jsonl", base / "score-summary.json"
    if output.exists() or summary.exists():
        raise SystemExit("refusing overwrite")
    config = JudgeConfig(
        metrics=("llm",), model="deepseek-v4-pro", base_url="https://api.deepseek.com",
        api_key=os.environ["TDAI_LLM_API_KEY"], concurrency=2, resume=False,
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
            "qa_id": row["qa_id"], "category": row.get("category", ""),
            "answer_route": row.get("answer_route", {}).get("route", ""),
            "label": label, "judge_response": raw, "judge_model": config.model,
            "usage": usage, "latency_seconds": time.monotonic() - started, "error": error,
        }

    completed = []
    with output.open("x", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="judge") as pool:
            futures = {pool.submit(judge, row): index for index, row in enumerate(rows)}
            pending = {}
            next_index = 0
            for future in concurrent.futures.as_completed(futures):
                pending[futures[future]] = future.result()
                while next_index in pending:
                    result = pending.pop(next_index)
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.append(result)
                    next_index += 1
    correct = sum(result["label"] is True for result in completed)
    errors = sum(result["label"] is None for result in completed)
    categories = {}
    for category in sorted({result["category"] for result in completed}):
        subset = [result for result in completed if result["category"] == category]
        categories[category] = {
            "count": len(subset), "correct": sum(result["label"] is True for result in subset),
            "errors": sum(result["label"] is None for result in subset),
        }
        categories[category]["accuracy"] = categories[category]["correct"] / len(subset)
    report = {
        "expected_count": len(rows), "result_count": len(completed), "correct": correct,
        "errors": errors, "accuracy": correct / len(rows), "by_category": categories,
        "judge_model": config.model, "results": str(output),
    }
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
