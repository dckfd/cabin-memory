from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cockpit_slots import extract_clarification_reply


ANSWERER_ID = "slurp_fragmented_command_deterministic_reader"


def extract_fragment_reply(question: str, context: str) -> str:
    """Compatibility wrapper around the shared conservative slot reader."""
    candidate = extract_clarification_reply(question, context)
    return candidate.value if candidate else ""


def build_predictions(input_path: Path, output_path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predictions = []
    for row in rows:
        question = row["question"]
        answer = extract_fragment_reply(
            str(question["text"]), str(row.get("context") or "")
        )
        predictions.append({
            "schema_version": 1,
            "framework": row.get("framework"),
            "qa_id": question["question_id"],
            "conversation_id": question["conversation_id"],
            "category": question.get("category", ""),
            "gold_answers": question.get("answers") or [],
            "predicted_answer": answer or "Insufficient evidence",
            "retrieval_metrics": row.get("metrics") or {},
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "answerer": ANSWERER_ID,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    return {
        "questions": len(predictions),
        "answers_extracted": sum(
            row["predicted_answer"] != "Insufficient evidence"
            for row in predictions
        ),
        "uses_llm": False,
        "uses_gold_or_evidence_ids": False,
        "answerer": ANSWERER_ID,
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically read fragmented SLURP replies from retrieval context"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    print(json.dumps(build_predictions(args.input, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
