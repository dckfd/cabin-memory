from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from .registry import load_registry
from .runner import RetrievalRunner
from .answering import AnswerConfig, OpenAIAnswerer, answer_retrieval_file
from .doctor import inspect_frameworks, write_doctor_report
from .report import summarize_run, write_markdown
from .scoring import score_locomo_predictions
from .planner import build_plan, load_profiles, write_plan
from .validator import validate_retrieval
from .sources import verify_locked_sources
from .plugins import PluginCatalog
from .judges.base import JudgeConfig
from .native_runner import NativeAnswerRunner, NativeRunPolicy


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "benchmarks/framework_eval/frameworks.json"
PROFILES = ROOT / "benchmarks/framework_eval/profiles.json"
SOURCE_LOCK = ROOT / "benchmarks/framework_eval/sources.lock.json"
DATASETS = ROOT / "benchmarks/framework_eval/datasets.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Framework-neutral memory evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List registered memory frameworks")
    doctor = sub.add_parser("doctor", help="Verify downloads, commits, and licenses")
    doctor.add_argument("--output", type=Path, default=ROOT / "benchmarks/framework_eval_runs/doctor.json")
    sources = sub.add_parser("sources", help="Verify downloaded repositories against the lock file")
    sources.add_argument("--output", type=Path,
                         default=ROOT / "benchmarks/framework_eval_runs/sources-verification.json")
    plan = sub.add_parser("plan", help="Create a framework x dataset matrix without executing it")
    plan.add_argument("--framework", action="append", help="Repeatable; default is every profile")
    plan.add_argument("--dataset", action="append", default=[])
    plan.add_argument("--split", default="gate4")
    plan.add_argument("--answer-model", default="qwen3.8-max")
    plan.add_argument("--judge", default="qwen3-14b-refined")
    plan.add_argument("--track", choices=("unified", "native"), default="unified")
    plan.add_argument("--top-k", type=int, default=8)
    plan.add_argument("--max-context-chars", type=int, default=20000)
    plan.add_argument("--output", type=Path,
                      default=ROOT / "benchmarks/framework_eval_runs/evaluation-plan.json")
    run = sub.add_parser("retrieval", help="Run an offline retrieval Gate")
    run.add_argument("--adapter", required=True,
                     help="full_context, bm25, tencentdb, or a registered external framework id")
    run.add_argument("--dataset", default="locomo_refined")
    run.add_argument("--dataset-root", type=Path)
    run.add_argument("--conversation", action="append")
    run.add_argument("--selection-manifest", type=Path,
                     help="JSON file containing question_ids; preserves an existing split")
    run.add_argument("--question-limit", type=int)
    run.add_argument("--top-k", type=int, default=8)
    run.add_argument("--max-context-chars", type=int,
                     help="Shared answer-context budget; retrieval metrics retain pre-truncation size")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base-url", help="HTTP base URL for tencentdb")
    run.add_argument("--api-key-env", default="TDAI_API_KEY",
                     help="Environment variable containing the HTTP API key")
    run.add_argument("--external-command",
                     help="JSON-protocol bridge command for an external framework")
    run.add_argument("--external-cwd", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--skip-ingest", action="store_true",
        help="Reuse an already-built isolated store and run readiness/search only",
    )
    run.add_argument(
        "--ingest-only", action="store_true",
        help="Ingest every selected conversation without waiting or searching",
    )
    run.add_argument("--reflect-after-ingest", action="store_true")
    run.add_argument("--ready-timeout", type=float)
    native = sub.add_parser(
        "native-answer",
        help="Run a framework-owned answer capability (reported as a separate native track)",
    )
    native.add_argument("--adapter", required=True)
    native.add_argument("--dataset", default="locomo_refined")
    native.add_argument("--dataset-root", type=Path)
    native.add_argument("--conversation", action="append")
    native.add_argument("--selection-manifest", type=Path)
    native.add_argument("--question-limit", type=int)
    native.add_argument("--top-k", type=int, default=8)
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--base-url")
    native.add_argument("--api-key-env", default="TDAI_API_KEY")
    native.add_argument("--external-command")
    native.add_argument("--external-cwd", type=Path)
    native.add_argument("--reflect-after-ingest", action="store_true")
    native.add_argument("--ready-timeout", type=float)
    native.add_argument("--resume", action="store_true")
    answer = sub.add_parser("answer", help="Generate answers from canonical retrieval JSONL")
    answer.add_argument("--input", type=Path, required=True)
    answer.add_argument("--output", type=Path, required=True)
    answer.add_argument("--config", type=Path,
                        help="Existing project JSON config; secrets are never copied to outputs")
    answer.add_argument("--model", help="Override only the answer model name")
    answer.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent answer-model requests (default: 1)",
    )
    answer.add_argument(
        "--multimodal", action="store_true",
        help="Send retrieved image ContentParts to a compatible answer model",
    )
    answer.add_argument("--resume", action="store_true")
    score = sub.add_parser("score-locomo", help="Run the official LoCoMo-Refined scorer")
    score.add_argument("--input", type=Path, required=True, help="Canonical predictions JSONL")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--metrics", nargs="+", choices=("f1", "bleu", "llm"), default=("f1", "bleu"))
    score.add_argument("--concurrency", type=int, default=4)
    score.add_argument("--llm-judge", default="refined")
    generic_score = sub.add_parser("score", help="Load the dataset-owned Judge plugin")
    generic_score.add_argument("--dataset", required=True)
    generic_score.add_argument("--dataset-root", type=Path)
    generic_score.add_argument("--input", type=Path, required=True)
    generic_score.add_argument("--output-dir", type=Path, required=True)
    generic_score.add_argument("--metrics", nargs="+", required=True)
    generic_score.add_argument("--judge-model")
    generic_score.add_argument("--judge-base-url")
    generic_score.add_argument("--judge-api-key-env", default="MEMEVAL_JUDGE_API_KEY")
    generic_score.add_argument("--concurrency", type=int, default=4)
    generic_score.add_argument("--resume", action="store_true")
    report = sub.add_parser("report", help="Summarize canonical retrieval and answering outputs")
    report.add_argument("--retrieval", type=Path, required=True)
    report.add_argument("--predictions", type=Path)
    report.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate", help="Validate a canonical artifact without model calls")
    validate.add_argument("--retrieval", type=Path, required=True)
    validate.add_argument("--expected-count", type=int)
    validate.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    catalog = PluginCatalog(root=ROOT, framework_profiles=PROFILES, dataset_profiles=DATASETS)
    if args.command == "list":
        for spec in load_registry(REGISTRY).values():
            print(f"{spec.framework_id:16} {spec.status:22} {spec.name}")
        return 0
    if args.command == "doctor":
        result = inspect_frameworks(ROOT, load_registry(REGISTRY))
        write_doctor_report(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "sources":
        result = verify_locked_sources(ROOT, SOURCE_LOCK)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"pass": result["pass"], "verified": result["verified_count"],
                          "total": result["source_count"], "output": str(args.output)},
                         ensure_ascii=False, indent=2))
        return 0 if result["pass"] else 1
    if args.command == "plan":
        profiles = load_profiles(PROFILES)
        frameworks = args.framework or list(profiles)
        datasets = args.dataset or ["locomo_refined", "longmemeval"]
        result = build_plan(
            root=ROOT, profiles=profiles, framework_ids=frameworks,
            datasets=datasets, split=args.split,
            answer_model=args.answer_model, judge=args.judge,
            track=args.track, top_k=args.top_k,
            max_context_chars=args.max_context_chars,
        )
        markdown = args.output.with_suffix(".md")
        write_plan(result, args.output, markdown)
        print(json.dumps({"json": str(args.output), "markdown": str(markdown),
                          "runs": result["run_count"],
                          "adapter_ready": result["adapter_ready_count"]},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "answer":
        config = (AnswerConfig.from_json(args.config, model_override=args.model)
                  if args.config else AnswerConfig.from_env())
        if args.multimodal:
            config = replace(config, multimodal=True)
        result = answer_retrieval_file(
            args.input, args.output, OpenAIAnswerer(config), resume=args.resume,
            concurrency=args.concurrency,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "score-locomo":
        result = score_locomo_predictions(
            predictions_path=args.input,
            output_dir=args.output_dir,
            locomo_root=ROOT / "LoCoMo_refined",
            metrics=tuple(args.metrics),
            concurrency=args.concurrency,
            llm_judge=args.llm_judge,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "score":
        judge = catalog.create_judge(args.dataset, root=args.dataset_root)
        config = JudgeConfig(
            metrics=tuple(args.metrics),
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key=os.getenv(args.judge_api_key_env, ""),
            concurrency=args.concurrency,
            resume=args.resume,
        )
        result = judge.score(args.input, args.output_dir, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        result = summarize_run(args.retrieval, args.predictions)
        write_markdown(result, args.output)
        json_output = args.output.with_suffix(".json")
        json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"markdown": str(args.output), "json": str(json_output)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        result = validate_retrieval(args.retrieval, expected_count=args.expected_count)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["pass"] else 1
    dataset = catalog.create_dataset(args.dataset, root=args.dataset_root)
    requested_conversations = set(args.conversation or [])
    selected_ids: set[str] | None = None
    if args.selection_manifest:
        manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        selected_ids = {str(value) for value in manifest["question_ids"]}
    if not requested_conversations and selected_ids is None:
        raise SystemExit("provide --conversation or --selection-manifest")
    questions = dataset.questions(requested_conversations or None)
    if selected_ids is not None:
        questions = [question for question in questions if question.question_id in selected_ids]
        found = {question.question_id for question in questions}
        missing = selected_ids - found
        if missing:
            raise SystemExit(f"selection contains unknown question IDs: {sorted(missing)[:5]}")
    if args.question_limit is not None:
        questions = questions[:args.question_limit]
    conversation_ids = list(dict.fromkeys(question.conversation_id for question in questions))
    conversations = [dataset.conversation(value) for value in conversation_ids]
    try:
        adapter = catalog.create_memory_adapter(
            args.adapter,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            external_command=args.external_command,
            external_cwd=args.external_cwd,
        )
    except (KeyError, ValueError, RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.command == "native-answer":
        result = NativeAnswerRunner(
            adapter,
            args.output,
            limit=args.top_k,
            resume=args.resume,
            policy=NativeRunPolicy(
                reflect_after_ingest=args.reflect_after_ingest,
                ready_timeout=args.ready_timeout,
            ),
        ).run(conversations, questions)
    else:
        if args.ingest_only and args.skip_ingest:
            raise SystemExit("--ingest-only cannot be combined with --skip-ingest")
        result = RetrievalRunner(
            adapter, args.output, limit=args.top_k, resume=args.resume,
            max_context_chars=args.max_context_chars,
            reflect_after_ingest=args.reflect_after_ingest,
            ready_timeout=args.ready_timeout,
            ingest=not args.skip_ingest,
            ingest_only=args.ingest_only,
        ).run(
            conversations, questions
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
