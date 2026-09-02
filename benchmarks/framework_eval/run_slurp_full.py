#!/usr/bin/env python3
"""Run the exhaustive SLURP cockpit-memory experiment with safe sharding."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
# Keep direct executable usage (``./run_slurp_full.py``) equivalent to
# ``python -m benchmarks.framework_eval.run_slurp_full``.  Without this,
# imports used by the score stage resolve relative to this script directory
# and cannot see the repository-level ``benchmarks`` package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "benchmarks/data/SLURP/normalized-full-1759-v1"
RUN_ROOT = (
    ROOT
    / "benchmarks/framework_eval_runs/slurp-memory-full-1759-v1"
    / "tencentdb-cockpit-l0l1-v2"
)
ISOLATION = RUN_ROOT / "isolation.json"
USER_KEY_FILE = RUN_ROOT / ".user-key"
BASE_URL = "http://127.0.0.1:8420"
EXPECTED_QUESTIONS = 1759
SMOKE_CONVERSATION = "slurp-memory-full-122"

ARM_ENV = {
    "l0-window": {
        "TDAI_EVAL_MEMORY_LAYERS": "L0",
        "TDAI_EVAL_L0_MIN_RESULTS": "8",
        "TDAI_EVAL_L0_MIN_FRACTION": "0",
        "TDAI_EVAL_L0_FIRST": "true",
    },
    "l1-only": {
        "TDAI_EVAL_MEMORY_LAYERS": "L1",
        "TDAI_EVAL_L0_MIN_RESULTS": "0",
        "TDAI_EVAL_L0_MIN_FRACTION": "0",
        "TDAI_EVAL_L0_FIRST": "false",
    },
    "hybrid": {
        "TDAI_EVAL_MEMORY_LAYERS": "L0,L1",
        "TDAI_EVAL_L0_MIN_RESULTS": "0",
        "TDAI_EVAL_L0_MIN_FRACTION": "0.5",
        "TDAI_EVAL_L0_FIRST": "true",
    },
}


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        # MemoryCore's v2/v3 data router requires a syntactically non-empty
        # Bearer even when the optional shared-secret comparison is disabled.
        "TDAI_API_KEY": "local-benchmark-bearer",
        "TDAI_HTTP_API_VERSION": "v3",
        "TDAI_EVAL_ISOLATION_MAP": str(ISOLATION),
        "TDAI_EVAL_USER_KEY_FILE": str(USER_KEY_FILE),
        "TDAI_EVAL_PERSPECTIVE_MODE": "single",
        "TDAI_EVAL_CANDIDATE_MULTIPLIER": "5",
        "TDAI_EVAL_L0_WINDOW_BEFORE": "1",
        "TDAI_EVAL_L0_WINDOW_AFTER": "12",
        "TDAI_EVAL_L0_EXPLICIT_DATE_BOOST": "true",
        "TDAI_EVAL_READY_SETTLE_SECONDS": "10",
    })
    return env


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _conversation_weights() -> list[tuple[str, int]]:
    return [
        (str(row["sample_id"]), len(row["sessions"]))
        for row in _rows(DATA_ROOT / "conversations.jsonl")
    ]


def _balanced_shards(
    items: list[tuple[str, int]], count: int,
) -> list[list[str]]:
    bins: list[tuple[int, list[str]]] = [(0, []) for _ in range(max(1, count))]
    for conversation_id, weight in sorted(items, key=lambda item: (-item[1], item[0])):
        index = min(range(len(bins)), key=lambda value: (bins[value][0], value))
        total, values = bins[index]
        values.append(conversation_id)
        bins[index] = (total + weight, values)
    return [values for _, values in bins if values]


def _cli_retrieval(
    conversations: list[str], output: Path, *, ingest_only: bool = False,
    skip_ingest: bool = False, ready_timeout: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m", "benchmarks.framework_eval.cli", "retrieval",
        "--adapter", "tencentdb",
        "--dataset", "slurp_memory",
        "--dataset-root", str(DATA_ROOT),
        "--top-k", "8",
        "--max-context-chars", "3000",
        "--base-url", BASE_URL,
        "--output", str(output),
    ]
    for conversation_id in conversations:
        command.extend(["--conversation", conversation_id])
    if ingest_only:
        command.append("--ingest-only")
    if skip_ingest:
        command.extend(["--skip-ingest", "--resume"])
    if ready_timeout is not None:
        command.extend(["--ready-timeout", str(ready_timeout)])
    return command


def _run_batch(
    label: str,
    specs: list[tuple[str, list[str], dict[str, str], Path, Path | None]],
    *, max_parallel: int,
) -> None:
    pending = list(specs)
    failures: list[str] = []
    while pending:
        batch = pending[:max(1, max_parallel)]
        pending = pending[max(1, max_parallel):]
        active = []
        for name, command, env, log_path, marker in batch:
            if marker is not None and marker.exists():
                print(f"[{label}] skip completed {name}", flush=True)
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append((name, process, handle, marker))
        started = time.monotonic()
        while active:
            remaining = []
            for name, process, handle, marker in active:
                code = process.poll()
                if code is None:
                    remaining.append((name, process, handle, marker))
                    continue
                handle.close()
                if code:
                    failures.append(name)
                    print(f"[{label}] FAILED {name} exit={code}", flush=True)
                else:
                    if marker is not None:
                        marker.write_text(
                            json.dumps({"completed": True, "name": name}) + "\n",
                            encoding="utf-8",
                        )
                    print(f"[{label}] completed {name}", flush=True)
            active = remaining
            if active:
                elapsed = int(time.monotonic() - started)
                print(
                    f"[{label}] active={len(active)} elapsed={elapsed}s",
                    flush=True,
                )
                time.sleep(10)
    if failures:
        raise RuntimeError(f"{label} failed shards: {', '.join(failures)}")


def _smoke() -> None:
    output = RUN_ROOT / "smoke/retrieval.jsonl"
    if output.exists() and len(_rows(output)) == 1:
        print("[smoke] already complete", flush=True)
        return
    env = _base_env()
    env.update(ARM_ENV["hybrid"])
    command = _cli_retrieval(
        [SMOKE_CONVERSATION], output, ready_timeout=900,
    )
    _run_batch(
        "smoke",
        [("one-session", command, env, output.with_suffix(".log"), None)],
        max_parallel=1,
    )
    rows = _rows(output)
    if len(rows) != 1:
        raise RuntimeError(f"smoke expected one row, found {len(rows)}")
    print(
        "[smoke] pass "
        f"hits={rows[0]['metrics']['hit_count']} "
        f"evidence_recall={rows[0]['metrics']['evidence_recall']}",
        flush=True,
    )


def _ingest(workers: int) -> None:
    items = _conversation_weights()
    smoke_done = (
        (RUN_ROOT / "smoke/retrieval.jsonl").exists()
        and len(_rows(RUN_ROOT / "smoke/retrieval.jsonl")) == 1
    )
    if smoke_done:
        items = [item for item in items if item[0] != SMOKE_CONVERSATION]
    shards = _balanced_shards(items, workers)
    env = _base_env()
    env.update(ARM_ENV["hybrid"])
    specs = []
    for index, conversations in enumerate(shards):
        shard_dir = RUN_ROOT / "ingest"
        specs.append((
            f"shard-{index:02d}",
            _cli_retrieval(
                conversations,
                shard_dir / f"shard-{index:02d}.jsonl",
                ingest_only=True,
            ),
            env,
            shard_dir / f"shard-{index:02d}.log",
            shard_dir / f"shard-{index:02d}.done.json",
        ))
    _run_batch("ingest", specs, max_parallel=workers)


def _merge_retrieval(arm: str, shard_count: int) -> Path:
    arm_dir = RUN_ROOT / arm
    rows: list[dict] = []
    for index in range(shard_count):
        path = arm_dir / "shards" / f"shard-{index:02d}.jsonl"
        rows.extend(_rows(path))
    unique = {str(row["question"]["question_id"]): row for row in rows}
    if len(unique) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"{arm} expected {EXPECTED_QUESTIONS} unique questions, "
            f"found {len(unique)}"
        )
    output = arm_dir / "retrieval.jsonl"
    output.write_text(
        "".join(
            json.dumps(unique[key], ensure_ascii=False) + "\n"
            for key in sorted(unique)
        ),
        encoding="utf-8",
    )
    print(f"[retrieve] merged {arm}: {len(unique)} rows", flush=True)
    return output


def _retrieve(arms: list[str], workers: int, total_parallel: int) -> None:
    shards = _balanced_shards(_conversation_weights(), workers)
    specs = []
    for arm in arms:
        env = _base_env()
        env.update(ARM_ENV[arm])
        arm_dir = RUN_ROOT / arm / "shards"
        for index, conversations in enumerate(shards):
            output = arm_dir / f"shard-{index:02d}.jsonl"
            specs.append((
                f"{arm}-shard-{index:02d}",
                _cli_retrieval(
                    conversations,
                    output,
                    skip_ingest=True,
                    ready_timeout=21600,
                ),
                env,
                arm_dir / f"shard-{index:02d}.log",
                None,
            ))
    _run_batch("retrieve", specs, max_parallel=total_parallel)
    for arm in arms:
        _merge_retrieval(arm, len(shards))


def _load_answer_env() -> dict[str, str]:
    deploy = ROOT / "third_party/tencentdb-agent-memory-v2/deploy/global-images"
    raw = subprocess.check_output(
        ["bash", "-lc", "set -a; source ./.env >/dev/null 2>&1; env -0"],
        cwd=deploy,
    )
    shell_env = {}
    for item in raw.decode(errors="replace").split("\0"):
        if "=" in item:
            key, value = item.split("=", 1)
            shell_env[key] = value
    required = ("MEMORY_LLM_BASE_URL", "MEMORY_LLM_API_KEY", "MEMORY_LLM_MODEL")
    missing = [name for name in required if not shell_env.get(name)]
    if missing:
        raise RuntimeError(f"missing answer settings: {', '.join(missing)}")
    env = dict(os.environ)
    env.update({
        "MEMEVAL_ANSWER_BASE_URL": shell_env["MEMORY_LLM_BASE_URL"],
        "MEMEVAL_ANSWER_API_KEY": shell_env["MEMORY_LLM_API_KEY"],
        "MEMEVAL_ANSWER_MODEL": shell_env["MEMORY_LLM_MODEL"],
        "MEMEVAL_ANSWER_MAX_TOKENS": "64",
        "MEMEVAL_ANSWER_TIMEOUT": "300",
    })
    return env


def _answer(arms: list[str], concurrency: int) -> None:
    env = _load_answer_env()
    for arm in arms:
        arm_dir = RUN_ROOT / arm
        command = [
            sys.executable,
            "-m", "benchmarks.framework_eval.cli", "answer",
            "--input", str(arm_dir / "retrieval.jsonl"),
            "--output", str(arm_dir / "predictions.llm.jsonl"),
            "--concurrency", str(concurrency),
            "--resume",
        ]
        _run_batch(
            "answer",
            [(arm, command, env, arm_dir / "answer.log", None)],
            max_parallel=1,
        )


def _answer_baselines(concurrency: int) -> None:
    """Use otherwise-idle quota to make the capped baselines fully comparable."""
    env = _load_answer_env()
    baseline_root = RUN_ROOT.parent / "baselines"
    for name in ("full-context-c3000", "bm25-k16-c3000", "bm25-k8-c3000"):
        run_dir = baseline_root / name
        predictions = run_dir / "predictions.llm.jsonl"
        command = [
            sys.executable,
            "-m", "benchmarks.framework_eval.cli", "answer",
            "--input", str(run_dir / "retrieval.jsonl"),
            "--output", str(predictions),
            "--concurrency", str(concurrency),
            "--resume",
        ]
        _run_batch(
            "baseline-answer",
            [(name, command, env, run_dir / "answer.log", None)],
            max_parallel=1,
        )
        subprocess.run([
            sys.executable,
            "-m", "benchmarks.framework_eval.cli", "score",
            "--dataset", "slurp_memory",
            "--dataset-root", str(DATA_ROOT),
            "--input", str(predictions),
            "--output-dir", str(run_dir / "score-llm"),
            "--metrics", "exact", "contains",
        ], cwd=ROOT, check=True)
        print(f"[baseline-score] completed {name}", flush=True)


def _score(arms: list[str]) -> None:
    from benchmarks.framework_eval.slurp_extract_answer import build_predictions

    for arm in arms:
        arm_dir = RUN_ROOT / arm
        retrieval = arm_dir / "retrieval.jsonl"
        deterministic = arm_dir / "predictions.deterministic.jsonl"
        build_predictions(retrieval, deterministic)
        for label, predictions in (
            ("deterministic", deterministic),
            ("llm", arm_dir / "predictions.llm.jsonl"),
        ):
            if not predictions.exists():
                continue
            command = [
                sys.executable,
                "-m", "benchmarks.framework_eval.cli", "score",
                "--dataset", "slurp_memory",
                "--dataset-root", str(DATA_ROOT),
                "--input", str(predictions),
                "--output-dir", str(arm_dir / f"score-{label}"),
                "--metrics", "exact", "contains",
            ]
            subprocess.run(command, cwd=ROOT, check=True)
        print(f"[score] completed {arm}", flush=True)


def _write_run_config(args: argparse.Namespace) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = RUN_ROOT / "run-config.json"
    existing = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists() else {}
    )
    arms = list(dict.fromkeys([
        *[str(value) for value in existing.get("arms", [])],
        *args.arm,
    ]))
    payload = {
        "dataset": "slurp-memory-full-1759-v1",
        "questions": EXPECTED_QUESTIONS,
        "memory_core": BASE_URL,
        "prompt_mode": "cockpit",
        "arms": arms,
        "ingest_workers": args.ingest_workers,
        "retrieval_workers_per_arm": args.retrieval_workers,
        "retrieval_total_parallel": args.retrieval_parallel,
        "answer_concurrency": args.answer_concurrency,
        "top_k": 8,
        "max_context_chars": 3000,
        "judge": "deterministic exact/contains",
        "secrets_persisted_in_result": False,
    }
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "all", "smoke", "ingest", "retrieve", "answer", "score",
            "baseline-answer",
        ),
        default="all",
    )
    parser.add_argument(
        "--arm", action="append", choices=tuple(ARM_ENV),
        default=[],
    )
    parser.add_argument("--ingest-workers", type=int, default=4)
    parser.add_argument("--retrieval-workers", type=int, default=4)
    parser.add_argument("--retrieval-parallel", type=int, default=8)
    parser.add_argument("--answer-concurrency", type=int, default=8)
    args = parser.parse_args()
    args.arm = args.arm or ["hybrid", "l1-only", "l0-window"]
    if not ISOLATION.exists() or not USER_KEY_FILE.exists():
        raise SystemExit("missing isolation.json or .user-key; provision first")
    _write_run_config(args)

    if args.stage in {"all", "smoke"}:
        _smoke()
    if args.stage in {"all", "ingest"}:
        _ingest(args.ingest_workers)
    if args.stage in {"all", "retrieve"}:
        _retrieve(args.arm, args.retrieval_workers, args.retrieval_parallel)
    if args.stage in {"all", "answer"}:
        _answer(args.arm, args.answer_concurrency)
    if args.stage in {"all", "score"}:
        _score(args.arm)
    if args.stage == "baseline-answer":
        _answer_baselines(args.answer_concurrency)
    print("[done] requested stages complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
