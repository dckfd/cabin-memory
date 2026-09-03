#!/usr/bin/env bash
set -euo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
dataset=${COCKPIT_DATASET_ROOT:-$release_root/benchmarks/framework_eval/challenges/cockpit_zh_public_mix_500_v7}
run_root=${1:?usage: run-recovery33-rc.sh NEW_OUTPUT_DIRECTORY}
service_url=${TDAI_SERVICE_URL:-http://127.0.0.1:18507}

if [[ -e "$run_root" ]]; then
  printf 'Refusing to overwrite existing run directory: %s\n' "$run_root" >&2
  exit 2
fi
mkdir -p "$run_root/logs"

python3 "$release_root/scripts/release/validate-cockpit-dataset.py" \
  "$dataset" --output "$run_root/dataset-validation.json" \
  2>&1 | tee "$run_root/logs/validate.log"

python3 "$release_root/scripts/release/run-recovery33-retrieval.py" \
  --dataset "$dataset" --service-url "$service_url" \
  --output "$run_root/retrieval.jsonl" \
  2>&1 | tee "$run_root/logs/retrieval.log"

: "${MEMEVAL_ANSWER_BASE_URL:=https://api.deepseek.com}"
: "${MEMEVAL_ANSWER_MODEL:=deepseek-v4-flash}"
: "${MEMEVAL_ANSWER_API_KEY:=${OPENAI_API_KEY:-}}"
export MEMEVAL_ANSWER_BASE_URL MEMEVAL_ANSWER_MODEL MEMEVAL_ANSWER_API_KEY
export MEMEVAL_ANSWER_TEMPORAL_QUERY_MODE=interval_v1
export MEMEVAL_ANSWER_TEMPORAL_DEFAULT_TIMEZONE=Asia/Shanghai
export MEMEVAL_ANSWER_DETERMINISTIC_SLOT_MODE=cockpit_v1
if [[ -z "$MEMEVAL_ANSWER_API_KEY" ]]; then
  printf '%s\n' 'MEMEVAL_ANSWER_API_KEY or OPENAI_API_KEY is required' >&2
  exit 3
fi
python3 -m benchmarks.framework_eval.cli answer \
  --input "$run_root/retrieval.jsonl" \
  --output "$run_root/predictions-flash.jsonl" --concurrency 2 \
  2>&1 | tee "$run_root/logs/answer.log"

python3 "$release_root/scripts/release/run-deepseek-judge.py" \
  --predictions "$run_root/predictions-flash.jsonl" \
  --output-dir "$run_root" --concurrency 2 \
  2>&1 | tee "$run_root/logs/judge.log"

python3 - "$run_root" <<'PY'
import hashlib, json, sys
from datetime import datetime
from pathlib import Path
root = Path(sys.argv[1])
score = json.loads((root / "score-summary.json").read_text())
manifest = {
    "schema_version": 1,
    "created_at": datetime.now().astimezone().isoformat(),
    "memory": "Recovery33 reused read-only; memory_rebuilt=false",
    "answer_model": "deepseek-v4-flash",
    "judge_model": "deepseek-v4-pro",
    "result": {key: score[key] for key in ("correct", "expected_count", "accuracy", "errors")},
}
(root / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(root / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
    for path in files
))
print(json.dumps(manifest, ensure_ascii=False))
PY
