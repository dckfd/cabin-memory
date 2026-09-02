#!/usr/bin/env bash
set -euo pipefail

: "${TDAI_AUDIT_DIR:?set TDAI_AUDIT_DIR}"
: "${TDAI_CONTAINER_NAME:?set TDAI_CONTAINER_NAME}"
: "${TDAI_BASE_URL:?set TDAI_BASE_URL}"
: "${TDAI_NOOP_MANIFEST:?set TDAI_NOOP_MANIFEST}"

audit_dir=$TDAI_AUDIT_DIR
container_name=$TDAI_CONTAINER_NAME
base_url=${TDAI_BASE_URL%/}
manifest=$TDAI_NOOP_MANIFEST
settle_seconds=${TDAI_NOOP_SETTLE_SECONDS:-190}
poll_seconds=${TDAI_NOOP_POLL_SECONDS:-5}
prefix=${TDAI_NOOP_AUDIT_PREFIX:-noop}
auth_header=${TDAI_PIPELINE_AUTH_HEADER:-Authorization: Bearer pipeline-recovery-placeholder}
service_header=${TDAI_PIPELINE_SERVICE_HEADER:-x-tdai-service-id: default}

test -d "$audit_dir"
test -f "$manifest"
test "$settle_seconds" -ge 1
test "$poll_seconds" -ge 1
expected_tasks=$(jq -er '.expected_task_count | select(type == "number" and . > 0 and floor == .)' "$manifest")

pipeline_status() {
  curl -fsS -X POST -H "$auth_header" -H "$service_header" \
    -H 'Content-Type: application/json' --data '{}' "$base_url/v2/pipeline/status"
}

pipeline_health() {
  curl -fsS "$base_url/health"
}

assert_timer_delivery_healthy() {
  jq -e '
    .status == "ok"
    and .services.timerScanner != null
    and .services.timerScanner.timerRearmFailures == 0
    and .services.timerScanner.pendingTimerRearms == 0
  ' >/dev/null
}

model_run_count() {
  docker logs "$container_name" 2>&1 \
    | awk '/\[standalone-runner\] run\(\) start:/ { count += 1 } END { print count + 0 }'
}

hash_persisted_tree() {
  docker exec "$container_name" node --no-warnings -e '
    const { createHash } = require("node:crypto");
    const { readFileSync, readdirSync, statSync } = require("node:fs");
    const { join, relative } = require("node:path");
    const root = "/data/tdai-memory";
    const ignored = /(?:^|\/)(?:[^/]+\.(?:db-(?:shm|wal))|\.metadata\/staging(?:\/|$))/u;
    const files = [];
    const walk = (dir) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const path = join(dir, entry.name);
        if (entry.isDirectory()) walk(path);
        else if (entry.isFile()) files.push(path);
      }
    };
    walk(root);
    for (const path of files.sort()) {
      const name = relative(root, path);
      if (ignored.test(name)) continue;
      const bytes = readFileSync(path);
      console.log(`${createHash("sha256").update(bytes).digest("hex")} ${bytes.length} ${name}`);
    }
  '
}

before_status=$(pipeline_status)
jq . <<<"$before_status" > "$audit_dir/${prefix}-before-status.json"
before_health=$(pipeline_health)
jq . <<<"$before_health" > "$audit_dir/${prefix}-before-health.json"
assert_timer_delivery_healthy <<<"$before_health"
jq -e '
  .code == 0
  and .data.l1.idle == true and .data.l2.idle == true and .data.l3.idle == true
  and .data.worker.tasksFailed == 0 and .data.worker.tasksRetried == 0
  and .data.worker.tasksDeadLettered == 0 and .data.worker.deadLetterCount == 0
' <<<"$before_status" >/dev/null
before_consumed=$(jq -er '.data.worker.tasksConsumed' <<<"$before_status")
before_completed=$(jq -er '.data.worker.tasksCompleted' <<<"$before_status")
before_model_runs=$(model_run_count)
hash_persisted_tree > "$audit_dir/${prefix}-before-tree.log"

curl -fsS -X POST -H "$auth_header" -H "$service_header" \
  -H 'Content-Type: application/json' --data-binary "@$manifest" \
  "$base_url/v2/pipeline/recover" | jq . > "$audit_dir/${prefix}-enqueue-response.json"
jq -e --argjson expected "$expected_tasks" '
  .code == 0 and .data.ok == true and .data.enqueued_count == $expected
' "$audit_dir/${prefix}-enqueue-response.json" >/dev/null

expected_consumed=$((before_consumed + expected_tasks))
expected_completed=$((before_completed + expected_tasks))
: > "$audit_dir/${prefix}-completion-history.jsonl"
for _attempt in $(seq 1 120); do
  status=$(pipeline_status)
  health=$(pipeline_health)
  assert_timer_delivery_healthy <<<"$health"
  jq -c . <<<"$status" >> "$audit_dir/${prefix}-completion-history.jsonl"
  jq -e '
    .data.worker.tasksFailed == 0 and .data.worker.tasksRetried == 0
    and .data.worker.tasksDeadLettered == 0 and .data.worker.deadLetterCount == 0
  ' <<<"$status" >/dev/null
  if jq -e --argjson consumed "$expected_consumed" --argjson completed "$expected_completed" '
    .data.worker.tasksConsumed == $consumed
    and .data.worker.tasksCompleted == $completed
    and .data.l1.idle == true and .data.l2.idle == true and .data.l3.idle == true
  ' <<<"$status" >/dev/null; then
    jq . <<<"$status" > "$audit_dir/${prefix}-completion-status.json"
    break
  fi
  test "$(docker inspect -f '{{.State.Running}}' "$container_name")" = true
  sleep 1
done
test -s "$audit_dir/${prefix}-completion-status.json"

# A no-op L1 used to arm a delayed L2 timer and falsely pass an immediate
# check. Observe beyond that configured delay and require absolute quiescence.
: > "$audit_dir/${prefix}-settle-history.jsonl"
settle_elapsed=0
while test "$settle_elapsed" -lt "$settle_seconds"; do
  remaining=$((settle_seconds - settle_elapsed))
  current_sleep=$poll_seconds
  if test "$current_sleep" -gt "$remaining"; then current_sleep=$remaining; fi
  sleep "$current_sleep"
  settle_elapsed=$((settle_elapsed + current_sleep))
  status=$(pipeline_status)
  health=$(pipeline_health)
  jq -c --argjson settle_elapsed "$settle_elapsed" \
    '. + {gate_observation_elapsed_seconds: $settle_elapsed}' <<<"$status" \
    >> "$audit_dir/${prefix}-settle-history.jsonl"
  jq -e --argjson consumed "$expected_consumed" --argjson completed "$expected_completed" '
    .data.worker.tasksConsumed == $consumed
    and .data.worker.tasksCompleted == $completed
    and .data.worker.tasksFailed == 0 and .data.worker.tasksRetried == 0
    and .data.worker.tasksDeadLettered == 0 and .data.worker.deadLetterCount == 0
    and .data.l1.idle == true and .data.l2.idle == true and .data.l3.idle == true
  ' <<<"$status" >/dev/null
  assert_timer_delivery_healthy <<<"$health"
  test "$(model_run_count)" -eq "$before_model_runs"
  test "$(docker inspect -f '{{.State.Running}}' "$container_name")" = true
done

final_status=$(pipeline_status)
jq . <<<"$final_status" > "$audit_dir/${prefix}-final-status.json"
final_health=$(pipeline_health)
jq . <<<"$final_health" > "$audit_dir/${prefix}-final-health.json"
assert_timer_delivery_healthy <<<"$final_health"
after_model_runs=$(model_run_count)
hash_persisted_tree > "$audit_dir/${prefix}-after-tree.log"
cmp "$audit_dir/${prefix}-before-tree.log" "$audit_dir/${prefix}-after-tree.log" \
  > "$audit_dir/${prefix}-tree-diff.log"
model_delta=$((after_model_runs - before_model_runs))
test "$model_delta" -eq 0

{
  echo "expected_tasks=$expected_tasks"
  echo "tasks_consumed_delta=$(( $(jq -er '.data.worker.tasksConsumed' <<<"$final_status") - before_consumed ))"
  echo "tasks_completed_delta=$(( $(jq -er '.data.worker.tasksCompleted' <<<"$final_status") - before_completed ))"
  echo "model_run_delta=$model_delta"
  echo "persisted_tree_match=true"
  echo "settle_seconds=$settle_seconds"
  echo "pipeline_idle=true"
  echo "worker_errors_zero=true"
  echo "timer_rearm_failures_zero=true"
  echo "pending_timer_rearms_zero=true"
  echo "delayed_cascade_absent=true"
  echo "noop_cascade_gate=true"
} | tee "$audit_dir/${prefix}-gate.log"
