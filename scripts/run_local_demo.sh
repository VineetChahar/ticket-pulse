#!/usr/bin/env bash
# Runs the full pipeline as local OS processes (not docker) against the
# ticket-pulse-redis container started separately -- used when RAM is tight
# and a full docker-compose stack (own image, own VM overhead) would add to
# that pressure. Same code, same Redis protocol, same multi-process
# architecture; just not containerized for this particular run.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${1:?usage: run_local_demo.sh <run_id> <data_file> [day_seconds]}"
DATA_FILE="${2:?usage: run_local_demo.sh <run_id> <data_file> [day_seconds]}"
DAY_SECONDS="${3:-0.1}"

export REDIS_URL="redis://localhost:6390/0"
mkdir -p logs
PIDS=()

cleanup() {
  echo "--- stopping workers ---"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

.venv/bin/python -m src.agents.translation_agent.main   > "logs/${RUN_ID}_translation.log"   2>&1 & PIDS+=($!)
.venv/bin/python -m src.agents.classification_agent.main > "logs/${RUN_ID}_classification.log" 2>&1 & PIDS+=($!)
.venv/bin/python -m src.agents.anomaly_agent.main        > "logs/${RUN_ID}_anomaly.log"        2>&1 & PIDS+=($!)
.venv/bin/python -m src.agents.synthesis_agent.main      > "logs/${RUN_ID}_synthesis.log"      2>&1 & PIDS+=($!)

echo "workers started: ${PIDS[*]}"
sleep 2

.venv/bin/python -m src.agents.ingestion_agent.main --data "$DATA_FILE" --run-id "$RUN_ID" --day-seconds "$DAY_SECONDS" \
  > "logs/${RUN_ID}_ingestion.log" 2>&1

echo "--- ingestion done, waiting for pipeline to drain ---"
SUMMARY="reports/${RUN_ID}/summary.jsonl"
for i in $(seq 1 720); do
  if [ -f "$SUMMARY" ] && grep -q '"is_final": true' "$SUMMARY"; then
    echo "--- final day summary written after ${i}0s ---"
    break
  fi
  sleep 10
done

if ! grep -q '"is_final": true' "$SUMMARY" 2>/dev/null; then
  echo "--- TIMED OUT waiting for final summary ---"
  echo "--- last 30 lines of each log ---"
  tail -n 30 logs/${RUN_ID}_*.log
  exit 1
fi

echo "--- run complete: ${RUN_ID} ---"
wc -l "reports/${RUN_ID}/tickets_scored.jsonl"
cat "reports/${RUN_ID}/summary.jsonl" | python3 -c "import sys,json; print(sum(1 for _ in sys.stdin))" || true
