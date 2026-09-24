"""Anomaly agent worker: consumes classified tickets, maintains a running
per-category daily count matrix, and on each day boundary runs both
detectors (src/agents/anomaly_agent/detector.py) over the series-so-far,
emitting a DaySummary for the synthesis agent.

Also appends the day's counts to reports/<run_id>/day_counts.csv so
eval/tune_anomaly_threshold.py can re-sweep detector thresholds offline
against the exact counts a real run produced, without re-running the whole
pipeline.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from src.common import config
from src.common.broker import Broker
from src.common.schemas import AnomalyFlag, CategoryDayCount, DaySummary, TicketMessage
from src.agents.anomaly_agent.detector import cusum_detect, poisson_rate_detect

logging.basicConfig(level=logging.INFO, format="[anomaly_agent] %(message)s")
log = logging.getLogger(__name__)


class RunState:
    """One instance per run_id -- a long-lived process may see several runs
    in sequence (e.g. repeated `make demo`), each with its own count matrix."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.today_counts: dict[str, int] = defaultdict(int)
        self.history: dict[str, list[int]] = defaultdict(list)  # category -> counts per day, 0-filled
        self.day0_date = datetime.now(tz=timezone.utc)
        self.csv_path = os.path.join(config.REPORTS_DIR, run_id, "day_counts.csv")
        self.tickets_path = os.path.join(config.REPORTS_DIR, run_id, "tickets_scored.jsonl")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["day_index", "category", "count"])
        open(self.tickets_path, "w").close()  # truncate; per-ticket eval records appended as they arrive

    def record_ticket(self, msg: TicketMessage) -> None:
        category = msg.predicted_category or "unknown"
        self.today_counts[category] += 1
        with open(self.tickets_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ticket_id": msg.ticket_id,
                        "day_index": msg.day_index,
                        "locale": msg.locale,
                        "true_category": msg.true_category,
                        "predicted_category": msg.predicted_category,
                        "predicted_category_custom": msg.predicted_category_custom,
                        "classify_confidence": msg.classify_confidence,
                        "back_translation_score": msg.back_translation_score,
                        "injected_anomaly": msg.injected_anomaly,
                    }
                )
                + "\n"
            )

    def close_day(self, day_index: int) -> dict[str, int]:
        closed = dict(self.today_counts)
        categories = set(self.history) | set(closed)
        for category in categories:
            series = self.history[category]
            while len(series) < day_index:
                series.append(0)
            series.append(closed.get(category, 0))

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for category, count in sorted(closed.items()):
                w.writerow([day_index, category, count])

        self.today_counts = defaultdict(int)
        return closed


def detect_today(history: dict[str, list[int]], day_index: int) -> list[AnomalyFlag]:
    flags: list[AnomalyFlag] = []
    for category, series in history.items():
        arr = np.array(series, dtype=float)

        for cp in cusum_detect(arr, config.CUSUM_THRESHOLD_STD, config.CUSUM_DRIFT_STD, config.CUSUM_BASELINE_DAYS):
            if cp.index == day_index:
                flags.append(
                    AnomalyFlag(
                        category=category,
                        day_index=day_index,
                        method="cusum",
                        baseline_rate=cp.baseline_value,
                        observed_count=int(cp.observed_value),
                        score=cp.score,
                        threshold=cp.threshold,
                        description=(
                            f"{category}: CUSUM flagged day {day_index} "
                            f"(observed={int(cp.observed_value)}, baseline={cp.baseline_value:.1f})"
                        ),
                    )
                )

        for cp in poisson_rate_detect(arr, config.POISSON_ALPHA, config.CUSUM_BASELINE_DAYS):
            if cp.index == day_index:
                flags.append(
                    AnomalyFlag(
                        category=category,
                        day_index=day_index,
                        method="poisson",
                        baseline_rate=cp.baseline_value,
                        observed_count=int(cp.observed_value),
                        score=cp.score,
                        threshold=cp.threshold,
                        description=(
                            f"{category}: Poisson-rate test flagged day {day_index} "
                            f"(observed={int(cp.observed_value)}, baseline_rate={cp.baseline_value:.1f})"
                        ),
                    )
                )
    return flags


def close_day_and_summarize(state: RunState, day_index: int, is_final: bool) -> DaySummary:
    """Closes the day on `state` and builds the DaySummary the synthesis
    agent consumes. Factored out of run() so tests can drive it directly
    without a broker."""
    closed_counts = state.close_day(day_index)
    flags = detect_today(state.history, day_index)
    return DaySummary(
        run_id=state.run_id,
        day_index=day_index,
        date=state.day0_date + timedelta(days=day_index),
        is_final=is_final,
        counts=[CategoryDayCount(category=c, count=n) for c, n in sorted(closed_counts.items())],
        anomalies=flags,
    )


def get_or_create_state(states: dict[str, RunState], run_id: str) -> RunState:
    """NOT states.setdefault(run_id, RunState(run_id)) -- dict.setdefault
    evaluates its default argument eagerly, every call, whether or not the
    key is already present. RunState.__init__ truncates day_counts.csv and
    tickets_scored.jsonl on construction, so that form silently re-truncated
    both files on EVERY message, leaving only the last day's/ticket's data
    after a run finished. Found by noticing day_counts.csv only had the
    final day's rows after a 2-day smoke test -- see LEARNING.md."""
    if run_id not in states:
        states[run_id] = RunState(run_id)
    return states[run_id]


def run(broker: Broker) -> None:
    log.info("started, waiting on %s", config.Q_CLASSIFIED)
    states: dict[str, RunState] = {}

    while True:
        msg = broker.blocking_pop(config.Q_CLASSIFIED, TicketMessage, timeout=0)
        if msg is None:
            continue

        state = get_or_create_state(states, msg.run_id)

        if msg.kind == "boundary":
            summary = close_day_and_summarize(state, msg.day_index, msg.is_final)
            broker.push(config.Q_DAY_SUMMARIES, summary)
            log.info(
                "day %d closed: %d tickets, %d categories, %d anomalies%s",
                msg.day_index,
                sum(c.count for c in summary.counts),
                len(summary.counts),
                len(summary.anomalies),
                " (final)" if msg.is_final else "",
            )
            if msg.is_final:
                del states[msg.run_id]
            continue

        state.record_ticket(msg)


def main() -> None:
    broker = Broker(config.REDIS_URL)
    try:
        run(broker)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
