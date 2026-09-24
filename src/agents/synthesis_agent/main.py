"""Synthesis agent worker: consumes DaySummary messages, generates the
grounded daily report, and writes it to reports/<run_id>/day_NNN.md plus one
line to reports/<run_id>/summary.jsonl (which eval/run_eval.py reads back)."""

from __future__ import annotations

import json
import logging
import os
import sys

from src.common import config
from src.common.broker import Broker
from src.common.schemas import DaySummary
from src.agents.synthesis_agent.narrative import generate_report

logging.basicConfig(level=logging.INFO, format="[synthesis_agent] %(message)s")
log = logging.getLogger(__name__)


def run(broker: Broker) -> None:
    log.info("started, waiting on %s", config.Q_DAY_SUMMARIES)
    while True:
        summary = broker.blocking_pop(config.Q_DAY_SUMMARIES, DaySummary, timeout=0)
        if summary is None:
            continue

        run_dir = os.path.join(config.REPORTS_DIR, summary.run_id)
        os.makedirs(run_dir, exist_ok=True)

        markdown, source = generate_report(summary)

        report_path = os.path.join(run_dir, f"day_{summary.day_index:03d}.md")
        with open(report_path, "w") as f:
            f.write(markdown)

        summary_line = {
            "day_index": summary.day_index,
            "date": summary.date.isoformat(),
            "counts": {c.category: c.count for c in summary.counts},
            "anomalies": [a.model_dump(mode="json") for a in summary.anomalies],
            "narrative_source": source,
            "is_final": summary.is_final,
        }
        with open(os.path.join(run_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(summary_line) + "\n")

        log.info("day %d report written (%s)%s", summary.day_index, source, " (final)" if summary.is_final else "")


def main() -> None:
    broker = Broker(config.REDIS_URL)
    try:
        run(broker)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
