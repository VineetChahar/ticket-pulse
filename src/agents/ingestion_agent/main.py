"""Ingestion agent: the run trigger. Streams a synthetic ticket file
(scripts/generate_tickets.py's output) into the raw-tickets queue at a
simulated day cadence, with a day_boundary barrier after each day so every
downstream stage learns "day N is complete" only once every ticket for day
N has actually cleared it.

Run as a one-shot task (`docker compose run ingestion_agent ...`), not a
long-lived service, mirroring repo-health-radar's orchestrator/worker split.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict

import typer

from src.common import config
from src.common.broker import Broker
from src.common.schemas import TicketMessage

logging.basicConfig(level=logging.INFO, format="[ingestion_agent] %(message)s")
log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data: str = typer.Option(f"{config.DATA_DIR}/tickets.jsonl", help="path to a tickets.jsonl fixture"),
    run_id: str = typer.Option(..., help="unique id for this run; tags every message and the reports/ subdir"),
    day_seconds: float = typer.Option(1.0, help="wall-clock seconds to sleep between simulated days"),
) -> None:
    broker = Broker(config.REDIS_URL)

    tickets_by_day: dict[int, list[dict]] = defaultdict(list)
    with open(data) as f:
        for line in f:
            t = json.loads(line)
            tickets_by_day[t["day_index"]].append(t)

    days = sorted(tickets_by_day)
    total = sum(len(v) for v in tickets_by_day.values())
    log.info("run_id=%s: streaming %d days, %d tickets, %.1fs/day", run_id, len(days), total, day_seconds)

    for i, day in enumerate(days):
        for t in tickets_by_day[day]:
            broker.push(
                config.Q_RAW_TICKETS,
                TicketMessage(
                    kind="ticket",
                    run_id=run_id,
                    day_index=day,
                    ticket_id=t["ticket_id"],
                    timestamp=t["timestamp"],
                    locale=t["locale"],
                    true_category=t["true_category"],
                    injected_anomaly=t.get("injected_anomaly", False),
                    text=t["text"],
                ),
            )

        is_final = i == len(days) - 1
        broker.push(config.Q_RAW_TICKETS, TicketMessage(kind="boundary", run_id=run_id, day_index=day, is_final=is_final))
        log.info("day %d streamed: %d tickets%s", day, len(tickets_by_day[day]), " (final)" if is_final else "")

        if not is_final:
            time.sleep(day_seconds)

    log.info("run_id=%s complete", run_id)


if __name__ == "__main__":
    app()
