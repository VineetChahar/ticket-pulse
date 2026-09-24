"""Classification agent worker: consumes translated tickets, fills in
predicted_category / predicted_category_custom / classify_confidence,
forwards to the anomaly queue."""

from __future__ import annotations

import logging
import sys

from src.common import config
from src.common.broker import Broker
from src.common.schemas import TicketMessage
from src.agents.classification_agent.classifier import classify

logging.basicConfig(level=logging.INFO, format="[classification_agent] %(message)s")
log = logging.getLogger(__name__)


def process_ticket(msg: TicketMessage) -> TicketMessage:
    """Pure per-ticket transform, exercised directly by tests (mocking
    classify) without needing a broker or Redis."""
    try:
        result = classify(msg.translated_text or msg.text or "")
        msg.predicted_category = result.scenario_label
        msg.predicted_category_custom = result.custom_label
        msg.classify_confidence = result.scenario_score
    except Exception:
        log.exception("classification failed for ticket %s", msg.ticket_id)
        msg.predicted_category = "unknown"
        msg.predicted_category_custom = "unknown"
        msg.classify_confidence = 0.0
    return msg


def run(broker: Broker) -> None:
    log.info("started, waiting on %s", config.Q_TRANSLATED)
    while True:
        msg = broker.blocking_pop(config.Q_TRANSLATED, TicketMessage, timeout=0)
        if msg is None:
            continue

        if msg.kind == "boundary":
            broker.push(config.Q_CLASSIFIED, msg)
            log.info("day %d boundary forwarded%s", msg.day_index, " (final)" if msg.is_final else "")
            continue

        broker.push(config.Q_CLASSIFIED, process_ticket(msg))


def main() -> None:
    broker = Broker(config.REDIS_URL)
    try:
        run(broker)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
