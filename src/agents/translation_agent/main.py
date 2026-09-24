"""Translation agent worker: consumes raw tickets, fills in translated_text
and back_translation_score, forwards to the classification queue.

Boundary messages are forwarded immediately, before any ticket-processing
work -- a stage must never sit on a day's boundary marker.
"""

from __future__ import annotations

import logging
import sys

from src.common import config
from src.common.broker import Broker
from src.common.schemas import TicketMessage
from src.agents.translation_agent.translator import translate_with_consistency_check

logging.basicConfig(level=logging.INFO, format="[translation_agent] %(message)s")
log = logging.getLogger(__name__)


def process_ticket(msg: TicketMessage) -> TicketMessage:
    """Pure per-ticket transform, exercised directly by tests (mocking
    translate_with_consistency_check) without needing a broker or Redis."""
    try:
        result = translate_with_consistency_check(msg.text or "", msg.locale or config.CANONICAL_LANG)
        msg.translated_text = result.translated_text
        msg.back_translation_score = result.back_translation_score
    except Exception:
        log.exception("translation failed for ticket %s, passing through untranslated", msg.ticket_id)
        msg.translated_text = msg.text
        msg.back_translation_score = None
    return msg


def run(broker: Broker) -> None:
    log.info("started, waiting on %s", config.Q_RAW_TICKETS)
    while True:
        msg = broker.blocking_pop(config.Q_RAW_TICKETS, TicketMessage, timeout=0)
        if msg is None:
            continue

        if msg.kind == "boundary":
            broker.push(config.Q_TRANSLATED, msg)
            log.info("day %d boundary forwarded%s", msg.day_index, " (final)" if msg.is_final else "")
            continue

        broker.push(config.Q_TRANSLATED, process_ticket(msg))


def main() -> None:
    broker = Broker(config.REDIS_URL)
    try:
        run(broker)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
