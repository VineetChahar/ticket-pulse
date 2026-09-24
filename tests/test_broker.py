"""Broker push/pop round trip against fakeredis: real queue semantics
(FIFO, blocking timeout, boundary-then-ticket ordering) without a live
Redis server."""

from __future__ import annotations

import fakeredis
import pytest
import redis

from src.common.broker import Broker
from src.common.schemas import TicketMessage


@pytest.fixture
def broker(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: fake))
    return Broker("redis://ignored")


def test_push_then_pop_round_trips(broker):
    msg = TicketMessage(kind="ticket", run_id="r1", day_index=0, ticket_id="t1", text="hi")
    broker.push("q", msg)
    out = broker.blocking_pop("q", TicketMessage, timeout=1)
    assert out is not None
    assert out.ticket_id == "t1"
    assert out.text == "hi"


def test_pop_on_empty_queue_times_out(broker):
    out = broker.blocking_pop("empty-q", TicketMessage, timeout=1)
    assert out is None


def test_fifo_ordering_preserved(broker):
    for i in range(3):
        broker.push("q", TicketMessage(kind="ticket", run_id="r1", day_index=0, ticket_id=f"t{i}"))
    seen = [broker.blocking_pop("q", TicketMessage, timeout=1).ticket_id for _ in range(3)]
    assert seen == ["t0", "t1", "t2"]


def test_queue_depth(broker):
    assert broker.queue_depth("q") == 0
    broker.push("q", TicketMessage(kind="ticket", run_id="r1", day_index=0, ticket_id="t1"))
    assert broker.queue_depth("q") == 1
