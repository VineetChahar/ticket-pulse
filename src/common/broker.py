"""Thin JSON task-queue wrapper over Redis lists.

Deliberately Redis lists with BRPOP/LPUSH, not PUBLISH/SUBSCRIBE: pub/sub has
no persistence, so a message published before a worker subscribes is dropped
forever -- a real race on container startup. A list holds a message until a
worker is ready to pop it, giving at-least-once delivery with the same
lightweight single-broker-container footprint.
"""

from __future__ import annotations

import time
from typing import TypeVar

import redis
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Docker Desktop for Mac's host<->container port-forwarding proxy silently
# drops idle forwarded connections after ~5s. Polling BRPOP in short chunks
# avoids ever holding one open long enough to hit that, and is harmless
# container-to-container where the proxy isn't in the path at all.
POLL_INTERVAL_SECONDS = 3


class Broker:
    def __init__(self, redis_url: str) -> None:
        self._r = redis.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        return bool(self._r.ping())

    def push(self, channel: str, message: BaseModel) -> None:
        self._r.lpush(channel, message.model_dump_json())

    def blocking_pop(self, channel: str, model: type[T], timeout: int) -> T | None:
        """timeout=0 blocks indefinitely (in short polled chunks); timeout>0
        gives up and returns None after that many seconds."""
        deadline = None if timeout == 0 else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            chunk = POLL_INTERVAL_SECONDS if deadline is None else min(POLL_INTERVAL_SECONDS, deadline - time.monotonic())
            if chunk <= 0:
                break
            item = self._r.brpop([channel], timeout=chunk)
            if item is not None:
                _, raw = item
                return model.model_validate_json(raw)
        return None

    def queue_depth(self, channel: str) -> int:
        return self._r.llen(channel)
