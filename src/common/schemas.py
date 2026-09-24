"""Message schemas passed between agents over the broker (src/common/broker.py).

Deliberate design choice: ONE evolving envelope (`TicketMessage`) carries a
ticket through all three per-ticket stages (raw -> translated -> classified)
instead of a distinct pydantic model per stage (as repo-health-radar uses
for its per-agent Task/Result pairs). Each stage only ever *adds* optional
fields; nothing is ever removed or reshaped. That trades away the stricter
"a translation-stage message cannot possibly have a category on it yet"
guarantee repo-health-radar's separate models gave, for a pipeline where a
message's shape doesn't change across three hops -- worth it here because,
unlike repo-health-radar's request/response-per-agent-call pattern, this is
a genuine streaming pipeline and re-wrapping the same ticket into three
different classes at each hop would just be ceremony. See LEARNING.md.

`kind="boundary"` messages are a barrier: every agent that reads from a
queue must forward the boundary for the day it just saw before doing any
end-of-day work of its own, so every downstream stage learns "day N is
complete" only after every ticket belonging to day N has actually cleared
that stage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TicketMessage(BaseModel):
    kind: Literal["ticket", "boundary"] = "ticket"
    run_id: str
    day_index: int
    is_final: bool = False  # boundary only: this is the last day of the run

    # --- ticket-only fields below; unset on boundary messages ---
    ticket_id: str | None = None
    timestamp: datetime | None = None
    locale: str | None = None
    true_category: str | None = None  # MASSIVE "scenario" label; kept for eval only
    injected_anomaly: bool = False

    text: str | None = None  # original-language text

    # filled in by translation_agent
    translated_text: str | None = None
    back_translation_score: float | None = None  # None if already canonical-language

    # filled in by classification_agent
    predicted_category: str | None = None  # zero-shot vs. the dataset's own taxonomy
    predicted_category_custom: str | None = None  # zero-shot vs. our own taxonomy
    classify_confidence: float | None = None


class CategoryDayCount(BaseModel):
    category: str
    count: int


class AnomalyFlag(BaseModel):
    category: str
    day_index: int
    method: Literal["cusum", "poisson"]
    baseline_rate: float
    observed_count: int
    score: float
    threshold: float
    description: str


class DaySummary(BaseModel):
    run_id: str
    day_index: int
    date: datetime
    is_final: bool = False
    counts: list[CategoryDayCount] = Field(default_factory=list)
    anomalies: list[AnomalyFlag] = Field(default_factory=list)
