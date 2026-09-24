"""RunState + detect_today: per-category day counts accumulate correctly
across ticket/boundary messages, and an injected spike produces an
AnomalyFlag on the right day."""

from __future__ import annotations

from src.common import config
from src.common.schemas import TicketMessage
from src.agents.anomaly_agent.main import RunState, detect_today


def make_ticket(day: int, category: str) -> TicketMessage:
    return TicketMessage(kind="ticket", run_id="r1", day_index=day, ticket_id=f"t{day}-{category}", predicted_category=category)


def test_counts_accumulate_and_close_day_resets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    state = RunState("r1")

    state.record_ticket(make_ticket(0, "weather"))
    state.record_ticket(make_ticket(0, "weather"))
    state.record_ticket(make_ticket(0, "iot"))

    closed = state.close_day(0)
    assert closed == {"weather": 2, "iot": 1}
    assert state.today_counts == {}  # reset after close


def test_history_zero_fills_categories_missing_on_a_given_day(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    state = RunState("r1")

    state.record_ticket(make_ticket(0, "weather"))
    state.close_day(0)
    # day 1: no "weather" tickets at all, but a new category appears
    state.record_ticket(make_ticket(1, "iot"))
    state.close_day(1)

    assert state.history["weather"] == [1, 0]
    assert state.history["iot"] == [0, 1]


def test_detect_today_flags_injected_spike(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CUSUM_BASELINE_DAYS", 5)
    state = RunState("r1")

    for day in range(5):
        for _ in range(4):
            state.record_ticket(make_ticket(day, "iot"))
        state.close_day(day)

    # day 5: a strong spike
    for _ in range(30):
        state.record_ticket(make_ticket(5, "iot"))
    state.close_day(5)

    flags = detect_today(state.history, day_index=5)
    assert any(f.category == "iot" and f.day_index == 5 for f in flags)


def test_tickets_scored_jsonl_is_appended_per_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    state = RunState("r1")
    state.record_ticket(make_ticket(0, "weather"))

    lines = open(state.tickets_path).read().strip().splitlines()
    assert len(lines) == 1
    assert '"predicted_category": "weather"' in lines[0]
