"""End-to-end flow on a small in-memory fixture: raw tickets -> translation
-> classification -> anomaly detection -> synthesis, chained directly
through each stage's process_ticket()/close_day_and_summarize() functions
with the heavy models monkeypatched out. This checks the DATA FLOW is
correct end to end (an injected spike really does produce an anomaly flag
and a report that mentions it); docker-compose's `make demo` is what
proves the pieces genuinely run as separate processes over real Redis.
"""

from __future__ import annotations

import requests

from src.agents.anomaly_agent.main import RunState, close_day_and_summarize
from src.agents.classification_agent import main as classification_main
from src.agents.classification_agent.classifier import ClassificationResult
from src.agents.synthesis_agent.narrative import generate_report
from src.agents.translation_agent import main as translation_main
from src.agents.translation_agent.translator import TranslationResult
from src.common import config
from src.common.schemas import TicketMessage

FIXTURE = [
    # (day, locale, category, text)
    (0, "en", "weather", "what's the weather like today"),
    (0, "en", "iot", "turn on the lights"),
    (1, "en", "weather", "will it rain tomorrow"),
    (1, "en", "iot", "turn off the fan"),
    (2, "en", "weather", "what's the temperature"),
    (2, "en", "iot", "dim the lights"),
    (3, "en", "weather", "is it sunny"),
    (3, "en", "iot", "lock the front door"),
    (4, "en", "weather", "chance of snow"),
    # day 4: injected iot spike
    *[(4, "en", "iot", f"turn on device {i}") for i in range(20)],
]


def fake_translate(text: str, locale: str, **kwargs) -> TranslationResult:
    return TranslationResult(translated_text=text, back_translation_score=None)


def fake_classify(text: str) -> ClassificationResult:
    # a "perfect" classifier keyed off a keyword, standing in for the real
    # zero-shot model so this test is deterministic and model-free
    category = "iot" if any(w in text for w in ("light", "fan", "door", "device")) else "weather"
    return ClassificationResult(category, 0.99, "Smart Home & Devices" if category == "iot" else "Information & Lookup", 0.9)


def test_injected_spike_flows_through_to_a_grounded_report(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CUSUM_BASELINE_DAYS", 4)
    monkeypatch.setattr(translation_main, "translate_with_consistency_check", fake_translate)
    monkeypatch.setattr(classification_main, "classify", fake_classify)

    def boom(*a, **k):
        raise ConnectionError("no ollama in tests")

    monkeypatch.setattr(requests, "post", boom)  # forces the template-fallback narrative path

    state = RunState("e2e-test")
    tickets_by_day: dict[int, list[TicketMessage]] = {}
    for day, locale, category, text in FIXTURE:
        tickets_by_day.setdefault(day, []).append(
            TicketMessage(kind="ticket", run_id="e2e-test", day_index=day, ticket_id=f"{day}-{text}", locale=locale, true_category=category, text=text)
        )

    summaries = []
    for day in sorted(tickets_by_day):
        for msg in tickets_by_day[day]:
            translated = translation_main.process_ticket(msg)
            classified = classification_main.process_ticket(translated)
            state.record_ticket(classified)
        summary = close_day_and_summarize(state, day, is_final=(day == max(tickets_by_day)))
        summaries.append(summary)

    day4 = summaries[4]
    iot_count = next(c.count for c in day4.counts if c.category == "iot")
    assert iot_count == 20  # the injected spike; day 4's only other ticket is weather

    anomaly_categories = {a.category for a in day4.anomalies}
    assert "iot" in anomaly_categories, f"expected iot flagged on day 4, got anomalies: {day4.anomalies}"
    assert "weather" not in anomaly_categories  # weather had no spike

    markdown, source = generate_report(day4)
    assert source == "template_fallback"
    assert "iot" in markdown
    assert "20" in markdown

    # every ticket was persisted for eval, in true/predicted-category form
    with open(state.tickets_path) as f:
        scored_lines = f.read().strip().splitlines()
    assert len(scored_lines) == len(FIXTURE)
