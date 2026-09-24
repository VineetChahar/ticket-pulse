"""Input-output contract tests for the translation and classification
stages: given a TicketMessage, each stage's process_ticket() must fill in
exactly its own fields and never touch the others, and must degrade
gracefully (not raise) when the underlying model call fails. The heavy
model calls themselves are monkeypatched out -- these are contract tests,
not model-quality tests (see eval/run_eval.py for the latter, run against
real models)."""

from __future__ import annotations

from src.common.schemas import TicketMessage
from src.agents.translation_agent import main as translation_main
from src.agents.translation_agent.translator import TranslationResult
from src.agents.classification_agent import main as classification_main
from src.agents.classification_agent.classifier import ClassificationResult


def make_ticket(**overrides) -> TicketMessage:
    base = dict(
        kind="ticket",
        run_id="test-run",
        day_index=0,
        ticket_id="t1",
        timestamp="2026-01-01T00:00:00Z",
        locale="es",
        true_category="weather",
        text="hola",
    )
    base.update(overrides)
    return TicketMessage(**base)


def test_translation_fills_only_its_own_fields(monkeypatch):
    monkeypatch.setattr(
        translation_main,
        "translate_with_consistency_check",
        lambda text, locale: TranslationResult("hello", 0.87),
    )
    msg = make_ticket()
    out = translation_main.process_ticket(msg)

    assert out.translated_text == "hello"
    assert out.back_translation_score == 0.87
    assert out.predicted_category is None  # classification hasn't run yet
    assert out.text == "hola"  # original untouched


def test_translation_degrades_gracefully_on_model_failure(monkeypatch):
    def boom(text, locale):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(translation_main, "translate_with_consistency_check", boom)
    msg = make_ticket()
    out = translation_main.process_ticket(msg)

    assert out.translated_text == msg.text  # passthrough fallback
    assert out.back_translation_score is None


def test_classification_fills_only_its_own_fields(monkeypatch):
    monkeypatch.setattr(
        classification_main,
        "classify",
        lambda text: ClassificationResult("weather", 0.9, "Information & Lookup", 0.8),
    )
    msg = make_ticket(translated_text="hello")
    out = classification_main.process_ticket(msg)

    assert out.predicted_category == "weather"
    assert out.predicted_category_custom == "Information & Lookup"
    assert out.classify_confidence == 0.9
    assert out.translated_text == "hello"  # untouched


def test_classification_degrades_gracefully_on_model_failure(monkeypatch):
    def boom(text):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(classification_main, "classify", boom)
    msg = make_ticket(translated_text="hello")
    out = classification_main.process_ticket(msg)

    assert out.predicted_category == "unknown"
    assert out.predicted_category_custom == "unknown"
    assert out.classify_confidence == 0.0
