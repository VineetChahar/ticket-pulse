"""Builds the daily triage report, grounded in a DaySummary's actual counts
and anomaly flags -- the prompt hands the model those numbers directly and
tells it not to invent others, rather than letting it free-generate.

Falls back to a template (no LLM call) if Ollama is unreachable, so the
pipeline still produces a report -- same pattern repo-health-radar uses for
its synthesis agent. `source` on the return value says which path ran, so
eval and the report itself can be honest about which reports are template
fallbacks.
"""

from __future__ import annotations

import logging
from typing import Literal

import requests

from src.common import config
from src.common.schemas import DaySummary

log = logging.getLogger(__name__)


def _template_report(summary: DaySummary) -> str:
    lines = [f"# Daily triage report -- day {summary.day_index} ({summary.date.date()})", ""]
    lines.append("## Ticket volume by category")
    for c in summary.counts:
        lines.append(f"- {c.category}: {c.count}")
    lines.append("")
    if summary.anomalies:
        lines.append("## Anomalies detected")
        for a in summary.anomalies:
            lines.append(f"- [{a.method}] {a.description} (score={a.score:.2f}, threshold={a.threshold})")
    else:
        lines.append("## Anomalies detected\n\nNone.")
    return "\n".join(lines)


def _build_prompt(summary: DaySummary) -> str:
    counts_block = "\n".join(f"- {c.category}: {c.count}" for c in summary.counts) or "(no tickets)"
    if summary.anomalies:
        anomalies_block = "\n".join(
            f"- {a.description} (method={a.method}, score={a.score:.2f}, threshold={a.threshold})"
            for a in summary.anomalies
        )
    else:
        anomalies_block = "(none detected)"

    return f"""You are writing a short daily support-ticket triage report for day \
{summary.day_index} ({summary.date.date()}).

Ticket counts per category today (these are the ONLY counts that exist --
do not invent, round, or restate them differently):
{counts_block}

Anomalies flagged by the statistical detector today (these are the ONLY
anomalies -- do not claim an anomaly exists for a category not listed here,
and do not invent a cause; the detector does not know causes, only that
volume deviated from baseline):
{anomalies_block}

Write a 3-5 sentence plain-English summary a support team lead could read
in 10 seconds: which categories are busiest, whether anything looks
anomalous today, and what (if anything) deserves attention. Cite the exact
numbers above. Do not mention categories with zero tickets. Do not add any
counts or categories not listed above."""


def generate_report(summary: DaySummary) -> tuple[str, Literal["ollama", "template_fallback"]]:
    prompt = _build_prompt(summary)
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        narrative = resp.json()["response"].strip()
        if not narrative:
            raise ValueError("empty response from ollama")
        markdown = f"# Daily triage report -- day {summary.day_index} ({summary.date.date()})\n\n{narrative}"
        return markdown, "ollama"
    except Exception:
        log.warning("ollama unavailable, falling back to template report", exc_info=True)
        return _template_report(summary), "template_fallback"
