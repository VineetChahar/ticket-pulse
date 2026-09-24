"""Environment-driven configuration, read once per process at import time.

No secrets here -- every value has a safe local default so the pipeline runs
with zero configuration against `docker compose up`. See .env.example.
"""

from __future__ import annotations

import os


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")

# facebook/m2m100_418M: public, no HF login required, ~1.9GB. Lighter than
# nllb-200-distilled-600M (the prompt's other suggested option) and its
# ISO 639-1 codes cover every locale sampled from mteb/amazon_massive_scenario
# (see LOCALE_TO_M2M100 below for the handful that don't match verbatim).
# AmazonScience/massive itself ships as a dataset-scripts loader that current
# `datasets` versions refuse to run -- mteb/amazon_massive_scenario is a
# Parquet mirror of the same MASSIVE corpus's scenario-classification split,
# same text/label pairs, no login either.
TRANSLATION_MODEL = os.environ.get("TRANSLATION_MODEL", "facebook/m2m100_418M")
CANONICAL_LANG = "en"

# Fraction of non-English tickets that get the (2x-cost) back-translation
# consistency check, rather than just the one forward translate needed for
# the pipeline itself. The eval spec asks for this metric "across a held-out
# sample," not on every ticket -- see translator.py's
# translate_with_consistency_check docstring.
BACK_TRANSLATION_SAMPLE_RATE = _float("BACK_TRANSLATION_SAMPLE_RATE", 0.20)

# mteb/amazon_massive_scenario's config names ("zh-CN") don't always match
# m2m100's ISO 639-1 codes ("zh") -- this project only samples the locales
# on the left, so an explicit table is simpler and safer than a generic
# BCP-47-to-ISO-639-1 conversion.
LOCALE_TO_M2M100 = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "hi": "hi",
    "ar": "ar",
    "zh-CN": "zh",
    "pt": "pt",
}

# Zero-shot NLI classifier, public, no login. Runs twice per ticket: once
# against the dataset's own taxonomy (for eval against real labels) and
# once against a custom support-desk taxonomy the model was never trained
# on (to demonstrate generalization -- see classification_agent/taxonomy.py).
CLASSIFICATION_MODEL = os.environ.get("CLASSIFICATION_MODEL", "facebook/bart-large-mnli")

# Queue names -- one per pipeline stage, plus a day-summary fan-in queue
# that the synthesis agent consumes.
Q_RAW_TICKETS = "tickets:raw"
Q_TRANSLATED = "tickets:translated"
Q_CLASSIFIED = "tickets:classified"
Q_DAY_SUMMARIES = "days:summary"

# Anomaly detector defaults. Tuned against the known injected anomaly
# windows by eval/run_eval.py's threshold sweep -- see README's "Anomaly
# detection" section for the real sweep numbers behind these values.
CUSUM_THRESHOLD_STD = _float("CUSUM_THRESHOLD_STD", 3.5)
CUSUM_DRIFT_STD = _float("CUSUM_DRIFT_STD", 0.5)
CUSUM_BASELINE_DAYS = _int("CUSUM_BASELINE_DAYS", 7)
POISSON_ALPHA = _float("POISSON_ALPHA", 0.01)  # significance level for the rate test

DATA_DIR = os.environ.get("DATA_DIR", "data/synthetic")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
