"""Builds the synthetic ticket stream: real multilingual utterances from
mteb/amazon_massive_scenario (a public, no-login Parquet mirror of MASSIVE's
scenario-classification split), sampled onto a 35-simulated-day timeline
with per-category Poisson-ish baseline volume, and deliberately injected
volume spikes with KNOWN (category, day-range) ground truth so anomaly
detection can be scored against real labels instead of "it looks right".

Also writes a small hard/ambiguous-examples set sampled from pairs of
categories this taxonomy makes genuinely easy to confuse (e.g. datetime vs.
calendar vs. alarm), for the classifier's failure-mode breakdown in eval.

Output (all under data/synthetic/):
  tickets.jsonl               one ticket per line: ticket_id, day_index,
                               timestamp, locale, true_category, text,
                               injected_anomaly
  ground_truth_anomalies.json the injected windows, as ground truth for eval
  hard_examples.jsonl         ~40 deliberately confusable examples
"""

from __future__ import annotations

import json
import os
import random
from datetime import UTC, datetime, timedelta

from datasets import load_dataset

N_DAYS = 35
LOCALES = ["en", "es", "fr", "de", "hi", "ar", "zh-CN", "pt"]
CANONICAL_LOCALE = "en"

# Per-category baseline daily volume (Poisson lambda). Deliberately uneven,
# like real ticket traffic: some categories are just chattier than others.
# Kept modest (~46/day, ~1600 tickets over 35 days) rather than the ~72/day
# an earlier pass used -- CPU-only zero-shot classification (2 taxonomies)
# and translation (2 model calls for a sampled subset, see
# config.BACK_TRANSLATION_SAMPLE_RATE) each cost real wall-clock per ticket,
# and this project runs on a laptop, not a GPU box. Still comfortably enough
# volume for every injected window to produce a real, visible spike.
BASELINE_LAMBDA = {
    "alarm": 2, "audio": 2, "calendar": 4, "cooking": 1, "datetime": 3,
    "email": 4, "general": 4, "iot": 2, "lists": 2, "music": 3, "news": 2,
    "play": 3, "qa": 4, "recommendation": 2, "social": 2, "takeaway": 1,
    "transport": 2, "weather": 3,
}

# Ground truth: known injected anomaly windows (day range inclusive).
# Chosen to be genuinely varied in difficulty: a long strong spike, a short
# strong spike, and a short WEAK spike that a well-tuned detector should
# just barely catch (or not) -- see eval's threshold sweep for how each
# one performs at different thresholds.
ANOMALY_WINDOWS = [
    {"category": "iot", "start_day": 10, "end_day": 13, "multiplier": 5.0},
    {"category": "weather", "start_day": 22, "end_day": 23, "multiplier": 6.0},
    {"category": "transport", "start_day": 29, "end_day": 30, "multiplier": 2.5},
]

# Category pairs sampled for the hard-examples set: scenarios this
# project's own taxonomy design (see classification_agent/taxonomy.py)
# already treats as close, or that are just semantically adjacent in
# MASSIVE's own labeling (e.g. "play" vs "music" vs "audio" all cover
# "play me something").
CONFUSABLE_GROUPS = [
    ("datetime", "calendar", "alarm"),
    ("music", "audio", "play"),
    ("qa", "general", "news"),
    ("recommendation", "cooking", "takeaway"),
]

SEED = 20260101


def load_pools() -> dict[str, dict[str, list[str]]]:
    """pools[locale][category] -> list of utterance texts, row-aligned
    across locales since MASSIVE's splits are the same utterance IDs
    translated into every language."""
    pools: dict[str, dict[str, list[str]]] = {}
    for locale in LOCALES:
        ds = load_dataset("mteb/amazon_massive_scenario", locale, split="train")
        by_cat: dict[str, list[str]] = {}
        for row in ds:
            by_cat.setdefault(row["label"], []).append(row["text"])
        pools[locale] = by_cat
    return pools


def anomaly_multiplier(category: str, day: int) -> float:
    for w in ANOMALY_WINDOWS:
        if w["category"] == category and w["start_day"] <= day <= w["end_day"]:
            return w["multiplier"]
    return 1.0


def generate_tickets(pools: dict[str, dict[str, list[str]]], rng: random.Random) -> list[dict]:
    day0 = datetime(2026, 1, 1, tzinfo=UTC)
    tickets: list[dict] = []
    ticket_seq = 0

    for day in range(N_DAYS):
        for category, lam in BASELINE_LAMBDA.items():
            mult = anomaly_multiplier(category, day)
            count = poisson_sample(rng, lam * mult)
            for _ in range(count):
                locale = rng.choice(LOCALES)
                pool = pools[locale].get(category)
                if not pool:
                    continue
                text = rng.choice(pool)
                ticket_seq += 1
                tickets.append(
                    {
                        "ticket_id": f"t{ticket_seq:06d}",
                        "day_index": day,
                        "timestamp": (day0 + timedelta(days=day)).isoformat(),
                        "locale": locale,
                        "true_category": category,
                        "text": text,
                        "injected_anomaly": mult > 1.0,
                    }
                )
    return tickets


def poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm -- avoids a numpy dependency in this script for
    one small thing."""
    import math

    l = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


def generate_hard_examples(pools: dict[str, dict[str, list[str]]], rng: random.Random, per_category: int = 4) -> list[dict]:
    examples = []
    for group in CONFUSABLE_GROUPS:
        for category in group:
            for locale in ["en"]:  # keep the hard set in English so classification errors aren't confounded by translation noise
                pool = pools[locale].get(category, [])
                sample = rng.sample(pool, min(per_category, len(pool)))
                for text in sample:
                    examples.append({"text": text, "locale": locale, "true_category": category, "confusable_with": [c for c in group if c != category]})
    return examples


def main() -> None:
    out_dir = "data/synthetic"
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)

    print("Loading mteb/amazon_massive_scenario for locales:", LOCALES)
    pools = load_pools()

    tickets = generate_tickets(pools, rng)
    print(f"Generated {len(tickets)} tickets across {N_DAYS} days")

    with open(os.path.join(out_dir, "tickets.jsonl"), "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in tickets)

    with open(os.path.join(out_dir, "ground_truth_anomalies.json"), "w") as f:
        json.dump({"windows": ANOMALY_WINDOWS, "n_days": N_DAYS}, f, indent=2)

    hard_examples = generate_hard_examples(pools, rng)
    print(f"Generated {len(hard_examples)} hard/ambiguous examples")
    with open(os.path.join(out_dir, "hard_examples.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in hard_examples)

    print("Wrote tickets.jsonl, ground_truth_anomalies.json, hard_examples.jsonl to", out_dir)


if __name__ == "__main__":
    main()
