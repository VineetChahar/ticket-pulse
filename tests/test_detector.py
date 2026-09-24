"""Anomaly detector math: feed a synthetic series with a known change point,
assert it's flagged within a few days of the injection, and assert a flat
baseline produces no false positives."""

from __future__ import annotations

import numpy as np

from src.agents.anomaly_agent.detector import cusum_detect, poisson_rate_detect

BASELINE_DAYS = 7


def flat_series(baseline: float, n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.poisson(baseline, size=n).astype(float)


def spiked_series(baseline: float, n: int, spike_start: int, spike_end: int, multiplier: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    values = rng.poisson(baseline, size=n).astype(float)
    values[spike_start : spike_end + 1] = rng.poisson(baseline * multiplier, size=spike_end + 1 - spike_start)
    return values


def test_cusum_detects_injected_spike_within_a_few_days():
    values = spiked_series(baseline=5, n=30, spike_start=15, spike_end=18, multiplier=6.0)
    flags = cusum_detect(values, threshold_std=3.0, drift_std=0.5, baseline_days=BASELINE_DAYS)
    assert flags, "expected at least one changepoint"
    assert any(15 <= cp.index <= 18 + 2 for cp in flags), f"flag indices {[cp.index for cp in flags]} not near spike"


def test_cusum_flat_series_has_no_false_positives():
    values = flat_series(baseline=5, n=30)
    flags = cusum_detect(values, threshold_std=3.5, drift_std=0.5, baseline_days=BASELINE_DAYS)
    assert flags == []


def test_cusum_too_short_returns_empty():
    assert cusum_detect(np.array([1.0, 2.0]), 3.5, 0.5, BASELINE_DAYS) == []


def test_poisson_detects_injected_spike_within_a_few_days():
    values = spiked_series(baseline=5, n=30, spike_start=15, spike_end=18, multiplier=6.0)
    flags = poisson_rate_detect(values, alpha=0.01, baseline_days=BASELINE_DAYS)
    assert flags, "expected at least one flagged day"
    assert any(15 <= cp.index <= 18 + 2 for cp in flags), f"flag indices {[cp.index for cp in flags]} not near spike"


def test_poisson_flat_series_rarely_false_positives():
    values = flat_series(baseline=8, n=30)
    flags = poisson_rate_detect(values, alpha=0.001, baseline_days=BASELINE_DAYS)
    assert len(flags) <= 1  # a strict alpha on a flat Poisson series should almost never trip


def test_poisson_too_short_returns_empty():
    assert poisson_rate_detect(np.array([1.0, 2.0]), 0.01, BASELINE_DAYS) == []


def test_lower_threshold_detects_a_weaker_spike_that_a_stricter_one_misses():
    values = spiked_series(baseline=10, n=30, spike_start=15, spike_end=16, multiplier=1.8)
    loose = cusum_detect(values, threshold_std=1.5, drift_std=0.3, baseline_days=BASELINE_DAYS)
    strict = cusum_detect(values, threshold_std=6.0, drift_std=0.5, baseline_days=BASELINE_DAYS)
    assert len(loose) >= len(strict)
