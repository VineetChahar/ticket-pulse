"""Per-category temporal anomaly detection over daily ticket counts.

Two independent methods, both pure functions of a count series (no I/O, no
agent state) so eval/tune_anomaly_threshold.py can sweep their thresholds
offline against known injected anomaly windows without touching Redis or
any model:

- CUSUM: a two-sided mean-shift detector. Cheap, explainable, catches a
  step-change in volume. Adapted from the same algorithm used for CI
  duration anomalies in the repo-health-radar project, generalized here to
  run independently per ticket category.
- Poisson-rate test: ticket counts are a counting process, so "is today's
  count surprising given the baseline rate" is naturally a Poisson tail
  test, not a Gaussian one -- meaningful even for small counts (a handful
  of tickets/day per category) where a std-dev-based test is noisy. This
  project uses this instead of STL/seasonal-decomposition (statsmodels'
  seasonal_decompose needs >= 2 full periods; a 35-day run has no reliable
  weekly seasonality signal to decompose against).

Both return one ChangePoint per flagged day; `index` is a position in the
input array, not a calendar day, so callers translate it back to a day
index/category themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass
class ChangePoint:
    index: int
    baseline_value: float
    observed_value: float
    score: float
    threshold: float
    method: str


def cusum_detect(values: np.ndarray, threshold_std: float, drift_std: float, baseline_days: int) -> list[ChangePoint]:
    if len(values) < baseline_days + 2:
        return []

    baseline = values[:baseline_days]
    mean0 = float(np.mean(baseline))
    std0 = float(np.std(baseline)) or 1.0  # count series can have zero-variance baselines

    drift = drift_std * std0
    threshold = threshold_std * std0

    s_hi, s_lo = 0.0, 0.0
    changepoints: list[ChangePoint] = []

    for i in range(baseline_days, len(values)):
        x = values[i]
        s_hi = max(0.0, s_hi + (x - mean0) - drift)
        s_lo = max(0.0, s_lo + (mean0 - x) - drift)

        if s_hi > threshold or s_lo > threshold:
            changepoints.append(
                ChangePoint(
                    index=i,
                    baseline_value=mean0,
                    observed_value=float(x),
                    score=max(s_hi, s_lo) / std0,
                    threshold=threshold_std,
                    method="cusum",
                )
            )
            s_hi, s_lo = 0.0, 0.0  # don't re-trigger every day of a sustained spike

    return changepoints


def poisson_rate_detect(values: np.ndarray, alpha: float, baseline_days: int) -> list[ChangePoint]:
    if len(values) < baseline_days + 1:
        return []

    baseline_rate = float(np.mean(values[:baseline_days])) or 1e-6
    changepoints: list[ChangePoint] = []

    for i in range(baseline_days, len(values)):
        k = int(values[i])
        # Two-sided tail probability of observing a count this far from the
        # baseline rate under Poisson(baseline_rate).
        p_upper = poisson.sf(k - 1, baseline_rate)
        p_lower = poisson.cdf(k, baseline_rate)
        p = 2 * min(p_upper, p_lower)
        if p < alpha:
            changepoints.append(
                ChangePoint(
                    index=i,
                    baseline_value=baseline_rate,
                    observed_value=float(k),
                    score=-np.log10(max(p, 1e-300)),
                    threshold=alpha,
                    method="poisson",
                )
            )

    return changepoints
