"""Evaluation harness for a completed TicketPulse run. Reads only files a
real run leaves behind (reports/<run_id>/tickets_scored.jsonl,
reports/<run_id>/day_counts.csv) plus the synthetic ground truth
(data/synthetic/ground_truth_anomalies.json, hard_examples.jsonl) -- never
touches Redis, so it works equally against a live docker-compose run or a
saved fixture.

Reports, with real numbers, not a single cherry-picked line:
  1. Translation quality: back-translation consistency score distribution.
  2. Classification: accuracy/F1 against real MASSIVE scenario labels, and
     against the silver-standard custom-taxonomy mapping.
  3. Anomaly detection: precision/recall/F1 against the injected ground
     truth windows, swept across a range of thresholds for both detectors
     (not just the configured operating point), saved as a tradeoff plot.
  4. The hard/ambiguous-examples subset, scored separately from the
     overall average.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common import config
from src.agents.anomaly_agent.detector import cusum_detect, poisson_rate_detect
from src.agents.classification_agent.taxonomy import SCENARIO_TO_CUSTOM

app = typer.Typer(add_completion=False)


def load_tickets_scored(run_id: str) -> pd.DataFrame:
    path = os.path.join(config.REPORTS_DIR, run_id, "tickets_scored.jsonl")
    rows = [json.loads(line) for line in open(path)]
    return pd.DataFrame(rows)


def eval_translation(df: pd.DataFrame) -> dict:
    scored = df[df["back_translation_score"].notna()]
    if scored.empty:
        return {"n_scored": 0}
    return {
        "n_scored": int(len(scored)),
        "n_canonical_language_skipped": int(len(df) - len(scored)),
        "mean": float(scored["back_translation_score"].mean()),
        "median": float(scored["back_translation_score"].median()),
        "p10": float(scored["back_translation_score"].quantile(0.10)),
        "by_locale": scored.groupby("locale")["back_translation_score"].mean().round(3).to_dict(),
    }


def eval_classification(df: pd.DataFrame) -> dict:
    y_true = df["true_category"]
    y_pred = df["predicted_category"]
    scenario_acc = accuracy_score(y_true, y_pred)
    scenario_f1 = f1_score(y_true, y_pred, average="macro")

    custom_true = y_true.map(SCENARIO_TO_CUSTOM)
    custom_pred = df["predicted_category_custom"]
    custom_acc = accuracy_score(custom_true, custom_pred)
    custom_f1 = f1_score(custom_true, custom_pred, average="macro")

    return {
        "scenario_taxonomy": {"accuracy": float(scenario_acc), "macro_f1": float(scenario_f1), "n": int(len(df))},
        "custom_taxonomy_generalization": {
            "accuracy": float(custom_acc),
            "macro_f1": float(custom_f1),
            "n": int(len(df)),
            "note": "ground truth here is the silver-standard SCENARIO_TO_CUSTOM mapping, not a human label",
        },
    }


def eval_hard_examples(run_id: str) -> dict:
    from src.agents.classification_agent.classifier import classify

    path = os.path.join(config.DATA_DIR, "hard_examples.jsonl")
    if not os.path.exists(path):
        return {"n": 0}
    examples = [json.loads(line) for line in open(path)]

    correct = 0
    rows = []
    for ex in examples:
        result = classify(ex["text"])
        is_correct = result.scenario_label == ex["true_category"]
        correct += is_correct
        rows.append({**ex, "predicted": result.scenario_label, "confidence": result.scenario_score, "correct": is_correct})

    out_path = os.path.join(config.REPORTS_DIR, run_id, "hard_examples_scored.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return {"n": len(examples), "accuracy": correct / len(examples) if examples else 0.0, "detail_path": out_path}


def load_ground_truth_windows() -> tuple[list[dict], int]:
    gt = json.load(open(os.path.join(config.DATA_DIR, "ground_truth_anomalies.json")))
    return gt["windows"], gt["n_days"]


def build_count_matrix(run_id: str) -> pd.DataFrame:
    path = os.path.join(config.REPORTS_DIR, run_id, "day_counts.csv")
    df = pd.read_csv(path)
    return df.pivot_table(index="category", columns="day_index", values="count", fill_value=0)


def positive_set(windows: list[dict]) -> set[tuple[str, int]]:
    return {(w["category"], d) for w in windows for d in range(w["start_day"], w["end_day"] + 1)}


def score_flags(flags: set[tuple[str, int]], positives: set[tuple[str, int]], universe_size: int) -> dict:
    tp = len(flags & positives)
    fp = len(flags - positives)
    fn = len(positives - flags)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / max(universe_size - len(positives), 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def sweep_anomaly_thresholds(count_matrix: pd.DataFrame, windows: list[dict]) -> dict:
    positives = positive_set(windows)
    universe_size = count_matrix.shape[0] * count_matrix.shape[1]

    cusum_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    poisson_alphas = [0.20, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0001]

    cusum_curve = []
    for t in cusum_thresholds:
        flags: set[tuple[str, int]] = set()
        for category, series in count_matrix.iterrows():
            for cp in cusum_detect(series.values.astype(float), t, config.CUSUM_DRIFT_STD, config.CUSUM_BASELINE_DAYS):
                flags.add((category, int(cp.index)))
        cusum_curve.append({"threshold_std": t, **score_flags(flags, positives, universe_size)})

    poisson_curve = []
    for a in poisson_alphas:
        flags = set()
        for category, series in count_matrix.iterrows():
            for cp in poisson_rate_detect(series.values.astype(float), a, config.CUSUM_BASELINE_DAYS):
                flags.add((category, int(cp.index)))
        poisson_curve.append({"alpha": a, **score_flags(flags, positives, universe_size)})

    best_cusum = max(cusum_curve, key=lambda r: r["f1"])
    best_poisson = max(poisson_curve, key=lambda r: r["f1"])

    return {
        "n_injected_category_days": len(positives),
        "cusum_sweep": cusum_curve,
        "poisson_sweep": poisson_curve,
        "cusum_best": best_cusum,
        "poisson_best": best_poisson,
        "configured_operating_point": {
            "cusum": score_flags(
                {
                    (cat, int(cp.index))
                    for cat, series in count_matrix.iterrows()
                    for cp in cusum_detect(series.values.astype(float), config.CUSUM_THRESHOLD_STD, config.CUSUM_DRIFT_STD, config.CUSUM_BASELINE_DAYS)
                },
                positives,
                universe_size,
            ),
            "poisson": score_flags(
                {
                    (cat, int(cp.index))
                    for cat, series in count_matrix.iterrows()
                    for cp in poisson_rate_detect(series.values.astype(float), config.POISSON_ALPHA, config.CUSUM_BASELINE_DAYS)
                },
                positives,
                universe_size,
            ),
        },
    }


def plot_sweep(sweep: dict, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, curve, key in [("CUSUM", sweep["cusum_sweep"], "threshold_std"), ("Poisson-rate", sweep["poisson_sweep"], "alpha")]:
        recalls = [r["recall"] for r in curve]
        precisions = [r["precision"] for r in curve]
        ax.plot(recalls, precisions, marker="o", label=label)
        for r in curve:
            ax.annotate(str(r[key]), (r["recall"], r["precision"]), fontsize=7, alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Anomaly detection: precision/recall across thresholds")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


@app.command()
def main(run_id: str = typer.Option(..., help="run_id used when the pipeline was run")) -> None:
    df = load_tickets_scored(run_id)
    print(f"Loaded {len(df)} scored tickets for run {run_id}")

    translation = eval_translation(df)
    classification = eval_classification(df)
    hard = eval_hard_examples(run_id)

    windows, n_days = load_ground_truth_windows()
    count_matrix = build_count_matrix(run_id)
    sweep = sweep_anomaly_thresholds(count_matrix, windows)

    plot_path = os.path.join(config.REPORTS_DIR, run_id, "threshold_sweep.png")
    plot_sweep(sweep, plot_path)

    results = {
        "run_id": run_id,
        "translation": translation,
        "classification": classification,
        "hard_examples": hard,
        "anomaly_detection": sweep,
    }
    out_path = os.path.join(config.REPORTS_DIR, run_id, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in results.items() if k != "anomaly_detection"}, indent=2, default=str))
    print("\nAnomaly detection (configured operating point):")
    print(json.dumps(sweep["configured_operating_point"], indent=2))
    print("\nAnomaly detection (best-F1 in sweep):")
    print("  cusum:  ", sweep["cusum_best"])
    print("  poisson:", sweep["poisson_best"])
    print(f"\nFull results: {out_path}")
    print(f"Threshold sweep plot: {plot_path}")


if __name__ == "__main__":
    app()
