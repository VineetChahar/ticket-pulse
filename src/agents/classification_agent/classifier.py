"""Zero-shot ticket classification via facebook/bart-large-mnli.

Every ticket is classified twice: once against SCENARIO_LABELS (scored
against real dataset ground truth in eval) and once against CUSTOM_LABELS
(scored against the silver-standard SCENARIO_TO_CUSTOM mapping) -- see
taxonomy.py for why both exist.
"""

from __future__ import annotations

from functools import lru_cache

from src.common import config
from src.agents.classification_agent.taxonomy import CUSTOM_LABELS, SCENARIO_LABELS

_HYPOTHESIS_TEMPLATE = "This customer support ticket is about {}."


@lru_cache(maxsize=1)
def _pipeline():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=config.CLASSIFICATION_MODEL, device=device)


class ClassificationResult:
    __slots__ = ("scenario_label", "scenario_score", "custom_label", "custom_score")

    def __init__(self, scenario_label: str, scenario_score: float, custom_label: str, custom_score: float) -> None:
        self.scenario_label = scenario_label
        self.scenario_score = scenario_score
        self.custom_label = custom_label
        self.custom_score = custom_score


def classify(text: str) -> ClassificationResult:
    clf = _pipeline()

    scenario_out = clf(text, candidate_labels=SCENARIO_LABELS, hypothesis_template=_HYPOTHESIS_TEMPLATE)
    custom_out = clf(text, candidate_labels=CUSTOM_LABELS, hypothesis_template=_HYPOTHESIS_TEMPLATE)

    return ClassificationResult(
        scenario_label=scenario_out["labels"][0],
        scenario_score=float(scenario_out["scores"][0]),
        custom_label=custom_out["labels"][0],
        custom_score=float(custom_out["scores"][0]),
    )
