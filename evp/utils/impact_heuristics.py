from __future__ import annotations

from typing import Any, Dict


HIGH_NOVELTY = {
    "contrastive",
    "self-supervised",
    "multimodal",
    "multi-modal",
    "fusion",
    "graph",
    "transformer",
    "vit",
}

MEDIUM_NOVELTY = {
    "transfer learning",
    "fine-tune",
    "finetune",
    "ensemble",
    "augmentation",
    "regularization",
}

LOW_NOVELTY = {
    "baseline",
    "ablation",
    "logistic regression",
    "svm",
    "random forest",
}


def estimate_impact_for_experiment(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic fallback for novelty/expected gain estimation."""
    text = " ".join(
        str(experiment.get(k, ""))
        for k in ("title", "model", "description", "method")
    ).lower()

    if any(marker in text for marker in HIGH_NOVELTY):
        return {
            "novelty_score": 7,
            "expected_gain": 3.0,
            "impact_rationale": "Idea suggests a more novel technique likely to improve results.",
        }

    if any(marker in text for marker in MEDIUM_NOVELTY):
        return {
            "novelty_score": 5,
            "expected_gain": 2.0,
            "impact_rationale": "Idea extends common methods with moderate potential gains.",
        }

    if any(marker in text for marker in LOW_NOVELTY):
        return {
            "novelty_score": 2,
            "expected_gain": 1.0,
            "impact_rationale": "Idea is a baseline or incremental change with limited novelty.",
        }

    return {
        "novelty_score": 4,
        "expected_gain": 1.5,
        "impact_rationale": "Defaulted to moderate novelty due to limited signals.",
    }
