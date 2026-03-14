from __future__ import annotations

from typing import Any, Dict


HIGH_MARKERS = {
    "large language model training",
    "llm training",
    "pretrain",
    "pre-training",
    "vision transformer",
    "vit",
    "diffusion",
    "multi-modal",
    "multimodal",
    "rlhf",
}

MEDIUM_MARKERS = {
    "contrastive",
    "transfer learning",
    "fine-tune",
    "finetune",
    "transformer",
    "bert",
    "gan",
    "graph neural network",
    "gnn",
}

LOW_MARKERS = {
    "data analysis",
    "feature engineering",
    "logistic regression",
    "svm",
    "random forest",
    "baseline",
    "ablation",
}


def estimate_resource_for_experiment(experiment: Dict[str, Any]) -> Dict[str, str]:
    """Heuristic fallback for compute unit estimation."""
    text = " ".join(
        str(experiment.get(k, ""))
        for k in ("title", "model", "description", "method")
    ).lower()

    if any(marker in text for marker in HIGH_MARKERS):
        return {
            "compute_units": "High",
            "resource_rationale": "Hypothesis suggests large model training or heavy compute architecture.",
        }

    if any(marker in text for marker in MEDIUM_MARKERS):
        return {
            "compute_units": "Medium",
            "resource_rationale": "Hypothesis likely needs moderate GPU time and memory for tuning/training.",
        }

    if any(marker in text for marker in LOW_MARKERS):
        return {
            "compute_units": "Low",
            "resource_rationale": "Hypothesis can likely run with lightweight training or analysis workflows.",
        }

    return {
        "compute_units": "Medium",
        "resource_rationale": "Defaulted to medium due to limited complexity signals.",
    }
