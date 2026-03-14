import json
from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class LocalLLMClient(LLMClient):
    """Placeholder local LLM client. Replace with real endpoint later."""

    def generate(self, prompt: str) -> str:
        return (
            "{\"note\": \"LocalLLMClient stub: no model connected.\"}"
        )


class MockLLMClient(LLMClient):
    """Deterministic mock outputs for demo/testing."""

    def generate(self, prompt: str) -> str:
        if "LiteratureAgent" in prompt:
            return json.dumps(
                {
                    "summary": "Key papers compare CNN and ViT on medical imaging.",
                    "key_findings": [
                        "ViT improves accuracy by ~2-4% with large datasets",
                        "CNNs are more stable on small datasets",
                    ],
                    "limitations": ["ViT needs large labeled data"],
                    "hypotheses": [],
                }
            )
        if "HypothesisAgent" in prompt:
            return json.dumps(
                {
                    "summary": "Generated candidate experiments.",
                    "key_findings": [],
                    "limitations": [],
                    "hypotheses": [
                        {
                            "id": "exp_1",
                            "title": "ViT with transfer learning",
                            "model": "Vision Transformer",
                        },
                        {
                            "id": "exp_2",
                            "title": "CNN with contrastive learning",
                            "model": "CNN + contrastive",
                        },
                        {
                            "id": "exp_3",
                            "title": "EfficientNet baseline",
                            "model": "EfficientNet",
                        },
                    ],
                }
            )
        if "ResourceEstimatorAgent" in prompt:
            return json.dumps(
                {
                    "compute_units": "Medium",
                    "resource_rationale": "Moderate training time and memory needs.",
                }
            )
        if "ImpactPredictorAgent" in prompt:
            return json.dumps(
                {
                    "novelty_score": 7,
                    "expected_gain": 2.5,
                    "impact_rationale": "Good improvement-to-cost ratio.",
                }
            )
        return "{}"


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
