import json
import os
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class LocalLLMClient(LLMClient):
    """ACPX CLI-backed local LLM client for Gemini/Qwen integration."""

    def __init__(self) -> None:
        self.model = os.getenv("EVP_LOCAL_MODEL", "gemini")
        self.timeout_seconds = int(os.getenv("EVP_ACPX_TIMEOUT_SECONDS", "120"))
        self.cmd_template = os.getenv("EVP_ACPX_CMD", "acpx run --model {model}")

    def _build_command(self) -> list[str]:
        raw = self.cmd_template.format(model=self.model)
        return shlex.split(raw)

    def generate(self, prompt: str) -> str:
        try:
            proc = subprocess.run(
                self._build_command(),
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "{\"note\": \"ACPX invocation failed.\"}"

        if proc.returncode != 0:
            err = proc.stderr.strip() or "unknown ACPX CLI error"
            return json.dumps({"note": f"ACPX CLI failed: {err}"})

        output = proc.stdout.strip()
        if output:
            return _extract_json_from_text(output)
        return "{\"note\": \"ACPX returned empty output.\"}"


class MockLLMClient(LLMClient):
    """Deterministic mock outputs for demo/testing."""

    def generate(self, prompt: str) -> str:
        if "ImpactPredictorAgent" in prompt:
            return json.dumps(
                {
                    "novelty_score": 7,
                    "expected_gain": 2.5,
                    "impact_rationale": "Good improvement-to-cost ratio.",
                }
            )
        if "ResourceEstimatorAgent" in prompt:
            return json.dumps(
                {
                    "compute_units": "Medium",
                    "resource_rationale": "Moderate training time and memory needs.",
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
        return "{}"


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _extract_json_from_text(text: str) -> str:
    """Extract likely JSON object from mixed CLI output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text
