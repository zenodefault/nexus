from typing import Dict

from evp.agents.base import BaseAgent
from evp.utils.context import RunContext
from evp.utils.validation import fill_missing, require_fields


class HypothesisAgent(BaseAgent):
    role = "Experiment generator"
    goal = "Generate candidate experiments based on literature context."
    prompt_template = (
        "You are HypothesisAgent. Given the context: {context} "
        "generate 3 experiment hypotheses. "
        "Return JSON with fields: summary (string), key_findings (list), limitations (list), hypotheses (list of objects with id, title, model). "
        "If unknown, state unknown."
    )

    def run_with_context(self, context: RunContext) -> Dict:
        memory_text = f"Topic: {context.topic}. Goal: {context.goal}. Memory: {context.memory}"
        payload = self.run_sync(memory_text)
        payload = fill_missing(
            payload,
            {
                "summary": "Unknown",
                "key_findings": [],
                "limitations": [],
                "hypotheses": [],
            },
        )
        if not payload.get("hypotheses"):
            payload["hypotheses"] = [
                {
                    "id": "exp_1",
                    "title": f"{context.topic} baseline with data augmentation",
                    "model": "Baseline + augmentation",
                },
                {
                    "id": "exp_2",
                    "title": f"{context.topic} with transfer learning",
                    "model": "Transfer learning",
                },
                {
                    "id": "exp_3",
                    "title": f"{context.topic} with lightweight architecture",
                    "model": "Efficient model",
                },
            ]
        require_fields(payload, ["summary", "key_findings", "limitations", "hypotheses"], "HypothesisAgent")
        context.add_memory("HypothesisAgent", payload)
        return payload
