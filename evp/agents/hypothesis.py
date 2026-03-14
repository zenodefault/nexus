from typing import Dict

from evp.agents.base import BaseAgent
from evp.utils.context import RunContext
from evp.utils.validation import fill_missing, require_fields


class HypothesisAgent(BaseAgent):
    role = "Experiment generator"
    goal = "Generate candidate experiments based on literature context."
    prompt_template = (
        "You are HypothesisAgent. Given the context: {context} "
        "generate 3 experiment hypotheses tailored to the dataset and goal. "
        "Prefer data analysis or modeling steps grounded in the dataset profile over generic architecture swaps. "
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
            dataset_profile = context.constraints.get("dataset_profile", "")
            goal = context.goal or "the stated goal"
            focus = "dataset-driven analysis"
            if dataset_profile:
                focus = "analysis grounded in the dataset profile"
            payload["hypotheses"] = [
                {
                    "id": "exp_1",
                    "title": f"Exploratory analysis to answer: {goal}",
                    "model": f"EDA + summary ({focus})",
                },
                {
                    "id": "exp_2",
                    "title": "Group comparison on key outcome columns",
                    "model": "Statistical comparison",
                },
                {
                    "id": "exp_3",
                    "title": "Predict outcome drivers using a simple baseline model",
                    "model": "Baseline regression/classification",
                },
            ]
        require_fields(payload, ["summary", "key_findings", "limitations", "hypotheses"], "HypothesisAgent")
        context.add_memory("HypothesisAgent", payload)
        return payload
