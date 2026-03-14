from typing import Dict

from evp.agents.base import BaseAgent
from evp.utils.context import RunContext
from evp.utils.impact_heuristics import estimate_impact_for_experiment
from evp.utils.validation import fill_missing, require_fields


class ImpactPredictorAgent(BaseAgent):
    role = "Experiment impact scorer"
    goal = "Score novelty and expected improvement."
    prompt_template = (
        "You are ImpactPredictorAgent. Given experiment and literature context: {context} "
        "score novelty from 1-10 and expected_gain as percent improvement. "
        "Return JSON with fields: novelty_score (int), expected_gain (float), impact_rationale (string). "
        "If unknown, state unknown."
    )

    def run_with_context(self, context: RunContext, experiment: Dict) -> Dict:
        memory_text = f"Experiment: {experiment}. Memory: {context.memory}"
        payload = self.run_sync(memory_text)
        payload = fill_missing(
            payload,
            {
                "novelty_score": 0,
                "expected_gain": 0.0,
                "impact_rationale": "No impact rationale returned.",
            },
        )
        if payload.get("novelty_score", 0) == 0 and payload.get("expected_gain", 0) == 0:
            payload = estimate_impact_for_experiment(experiment)
        require_fields(payload, ["novelty_score", "expected_gain", "impact_rationale"], "ImpactPredictorAgent")
        context.add_memory("ImpactPredictorAgent", {"experiment": experiment, **payload})
        return payload
