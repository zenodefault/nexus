from typing import Dict

from evp.agents.base import BaseAgent
from evp.utils.context import RunContext
from evp.utils.validation import require_fields


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
        require_fields(payload, ["novelty_score", "expected_gain", "impact_rationale"], "ImpactPredictorAgent")
        context.add_memory("ImpactPredictorAgent", {"experiment": experiment, **payload})
        return payload
