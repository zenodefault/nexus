from typing import Dict

from evp.agents.base import BaseAgent
from evp.utils.context import RunContext
from evp.utils.resource_heuristics import estimate_resource_for_experiment
from evp.utils.validation import require_fields


class ResourceEstimatorAgent(BaseAgent):
    role = "Compute cost estimator"
    goal = "Estimate compute units for each experiment."
    prompt_template = (
        "You are ResourceEstimatorAgent. Given experiment: {context} "
        "estimate compute units (Low/Medium/High) and a short rationale. "
        "Return JSON with fields: compute_units (string), resource_rationale (string). "
        "If unknown, state unknown."
    )

    def run_with_context(self, context: RunContext, experiment: Dict) -> Dict:
        heuristic_payload = estimate_resource_for_experiment(experiment)
        payload = self.run_sync(f"Experiment: {experiment}. Prior estimate: {heuristic_payload}")

        if not payload or "compute_units" not in payload or "resource_rationale" not in payload:
            payload = heuristic_payload

        require_fields(payload, ["compute_units", "resource_rationale"], "ResourceEstimatorAgent")
        context.add_memory("ResourceEstimatorAgent", {"experiment": experiment, **payload})
        return payload
