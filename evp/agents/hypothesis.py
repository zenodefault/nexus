from typing import Dict

from evp.agents.base import BaseAgent
from evp.utils.context import RunContext
from evp.utils.validation import require_fields


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
        require_fields(payload, ["summary", "key_findings", "limitations", "hypotheses"], "HypothesisAgent")
        context.add_memory("HypothesisAgent", payload)
        return payload
