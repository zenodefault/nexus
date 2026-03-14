from typing import Dict

from evp.agents.base import BaseAgent
from evp.utils.context import RunContext
from evp.utils.validation import fill_missing, require_fields


class LiteratureAgent(BaseAgent):
    role = "Literature summarizer"
    goal = "Extract key findings and limitations from relevant papers."
    prompt_template = (
        "You are LiteratureAgent. Summarize recent research for the topic: {context}. "
        "Return JSON with fields: summary (string), key_findings (list), limitations (list), hypotheses (list). "
        "If unknown, state unknown."
    )

    def run_with_context(self, context: RunContext) -> Dict:
        digest = context.constraints.get("literature_digest", "No paper context available.")
        payload = self.run_sync(f"Topic: {context.topic}. Goal: {context.goal}. {digest}")
        payload = fill_missing(
            payload,
            {
                "summary": "Unknown",
                "key_findings": [],
                "limitations": [],
                "hypotheses": [],
            },
        )
        require_fields(payload, ["summary", "key_findings", "limitations", "hypotheses"], "LiteratureAgent")
        context.add_memory("LiteratureAgent", payload)
        return payload
