from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

from evp.utils.llm import LLMClient, safe_json_loads
from evp.utils.logging_utils import get_logger


@dataclass
class BaseAgent:
    name: str
    role: str
    goal: str
    prompt_template: str
    llm: LLMClient

    def build_prompt(self, context_text: str) -> str:
        return self.prompt_template.format(context=context_text)

    def run_sync(self, context_text: str) -> Dict[str, Any]:
        logger = get_logger(self.name)
        prompt = self.build_prompt(context_text)
        logger.debug("Prompt built")
        raw = self.llm.generate(prompt)
        logger.debug("Raw response: %s", raw)
        return safe_json_loads(raw)

    async def run(self, context_text: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run_sync, context_text)
