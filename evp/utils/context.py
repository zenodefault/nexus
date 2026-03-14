from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RunContext:
    """Shared context across agents for memory passing."""

    topic: str
    goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)

    def add_memory(self, agent: str, payload: Dict[str, Any]) -> None:
        self.memory.append({"agent": agent, "payload": payload})
