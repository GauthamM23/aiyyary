"""Abstract base class for all Agent 2 executors.

An executor takes a surviving, executable opportunity and actually activates
it for the business — connecting to a market tool, building a deployable
artifact, or scaffolding a partial build — rather than just recommending it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExecutorResult:
    mode: str  # "market_connect" | "full_build" | "partial_build"
    status: str  # "complete" | "partial" | "connected" | "failed"
    deliverables: list = field(default_factory=list)  # list of {name, type, content, instructions}
    missing_prerequisites: list = field(default_factory=list)
    next_steps: list = field(default_factory=list)
    estimated_activation_time: str = ""
    market_tools_recommended: list = field(default_factory=list)
    llm_calls_made: int = 0
    error: str = None


class BaseExecutor(ABC):
    """Every executor must implement execute() and never raise."""

    def __init__(self, llm_client):
        self.llm = llm_client

    @abstractmethod
    def execute(self, profile: dict, opportunity: dict, agent1_output: dict) -> ExecutorResult:
        """Activate `opportunity` for this business. Must catch all exceptions internally."""
        raise NotImplementedError
