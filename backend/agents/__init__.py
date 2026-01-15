"""
DataChat Agents Module

Multi-agent pipeline for natural language to SQL conversion.

Available Agents:
    - BaseAgent: Abstract base class for all agents
    - ContextAgent: Knowledge graph and vector retrieval (pure retrieval, no LLM)
    - ClassifierAgent: Intent detection and entity extraction (TODO)
    - SQLAgent: SQL query generation (TODO)
    - ValidatorAgent: Query validation and security checks (TODO)
    - ExecutorAgent: Query execution and response formatting (TODO)

Usage:
    from backend.agents import BaseAgent, ContextAgent

    class MyAgent(BaseAgent):
        async def execute(self, input: AgentInput) -> AgentOutput:
            return AgentOutput(
                success=True,
                data={"result": "value"},
                metadata=self._create_metadata()
            )
"""

from backend.agents.base import BaseAgent
from backend.agents.context import ContextAgent

__all__ = ["BaseAgent", "ContextAgent"]
