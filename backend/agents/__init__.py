"""
DataChat Agents Module

Multi-agent pipeline for natural language to SQL conversion.

Available Agents:
    - BaseAgent: Abstract base class for all agents
    - ClassifierAgent: Intent detection and entity extraction (TODO)
    - ContextAgent: Knowledge graph and vector retrieval (TODO)
    - SQLAgent: SQL query generation (TODO)
    - ValidatorAgent: Query validation and security checks (TODO)
    - ExecutorAgent: Query execution and response formatting (TODO)

Usage:
    from backend.agents import BaseAgent

    class MyAgent(BaseAgent):
        async def execute(self, input: AgentInput) -> AgentOutput:
            return AgentOutput(
                success=True,
                data={"result": "value"},
                metadata=self._create_metadata()
            )
"""

from backend.agents.base import BaseAgent

__all__ = ["BaseAgent"]
