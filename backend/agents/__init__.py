"""
DataChat Agents Module

Multi-agent pipeline for natural language to SQL conversion.

Available Agents:
    - BaseAgent: Abstract base class for all agents
    - ContextAgent: Knowledge graph and vector retrieval (pure retrieval, no LLM)
    - SQLAgent: SQL query generation with self-correction
    - ValidatorAgent: SQL validation with security and performance checks
    - ClassifierAgent: Intent detection and entity extraction (TODO)
    - ExecutorAgent: Query execution and response formatting (TODO)

Usage:
    from backend.agents import BaseAgent, ContextAgent, SQLAgent, ValidatorAgent

    class MyAgent(BaseAgent):
        async def execute(self, input: AgentInput) -> AgentOutput:
            return AgentOutput(
                success=True,
                data={"result": "value"},
                metadata=self._create_metadata()
            )
"""

from backend.agents.base import BaseAgent
from backend.agents.classifier import ClassifierAgent
from backend.agents.context import ContextAgent
from backend.agents.context_answer import ContextAnswerAgent
from backend.agents.executor import ExecutorAgent
from backend.agents.tool_planner import ToolPlannerAgent
from backend.agents.response_synthesis import ResponseSynthesisAgent
from backend.agents.sql import SQLAgent
from backend.agents.validator import ValidatorAgent

__all__ = [
    "BaseAgent",
    "ClassifierAgent",
    "ContextAgent",
    "ContextAnswerAgent",
    "ExecutorAgent",
    "ToolPlannerAgent",
    "ResponseSynthesisAgent",
    "SQLAgent",
    "ValidatorAgent",
]
