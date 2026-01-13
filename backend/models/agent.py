"""
Agent I/O Models

Pydantic models for agent inputs, outputs, and error handling.
All agents in the pipeline use these base models to ensure type safety
and consistent data structures throughout the system.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class Message(BaseModel):
    """Single message in conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class AgentMetadata(BaseModel):
    """Metadata about agent execution."""

    agent_name: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    llm_calls: int = 0
    tokens_used: Optional[int] = None
    error: Optional[str] = None

    model_config = ConfigDict(frozen=False)

    def mark_complete(self) -> None:
        """Mark execution as complete and calculate duration."""
        self.completed_at = datetime.utcnow()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000


class AgentInput(BaseModel):
    """
    Base input model for all agents.

    Each agent should extend this with their specific input fields.
    The conversation history and metadata are common across all agents.
    """

    query: str = Field(..., description="User's natural language query")
    conversation_history: List[Message] = Field(
        default_factory=list,
        description="Previous messages in the conversation"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context passed between agents"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were our top selling products last quarter?",
                "conversation_history": [],
                "context": {}
            }
        }
    )


class AgentOutput(BaseModel):
    """
    Base output model for all agents.

    Each agent should extend this with their specific output fields.
    The metadata field tracks execution details for observability.
    """

    success: bool = Field(..., description="Whether the agent executed successfully")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific output data"
    )
    metadata: AgentMetadata = Field(..., description="Execution metadata")
    next_agent: Optional[str] = Field(
        None,
        description="Name of next agent to execute (for pipeline routing)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {"intent": "data_query"},
                "metadata": {
                    "agent_name": "ClassifierAgent",
                    "duration_ms": 234.5,
                    "llm_calls": 1
                },
                "next_agent": "ContextAgent"
            }
        }
    )


class AgentError(Exception):
    """
    Custom exception for agent execution errors.

    Attributes:
        agent: Name of the agent that raised the error
        message: Error description
        recoverable: Whether the pipeline can retry or continue
        context: Additional context for debugging
    """

    def __init__(
        self,
        agent: str,
        message: str,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        self.agent = agent
        self.message = message
        self.recoverable = recoverable
        self.context = context or {}
        super().__init__(f"[{agent}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/API responses."""
        return {
            "agent": self.agent,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context,
            "type": self.__class__.__name__
        }


class ValidationError(AgentError):
    """Error during data validation (usually not recoverable)."""

    def __init__(self, agent: str, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(agent, message, recoverable=False, context=context)


class LLMError(AgentError):
    """Error during LLM API call (usually recoverable with retry)."""

    def __init__(self, agent: str, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(agent, message, recoverable=True, context=context)


class DatabaseError(AgentError):
    """Error during database operation (may or may not be recoverable)."""

    def __init__(
        self,
        agent: str,
        message: str,
        recoverable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent, message, recoverable=recoverable, context=context)


class RetrievalError(AgentError):
    """Error during context retrieval (usually recoverable)."""

    def __init__(self, agent: str, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(agent, message, recoverable=True, context=context)
