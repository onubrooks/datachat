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


# ============================================================================
# ContextAgent Models
# ============================================================================


class ExtractedEntity(BaseModel):
    """Entity extracted from user query (optional for ContextAgent)."""

    entity_type: Literal["table", "column", "metric", "process", "value"]
    value: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

    model_config = ConfigDict(frozen=True)


class ContextAgentInput(AgentInput):
    """
    Input for ContextAgent.

    ContextAgent performs pure retrieval (no LLM calls) to gather relevant
    DataPoints for the user's query. Can optionally use extracted entities
    to improve retrieval precision.
    """

    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="Entities extracted from query (optional, from ClassifierAgent)"
    )
    retrieval_mode: Literal["local", "global", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval mode: local (vector), global (graph), hybrid (both)"
    )
    max_datapoints: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of DataPoints to retrieve"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were total sales last quarter?",
                "entities": [
                    {"entity_type": "metric", "value": "sales", "confidence": 0.9},
                    {"entity_type": "table", "value": "fact_sales", "confidence": 0.8}
                ],
                "retrieval_mode": "hybrid",
                "max_datapoints": 10
            }
        }
    )


class RetrievedDataPoint(BaseModel):
    """A single retrieved DataPoint with retrieval metadata."""

    datapoint_id: str = Field(..., description="DataPoint identifier")
    datapoint_type: Literal["Schema", "Business", "Process"]
    name: str = Field(..., description="Human-readable name")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1, higher is better)"
    )
    source: str = Field(..., description="Retrieval source (vector/graph/hybrid)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full DataPoint metadata"
    )

    model_config = ConfigDict(frozen=True)


class InvestigationMemory(BaseModel):
    """
    Contextual memory for the investigation.

    Contains relevant DataPoints retrieved for the query, organized by type
    and ranked by relevance. This memory is passed to downstream agents
    (SQLAgent, etc.) to inform their reasoning.
    """

    query: str = Field(..., description="Original user query")
    datapoints: List[RetrievedDataPoint] = Field(
        default_factory=list,
        description="Retrieved DataPoints ranked by relevance"
    )
    total_retrieved: int = Field(..., description="Total number of DataPoints retrieved")
    retrieval_mode: str = Field(..., description="Retrieval mode used")
    sources_used: List[str] = Field(
        default_factory=list,
        description="Unique sources (for citation tracking)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were total sales last quarter?",
                "datapoints": [
                    {
                        "datapoint_id": "metric_revenue_001",
                        "datapoint_type": "Business",
                        "name": "Revenue",
                        "score": 0.95,
                        "source": "hybrid",
                        "metadata": {"calculation": "SUM(amount)"}
                    }
                ],
                "total_retrieved": 5,
                "retrieval_mode": "hybrid",
                "sources_used": ["metric_revenue_001", "table_sales_001"]
            }
        }
    )


class ContextAgentOutput(AgentOutput):
    """
    Output from ContextAgent.

    Contains InvestigationMemory with retrieved DataPoints that will be used
    by downstream agents for SQL generation and query planning.
    """

    investigation_memory: InvestigationMemory = Field(
        ...,
        description="Retrieved context and DataPoints"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {},
                "metadata": {
                    "agent_name": "ContextAgent",
                    "duration_ms": 45.2,
                    "llm_calls": 0  # ContextAgent doesn't use LLM
                },
                "next_agent": "SQLAgent",
                "investigation_memory": {
                    "query": "What were total sales last quarter?",
                    "datapoints": [],
                    "total_retrieved": 5,
                    "retrieval_mode": "hybrid",
                    "sources_used": []
                }
            }
        }
    )
