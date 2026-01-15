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


class SQLGenerationError(AgentError):
    """Error during SQL generation (may be recoverable with self-correction)."""

    def __init__(
        self,
        agent: str,
        message: str,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent, message, recoverable=recoverable, context=context)


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


# ============================================================================
# SQLAgent Models
# ============================================================================


class ValidationIssue(BaseModel):
    """
    Issue found during SQL validation.

    Used for self-correction: tracks syntax errors, missing columns,
    table name errors, etc. that need to be fixed.
    """

    issue_type: Literal["syntax", "missing_column", "missing_table", "ambiguous", "other"] = Field(
        ...,
        description="Type of validation issue"
    )
    message: str = Field(
        ...,
        description="Human-readable description of the issue"
    )
    location: Optional[str] = Field(
        None,
        description="Location in SQL where issue was found (line/column if available)"
    )
    suggested_fix: Optional[str] = Field(
        None,
        description="Suggested correction for the issue"
    )


class GeneratedSQL(BaseModel):
    """
    SQL query generated by SQLAgent.

    Contains the query, explanation, and metadata about what DataPoints
    were used to generate it.
    """

    sql: str = Field(
        ...,
        description="Generated SQL query",
        min_length=1
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of what the query does"
    )
    used_datapoints: List[str] = Field(
        default_factory=list,
        description="DataPoint IDs used in query generation (for citation)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the generated query (0-1)"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made during query generation"
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Questions for user if query is ambiguous"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sql": "SELECT SUM(amount) FROM fact_sales WHERE date >= '2024-07-01' AND date < '2024-10-01'",
                "explanation": "This query calculates total sales for Q3 2024 by summing the amount column from the fact_sales table, filtered for dates in the third quarter.",
                "used_datapoints": ["table_fact_sales_001", "metric_revenue_001"],
                "confidence": 0.95,
                "assumptions": ["'last quarter' refers to Q3 2024 (most recent complete quarter)"],
                "clarifying_questions": []
            }
        }
    )


class CorrectionAttempt(BaseModel):
    """
    Record of a self-correction attempt.

    Tracks validation issues found and the corrected SQL.
    """

    attempt_number: int = Field(..., ge=1, description="Correction attempt number")
    original_sql: str = Field(..., description="SQL before correction")
    issues_found: List[ValidationIssue] = Field(
        ...,
        description="Validation issues that triggered correction"
    )
    corrected_sql: str = Field(..., description="SQL after correction")
    success: bool = Field(..., description="Whether correction resolved the issues")


class SQLAgentInput(AgentInput):
    """
    Input for SQLAgent.

    Receives query and InvestigationMemory from ContextAgent to generate SQL.
    """

    investigation_memory: InvestigationMemory = Field(
        ...,
        description="Context retrieved by ContextAgent"
    )
    max_correction_attempts: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of self-correction attempts"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were total sales last quarter?",
                "investigation_memory": {
                    "query": "What were total sales last quarter?",
                    "datapoints": [],
                    "total_retrieved": 5,
                    "retrieval_mode": "hybrid",
                    "sources_used": []
                },
                "max_correction_attempts": 3
            }
        }
    )


class SQLAgentOutput(AgentOutput):
    """
    Output from SQLAgent.

    Contains generated SQL, explanation, and correction history if applicable.
    """

    generated_sql: GeneratedSQL = Field(
        ...,
        description="Generated SQL query with metadata"
    )
    correction_attempts: List[CorrectionAttempt] = Field(
        default_factory=list,
        description="Self-correction attempts made (empty if first attempt succeeded)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether query needs user clarification"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {},
                "metadata": {
                    "agent_name": "SQLAgent",
                    "duration_ms": 1234.5,
                    "llm_calls": 1,
                    "tokens_used": 850
                },
                "next_agent": "ValidatorAgent",
                "generated_sql": {
                    "sql": "SELECT SUM(amount) FROM fact_sales WHERE date >= '2024-07-01'",
                    "explanation": "Sums sales amounts for Q3 2024",
                    "used_datapoints": ["table_fact_sales_001"],
                    "confidence": 0.95,
                    "assumptions": [],
                    "clarifying_questions": []
                },
                "correction_attempts": [],
                "needs_clarification": False
            }
        }
    )
