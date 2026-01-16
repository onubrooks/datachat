"""
Agent I/O Models

Pydantic models for agent inputs, outputs, and error handling.
All agents in the pipeline use these base models to ensure type safety
and consistent data structures throughout the system.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    completed_at: datetime | None = None
    duration_ms: float | None = None
    llm_calls: int = 0
    tokens_used: int | None = None
    error: str | None = None

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
    conversation_history: list[Message] = Field(
        default_factory=list, description="Previous messages in the conversation"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context passed between agents"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were our top selling products last quarter?",
                "conversation_history": [],
                "context": {},
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
    data: dict[str, Any] = Field(default_factory=dict, description="Agent-specific output data")
    metadata: AgentMetadata = Field(..., description="Execution metadata")
    next_agent: str | None = Field(
        None, description="Name of next agent to execute (for pipeline routing)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {"intent": "data_query"},
                "metadata": {"agent_name": "ClassifierAgent", "duration_ms": 234.5, "llm_calls": 1},
                "next_agent": "ContextAgent",
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
        context: dict[str, Any] | None = None,
    ):
        self.agent = agent
        self.message = message
        self.recoverable = recoverable
        self.context = context or {}
        super().__init__(f"[{agent}] {message}")

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/API responses."""
        return {
            "agent": self.agent,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context,
            "type": self.__class__.__name__,
        }


class ValidationError(AgentError):
    """Error during data validation (usually not recoverable)."""

    def __init__(self, agent: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(agent, message, recoverable=False, context=context)


class LLMError(AgentError):
    """Error during LLM API call (usually recoverable with retry)."""

    def __init__(self, agent: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(agent, message, recoverable=True, context=context)


class DatabaseError(AgentError):
    """Error during database operation (may or may not be recoverable)."""

    def __init__(
        self,
        agent: str,
        message: str,
        recoverable: bool = False,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(agent, message, recoverable=recoverable, context=context)


class RetrievalError(AgentError):
    """Error during context retrieval (usually recoverable)."""

    def __init__(self, agent: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(agent, message, recoverable=True, context=context)


class SQLGenerationError(AgentError):
    """Error during SQL generation (may be recoverable with self-correction)."""

    def __init__(
        self,
        agent: str,
        message: str,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
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

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Entities extracted from query (optional, from ClassifierAgent)",
    )
    retrieval_mode: Literal["local", "global", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval mode: local (vector), global (graph), hybrid (both)",
    )
    max_datapoints: int = Field(
        default=10, ge=1, le=50, description="Maximum number of DataPoints to retrieve"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were total sales last quarter?",
                "entities": [
                    {"entity_type": "metric", "value": "sales", "confidence": 0.9},
                    {"entity_type": "table", "value": "fact_sales", "confidence": 0.8},
                ],
                "retrieval_mode": "hybrid",
                "max_datapoints": 10,
            }
        }
    )


class RetrievedDataPoint(BaseModel):
    """A single retrieved DataPoint with retrieval metadata."""

    datapoint_id: str = Field(..., description="DataPoint identifier")
    datapoint_type: Literal["Schema", "Business", "Process"]
    name: str = Field(..., description="Human-readable name")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1, higher is better)")
    source: str = Field(..., description="Retrieval source (vector/graph/hybrid)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Full DataPoint metadata")

    model_config = ConfigDict(frozen=True)


class InvestigationMemory(BaseModel):
    """
    Contextual memory for the investigation.

    Contains relevant DataPoints retrieved for the query, organized by type
    and ranked by relevance. This memory is passed to downstream agents
    (SQLAgent, etc.) to inform their reasoning.
    """

    query: str = Field(..., description="Original user query")
    datapoints: list[RetrievedDataPoint] = Field(
        default_factory=list, description="Retrieved DataPoints ranked by relevance"
    )
    total_retrieved: int = Field(..., description="Total number of DataPoints retrieved")
    retrieval_mode: str = Field(..., description="Retrieval mode used")
    sources_used: list[str] = Field(
        default_factory=list, description="Unique sources (for citation tracking)"
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
                        "metadata": {"calculation": "SUM(amount)"},
                    }
                ],
                "total_retrieved": 5,
                "retrieval_mode": "hybrid",
                "sources_used": ["metric_revenue_001", "table_sales_001"],
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
        ..., description="Retrieved context and DataPoints"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {},
                "metadata": {
                    "agent_name": "ContextAgent",
                    "duration_ms": 45.2,
                    "llm_calls": 0,  # ContextAgent doesn't use LLM
                },
                "next_agent": "SQLAgent",
                "investigation_memory": {
                    "query": "What were total sales last quarter?",
                    "datapoints": [],
                    "total_retrieved": 5,
                    "retrieval_mode": "hybrid",
                    "sources_used": [],
                },
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
        ..., description="Type of validation issue"
    )
    message: str = Field(..., description="Human-readable description of the issue")
    location: str | None = Field(
        None, description="Location in SQL where issue was found (line/column if available)"
    )
    suggested_fix: str | None = Field(None, description="Suggested correction for the issue")


class GeneratedSQL(BaseModel):
    """
    SQL query generated by SQLAgent.

    Contains the query, explanation, and metadata about what DataPoints
    were used to generate it.
    """

    sql: str = Field(..., description="Generated SQL query", min_length=1)
    explanation: str = Field(..., description="Human-readable explanation of what the query does")
    used_datapoints: list[str] = Field(
        default_factory=list, description="DataPoint IDs used in query generation (for citation)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the generated query (0-1)"
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions made during query generation"
    )
    clarifying_questions: list[str] = Field(
        default_factory=list, description="Questions for user if query is ambiguous"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sql": "SELECT SUM(amount) FROM fact_sales WHERE date >= '2024-07-01' AND date < '2024-10-01'",
                "explanation": "This query calculates total sales for Q3 2024 by summing the amount column from the fact_sales table, filtered for dates in the third quarter.",
                "used_datapoints": ["table_fact_sales_001", "metric_revenue_001"],
                "confidence": 0.95,
                "assumptions": ["'last quarter' refers to Q3 2024 (most recent complete quarter)"],
                "clarifying_questions": [],
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
    issues_found: list[ValidationIssue] = Field(
        ..., description="Validation issues that triggered correction"
    )
    corrected_sql: str = Field(..., description="SQL after correction")
    success: bool = Field(..., description="Whether correction resolved the issues")


class SQLAgentInput(AgentInput):
    """
    Input for SQLAgent.

    Receives query and InvestigationMemory from ContextAgent to generate SQL.
    """

    investigation_memory: InvestigationMemory = Field(
        ..., description="Context retrieved by ContextAgent"
    )
    max_correction_attempts: int = Field(
        default=3, ge=1, le=5, description="Maximum number of self-correction attempts"
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
                    "sources_used": [],
                },
                "max_correction_attempts": 3,
            }
        }
    )


class SQLAgentOutput(AgentOutput):
    """
    Output from SQLAgent.

    Contains generated SQL, explanation, and correction history if applicable.
    """

    generated_sql: GeneratedSQL = Field(..., description="Generated SQL query with metadata")
    correction_attempts: list[CorrectionAttempt] = Field(
        default_factory=list,
        description="Self-correction attempts made (empty if first attempt succeeded)",
    )
    needs_clarification: bool = Field(
        default=False, description="Whether query needs user clarification"
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
                    "tokens_used": 850,
                },
                "next_agent": "ValidatorAgent",
                "generated_sql": {
                    "sql": "SELECT SUM(amount) FROM fact_sales WHERE date >= '2024-07-01'",
                    "explanation": "Sums sales amounts for Q3 2024",
                    "used_datapoints": ["table_fact_sales_001"],
                    "confidence": 0.95,
                    "assumptions": [],
                    "clarifying_questions": [],
                },
                "correction_attempts": [],
                "needs_clarification": False,
            }
        }
    )


# ============================================================================
# ValidatorAgent Models
# ============================================================================


class SQLValidationError(BaseModel):
    """
    SQL validation error found during validation.

    Represents a critical issue that prevents SQL execution.
    """

    error_type: Literal["syntax", "security", "schema", "other"] = Field(
        ..., description="Type of validation error"
    )
    message: str = Field(..., description="Human-readable error message")
    location: str | None = Field(None, description="Location in SQL where error was found")
    severity: Literal["critical", "high", "medium", "low"] = Field(
        default="critical", description="Error severity level"
    )


class ValidationWarning(BaseModel):
    """
    Validation warning for SQL.

    Represents a non-critical issue or performance concern.
    """

    warning_type: Literal["performance", "style", "compatibility", "other"] = Field(
        ..., description="Type of validation warning"
    )
    message: str = Field(..., description="Human-readable warning message")
    suggestion: str | None = Field(None, description="Suggested fix or improvement")


class ValidatedSQL(BaseModel):
    """
    SQL validation result.

    Contains validation status, errors, warnings, and suggestions.
    """

    is_valid: bool = Field(..., description="Whether SQL is valid and safe to execute")
    sql: str = Field(..., description="The validated SQL query")
    errors: list[SQLValidationError] = Field(
        default_factory=list, description="Critical errors that prevent execution"
    )
    warnings: list[ValidationWarning] = Field(
        default_factory=list, description="Non-critical issues and performance concerns"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="General suggestions for improvement"
    )
    is_safe: bool = Field(
        ..., description="Whether SQL passed security checks (no injection patterns)"
    )
    performance_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Performance score (0-1, higher is better)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_valid": True,
                "sql": "SELECT amount FROM fact_sales WHERE date >= '2024-01-01'",
                "errors": [],
                "warnings": [
                    {
                        "warning_type": "performance",
                        "message": "Query missing LIMIT clause",
                        "suggestion": "Add LIMIT to prevent returning too many rows",
                    }
                ],
                "suggestions": ["Consider adding an index on date column"],
                "is_safe": True,
                "performance_score": 0.85,
            }
        }
    )


class ValidatorAgentInput(AgentInput):
    """
    Input for ValidatorAgent.

    Receives generated SQL from SQLAgent for validation.
    """

    generated_sql: GeneratedSQL = Field(..., description="SQL generated by SQLAgent")
    target_database: Literal["postgresql", "clickhouse", "mysql", "generic"] = Field(
        default="postgresql", description="Target database type for syntax validation"
    )
    strict_mode: bool = Field(
        default=False, description="Enable strict validation (treat warnings as errors)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were total sales last quarter?",
                "generated_sql": {
                    "sql": "SELECT SUM(amount) FROM fact_sales",
                    "explanation": "Sums all sales",
                    "used_datapoints": ["table_fact_sales_001"],
                    "confidence": 0.95,
                    "assumptions": [],
                    "clarifying_questions": [],
                },
                "target_database": "postgresql",
                "strict_mode": False,
            }
        }
    )


class ValidatorAgentOutput(AgentOutput):
    """
    Output from ValidatorAgent.

    Contains validation results with errors, warnings, and suggestions.
    """

    validated_sql: ValidatedSQL = Field(..., description="Validation results")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {},
                "metadata": {
                    "agent_name": "ValidatorAgent",
                    "duration_ms": 45.2,
                    "llm_calls": 0,  # ValidatorAgent doesn't use LLM
                },
                "next_agent": "ExecutorAgent",
                "validated_sql": {
                    "is_valid": True,
                    "sql": "SELECT SUM(amount) FROM fact_sales WHERE date >= '2024-01-01'",
                    "errors": [],
                    "warnings": [],
                    "suggestions": [],
                    "is_safe": True,
                    "performance_score": 0.95,
                },
            }
        }
    )
