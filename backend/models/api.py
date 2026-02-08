"""
API Request/Response Models

Pydantic models for FastAPI endpoints.
"""

from pydantic import BaseModel, Field

from backend.models.agent import EvidenceItem, SQLValidationError, ValidationWarning


class Message(BaseModel):
    """Chat message in conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User's natural language query")
    conversation_id: str | None = Field(None, description="Optional conversation ID for context")
    target_database: str | None = Field(
        None, description="Optional database connection ID to target"
    )
    conversation_history: list[Message] = Field(
        default_factory=list,
        description="Previous messages in the conversation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What's the total revenue?",
                "conversation_id": "conv_123",
                "target_database": "3a1f2d3e-4b5c-6d7e-8f90-1234567890ab",
                "conversation_history": [],
            }
        }
    }


class DataSource(BaseModel):
    """Information about a data source used to answer the query."""

    datapoint_id: str = Field(..., description="DataPoint ID")
    type: str = Field(..., description="DataPoint type (Schema, Business, Process)")
    name: str = Field(..., description="Human-readable name")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")


class ChatMetrics(BaseModel):
    """Performance metrics for the chat request."""

    total_latency_ms: float = Field(..., description="Total request latency in ms")
    agent_timings: dict[str, float] = Field(..., description="Per-agent execution times in ms")
    llm_calls: int = Field(..., description="Total number of LLM API calls")
    retry_count: int = Field(default=0, description="Number of SQL retries")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="Natural language answer to the query")
    clarifying_questions: list[str] = Field(
        default_factory=list,
        description="Clarifying questions when more detail is required",
    )
    sql: str | None = Field(None, description="Generated SQL query (if applicable)")
    data: dict[str, list] | None = Field(None, description="Query results in columnar format")
    visualization_hint: str | None = Field(None, description="Suggested visualization type")
    sources: list[DataSource] = Field(
        default_factory=list, description="Data sources used to answer"
    )
    answer_source: str | None = Field(
        default=None, description="Answer source (context|sql|error)"
    )
    answer_confidence: float | None = Field(
        default=None, description="Confidence score for the answer"
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list, description="Evidence items supporting the answer"
    )
    validation_errors: list[SQLValidationError] = Field(
        default_factory=list, description="SQL validation errors (if any)"
    )
    validation_warnings: list[ValidationWarning] = Field(
        default_factory=list, description="SQL validation warnings (if any)"
    )
    tool_approval_required: bool = Field(
        default=False, description="Whether tool execution needs approval"
    )
    tool_approval_message: str | None = Field(
        default=None, description="Approval request message"
    )
    tool_approval_calls: list[dict] = Field(
        default_factory=list, description="Tool calls requiring approval"
    )
    metrics: ChatMetrics | None = Field(None, description="Performance metrics")
    conversation_id: str | None = Field(None, description="Conversation ID for follow-up")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The total revenue is $1,234,567.89",
                "sql": "SELECT SUM(amount) as total_revenue FROM analytics.fact_sales WHERE status = 'completed'",
                "data": {"total_revenue": [1234567.89]},
                "visualization_hint": "none",
                "sources": [
                    {
                        "datapoint_id": "table_fact_sales_001",
                        "type": "Schema",
                        "name": "Fact Sales Table",
                        "relevance_score": 0.95,
                    }
                ],
                "answer_source": "sql",
                "answer_confidence": 0.92,
                "evidence": [
                    {
                        "datapoint_id": "table_fact_sales_001",
                        "name": "Fact Sales Table",
                        "type": "Schema",
                        "reason": "Used for SQL generation",
                    }
                ],
                "metrics": {
                    "total_latency_ms": 1523.45,
                    "agent_timings": {
                        "classifier": 234.5,
                        "context": 123.4,
                        "sql": 567.8,
                        "validator": 45.6,
                        "executor": 552.15,
                    },
                    "llm_calls": 3,
                    "retry_count": 0,
                },
                "conversation_id": "conv_123",
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoints."""

    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp (ISO 8601)")


class ReadinessResponse(BaseModel):
    """Response model for readiness check endpoint."""

    status: str = Field(..., description="Readiness status: 'ready' or 'not_ready'")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp (ISO 8601)")
    checks: dict[str, bool] = Field(
        ..., description="Individual readiness checks (db, vector_store, etc.)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ready",
                "version": "0.1.0",
                "timestamp": "2026-01-16T12:00:00Z",
                "checks": {
                    "database": True,
                    "vector_store": True,
                    "pipeline": True,
                },
            }
        }
    }


class SetupStep(BaseModel):
    """Setup step required to initialize the system."""

    step: str = Field(..., description="Setup step identifier")
    title: str = Field(..., description="Short step title")
    description: str = Field(..., description="Description of the setup step")
    action: str = Field(..., description="Suggested action key for clients")


class SystemStatusResponse(BaseModel):
    """Initialization status response."""

    is_initialized: bool = Field(
        ...,
        description=(
            "Whether the system can answer queries (target DB connected). "
            "DataPoints are optional enrichment."
        ),
    )
    has_databases: bool = Field(..., description="Whether a database connection is available")
    has_system_database: bool = Field(
        ..., description="Whether a system database is available for registry/profiling"
    )
    has_datapoints: bool = Field(..., description="Whether DataPoints are loaded")
    setup_required: list[SetupStep] = Field(
        default_factory=list,
        description="Remaining setup/recommended steps",
    )


class SystemInitializeRequest(BaseModel):
    """Initialization request payload."""

    database_url: str | None = Field(
        None, description="Database URL to use for initialization"
    )
    system_database_url: str | None = Field(
        None, description="System database URL for registry/profiling/demo"
    )
    auto_profile: bool = Field(
        default=False,
        description="Whether to auto-profile the database (not implemented yet)",
    )


class SystemInitializeResponse(BaseModel):
    """Initialization response payload."""

    message: str = Field(..., description="Initialization status message")
    is_initialized: bool = Field(
        ...,
        description=(
            "Whether the system can answer queries (target DB connected). "
            "DataPoints are optional enrichment."
        ),
    )
    has_databases: bool = Field(..., description="Whether a database connection is available")
    has_system_database: bool = Field(
        ..., description="Whether a system database is available for registry/profiling"
    )
    has_datapoints: bool = Field(..., description="Whether DataPoints are loaded")
    setup_required: list[SetupStep] = Field(
        default_factory=list,
        description="Remaining setup/recommended steps",
    )


class ToolExecuteRequest(BaseModel):
    """Tool execution request payload."""

    name: str = Field(..., description="Tool name")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")
    target_database: str | None = Field(
        default=None,
        description="Optional database connection ID to use for this tool call",
    )
    approved: bool = Field(default=False, description="Whether tool execution is approved")
    user_id: str | None = Field(default=None, description="Optional user ID")
    correlation_id: str | None = Field(
        default=None, description="Optional correlation ID for audit logging"
    )


class ToolExecuteResponse(BaseModel):
    """Tool execution response payload."""

    tool: str = Field(..., description="Tool name")
    success: bool = Field(..., description="Whether execution succeeded")
    result: dict | None = Field(None, description="Tool result payload")
    error: str | None = Field(None, description="Error message if execution failed")


class ToolInfo(BaseModel):
    """Tool definition summary."""

    name: str
    description: str
    category: str
    requires_approval: bool
    enabled: bool
    parameters_schema: dict


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    agent: str | None = Field(None, description="Agent that caused the error")
    recoverable: bool = Field(default=False, description="Whether the error is recoverable")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "agent_error",
                "message": "Failed to generate SQL query",
                "agent": "SQLAgent",
                "recoverable": True,
            }
        }
    }
