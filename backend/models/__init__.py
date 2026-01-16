"""
DataChat Models Module

Pydantic models for type-safe data validation throughout the application.

Available Models:
    Agent Models:
        - AgentInput: Base input for all agents
        - AgentOutput: Base output for all agents
        - AgentMetadata: Execution tracking metadata
        - Message: Conversation message
        - AgentError: Base exception for agent errors
        - ValidationError: Data validation errors
        - LLMError: LLM API errors
        - DatabaseError: Database operation errors
        - RetrievalError: Context retrieval errors

    DataPoint Models:
        - DataPoint: Discriminated union of all DataPoint types
        - BaseDataPoint: Base class with common fields
        - SchemaDataPoint: Table/column metadata
        - BusinessDataPoint: Business logic definitions
        - ProcessDataPoint: ETL process information
        - ColumnMetadata: Column information
        - Relationship: Table relationships

    API Models (TODO):
        - ChatRequest: API request model
        - ChatResponse: API response model

Usage:
    from backend.models.agent import AgentInput, AgentOutput, AgentError
    from backend.models.datapoint import DataPoint, SchemaDataPoint
    from backend.models import Message
"""

from backend.models.agent import (
    AgentError,
    AgentInput,
    AgentMetadata,
    AgentOutput,
    CorrectionAttempt,
    DatabaseError,
    GeneratedSQL,
    LLMError,
    Message,
    RetrievalError,
    # SQLAgent models
    SQLAgentInput,
    SQLAgentOutput,
    SQLGenerationError,
    # ValidatorAgent models
    SQLValidationError,
    ValidatedSQL,
    ValidationError,
    ValidationIssue,
    ValidationWarning,
    ValidatorAgentInput,
    ValidatorAgentOutput,
    # ClassifierAgent models
    ClassifierAgentInput,
    ClassifierAgentOutput,
    ExtractedEntity,
    QueryClassification,
    # ExecutorAgent models
    ExecutedQuery,
    ExecutorAgentInput,
    ExecutorAgentOutput,
    QueryResult,
)
from backend.models.datapoint import (
    BaseDataPoint,
    BusinessDataPoint,
    ColumnMetadata,
    DataPoint,
    ProcessDataPoint,
    Relationship,
    SchemaDataPoint,
)

__all__ = [
    # Core agent models
    "AgentInput",
    "AgentOutput",
    "AgentMetadata",
    "Message",
    # Error types
    "AgentError",
    "ValidationError",
    "LLMError",
    "DatabaseError",
    "RetrievalError",
    "SQLGenerationError",
    # SQLAgent models
    "SQLAgentInput",
    "SQLAgentOutput",
    "GeneratedSQL",
    "ValidationIssue",
    "CorrectionAttempt",
    # ValidatorAgent models
    "SQLValidationError",
    "ValidationWarning",
    "ValidatedSQL",
    "ValidatorAgentInput",
    "ValidatorAgentOutput",
    # ClassifierAgent models
    "ClassifierAgentInput",
    "ClassifierAgentOutput",
    "ExtractedEntity",
    "QueryClassification",
    # ExecutorAgent models
    "ExecutorAgentInput",
    "ExecutorAgentOutput",
    "ExecutedQuery",
    "QueryResult",
    # DataPoint models
    "DataPoint",
    "BaseDataPoint",
    "SchemaDataPoint",
    "BusinessDataPoint",
    "ProcessDataPoint",
    "ColumnMetadata",
    "Relationship",
]
