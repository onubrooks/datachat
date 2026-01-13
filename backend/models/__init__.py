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

    DataPoint Models (TODO):
        - DataPoint: Base datapoint model
        - SchemaDataPoint: Table/column metadata
        - BusinessDataPoint: Business logic definitions
        - ProcessDataPoint: ETL process information

    API Models (TODO):
        - ChatRequest: API request model
        - ChatResponse: API response model

Usage:
    from backend.models.agent import AgentInput, AgentOutput, AgentError
    from backend.models import Message
"""

from backend.models.agent import (
    AgentError,
    AgentInput,
    AgentMetadata,
    AgentOutput,
    DatabaseError,
    LLMError,
    Message,
    RetrievalError,
    ValidationError,
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
]
