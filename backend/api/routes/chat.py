"""
Chat Routes

FastAPI endpoints for natural language chat interface.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from backend.models.api import ChatMetrics, ChatRequest, ChatResponse, DataSource

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    Process a natural language query and return structured response.

    Args:
        chat_request: User's message with optional conversation context

    Returns:
        ChatResponse with answer, SQL, data, and metadata

    Raises:
        HTTPException: If pipeline fails or is not initialized
    """
    logger.info(f"Chat request received: {chat_request.message[:100]}...")

    try:
        # Get pipeline from app state
        from backend.api.main import app_state

        pipeline = app_state.get("pipeline")
        if pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        # Convert conversation history to pipeline format
        conversation_history = [
            {"role": msg.role, "content": msg.content} for msg in chat_request.conversation_history
        ]

        # Run pipeline
        logger.info("Running pipeline...")
        result = await pipeline.run(
            query=chat_request.message,
            conversation_history=conversation_history,
        )

        # Extract data from pipeline state
        answer = result.get("natural_language_answer")
        if not answer:
            # Fallback if no answer generated
            if result.get("error"):
                answer = f"I encountered an error: {result.get('error')}"
            else:
                answer = "I was unable to process your query. Please try rephrasing."

        # Extract SQL
        sql_query = result.get("validated_sql") or result.get("generated_sql")

        # Extract query results
        query_result = result.get("query_result")
        data = None
        if query_result and isinstance(query_result, dict):
            data = query_result.get("data")

        # Extract visualization hint
        visualization_hint = result.get("visualization_hint")

        # Build sources from retrieved datapoints
        sources = _build_sources(result)

        # Build metrics
        metrics = _build_metrics(result)

        # Generate or use conversation ID
        conversation_id = chat_request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

        response = ChatResponse(
            answer=answer,
            sql=sql_query,
            data=data,
            visualization_hint=visualization_hint,
            sources=sources,
            metrics=metrics,
            conversation_id=conversation_id,
        )

        logger.info(
            "Chat request completed successfully",
            extra={
                "conversation_id": conversation_id,
                "latency_ms": metrics.total_latency_ms if metrics else None,
                "llm_calls": metrics.llm_calls if metrics else None,
            },
        )

        return response

    except RuntimeError as e:
        logger.error(f"Pipeline not initialized: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized. Please try again later.",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        ) from e


def _build_sources(result: dict[str, Any]) -> list[DataSource]:
    """Build DataSource list from pipeline result."""
    sources = []

    # Get retrieved datapoints from context agent
    retrieved_datapoints = result.get("retrieved_datapoints", [])

    for dp in retrieved_datapoints:
        # Handle both dict and object formats
        if isinstance(dp, dict):
            sources.append(
                DataSource(
                    datapoint_id=dp.get("datapoint_id", "unknown"),
                    type=dp.get("datapoint_type", dp.get("type", "unknown")),
                    name=dp.get("name", "Unknown"),
                    relevance_score=dp.get("score", 0.0),
                )
            )
        else:
            # Handle Pydantic model
            sources.append(
                DataSource(
                    datapoint_id=getattr(dp, "datapoint_id", "unknown"),
                    type=getattr(dp, "datapoint_type", "unknown"),
                    name=getattr(dp, "name", "Unknown"),
                    relevance_score=getattr(dp, "score", 0.0),
                )
            )

    return sources


def _build_metrics(result: dict[str, Any]) -> ChatMetrics | None:
    """Build ChatMetrics from pipeline result."""
    try:
        return ChatMetrics(
            total_latency_ms=result.get("total_latency_ms", 0.0),
            agent_timings=result.get("agent_timings", {}),
            llm_calls=result.get("llm_calls", 0),
            retry_count=result.get("retry_count", 0),
        )
    except Exception as e:
        logger.warning(f"Failed to build metrics: {e}")
        return None
