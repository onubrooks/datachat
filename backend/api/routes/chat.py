"""
Chat Routes

FastAPI endpoints for natural language chat interface.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.initialization.initializer import SystemInitializer
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

        database_type = "postgresql"
        database_url = None
        manager = app_state.get("database_manager")
        if chat_request.target_database:
            if manager is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database registry is unavailable. Set DATABASE_CREDENTIALS_KEY.",
                )
            try:
                connection = await manager.get_connection(chat_request.target_database)
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
            database_type = connection.database_type
            database_url = connection.database_url.get_secret_value()
        elif manager is not None:
            default_connection = await manager.get_default_connection()
            if default_connection is not None:
                database_type = default_connection.database_type
                database_url = default_connection.database_url.get_secret_value()

        initializer = SystemInitializer(app_state)
        status_state = await initializer.status()
        if not status_state.is_initialized:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "system_not_initialized",
                    "message": (
                        "DataChat requires setup. Run 'datachat setup' or "
                        "'datachat demo' to get started."
                    ),
                    "setup_steps": [
                        {
                            "step": step.step,
                            "title": step.title,
                            "description": step.description,
                            "action": step.action,
                        }
                        for step in status_state.setup_required
                    ],
                },
            )

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
            database_type=database_type,
            database_url=database_url,
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
            if data is None:
                rows = query_result.get("rows")
                columns = query_result.get("columns")
                if isinstance(rows, list) and isinstance(columns, list):
                    data = {col: [row.get(col) for row in rows] for col in columns}

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
            answer_source=result.get("answer_source"),
            answer_confidence=result.get("answer_confidence"),
            evidence=_build_evidence(result),
            validation_errors=result.get("validation_errors", []),
            validation_warnings=result.get("validation_warnings", []),
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


def _build_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_items = []
    for item in result.get("evidence", []):
        if isinstance(item, dict):
            evidence_items.append(
                {
                    "datapoint_id": item.get("datapoint_id", "unknown"),
                    "name": item.get("name"),
                    "type": item.get("type"),
                    "reason": item.get("reason"),
                }
            )
    return evidence_items
