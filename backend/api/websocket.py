"""
WebSocket Routes

Real-time streaming WebSocket endpoint for chat with agent status updates.
"""

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time chat with streaming updates.

    Event Types:
        - agent_start: Agent begins execution
        - agent_complete: Agent finishes execution
        - data_chunk: Intermediate data from agent
        - answer_chunk: Streaming answer text
        - complete: Final response with all data
        - error: Error occurred during processing

    Message Format:
        Client -> Server:
        {
            "message": "What's the total revenue?",
            "conversation_id": "conv_123",  # optional
            "conversation_history": [...]    # optional
        }

        Server -> Client:
        {
            "event": "agent_start",
            "agent": "ClassifierAgent",
            "timestamp": "2026-01-16T12:00:00Z"
        }
        {
            "event": "agent_complete",
            "agent": "ClassifierAgent",
            "data": {...},
            "duration_ms": 234.5
        }
        {
            "event": "complete",
            "answer": "The total revenue is $1,234,567.89",
            "sql": "SELECT ...",
            "data": {...},
            "sources": [...],
            "metrics": {...},
            "conversation_id": "conv_123"
        }
    """
    from backend.api.main import app_state

    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        # Receive initial message
        data = await websocket.receive_json()
        logger.info(f"Received WebSocket message: {data.get('message', '')[:100]}...")

        # Validate required fields
        if "message" not in data:
            await websocket.send_json(
                {
                    "event": "error",
                    "error": "validation_error",
                    "message": "Missing required field: message",
                }
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Get pipeline from app state
        pipeline = app_state.get("pipeline")
        if pipeline is None:
            await websocket.send_json(
                {
                    "event": "error",
                    "error": "service_unavailable",
                    "message": "Pipeline not initialized. Please try again later.",
                }
            )
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # Extract request data
        message = data["message"]
        conversation_id = data.get("conversation_id") or f"conv_{uuid.uuid4().hex[:12]}"
        conversation_history = data.get("conversation_history", [])

        # Convert conversation history to pipeline format
        history = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in conversation_history
        ]

        # Define callback for streaming events
        async def event_callback(event_type: str, event_data: dict[str, Any]) -> None:
            """Send event to WebSocket client."""
            try:
                await websocket.send_json(
                    {
                        "event": event_type,
                        **event_data,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to send event {event_type}: {e}")

        # Run pipeline with streaming
        logger.info("Running pipeline with streaming...")
        result = await pipeline.run_with_streaming(
            query=message,
            conversation_history=history,
            event_callback=event_callback,
        )

        # Build final response
        answer = result.get("natural_language_answer")
        if not answer:
            if result.get("error"):
                answer = f"I encountered an error: {result.get('error')}"
            else:
                answer = "I was unable to process your query. Please try rephrasing."

        sql_query = result.get("validated_sql") or result.get("generated_sql")
        query_result = result.get("query_result")
        data_result = None
        if query_result and isinstance(query_result, dict):
            data_result = query_result.get("data")

        visualization_hint = result.get("visualization_hint")

        # Build sources
        sources = []
        retrieved_datapoints = result.get("retrieved_datapoints", [])
        for dp in retrieved_datapoints:
            if isinstance(dp, dict):
                sources.append(
                    {
                        "datapoint_id": dp.get("datapoint_id", "unknown"),
                        "type": dp.get("datapoint_type", dp.get("type", "unknown")),
                        "name": dp.get("name", "Unknown"),
                        "relevance_score": dp.get("score", 0.0),
                    }
                )
            else:
                sources.append(
                    {
                        "datapoint_id": getattr(dp, "datapoint_id", "unknown"),
                        "type": getattr(dp, "datapoint_type", "unknown"),
                        "name": getattr(dp, "name", "Unknown"),
                        "relevance_score": getattr(dp, "score", 0.0),
                    }
                )

        # Build metrics
        metrics = {
            "total_latency_ms": result.get("total_latency_ms", 0.0),
            "agent_timings": result.get("agent_timings", {}),
            "llm_calls": result.get("llm_calls", 0),
            "retry_count": result.get("retry_count", 0),
        }

        # Send final complete event
        await websocket.send_json(
            {
                "event": "complete",
                "answer": answer,
                "sql": sql_query,
                "data": data_result,
                "visualization_hint": visualization_hint,
                "sources": sources,
                "metrics": metrics,
                "conversation_id": conversation_id,
            }
        )

        logger.info(
            "WebSocket request completed successfully",
            extra={
                "conversation_id": conversation_id,
                "latency_ms": metrics["total_latency_ms"],
                "llm_calls": metrics["llm_calls"],
            },
        )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON received: {e}")
        try:
            await websocket.send_json(
                {
                    "event": "error",
                    "error": "invalid_json",
                    "message": "Invalid JSON format",
                }
            )
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {e}", exc_info=True)
        try:
            await websocket.send_json(
                {
                    "event": "error",
                    "error": "internal_error",
                    "message": f"An unexpected error occurred: {str(e)}",
                }
            )
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
