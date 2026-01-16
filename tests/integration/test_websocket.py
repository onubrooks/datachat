"""
Integration Tests for WebSocket Streaming

Tests the /ws/chat WebSocket endpoint with real-time streaming.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.mark.integration
class TestWebSocketStreaming:
    """Integration tests for WebSocket streaming endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_pipeline_result(self):
        """Mock successful pipeline result."""
        return {
            "query": "What's the total revenue?",
            "natural_language_answer": "The total revenue is $1,234,567.89",
            "validated_sql": "SELECT SUM(amount) as total_revenue FROM analytics.fact_sales WHERE status = 'completed'",
            "generated_sql": "SELECT SUM(amount) as total_revenue FROM analytics.fact_sales WHERE status = 'completed'",
            "query_result": {
                "data": {"total_revenue": [1234567.89]},
                "row_count": 1,
                "column_names": ["total_revenue"],
            },
            "visualization_hint": "none",
            "retrieved_datapoints": [
                {
                    "datapoint_id": "table_fact_sales_001",
                    "datapoint_type": "Schema",
                    "name": "Fact Sales Table",
                    "score": 0.95,
                }
            ],
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
            "error": None,
        }

    def test_websocket_connects_successfully(self, client):
        """Test that WebSocket connection is established successfully."""
        with patch("backend.api.main.app_state") as mock_app_state:
            mock_pipeline = AsyncMock()
            mock_pipeline.run_with_streaming = AsyncMock(
                return_value={
                    "natural_language_answer": "Test answer",
                    "total_latency_ms": 1000.0,
                    "agent_timings": {},
                    "llm_calls": 1,
                    "retry_count": 0,
                }
            )
            mock_app_state.get.return_value = mock_pipeline

            with client.websocket_connect("/ws/chat") as websocket:
                # Send message
                websocket.send_json({"message": "Test query"})

                # Should receive complete event eventually
                messages = []
                try:
                    while True:
                        data = websocket.receive_json()
                        messages.append(data)
                        if data.get("event") == "complete":
                            break
                except Exception:
                    pass

                # Verify we got the complete event
                assert any(msg.get("event") == "complete" for msg in messages)

    def test_websocket_receives_agent_status_events(self, client):
        """Test that WebSocket receives agent_start and agent_complete events."""
        with patch("backend.api.main.app_state") as mock_app_state:
            # Create mock pipeline that will call the callback
            async def mock_run_with_streaming(query, conversation_history, event_callback):
                # Simulate agent events
                await event_callback(
                    "agent_start", {"agent": "ClassifierAgent", "timestamp": "2026-01-16T12:00:00Z"}
                )
                await event_callback(
                    "agent_complete",
                    {
                        "agent": "ClassifierAgent",
                        "data": {"intent": "data_query"},
                        "duration_ms": 234.5,
                        "timestamp": "2026-01-16T12:00:00Z",
                    },
                )
                return {
                    "natural_language_answer": "Test answer",
                    "total_latency_ms": 1000.0,
                    "agent_timings": {},
                    "llm_calls": 1,
                    "retry_count": 0,
                    "retrieved_datapoints": [],
                }

            mock_pipeline = AsyncMock()
            mock_pipeline.run_with_streaming = mock_run_with_streaming
            mock_app_state.get.return_value = mock_pipeline

            with client.websocket_connect("/ws/chat") as websocket:
                # Send message
                websocket.send_json({"message": "Test query"})

                # Collect all messages
                messages = []
                try:
                    while True:
                        data = websocket.receive_json()
                        messages.append(data)
                        if data.get("event") == "complete":
                            break
                except Exception:
                    pass

                # Verify we got agent events
                assert any(
                    msg.get("event") == "agent_start" and msg.get("agent") == "ClassifierAgent"
                    for msg in messages
                )
                assert any(
                    msg.get("event") == "agent_complete" and msg.get("agent") == "ClassifierAgent"
                    for msg in messages
                )

    def test_websocket_final_message_contains_complete_response(self, client, mock_pipeline_result):
        """Test that final WebSocket message contains complete response."""
        with patch("backend.api.main.app_state") as mock_app_state:
            mock_pipeline = AsyncMock()
            mock_pipeline.run_with_streaming = AsyncMock(return_value=mock_pipeline_result)
            mock_app_state.get.return_value = mock_pipeline

            with client.websocket_connect("/ws/chat") as websocket:
                # Send message
                websocket.send_json({"message": "What's the total revenue?"})

                # Collect all messages until complete
                final_message = None
                try:
                    while True:
                        data = websocket.receive_json()
                        if data.get("event") == "complete":
                            final_message = data
                            break
                except Exception:
                    pass

                # Verify final message has all required fields
                assert final_message is not None
                assert final_message["event"] == "complete"
                assert "answer" in final_message
                assert "sql" in final_message
                assert "data" in final_message
                assert "sources" in final_message
                assert "metrics" in final_message
                assert "conversation_id" in final_message

                # Verify content
                assert final_message["answer"] == "The total revenue is $1,234,567.89"
                assert "SELECT SUM(amount)" in final_message["sql"]
                assert final_message["data"]["total_revenue"] == [1234567.89]
                assert len(final_message["sources"]) == 1
                assert final_message["metrics"]["llm_calls"] == 3

    def test_websocket_handles_client_disconnect(self, client):
        """Test that WebSocket handles client disconnect gracefully."""
        with patch("backend.api.main.app_state") as mock_app_state:
            mock_pipeline = AsyncMock()
            mock_pipeline.run_with_streaming = AsyncMock(
                return_value={
                    "natural_language_answer": "Test answer",
                    "total_latency_ms": 1000.0,
                    "agent_timings": {},
                    "llm_calls": 1,
                    "retry_count": 0,
                }
            )
            mock_app_state.get.return_value = mock_pipeline

            # Connect and immediately disconnect
            with client.websocket_connect("/ws/chat") as websocket:
                websocket.send_json({"message": "Test query"})
                # Connection will be closed when exiting context

            # Should not raise an exception
            assert True

    def test_websocket_validates_request_message(self, client):
        """Test that WebSocket validates incoming message."""
        with patch("backend.api.main.app_state") as mock_app_state:
            mock_pipeline = AsyncMock()
            mock_app_state.get.return_value = mock_pipeline

            with client.websocket_connect("/ws/chat") as websocket:
                # Send invalid message (missing required field)
                websocket.send_json({})

                # Should receive error event
                data = websocket.receive_json()
                assert data["event"] == "error"
                assert data["error"] == "validation_error"

    def test_websocket_handles_pipeline_not_initialized(self, client):
        """Test that WebSocket handles uninitialized pipeline."""
        with patch("backend.api.main.app_state") as mock_app_state:
            # Pipeline not initialized
            mock_app_state.get.return_value = None

            with client.websocket_connect("/ws/chat") as websocket:
                # Send message
                websocket.send_json({"message": "Test query"})

                # Should receive error event
                data = websocket.receive_json()
                assert data["event"] == "error"
                assert data["error"] == "service_unavailable"

    def test_websocket_supports_conversation_id(self, client):
        """Test that WebSocket preserves conversation_id."""
        with patch("backend.api.main.app_state") as mock_app_state:
            mock_pipeline = AsyncMock()
            mock_pipeline.run_with_streaming = AsyncMock(
                return_value={
                    "natural_language_answer": "Test answer",
                    "total_latency_ms": 1000.0,
                    "agent_timings": {},
                    "llm_calls": 1,
                    "retry_count": 0,
                    "retrieved_datapoints": [],
                }
            )
            mock_app_state.get.return_value = mock_pipeline

            with client.websocket_connect("/ws/chat") as websocket:
                # Send message with conversation_id
                websocket.send_json({"message": "Test query", "conversation_id": "conv_custom_123"})

                # Get final message
                final_message = None
                try:
                    while True:
                        data = websocket.receive_json()
                        if data.get("event") == "complete":
                            final_message = data
                            break
                except Exception:
                    pass

                # Verify conversation_id is preserved
                assert final_message is not None
                assert final_message["conversation_id"] == "conv_custom_123"

    def test_websocket_generates_conversation_id_if_not_provided(self, client):
        """Test that WebSocket generates conversation_id if not provided."""
        with patch("backend.api.main.app_state") as mock_app_state:
            mock_pipeline = AsyncMock()
            mock_pipeline.run_with_streaming = AsyncMock(
                return_value={
                    "natural_language_answer": "Test answer",
                    "total_latency_ms": 1000.0,
                    "agent_timings": {},
                    "llm_calls": 1,
                    "retry_count": 0,
                    "retrieved_datapoints": [],
                }
            )
            mock_app_state.get.return_value = mock_pipeline

            with client.websocket_connect("/ws/chat") as websocket:
                # Send message without conversation_id
                websocket.send_json({"message": "Test query"})

                # Get final message
                final_message = None
                try:
                    while True:
                        data = websocket.receive_json()
                        if data.get("event") == "complete":
                            final_message = data
                            break
                except Exception:
                    pass

                # Verify conversation_id is generated
                assert final_message is not None
                assert "conversation_id" in final_message
                assert final_message["conversation_id"].startswith("conv_")
