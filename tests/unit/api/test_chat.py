"""
Unit Tests for Chat Endpoint

Tests the /api/v1/chat endpoint with mocked pipeline.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.initialization.initializer import SystemStatus


class TestChatEndpoint:
    """Test suite for chat endpoint."""

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

    @pytest.fixture
    def initialized_status(self):
        """Mock initialized system status."""
        return SystemStatus(
            is_initialized=True,
            has_databases=True,
            has_datapoints=True,
            setup_required=[],
        )

    @pytest.mark.asyncio
    async def test_chat_returns_200_for_valid_query(
        self, client, mock_pipeline_result, initialized_status
    ):
        """Test that chat endpoint returns 200 OK for valid query."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock
                mock_pipeline = AsyncMock()
                mock_pipeline.run = AsyncMock(return_value=mock_pipeline_result)
                mock_app_state.get.return_value = mock_pipeline

                # Make request
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "What's the total revenue?"},
                )

                # Assert
                assert response.status_code == 200
                assert mock_pipeline.run.called

    @pytest.mark.asyncio
    async def test_chat_returns_structured_response(
        self, client, mock_pipeline_result, initialized_status
    ):
        """Test that chat endpoint returns properly structured response."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock
                mock_pipeline = AsyncMock()
                mock_pipeline.run = AsyncMock(return_value=mock_pipeline_result)
                mock_app_state.get.return_value = mock_pipeline

                # Make request
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "What's the total revenue?"},
                )

                # Assert response structure
                data = response.json()
                assert "answer" in data
                assert "sql" in data
                assert "data" in data
                assert "visualization_hint" in data
                assert "sources" in data
                assert "metrics" in data
                assert "conversation_id" in data

                # Assert content
                assert data["answer"] == "The total revenue is $1,234,567.89"
                assert "SELECT SUM(amount)" in data["sql"]
                assert data["data"]["total_revenue"] == [1234567.89]
                assert len(data["sources"]) == 1
                assert data["sources"][0]["datapoint_id"] == "table_fact_sales_001"
                assert data["metrics"]["llm_calls"] == 3

    @pytest.mark.asyncio
    async def test_chat_handles_conversation_history(
        self, client, mock_pipeline_result, initialized_status
    ):
        """Test that chat endpoint passes conversation history to pipeline."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock
                mock_pipeline = AsyncMock()
                mock_pipeline.run = AsyncMock(return_value=mock_pipeline_result)
                mock_app_state.get.return_value = mock_pipeline

                # Make request with conversation history
                response = client.post(
                    "/api/v1/chat",
                    json={
                        "message": "What about last month?",
                        "conversation_id": "conv_123",
                        "conversation_history": [
                            {"role": "user", "content": "What's the revenue?"},
                            {"role": "assistant", "content": "The revenue is $1M"},
                        ],
                    },
                )

                # Assert
                assert response.status_code == 200
                assert mock_pipeline.run.called

                # Check that conversation history was passed
                call_args = mock_pipeline.run.call_args
                assert call_args.kwargs["query"] == "What about last month?"
                assert len(call_args.kwargs["conversation_history"]) == 2
                assert call_args.kwargs["conversation_history"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_chat_handles_errors_gracefully(self, client, initialized_status):
        """Test that chat endpoint handles pipeline errors gracefully."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock to raise exception
                mock_pipeline = AsyncMock()
                mock_pipeline.run = AsyncMock(side_effect=Exception("Pipeline failed"))
                mock_app_state.get.return_value = mock_pipeline

                # Make request
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "What's the total revenue?"},
                )

                # Assert error response
                assert response.status_code == 500
                data = response.json()
                assert "detail" in data
                assert "Pipeline failed" in data["detail"]

    @pytest.mark.asyncio
    async def test_chat_validates_request_body(self, client, initialized_status):
        """Test that chat endpoint validates request body."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            # Missing message field
            response = client.post(
                "/api/v1/chat",
                json={},
            )
            assert response.status_code == 422  # Unprocessable Entity

            # Empty message
            response = client.post(
                "/api/v1/chat",
                json={"message": ""},
            )
            assert response.status_code == 422

            # Invalid conversation history format
            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "test",
                    "conversation_history": "not a list",
                },
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_handles_pipeline_not_initialized(self, client, initialized_status):
        """Test that chat endpoint handles uninitialized pipeline."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock to return None (uninitialized)
                mock_app_state.get.return_value = None

                # Make request
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "What's the total revenue?"},
                )

                # Assert
                assert response.status_code == 503  # Service Unavailable
                data = response.json()
                assert "detail" in data
                assert "not initialized" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_chat_returns_fallback_answer_on_pipeline_error(
        self, client, initialized_status
    ):
        """Test that chat endpoint returns fallback answer when pipeline has errors."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock with error in result
                mock_pipeline = AsyncMock()
                mock_pipeline.run = AsyncMock(
                    return_value={
                        "error": "SQL generation failed",
                        "natural_language_answer": None,
                        "total_latency_ms": 500.0,
                        "agent_timings": {},
                        "llm_calls": 1,
                        "retry_count": 3,
                    }
                )
                mock_app_state.get.return_value = mock_pipeline

                # Make request
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "What's the total revenue?"},
                )

                # Assert fallback answer is returned
                assert response.status_code == 200
                data = response.json()
                assert "error" in data["answer"].lower()
                assert "SQL generation failed" in data["answer"]

    @pytest.mark.asyncio
    async def test_chat_generates_conversation_id_if_not_provided(
        self, client, mock_pipeline_result, initialized_status
    ):
        """Test that chat endpoint generates conversation ID if not provided."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock
                mock_pipeline = AsyncMock()
                mock_pipeline.run = AsyncMock(return_value=mock_pipeline_result)
                mock_app_state.get.return_value = mock_pipeline

                # Make request without conversation_id
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "What's the total revenue?"},
                )

                # Assert conversation_id is generated
                assert response.status_code == 200
                data = response.json()
                assert "conversation_id" in data
                assert data["conversation_id"].startswith("conv_")
                assert len(data["conversation_id"]) > 5

    @pytest.mark.asyncio
    async def test_chat_preserves_conversation_id_if_provided(
        self, client, mock_pipeline_result, initialized_status
    ):
        """Test that chat endpoint preserves provided conversation ID."""
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=initialized_status),
        ):
            with patch("backend.api.main.app_state") as mock_app_state:
                # Setup mock
                mock_pipeline = AsyncMock()
                mock_pipeline.run = AsyncMock(return_value=mock_pipeline_result)
                mock_app_state.get.return_value = mock_pipeline

                # Make request with conversation_id
                response = client.post(
                    "/api/v1/chat",
                    json={
                        "message": "What's the total revenue?",
                        "conversation_id": "my_custom_id",
                    },
                )

                # Assert conversation_id is preserved
                assert response.status_code == 200
                data = response.json()
                assert data["conversation_id"] == "my_custom_id"
