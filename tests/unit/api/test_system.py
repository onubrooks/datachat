"""
Unit tests for system initialization endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.initialization.initializer import SetupStep, SystemStatus


class TestSystemEndpoints:
    """Test system status and initialization endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def not_initialized_status(self):
        return SystemStatus(
            is_initialized=False,
            has_databases=False,
            has_datapoints=False,
            setup_required=[
                SetupStep(
                    step="database_connection",
                    title="Connect a database",
                    description="Configure the database connection used for queries.",
                    action="configure_database",
                ),
                SetupStep(
                    step="datapoints",
                    title="Load DataPoints",
                    description="Add DataPoints describing your schema and business logic.",
                    action="load_datapoints",
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_status_returns_initialization_state(self, client, not_initialized_status):
        with patch(
            "backend.api.routes.system.SystemInitializer.status",
            new=AsyncMock(return_value=not_initialized_status),
        ):
            response = client.get("/api/v1/system/status")
            assert response.status_code == 200
            data = response.json()
            assert data["is_initialized"] is False
            assert data["has_databases"] is False
            assert data["has_datapoints"] is False
            assert len(data["setup_required"]) == 2

    @pytest.mark.asyncio
    async def test_status_returns_setup_steps(self, client, not_initialized_status):
        with patch(
            "backend.api.routes.system.SystemInitializer.status",
            new=AsyncMock(return_value=not_initialized_status),
        ):
            response = client.get("/api/v1/system/status")
            data = response.json()
            steps = {step["step"] for step in data["setup_required"]}
            assert "database_connection" in steps
            assert "datapoints" in steps

    @pytest.mark.asyncio
    async def test_initialize_validates_input(self, client):
        with patch(
            "backend.api.routes.system.SystemInitializer.initialize",
            new=AsyncMock(side_effect=Exception("Invalid database URL")),
        ):
            response = client.post(
                "/api/v1/system/initialize",
                json={"database_url": "not-a-url", "auto_profile": False},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_returns_empty_state_error(self, client, not_initialized_status):
        with patch(
            "backend.api.routes.chat.SystemInitializer.status",
            new=AsyncMock(return_value=not_initialized_status),
        ):
            response = client.post("/api/v1/chat", json={"message": "Test query"})
            assert response.status_code == 400
            data = response.json()
            assert data["error"] == "system_not_initialized"
            assert "setup_steps" in data
