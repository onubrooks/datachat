"""Unit tests for database registry endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from backend.api.main import app
from backend.models.database import DatabaseConnection


class TestDatabaseEndpoints:
    """Test CRUD endpoints for database connections."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def sample_connection(self):
        return DatabaseConnection(
            connection_id=uuid4(),
            name="Warehouse",
            database_url=SecretStr("postgresql://user:pass@localhost:5432/warehouse"),
            database_type="postgresql",
            is_active=True,
            is_default=False,
            tags=["prod"],
            description="Primary warehouse",
            datapoint_count=0,
        )

    def test_create_connection(self, client, sample_connection):
        manager = AsyncMock()
        manager.add_connection = AsyncMock(return_value=sample_connection)

        with patch("backend.api.routes.databases._get_manager", return_value=manager):
            response = client.post(
                "/api/v1/databases",
                json={
                    "name": "Warehouse",
                    "database_url": "postgresql://user:pass@localhost:5432/warehouse",
                    "database_type": "postgresql",
                    "tags": ["prod"],
                    "description": "Primary warehouse",
                    "is_default": False,
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Warehouse"
        manager.add_connection.assert_awaited_once()

    def test_list_connections(self, client, sample_connection):
        manager = AsyncMock()
        manager.list_connections = AsyncMock(return_value=[sample_connection])

        with patch("backend.api.routes.databases._get_manager", return_value=manager):
            response = client.get("/api/v1/databases")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Warehouse"

    def test_get_connection(self, client, sample_connection):
        manager = AsyncMock()
        manager.get_connection = AsyncMock(return_value=sample_connection)

        with patch("backend.api.routes.databases._get_manager", return_value=manager):
            response = client.get(f"/api/v1/databases/{sample_connection.connection_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Warehouse"

    def test_get_connection_not_found(self, client):
        manager = AsyncMock()
        manager.get_connection = AsyncMock(side_effect=KeyError("Not found"))

        with patch("backend.api.routes.databases._get_manager", return_value=manager):
            response = client.get("/api/v1/databases/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404

    def test_set_default_connection(self, client, sample_connection):
        manager = AsyncMock()
        manager.set_default = AsyncMock(return_value=None)

        with patch("backend.api.routes.databases._get_manager", return_value=manager):
            response = client.put(
                f"/api/v1/databases/{sample_connection.connection_id}/default",
                json={"is_default": True},
            )

        assert response.status_code == 204
        manager.set_default.assert_awaited_once()

    def test_delete_connection(self, client, sample_connection):
        manager = AsyncMock()
        manager.remove_connection = AsyncMock(return_value=None)

        with patch("backend.api.routes.databases._get_manager", return_value=manager):
            response = client.delete(
                f"/api/v1/databases/{sample_connection.connection_id}"
            )

        assert response.status_code == 204
        manager.remove_connection.assert_awaited_once()

    def test_create_connection_validation_error(self, client):
        manager = AsyncMock()
        manager.add_connection = AsyncMock(side_effect=ValueError("Invalid URL"))

        with patch("backend.api.routes.databases._get_manager", return_value=manager):
            response = client.post(
                "/api/v1/databases",
                json={
                    "name": "Bad",
                    "database_url": "not-a-url",
                    "database_type": "postgresql",
                },
            )

        assert response.status_code == 400
