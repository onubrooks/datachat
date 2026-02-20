"""Unit tests for datapoint listing endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


class TestDatapointEndpoints:
    """Test datapoint listing behavior from vector-store state."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_list_datapoints_reads_from_vector_store(self, client):
        mock_store = AsyncMock()
        mock_store.list_datapoints.return_value = [
            {
                "datapoint_id": "table_grocery_stores_001",
                "metadata": {
                    "type": "Schema",
                    "name": "Grocery Stores",
                    "source_tier": "example",
                    "source_path": "/tmp/datapoints/examples/grocery_store/table_grocery_stores_001.json",
                },
            },
            {
                "datapoint_id": "metric_total_revenue_001",
                "metadata": {
                    "type": "Business",
                    "name": "Total Revenue",
                    "source_tier": "managed",
                    "source_path": "/tmp/datapoints/managed/metric_total_revenue_001.json",
                },
            },
        ]

        with patch("backend.api.routes.datapoints._get_vector_store", return_value=mock_store):
            response = client.get("/api/v1/datapoints")

        assert response.status_code == 200
        payload = response.json()
        assert len(payload["datapoints"]) == 2
        # Managed should be ranked ahead of example.
        assert payload["datapoints"][0]["datapoint_id"] == "metric_total_revenue_001"
        by_id = {item["datapoint_id"]: item for item in payload["datapoints"]}
        assert by_id["table_grocery_stores_001"]["source_tier"] == "example"
        assert by_id["metric_total_revenue_001"]["source_tier"] == "managed"
        assert by_id["table_grocery_stores_001"]["source_path"].endswith(
            "table_grocery_stores_001.json"
        )

    def test_list_datapoints_prefers_higher_source_tier_for_duplicate_ids(self, client):
        duplicate_id = "table_grocery_products_001"
        mock_store = AsyncMock()
        mock_store.list_datapoints.return_value = [
            {
                "datapoint_id": duplicate_id,
                "metadata": {
                    "type": "Schema",
                    "name": "Example Products",
                    "source_tier": "example",
                },
            },
            {
                "datapoint_id": duplicate_id,
                "metadata": {
                    "type": "Schema",
                    "name": "Managed Products",
                    "source_tier": "managed",
                },
            },
        ]

        with patch("backend.api.routes.datapoints._get_vector_store", return_value=mock_store):
            response = client.get("/api/v1/datapoints")

        assert response.status_code == 200
        datapoints = response.json()["datapoints"]
        assert len(datapoints) == 1
        assert datapoints[0]["datapoint_id"] == duplicate_id
        assert datapoints[0]["name"] == "Managed Products"
        assert datapoints[0]["source_tier"] == "managed"

    def test_trigger_sync_accepts_database_scope(self, client):
        orchestrator = AsyncMock()
        orchestrator.enqueue_sync_all.return_value = "11111111-1111-1111-1111-111111111111"

        with patch("backend.api.routes.datapoints._get_orchestrator", return_value=orchestrator):
            response = client.post(
                "/api/v1/sync",
                json={"scope": "database", "connection_id": "db-123"},
            )

        assert response.status_code == 200
        orchestrator.enqueue_sync_all.assert_called_once_with(
            scope="database", connection_id="db-123"
        )

    def test_trigger_sync_rejects_database_scope_without_connection(self, client):
        orchestrator = AsyncMock()

        with patch("backend.api.routes.datapoints._get_orchestrator", return_value=orchestrator):
            response = client.post("/api/v1/sync", json={"scope": "database"})

        assert response.status_code == 400
        assert "connection_id is required" in response.json()["detail"]

    def test_trigger_sync_rejects_connection_for_global_scope(self, client):
        orchestrator = AsyncMock()

        with patch("backend.api.routes.datapoints._get_orchestrator", return_value=orchestrator):
            response = client.post(
                "/api/v1/sync",
                json={"scope": "global", "connection_id": "db-123"},
            )

        assert response.status_code == 400
        assert "only allowed when scope=database" in response.json()["detail"]

    def test_get_datapoint_returns_managed_json(self, client, tmp_path):
        datapoint_payload = {
            "datapoint_id": "query_top_customers_001",
            "type": "Query",
            "name": "Top customers by revenue",
            "owner": "data-team@example.com",
            "tags": ["manual"],
            "metadata": {},
            "description": "Top customers by completed revenue.",
            "sql_template": "SELECT customer_id, SUM(amount) AS revenue FROM public.transactions GROUP BY customer_id LIMIT {limit}",
            "parameters": {
                "limit": {
                    "type": "integer",
                    "required": False,
                    "default": 20,
                    "description": "Max rows.",
                }
            },
            "related_tables": ["public.transactions"],
        }
        datapoint_path = tmp_path / "query_top_customers_001.json"
        datapoint_path.write_text(json.dumps(datapoint_payload), encoding="utf-8")

        with patch("backend.api.routes.datapoints._file_path", return_value=datapoint_path):
            response = client.get("/api/v1/datapoints/query_top_customers_001")

        assert response.status_code == 200
        payload = response.json()
        assert payload["datapoint_id"] == "query_top_customers_001"
        assert payload["type"] == "Query"
