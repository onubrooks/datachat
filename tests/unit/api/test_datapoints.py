"""Unit tests for datapoint listing endpoints."""

from __future__ import annotations

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
