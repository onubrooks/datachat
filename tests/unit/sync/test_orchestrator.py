"""Unit tests for SyncOrchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.sync.orchestrator import SyncOrchestrator


def _schema_datapoint(datapoint_id: str) -> dict:
    return {
        "datapoint_id": datapoint_id,
        "type": "Schema",
        "name": "Orders",
        "owner": "data@example.com",
        "tags": [],
        "metadata": {},
        "table_name": "public.orders",
        "schema": "public",
        "business_purpose": "Orders table for testing.",
        "key_columns": [
            {
                "name": "order_id",
                "type": "integer",
                "business_meaning": "Order identifier",
                "nullable": False,
                "default_value": None,
            }
        ],
        "relationships": [],
        "common_queries": [],
        "gotchas": [],
        "freshness": None,
        "row_count": 100,
    }


def _write_datapoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


@pytest.mark.asyncio
async def test_full_sync_rebuilds_everything(tmp_path: Path):
    vector_store = AsyncMock()
    vector_store.clear = AsyncMock()
    vector_store.add_datapoints = AsyncMock()

    graph = MagicMock()
    graph.clear = MagicMock()
    graph.add_datapoint = MagicMock()

    datapoints_dir = tmp_path / "datapoints"
    _write_datapoint(datapoints_dir / "table_orders_001.json", _schema_datapoint("table_orders_001"))
    _write_datapoint(datapoints_dir / "table_customers_001.json", _schema_datapoint("table_customers_001"))

    orchestrator = SyncOrchestrator(
        vector_store=vector_store,
        knowledge_graph=graph,
        datapoints_dir=datapoints_dir,
    )

    job = await orchestrator.sync_all()

    vector_store.clear.assert_awaited_once()
    vector_store.add_datapoints.assert_awaited_once()
    graph.clear.assert_called_once()
    assert job.status == "completed"
    assert job.total_datapoints == 2


@pytest.mark.asyncio
async def test_incremental_sync_updates_specific_ids(tmp_path: Path):
    vector_store = AsyncMock()
    vector_store.delete = AsyncMock()
    vector_store.add_datapoints = AsyncMock()

    graph = MagicMock()
    graph.remove_datapoint = MagicMock()
    graph.add_datapoint = MagicMock()

    datapoints_dir = tmp_path / "datapoints"
    _write_datapoint(datapoints_dir / "table_orders_001.json", _schema_datapoint("table_orders_001"))
    _write_datapoint(datapoints_dir / "table_customers_001.json", _schema_datapoint("table_customers_001"))

    orchestrator = SyncOrchestrator(
        vector_store=vector_store,
        knowledge_graph=graph,
        datapoints_dir=datapoints_dir,
    )

    job = await orchestrator.sync_incremental(["table_orders_001"])

    vector_store.delete.assert_awaited_once_with(["table_orders_001"])
    graph.remove_datapoint.assert_called_once_with("table_orders_001")
    vector_store.add_datapoints.assert_awaited_once()
    assert job.status == "completed"


@pytest.mark.asyncio
async def test_status_tracking_updates(tmp_path: Path):
    vector_store = AsyncMock()
    vector_store.clear = AsyncMock()
    vector_store.add_datapoints = AsyncMock()

    graph = MagicMock()
    graph.clear = MagicMock()
    graph.add_datapoint = MagicMock()

    datapoints_dir = tmp_path / "datapoints"
    _write_datapoint(datapoints_dir / "table_orders_001.json", _schema_datapoint("table_orders_001"))

    orchestrator = SyncOrchestrator(
        vector_store=vector_store,
        knowledge_graph=graph,
        datapoints_dir=datapoints_dir,
    )

    job = await orchestrator.sync_all()

    status = orchestrator.get_status()
    assert status["status"] == "completed"
    assert status["total_datapoints"] == 1
    assert status["processed_datapoints"] == 1
    assert job.finished_at is not None


@pytest.mark.asyncio
async def test_full_sync_applies_database_scope_metadata(tmp_path: Path):
    vector_store = AsyncMock()
    vector_store.clear = AsyncMock()
    vector_store.add_datapoints = AsyncMock()

    graph = MagicMock()
    graph.clear = MagicMock()
    graph.add_datapoint = MagicMock()

    datapoints_dir = tmp_path / "datapoints"
    _write_datapoint(datapoints_dir / "table_orders_001.json", _schema_datapoint("table_orders_001"))

    orchestrator = SyncOrchestrator(
        vector_store=vector_store,
        knowledge_graph=graph,
        datapoints_dir=datapoints_dir,
    )

    await orchestrator.sync_all(scope="database", connection_id="conn-fintech")

    synced_datapoints = vector_store.add_datapoints.await_args.args[0]
    assert synced_datapoints[0].metadata["scope"] == "database"
    assert synced_datapoints[0].metadata["connection_id"] == "conn-fintech"
