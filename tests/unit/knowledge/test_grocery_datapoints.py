"""Validation tests for grocery sample DataPoints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from backend.knowledge.datapoints import DataPointLoader

ROOT = Path(__file__).resolve().parents[3]
GROCERY_DATAPOINT_DIR = ROOT / "datapoints" / "examples" / "grocery_store"


def test_grocery_datapoints_load_successfully():
    loader = DataPointLoader()
    datapoints = loader.load_directory(GROCERY_DATAPOINT_DIR)
    stats = loader.get_stats()

    assert stats["failed_count"] == 0
    assert len(datapoints) == 13

    by_type = Counter(dp.type for dp in datapoints)
    assert by_type["Schema"] == 7
    assert by_type["Business"] == 4
    assert by_type["Process"] == 2


def test_grocery_datapoints_have_unique_ids():
    loader = DataPointLoader()
    datapoints = loader.load_directory(GROCERY_DATAPOINT_DIR)

    ids = [dp.datapoint_id for dp in datapoints]
    assert len(ids) == len(set(ids))
    assert "table_grocery_sales_transactions_001" in ids
    assert "metric_total_revenue_grocery_001" in ids
    assert "proc_nightly_inventory_snapshot_001" in ids
