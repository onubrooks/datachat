"""Validation tests for fintech sample DataPoints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from backend.knowledge.datapoints import DataPointLoader

ROOT = Path(__file__).resolve().parents[3]
FINTECH_DATAPOINT_DIR = ROOT / "datapoints" / "examples" / "fintech_bank"


def test_fintech_datapoints_load_successfully():
    loader = DataPointLoader()
    datapoints = loader.load_directory(FINTECH_DATAPOINT_DIR)
    stats = loader.get_stats()

    assert stats["failed_count"] == 0
    assert len(datapoints) == 13

    by_type = Counter(dp.type for dp in datapoints)
    assert by_type["Schema"] == 7
    assert by_type["Business"] == 4
    assert by_type["Process"] == 2


def test_fintech_datapoints_have_unique_ids():
    loader = DataPointLoader()
    datapoints = loader.load_directory(FINTECH_DATAPOINT_DIR)

    ids = [dp.datapoint_id for dp in datapoints]
    assert len(ids) == len(set(ids))
    assert "table_bank_transactions_001" in ids
    assert "metric_total_deposits_bank_001" in ids
    assert "proc_nightly_credit_risk_snapshot_bank_001" in ids
