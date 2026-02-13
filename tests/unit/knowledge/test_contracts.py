"""Tests for DataPoint metadata contract validation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from backend.knowledge.contracts import validate_datapoint_contract
from backend.models.datapoint import DataPoint

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "datapoints"
EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "datapoints" / "examples"
datapoint_adapter = TypeAdapter(DataPoint)


def _load_fixture(name: str):
    with open(FIXTURES_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def test_contract_allows_fixture_datapoint_without_blocking_errors():
    payload = _load_fixture("metric_revenue_001.json")
    datapoint = datapoint_adapter.validate_python(payload)
    report = validate_datapoint_contract(datapoint)

    assert report.is_valid is True
    assert any(issue.code == "missing_grain" for issue in report.warnings)


def test_contract_requires_business_units():
    payload = _load_fixture("metric_revenue_001.json")
    payload.pop("unit", None)
    datapoint = datapoint_adapter.validate_python(payload)
    report = validate_datapoint_contract(datapoint)

    assert report.is_valid is False
    assert any(issue.code == "missing_units" for issue in report.errors)


def test_contract_requires_schema_freshness():
    payload = _load_fixture("table_fact_sales_001.json")
    payload.pop("freshness", None)
    datapoint = datapoint_adapter.validate_python(payload)
    report = validate_datapoint_contract(datapoint)

    assert report.is_valid is False
    assert any(issue.code == "missing_freshness" for issue in report.errors)


def test_contract_strict_escalates_advisory_gaps():
    payload = _load_fixture("proc_daily_etl_001.json")
    datapoint = datapoint_adapter.validate_python(payload)
    report = validate_datapoint_contract(datapoint, strict=True)

    assert report.is_valid is False
    assert any(issue.code == "missing_grain" for issue in report.errors)


def _validate_directory_without_warnings(directory: Path):
    reports = []
    for path in sorted(directory.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        datapoint = datapoint_adapter.validate_python(payload)
        reports.append(validate_datapoint_contract(datapoint))
    return reports


def test_grocery_example_contracts_have_no_warnings():
    reports = _validate_directory_without_warnings(EXAMPLES_DIR / "grocery_store")
    assert reports, "Expected grocery example datapoints"
    for report in reports:
        assert report.is_valid is True
        assert report.warnings == []


def test_fintech_example_contracts_have_no_warnings():
    reports = _validate_directory_without_warnings(EXAMPLES_DIR / "fintech_bank")
    assert reports, "Expected fintech example datapoints"
    for report in reports:
        assert report.is_valid is True
        assert report.warnings == []
