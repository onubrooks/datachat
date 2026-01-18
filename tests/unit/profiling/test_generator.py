"""Unit tests for DataPointGenerator."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.llm.models import LLMResponse
from backend.profiling.generator import DataPointGenerator
from backend.profiling.models import ColumnProfile, DatabaseProfile, TableProfile


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = responses
        self._calls = 0

    async def generate(self, _request):
        content = self._responses[self._calls]
        self._calls += 1
        return LLMResponse(
            content=content,
            model="mock",
            provider="mock",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
        )


def _sample_profile():
    table = TableProfile(
        schema="public",
        name="orders",
        row_count=100,
        columns=[
            ColumnProfile(
                name="order_id",
                data_type="integer",
                nullable=False,
                sample_values=["1", "2"],
            ),
            ColumnProfile(
                name="total_amount",
                data_type="numeric",
                nullable=False,
                sample_values=["10.5", "20.0"],
            ),
            ColumnProfile(
                name="created_at",
                data_type="timestamp",
                nullable=False,
                sample_values=["2024-01-01"],
            ),
        ],
        relationships=[],
        sample_size=2,
    )
    return DatabaseProfile(
        profile_id=uuid4(),
        connection_id=uuid4(),
        tables=[table],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_generates_schema_datapoints_from_profile():
    profile = _sample_profile()
    llm = FakeLLM(
        [
            '{"business_purpose": "Track customer orders", "columns": {"order_id": "Order id", "total_amount": "Total", "created_at": "Date"}, "common_queries": ["SUM(total_amount)"], "gotchas": [], "freshness": "T-1", "confidence": 0.8}',
            '{"metrics": []}',
        ]
    )

    generator = DataPointGenerator(llm_provider=llm)
    generated = await generator.generate_from_profile(profile)

    assert generated.schema_datapoints
    datapoint = generated.schema_datapoints[0].datapoint
    assert datapoint["type"] == "Schema"
    assert "Track customer orders" in datapoint["business_purpose"]


@pytest.mark.asyncio
async def test_suggests_metrics_from_numeric_columns():
    profile = _sample_profile()
    llm = FakeLLM(
        [
            '{"business_purpose": "Orders", "columns": {"order_id": "Order id", "total_amount": "Total", "created_at": "Date"}}',
            '{"metrics": [{"name": "Total Order Value", "calculation": "SUM(total_amount)", "aggregation": "SUM", "unit": "USD", "confidence": 0.75}]}',
        ]
    )

    generator = DataPointGenerator(llm_provider=llm)
    generated = await generator.generate_from_profile(profile)

    assert generated.business_datapoints
    metric = generated.business_datapoints[0].datapoint
    assert metric["type"] == "Business"
    assert metric["aggregation"] == "SUM"


@pytest.mark.asyncio
async def test_identifies_time_series_patterns():
    profile = _sample_profile()
    llm = FakeLLM(
        [
            '{"business_purpose": "Orders", "columns": {"order_id": "Order id", "total_amount": "Total", "created_at": "Date"}}',
            '{"metrics": []}',
        ]
    )

    generator = DataPointGenerator(llm_provider=llm)
    generated = await generator.generate_from_profile(profile)

    datapoint = generated.schema_datapoints[0].datapoint
    common_queries = datapoint.get("common_queries", [])
    assert any("DATE_TRUNC" in query for query in common_queries)


@pytest.mark.asyncio
async def test_returns_confidence_scores():
    profile = _sample_profile()
    llm = FakeLLM(
        [
            '{"business_purpose": "Orders", "columns": {"order_id": "Order id", "total_amount": "Total", "created_at": "Date"}, "confidence": 0.9}',
            '{"metrics": []}',
        ]
    )

    generator = DataPointGenerator(llm_provider=llm)
    generated = await generator.generate_from_profile(profile)

    assert generated.schema_datapoints[0].confidence == 0.9
