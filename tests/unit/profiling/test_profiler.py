"""Unit tests for SchemaProfiler."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from backend.database.manager import DatabaseConnectionManager
from backend.models.database import DatabaseConnection
from backend.profiling.profiler import SchemaProfiler


class TestSchemaProfiler:
    """Test SchemaProfiler behavior with mocked asyncpg."""

    @pytest.fixture
    def connection(self):
        return DatabaseConnection(
            connection_id=uuid4(),
            name="Warehouse",
            database_url=SecretStr("postgresql://user:pass@localhost:5432/warehouse"),
            database_type="postgresql",
            is_active=True,
            is_default=True,
            tags=[],
            description=None,
            datapoint_count=0,
        )

    @pytest.fixture
    def manager(self, connection):
        manager = AsyncMock(spec=DatabaseConnectionManager)
        manager.get_connection = AsyncMock(return_value=connection)
        return manager

    def _build_connection(self):
        conn = AsyncMock()

        async def fetch(query, *args):
            if "information_schema.tables" in query:
                return [
                    {"table_schema": "public", "table_name": "orders"},
                ]
            if "information_schema.columns" in query:
                return [
                    {
                        "column_name": "order_id",
                        "data_type": "integer",
                        "is_nullable": "NO",
                        "column_default": None,
                    },
                    {
                        "column_name": "created_at",
                        "data_type": "timestamp",
                        "is_nullable": "YES",
                        "column_default": None,
                    },
                ]
            if "information_schema.table_constraints" in query:
                return [
                    {
                        "source_table": "orders",
                        "source_column": "customer_id",
                        "target_table": "customers",
                        "target_column": "id",
                    }
                ]
            if "SELECT" in query and "LIMIT" in query:
                return [{"value": "sample"}, {"value": "sample2"}]
            return []

        async def fetchrow(query, *args):
            if "pg_class" in query:
                return {"estimate": 1234}
            if "COUNT" in query:
                return {
                    "null_count": 1,
                    "distinct_count": 2,
                    "min_value": "1",
                    "max_value": "5",
                }
            return None

        conn.fetch.side_effect = fetch
        conn.fetchrow.side_effect = fetchrow
        return conn

    @pytest.mark.asyncio
    async def test_profiles_postgres_schema(self, manager):
        profiler = SchemaProfiler(manager)
        mock_connection = self._build_connection()

        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_connection)):
            profile = await profiler.profile_database(str(uuid4()), sample_size=2)

        assert profile.tables[0].name == "orders"
        assert profile.tables[0].row_count == 1234
        assert len(profile.tables[0].columns) == 2

    @pytest.mark.asyncio
    async def test_samples_data_with_correct_size(self, manager):
        profiler = SchemaProfiler(manager)
        mock_connection = self._build_connection()

        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_connection)):
            profile = await profiler.profile_database(str(uuid4()), sample_size=2)

        samples = profile.tables[0].columns[0].sample_values
        assert len(samples) == 2

    @pytest.mark.asyncio
    async def test_discovers_foreign_keys(self, manager):
        profiler = SchemaProfiler(manager)
        mock_connection = self._build_connection()

        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_connection)):
            profile = await profiler.profile_database(str(uuid4()), sample_size=2)

        relationships = profile.tables[0].relationships
        assert relationships
        assert relationships[0].target_table == "customers"

    @pytest.mark.asyncio
    async def test_calculates_statistics(self, manager):
        profiler = SchemaProfiler(manager)
        mock_connection = self._build_connection()

        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_connection)):
            profile = await profiler.profile_database(str(uuid4()), sample_size=2)

        column = profile.tables[0].columns[0]
        assert column.null_count == 1
        assert column.distinct_count == 2
        assert column.min_value == "1"
        assert column.max_value == "5"
