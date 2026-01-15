"""
Unit tests for PostgresConnector.

Tests the PostgreSQL connector with mocked asyncpg connections.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from backend.connectors.postgres import PostgresConnector
from backend.connectors.base import (
    ConnectionError,
    QueryError,
    SchemaError,
    QueryResult,
    TableInfo,
    ColumnInfo,
)


@pytest.fixture
def postgres_config():
    """PostgreSQL connection configuration."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "testdb",
        "user": "testuser",
        "password": "testpass",
        "pool_size": 5,
        "timeout": 30,
    }


@pytest.fixture
def mock_pool():
    """Mock asyncpg connection pool."""
    pool = AsyncMock()

    # Mock connection from pool
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value="PostgreSQL 15.0")
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()

    # Mock pool.acquire() context manager
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock()
    pool.close = AsyncMock()

    return pool, conn


class TestInitialization:
    """Test PostgresConnector initialization."""

    def test_initialization(self, postgres_config):
        """Test connector initializes with correct config."""
        connector = PostgresConnector(**postgres_config)

        assert connector.host == "localhost"
        assert connector.port == 5432
        assert connector.database == "testdb"
        assert connector.user == "testuser"
        assert connector.password == "testpass"
        assert connector.pool_size == 5
        assert connector.timeout == 30
        assert connector.is_connected is False

    def test_repr(self, postgres_config):
        """Test string representation."""
        connector = PostgresConnector(**postgres_config)
        repr_str = repr(connector)

        assert "PostgresConnector" in repr_str
        assert "testuser@localhost:5432/testdb" in repr_str
        assert "disconnected" in repr_str


class TestConnection:
    """Test connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self, postgres_config, mock_pool):
        """Test successful connection."""
        pool, conn = mock_pool

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            assert connector.is_connected is True
            assert connector._pool is not None
            conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, postgres_config, mock_pool):
        """Test connecting multiple times doesn't create multiple pools."""
        pool, _ = mock_pool

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)) as create_pool:
            connector = PostgresConnector(**postgres_config)
            await connector.connect()
            await connector.connect()  # Second call

            # Should only create pool once
            assert create_pool.call_count == 1

    @pytest.mark.asyncio
    async def test_connect_failure(self, postgres_config):
        """Test connection failure raises ConnectionError."""
        import asyncpg

        with patch(
            "asyncpg.create_pool",
            new=AsyncMock(side_effect=asyncpg.PostgresError("Connection refused")),
        ):
            connector = PostgresConnector(**postgres_config)

            with pytest.raises(ConnectionError, match="Failed to connect"):
                await connector.connect()

            assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_close(self, postgres_config, mock_pool):
        """Test closing connection."""
        pool, _ = mock_pool

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()
            await connector.close()

            assert connector.is_connected is False
            assert connector._pool is None
            pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, postgres_config):
        """Test closing multiple times is safe."""
        connector = PostgresConnector(**postgres_config)
        await connector.close()  # Close without connecting
        await connector.close()  # Close again

        # Should not raise


class TestQueryExecution:
    """Test query execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_query(self, postgres_config, mock_pool):
        """Test executing a simple query."""
        pool, conn = mock_pool

        # Mock query result
        mock_row = {"id": 1, "name": "Alice"}
        conn.fetch = AsyncMock(return_value=[mock_row])

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            result = await connector.execute("SELECT * FROM users")

            assert isinstance(result, QueryResult)
            assert result.row_count == 1
            assert result.rows == [mock_row]
            assert result.columns == ["id", "name"]
            assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_with_parameters(self, postgres_config, mock_pool):
        """Test executing query with parameters."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            await connector.execute(
                "SELECT * FROM users WHERE id = $1", params=[123]
            )

            # Verify fetch was called with unpacked params
            conn.fetch.assert_called_once()
            call_args = conn.fetch.call_args
            assert call_args[0][0] == "SELECT * FROM users WHERE id = $1"
            assert call_args[0][1] == 123

    @pytest.mark.asyncio
    async def test_execute_without_connection(self, postgres_config):
        """Test executing query without connection raises error."""
        connector = PostgresConnector(**postgres_config)

        with pytest.raises(ConnectionError, match="Not connected"):
            await connector.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_timeout(self, postgres_config, mock_pool):
        """Test query timeout."""
        pool, conn = mock_pool

        # Create a proper asyncpg exception mock
        from asyncpg.exceptions import QueryCanceledError as AsyncpgQueryCanceledError

        conn.fetch = AsyncMock(side_effect=AsyncpgQueryCanceledError("Timeout"))
        conn.execute = AsyncMock()

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            with pytest.raises(QueryError, match="Query timeout"):
                await connector.execute("SELECT pg_sleep(100)")

    @pytest.mark.asyncio
    async def test_execute_query_error(self, postgres_config, mock_pool):
        """Test query execution error."""
        pool, conn = mock_pool

        from asyncpg.exceptions import PostgresSyntaxError as AsyncpgSyntaxError

        conn.fetch = AsyncMock(side_effect=AsyncpgSyntaxError("Syntax error"))
        conn.execute = AsyncMock()

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            with pytest.raises(QueryError, match="Query execution failed"):
                await connector.execute("INVALID SQL")

    @pytest.mark.asyncio
    async def test_execute_custom_timeout(self, postgres_config, mock_pool):
        """Test query with custom timeout."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            await connector.execute("SELECT 1", timeout=60)

            # Verify timeout was set
            conn.execute.assert_called()
            call_args = conn.execute.call_args_list[0]
            assert "60000" in call_args[0][0]  # 60 seconds = 60000 ms


class TestSchemaIntrospection:
    """Test schema introspection."""

    @pytest.mark.asyncio
    async def test_get_schema_success(self, postgres_config, mock_pool):
        """Test successful schema introspection."""
        pool, conn = mock_pool

        # Mock tables query
        tables_result = [
            {"table_schema": "public", "table_name": "users", "table_type": "BASE TABLE"}
        ]

        # Mock columns query
        columns_result = [
            {
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
                "column_default": "nextval('users_id_seq'::regclass)",
            },
            {
                "column_name": "name",
                "data_type": "character varying",
                "is_nullable": "YES",
                "column_default": None,
            },
        ]

        # Mock primary keys query
        pk_result = [{"attname": "id"}]

        # Mock foreign keys query
        fk_result = []

        # Mock row count query
        row_count_result = 100

        # Setup fetch to return different results based on query
        async def mock_fetch(query, *args):
            if "information_schema.tables" in query:
                return tables_result
            elif "information_schema.columns" in query:
                return columns_result
            elif "pg_index" in query:
                return pk_result
            elif "FOREIGN KEY" in query:
                return fk_result
            return []

        conn.fetch = AsyncMock(side_effect=mock_fetch)
        conn.fetchval = AsyncMock(return_value=row_count_result)

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            tables = await connector.get_schema(schema_name="public")

            assert len(tables) == 1
            assert isinstance(tables[0], TableInfo)

            table = tables[0]
            assert table.schema == "public"
            assert table.table_name == "users"
            assert table.table_type == "BASE TABLE"
            assert len(table.columns) == 2
            assert table.row_count == 100

            # Check columns
            id_col = table.columns[0]
            assert id_col.name == "id"
            assert id_col.data_type == "integer"
            assert id_col.is_nullable is False
            assert id_col.is_primary_key is True

            name_col = table.columns[1]
            assert name_col.name == "name"
            assert name_col.is_nullable is True
            assert name_col.is_primary_key is False

    @pytest.mark.asyncio
    async def test_get_schema_without_connection(self, postgres_config):
        """Test schema introspection without connection."""
        connector = PostgresConnector(**postgres_config)

        with pytest.raises(ConnectionError, match="Not connected"):
            await connector.get_schema()

    @pytest.mark.asyncio
    async def test_get_schema_error(self, postgres_config, mock_pool):
        """Test schema introspection error."""
        pool, conn = mock_pool

        from asyncpg.exceptions import PostgresError as AsyncpgPostgresError

        conn.fetch = AsyncMock(side_effect=AsyncpgPostgresError("Schema error"))

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            connector = PostgresConnector(**postgres_config)
            await connector.connect()

            with pytest.raises(SchemaError, match="Failed to introspect schema"):
                await connector.get_schema()


class TestContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self, postgres_config, mock_pool):
        """Test using connector as async context manager."""
        pool, conn = mock_pool

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            async with PostgresConnector(**postgres_config) as connector:
                assert connector.is_connected is True

            # Should be closed after exiting context
            assert connector.is_connected is False
            pool.close.assert_called_once()
