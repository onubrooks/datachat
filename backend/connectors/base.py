"""
Base Database Connector

Abstract base class for all database connectors. Provides a consistent
async interface for connecting to, querying, and introspecting databases.

All connectors must implement:
- connect(): Establish connection with connection pooling
- execute(): Run queries with parameters and timeout
- get_schema(): Introspect database schema (tables, columns, types)
- close(): Clean up connections and pools
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class ColumnInfo(BaseModel):
    """Information about a database column."""

    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Column data type")
    is_nullable: bool = Field(..., description="Whether column can be NULL")
    default_value: Optional[str] = Field(None, description="Default value if any")
    is_primary_key: bool = Field(default=False, description="Is part of primary key")
    is_foreign_key: bool = Field(default=False, description="Is a foreign key")
    foreign_table: Optional[str] = Field(None, description="Referenced table if FK")
    foreign_column: Optional[str] = Field(None, description="Referenced column if FK")


class TableInfo(BaseModel):
    """Information about a database table."""

    schema: str = Field(..., description="Schema/database name")
    table_name: str = Field(..., description="Table name")
    columns: List[ColumnInfo] = Field(..., description="List of columns")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    table_type: str = Field(default="TABLE", description="TABLE, VIEW, etc.")


class QueryResult(BaseModel):
    """Result from query execution."""

    rows: List[Dict[str, Any]] = Field(..., description="Query result rows")
    row_count: int = Field(..., description="Number of rows returned")
    columns: List[str] = Field(..., description="Column names")
    execution_time_ms: float = Field(..., description="Query execution time in ms")


class ConnectorError(Exception):
    """Base exception for connector errors."""

    pass


class ConnectionError(ConnectorError):
    """Error establishing or managing database connection."""

    pass


class QueryError(ConnectorError):
    """Error executing database query."""

    pass


class SchemaError(ConnectorError):
    """Error introspecting database schema."""

    pass


# ============================================================================
# Base Connector
# ============================================================================


class BaseConnector(ABC):
    """
    Abstract base class for database connectors.

    All database connectors (PostgreSQL, ClickHouse, etc.) must implement
    this interface to ensure consistent behavior across the application.

    Features:
    - Async interface throughout
    - Connection pooling support
    - Query timeout configuration
    - Schema introspection
    - Parameterized query execution
    - Automatic resource cleanup

    Usage:
        class MyConnector(BaseConnector):
            async def connect(self):
                # Implementation
                pass

            async def execute(self, query, params=None):
                # Implementation
                pass

            # ... implement other methods

        connector = MyConnector(host="localhost", ...)
        await connector.connect()

        result = await connector.execute("SELECT * FROM users WHERE id = $1", [123])
        print(f"Found {result.row_count} rows")

        await connector.close()
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        pool_size: int = 10,
        timeout: int = 30,
        **kwargs,
    ):
        """
        Initialize connector.

        Args:
            host: Database host
            port: Database port
            database: Database/schema name
            user: Database user
            password: Database password
            pool_size: Connection pool size (default: 10)
            timeout: Query timeout in seconds (default: 30)
            **kwargs: Additional connector-specific parameters
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.timeout = timeout
        self.kwargs = kwargs

        self._pool = None
        self._connected = False

        logger.info(
            f"Initialized {self.__class__.__name__} for {user}@{host}:{port}/{database}"
        )

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish database connection and create connection pool.

        Should be idempotent - calling multiple times should not create
        multiple pools.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        timeout: Optional[int] = None,
    ) -> QueryResult:
        """
        Execute a SQL query.

        Args:
            query: SQL query string (use $1, $2 for parameters)
            params: Query parameters (optional)
            timeout: Query timeout in seconds (overrides default)

        Returns:
            QueryResult with rows, columns, and metadata

        Raises:
            QueryError: If query execution fails
            ConnectionError: If not connected
        """
        pass

    @abstractmethod
    async def get_schema(
        self, schema_name: Optional[str] = None
    ) -> List[TableInfo]:
        """
        Introspect database schema.

        Returns information about tables, columns, types, and relationships.

        Args:
            schema_name: Specific schema to introspect (None = all schemas)

        Returns:
            List of TableInfo objects

        Raises:
            SchemaError: If schema introspection fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close database connection and clean up pool.

        Should be idempotent - safe to call multiple times.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._connected

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        return f"<{self.__class__.__name__} {self.user}@{self.host}:{self.port}/{self.database} ({status})>"
