"""Built-in database tools."""

from __future__ import annotations

from typing import Any

from urllib.parse import urlparse

from backend.connectors.base import BaseConnector
from backend.connectors.clickhouse import ClickHouseConnector
from backend.connectors.postgres import PostgresConnector
from backend.database.manager import DatabaseConnectionManager
from backend.tools.base import ToolCategory, ToolContext, tool


async def _get_default_connector() -> BaseConnector:
    manager = DatabaseConnectionManager()
    await manager.initialize()
    try:
        connection = await manager.get_default_connection()
        if not connection:
            raise ValueError("No default database connection configured.")
        database_url = connection.database_url.get_secret_value()
        parsed = urlparse(database_url)
        scheme = parsed.scheme.split("+")[0]
        if connection.database_type == "clickhouse" or scheme == "clickhouse":
            connector = ClickHouseConnector(
                host=parsed.hostname or "localhost",
                port=parsed.port or 8123,
                database=parsed.path.lstrip("/") if parsed.path else "default",
                user=parsed.username or "default",
                password=parsed.password or "",
            )
        else:
            connector = PostgresConnector(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") if parsed.path else "datachat",
                user=parsed.username or "postgres",
                password=parsed.password or "",
            )
        await connector.connect()
        return connector
    finally:
        await manager.close()


@tool(
    name="list_tables",
    description="List tables available in the target database.",
    category=ToolCategory.DATABASE,
)
async def list_tables(schema: str | None = None, ctx: ToolContext | None = None) -> dict[str, Any]:
    connector = await _get_default_connector()
    try:
        schema_rows = await connector.get_schema(schema_name=schema)
        return {
            "tables": [
                {
                    "schema": table.schema_name,
                    "table": table.table_name,
                    "row_count": table.row_count,
                    "table_type": table.table_type,
                }
                for table in schema_rows
            ]
        }
    finally:
        await connector.close()


@tool(
    name="list_columns",
    description="List columns for a given table.",
    category=ToolCategory.DATABASE,
)
async def list_columns(
    table: str, schema: str | None = None, ctx: ToolContext | None = None
) -> dict[str, Any]:
    connector = await _get_default_connector()
    try:
        schema_rows = await connector.get_schema(schema_name=schema)
        match = None
        for table_info in schema_rows:
            if table_info.table_name == table:
                match = table_info
                break
        if not match:
            raise ValueError(f"Table not found: {table}")
        return {
            "table": f"{match.schema_name}.{match.table_name}",
            "columns": [
                {
                    "name": col.name,
                    "type": col.data_type,
                    "nullable": col.is_nullable,
                }
                for col in match.columns
            ],
        }
    finally:
        await connector.close()


@tool(
    name="get_table_sample",
    description="Fetch a small sample of rows from a table.",
    category=ToolCategory.DATABASE,
)
async def get_table_sample(
    table: str,
    schema: str | None = None,
    limit: int = 5,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    connector = await _get_default_connector()
    try:
        schema_prefix = f"{schema}." if schema else ""
        query = f"SELECT * FROM {schema_prefix}{table} LIMIT {limit}"
        result = await connector.execute(query)
        return {
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
        }
    finally:
        await connector.close()
