"""Schema profiling utilities."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

import asyncpg

from backend.database.manager import DatabaseConnectionManager
from backend.profiling.models import (
    ColumnProfile,
    DatabaseProfile,
    RelationshipProfile,
    TableProfile,
)


class SchemaProfiler:
    """Profiles database schemas and samples data."""

    def __init__(self, manager: DatabaseConnectionManager) -> None:
        self._manager = manager

    async def profile_database(
        self,
        connection_id: str,
        sample_size: int = 100,
        tables: Sequence[str] | None = None,
        progress_callback: callable | None = None,
    ) -> DatabaseProfile:
        connection = await self._manager.get_connection(connection_id)
        db_url = connection.database_url.get_secret_value()
        parsed = urlparse(db_url.replace("postgresql+asyncpg://", "postgresql://"))
        if parsed.scheme not in {"postgres", "postgresql"}:
            raise ValueError("Only PostgreSQL profiling is supported right now.")
        if not parsed.hostname:
            raise ValueError("Invalid database URL.")

        conn = await asyncpg.connect(dsn=db_url.replace("postgresql+asyncpg://", "postgresql://"))
        try:
            table_rows = await self._fetch_tables(conn, tables)
            table_profiles: list[TableProfile] = []

            total_tables = len(table_rows)
            completed = 0

            for row in table_rows:
                schema = row["table_schema"]
                table = row["table_name"]
                columns = await self._fetch_columns(conn, schema, table, sample_size)
                row_count = await self._estimate_row_count(conn, schema, table)
                relationships = await self._fetch_relationships(conn, schema, table)

                table_profiles.append(
                    TableProfile(
                        schema=schema,
                        name=table,
                        row_count=row_count,
                        columns=columns,
                        relationships=relationships,
                        sample_size=sample_size,
                    )
                )
                completed += 1
                if progress_callback:
                    await progress_callback(total_tables, completed)

            return DatabaseProfile(
                connection_id=connection.connection_id,
                tables=table_profiles,
            )
        finally:
            await conn.close()

    async def _fetch_tables(
        self, conn: asyncpg.Connection, tables: Sequence[str] | None
    ) -> list[asyncpg.Record]:
        base_query = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
        """
        if tables:
            return await conn.fetch(
                base_query + " AND table_name = ANY($1) ORDER BY table_schema, table_name",
                list(tables),
            )
        return await conn.fetch(base_query + " ORDER BY table_schema, table_name")

    async def _fetch_columns(
        self, conn: asyncpg.Connection, schema: str, table: str, sample_size: int
    ) -> list[ColumnProfile]:
        column_rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
            """,
            schema,
            table,
        )

        columns: list[ColumnProfile] = []
        for row in column_rows:
            column_name = row["column_name"]
            stats = await self._fetch_column_stats(conn, schema, table, column_name)
            samples = await self._fetch_column_samples(
                conn, schema, table, column_name, sample_size
            )
            columns.append(
                ColumnProfile(
                    name=column_name,
                    data_type=row["data_type"],
                    nullable=row["is_nullable"] == "YES",
                    default_value=row["column_default"],
                    sample_values=samples,
                    null_count=stats.get("null_count"),
                    distinct_count=stats.get("distinct_count"),
                    min_value=stats.get("min_value"),
                    max_value=stats.get("max_value"),
                )
            )
        return columns

    async def _fetch_column_stats(
        self, conn: asyncpg.Connection, schema: str, table: str, column: str
    ) -> dict[str, str | int | None]:
        qualified_table = f"{self._quote_identifier(schema)}.{self._quote_identifier(table)}"
        qualified_column = self._quote_identifier(column)
        query = (
            "SELECT "
            f"COUNT(*) FILTER (WHERE {qualified_column} IS NULL) AS null_count, "
            f"COUNT(DISTINCT {qualified_column}) AS distinct_count, "
            f"MIN({qualified_column})::text AS min_value, "
            f"MAX({qualified_column})::text AS max_value "
            f"FROM {qualified_table}"
        )
        try:
            row = await conn.fetchrow(query)
        except Exception:
            row = None

        return {
            "null_count": row["null_count"] if row else None,
            "distinct_count": row["distinct_count"] if row else None,
            "min_value": row["min_value"] if row else None,
            "max_value": row["max_value"] if row else None,
        }

    async def _fetch_column_samples(
        self,
        conn: asyncpg.Connection,
        schema: str,
        table: str,
        column: str,
        sample_size: int,
    ) -> list[str]:
        qualified_table = f"{self._quote_identifier(schema)}.{self._quote_identifier(table)}"
        qualified_column = self._quote_identifier(column)
        query = (
            f"SELECT {qualified_column}::text AS value "
            f"FROM {qualified_table} "
            f"WHERE {qualified_column} IS NOT NULL "
            f"LIMIT {sample_size}"
        )
        try:
            rows = await conn.fetch(query)
        except Exception:
            return []
        return [row["value"] for row in rows if row["value"] is not None]

    async def _estimate_row_count(
        self, conn: asyncpg.Connection, schema: str, table: str
    ) -> int | None:
        row = await conn.fetchrow(
            """
            SELECT reltuples::BIGINT AS estimate
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1 AND c.relname = $2
            """,
            schema,
            table,
        )
        if row is None:
            return None
        return int(row["estimate"])

    async def _fetch_relationships(
        self, conn: asyncpg.Connection, schema: str, table: str
    ) -> list[RelationshipProfile]:
        rows = await conn.fetch(
            """
            SELECT
                kcu.table_name AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name AS target_table,
                ccu.column_name AS target_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND kcu.table_schema = $1
              AND kcu.table_name = $2
            """,
            schema,
            table,
        )

        relationships: list[RelationshipProfile] = []
        for row in rows:
            relationships.append(
                RelationshipProfile(
                    source_table=row["source_table"],
                    source_column=row["source_column"],
                    target_table=row["target_table"],
                    target_column=row["target_column"],
                    relationship_type="foreign_key",
                    cardinality="N:1",
                )
            )
        return relationships

    @staticmethod
    def _quote_identifier(value: str) -> str:
        escaped = value.replace('"', '""')
        return f'"{escaped}"'
