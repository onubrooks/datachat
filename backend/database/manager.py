"""Database connection registry manager."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import UUID, uuid4

import asyncpg
from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

from backend.config import get_settings
from backend.connectors.clickhouse import ClickHouseConnector
from backend.connectors.postgres import PostgresConnector
from backend.models.database import DatabaseConnection

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS database_connections (
    connection_id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    database_url_encrypted TEXT NOT NULL,
    database_type TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    tags TEXT[] NOT NULL DEFAULT '{}',
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    last_profiled TIMESTAMPTZ,
    datapoint_count INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_DEFAULT_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS database_connections_default_idx
ON database_connections (is_default)
WHERE is_default;
"""


class DatabaseConnectionManager:
    """Manage database connections stored in the system database."""

    def __init__(
        self,
        system_database_url: str | None = None,
        encryption_key: str | bytes | None = None,
        pool: asyncpg.Pool | None = None,
    ) -> None:
        settings = get_settings()
        self._system_database_url = system_database_url or str(settings.database.url)
        self._pool = pool
        self._encryption_key = encryption_key or os.getenv("DATABASE_CREDENTIALS_KEY")
        self._cipher: Fernet | None = None

    async def initialize(self) -> None:
        """Initialize connection pool and ensure schema exists."""
        self._ensure_cipher()
        if self._pool is None:
            dsn = self._normalize_postgres_url(self._system_database_url)
            self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
        await self._pool.execute(_CREATE_TABLE_SQL)
        await self._pool.execute(_CREATE_DEFAULT_INDEX_SQL)

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def add_connection(
        self,
        name: str,
        database_url: str,
        database_type: str,
        tags: list[str] | None = None,
        description: str | None = None,
        is_default: bool = False,
    ) -> DatabaseConnection:
        """Add a new database connection after validation."""
        self._ensure_pool()
        await self._validate_connection(database_type, database_url)

        connection_id = uuid4()
        created_at = datetime.now(UTC)
        encrypted_url = self._encrypt_url(database_url)
        tags = tags or []

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if is_default:
                    await conn.execute(
                        "UPDATE database_connections SET is_default = FALSE WHERE is_default = TRUE"
                    )

                row = await conn.fetchrow(
                    """
                    INSERT INTO database_connections (
                        connection_id,
                        name,
                        database_url_encrypted,
                        database_type,
                        is_active,
                        is_default,
                        tags,
                        description,
                        created_at,
                        last_profiled,
                        datapoint_count
                    ) VALUES ($1, $2, $3, $4, TRUE, $5, $6, $7, $8, NULL, 0)
                    RETURNING
                        connection_id,
                        name,
                        database_url_encrypted,
                        database_type,
                        is_active,
                        is_default,
                        tags,
                        description,
                        created_at,
                        last_profiled,
                        datapoint_count
                    """,
                    connection_id,
                    name,
                    encrypted_url,
                    database_type,
                    is_default,
                    tags,
                    description,
                    created_at,
                )

        return self._row_to_connection(row)

    async def list_connections(self) -> list[DatabaseConnection]:
        """List active database connections."""
        self._ensure_pool()
        rows = await self._pool.fetch(
            """
            SELECT
                connection_id,
                name,
                database_url_encrypted,
                database_type,
                is_active,
                is_default,
                tags,
                description,
                created_at,
                last_profiled,
                datapoint_count
            FROM database_connections
            WHERE is_active = TRUE
            ORDER BY created_at DESC
            """
        )
        return [self._row_to_connection(row) for row in rows]

    async def get_connection(self, connection_id: UUID | str) -> DatabaseConnection:
        """Retrieve a single active connection."""
        self._ensure_pool()
        connection_uuid = self._coerce_uuid(connection_id)
        row = await self._pool.fetchrow(
            """
            SELECT
                connection_id,
                name,
                database_url_encrypted,
                database_type,
                is_active,
                is_default,
                tags,
                description,
                created_at,
                last_profiled,
                datapoint_count
            FROM database_connections
            WHERE connection_id = $1 AND is_active = TRUE
            """,
            connection_uuid,
        )
        if row is None:
            raise KeyError(f"Connection not found: {connection_id}")
        return self._row_to_connection(row)

    async def get_default_connection(self) -> DatabaseConnection | None:
        """Return the default connection if set."""
        self._ensure_pool()
        row = await self._pool.fetchrow(
            """
            SELECT
                connection_id,
                name,
                database_url_encrypted,
                database_type,
                is_active,
                is_default,
                tags,
                description,
                created_at,
                last_profiled,
                datapoint_count
            FROM database_connections
            WHERE is_default = TRUE AND is_active = TRUE
            """
        )
        if row is None:
            return None
        return self._row_to_connection(row)

    async def set_default(self, connection_id: UUID | str) -> None:
        """Mark a connection as the default."""
        self._ensure_pool()
        connection_uuid = self._coerce_uuid(connection_id)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                exists = await conn.fetchval(
                    """
                    SELECT 1
                    FROM database_connections
                    WHERE connection_id = $1 AND is_active = TRUE
                    """,
                    connection_uuid,
                )
                if not exists:
                    raise KeyError(f"Connection not found: {connection_id}")

                await conn.execute(
                    "UPDATE database_connections SET is_default = FALSE WHERE is_default = TRUE"
                )
                await conn.execute(
                    "UPDATE database_connections SET is_default = TRUE WHERE connection_id = $1",
                    connection_uuid,
                )

    async def remove_connection(self, connection_id: UUID | str) -> None:
        """Remove a connection from the registry."""
        self._ensure_pool()
        connection_uuid = self._coerce_uuid(connection_id)
        result = await self._pool.execute(
            "DELETE FROM database_connections WHERE connection_id = $1",
            connection_uuid,
        )
        deleted = int(result.split()[-1]) if result else 0
        if deleted == 0:
            raise KeyError(f"Connection not found: {connection_id}")

    def _row_to_connection(self, row: asyncpg.Record) -> DatabaseConnection:
        decrypted_url = self._decrypt_url(row["database_url_encrypted"])
        return DatabaseConnection(
            connection_id=row["connection_id"],
            name=row["name"],
            database_url=SecretStr(decrypted_url),
            database_type=row["database_type"],
            is_active=row["is_active"],
            is_default=row["is_default"],
            tags=list(row["tags"] or []),
            description=row["description"],
            created_at=row["created_at"],
            last_profiled=row["last_profiled"],
            datapoint_count=row["datapoint_count"],
        )

    async def _validate_connection(self, database_type: str, database_url: str) -> None:
        if database_type == "postgresql":
            connector = self._build_postgres_connector(database_url)
        elif database_type == "clickhouse":
            connector = self._build_clickhouse_connector(database_url)
        elif database_type == "mysql":
            raise ValueError("MySQL connections are not supported yet.")
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

        try:
            await connector.connect()
        finally:
            await connector.close()

    def _build_postgres_connector(self, database_url: str) -> PostgresConnector:
        parsed = urlparse(database_url)
        scheme = parsed.scheme.split("+")[0]
        if scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError("Invalid PostgreSQL database URL.")
        return PostgresConnector(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "datachat",
            user=parsed.username or "postgres",
            password=parsed.password or "",
        )

    def _build_clickhouse_connector(self, database_url: str) -> ClickHouseConnector:
        parsed = urlparse(database_url)
        scheme = parsed.scheme.split("+")[0]
        if scheme != "clickhouse" or not parsed.hostname:
            raise ValueError("Invalid ClickHouse database URL.")
        return ClickHouseConnector(
            host=parsed.hostname,
            port=parsed.port or 8123,
            database=parsed.path.lstrip("/") if parsed.path else "default",
            user=parsed.username or "default",
            password=parsed.password or "",
        )

    def _encrypt_url(self, database_url: str) -> str:
        cipher = self._ensure_cipher()
        return cipher.encrypt(database_url.encode("utf-8")).decode("utf-8")

    def _decrypt_url(self, encrypted_url: str) -> str:
        cipher = self._ensure_cipher()
        try:
            return cipher.decrypt(encrypted_url.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Failed to decrypt database URL.") from exc

    def _ensure_cipher(self) -> Fernet:
        if self._cipher is not None:
            return self._cipher
        if not self._encryption_key:
            raise ValueError(
                "DATABASE_CREDENTIALS_KEY must be set to store encrypted database URLs."
            )
        key = self._encryption_key
        if isinstance(key, str):
            key = key.encode("utf-8")
        try:
            self._cipher = Fernet(key)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Invalid DATABASE_CREDENTIALS_KEY. Use a Fernet-compatible base64 key."
            ) from exc
        return self._cipher

    def _ensure_pool(self) -> None:
        if self._pool is None:
            raise RuntimeError("DatabaseConnectionManager is not initialized")

    @staticmethod
    def _coerce_uuid(connection_id: UUID | str) -> UUID:
        if isinstance(connection_id, UUID):
            return connection_id
        try:
            return UUID(str(connection_id))
        except ValueError as exc:
            raise ValueError("Invalid connection ID.") from exc

    @staticmethod
    def _normalize_postgres_url(database_url: str) -> str:
        if database_url.startswith("postgresql+asyncpg://"):
            return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return database_url
