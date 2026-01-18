"""
Database connection models.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, SecretStr


class DatabaseConnection(BaseModel):
    """Stored database connection details."""

    connection_id: UUID = Field(default_factory=uuid4, description="Connection identifier")
    name: str = Field(..., min_length=1, description="User-friendly name")
    database_url: SecretStr = Field(..., description="Encrypted database URL")
    database_type: Literal["postgresql", "clickhouse", "mysql"] = Field(
        ..., description="Database engine type"
    )
    is_active: bool = Field(default=True, description="Whether the connection is active")
    is_default: bool = Field(default=False, description="Whether this is the default connection")
    tags: list[str] = Field(default_factory=list, description="Tags for grouping")
    description: str | None = Field(None, description="Optional description")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    last_profiled: datetime | None = Field(None, description="Last profiling timestamp")
    datapoint_count: int = Field(default=0, description="Linked DataPoint count")


class DatabaseConnectionCreate(BaseModel):
    """Payload for creating a database connection."""

    name: str = Field(..., min_length=1, description="User-friendly name")
    database_url: SecretStr = Field(..., description="Database URL")
    database_type: Literal["postgresql", "clickhouse", "mysql"] = Field(
        ..., description="Database engine type"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for grouping")
    description: str | None = Field(None, description="Optional description")
    is_default: bool = Field(default=False, description="Set as default connection")


class DatabaseConnectionUpdateDefault(BaseModel):
    """Payload for setting the default connection."""

    is_default: bool = Field(default=True, description="Set connection as default")
