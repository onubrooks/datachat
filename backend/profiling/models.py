"""Profiling and DataPoint generation models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ColumnProfile(BaseModel):
    """Profile for a database column."""

    name: str
    data_type: str
    nullable: bool
    default_value: str | None = None
    sample_values: list[str] = Field(default_factory=list)
    null_count: int | None = None
    distinct_count: int | None = None
    min_value: str | None = None
    max_value: str | None = None


class RelationshipProfile(BaseModel):
    """Profiled relationship between tables."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: Literal["foreign_key", "heuristic"] = "foreign_key"
    cardinality: Literal["1:1", "1:N", "N:1", "N:N"] = "N:1"


class TableProfile(BaseModel):
    """Profile for a database table."""

    schema_name: str = Field(..., alias="schema")
    name: str
    row_count: int | None
    columns: list[ColumnProfile]
    relationships: list[RelationshipProfile] = Field(default_factory=list)
    sample_size: int

    model_config = ConfigDict(populate_by_name=True)


class DatabaseProfile(BaseModel):
    """Profile for a database connection."""

    profile_id: UUID = Field(default_factory=uuid4)
    connection_id: UUID
    tables: list[TableProfile]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProfilingProgress(BaseModel):
    """Progress tracking for a profiling job."""

    total_tables: int
    tables_completed: int


class GenerationProgress(BaseModel):
    """Progress tracking for DataPoint generation."""

    total_tables: int
    tables_completed: int
    batch_size: int


class ProfilingJob(BaseModel):
    """Profiling job status."""

    job_id: UUID = Field(default_factory=uuid4)
    connection_id: UUID
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress: ProfilingProgress | None = None
    error: str | None = None
    profile_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GenerationJob(BaseModel):
    """DataPoint generation job status."""

    job_id: UUID = Field(default_factory=uuid4)
    profile_id: UUID
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress: GenerationProgress | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GeneratedDataPoint(BaseModel):
    """Generated DataPoint candidate with confidence score."""

    datapoint: dict
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str | None = None


class GeneratedDataPoints(BaseModel):
    """Collection of generated DataPoints."""

    profile_id: UUID
    schema_datapoints: list[GeneratedDataPoint]
    business_datapoints: list[GeneratedDataPoint]


class PendingDataPoint(BaseModel):
    """Pending DataPoint awaiting approval."""

    pending_id: UUID = Field(default_factory=uuid4)
    profile_id: UUID
    datapoint: dict
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None
    review_note: str | None = None
