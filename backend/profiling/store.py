"""Storage utilities for profiling jobs and pending DataPoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import asyncpg

from backend.config import get_settings
from backend.profiling.models import (
    DatabaseProfile,
    PendingDataPoint,
    ProfilingJob,
    ProfilingProgress,
)

_CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS profiling_jobs (
    job_id UUID PRIMARY KEY,
    connection_id UUID NOT NULL,
    status TEXT NOT NULL,
    progress JSONB,
    error TEXT,
    profile_id UUID,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
"""

_CREATE_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS profiling_profiles (
    profile_id UUID PRIMARY KEY,
    connection_id UUID NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
"""

_CREATE_PENDING_TABLE = """
CREATE TABLE IF NOT EXISTS pending_datapoints (
    pending_id UUID PRIMARY KEY,
    profile_id UUID NOT NULL,
    datapoint JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    reviewed_at TIMESTAMPTZ,
    review_note TEXT
);
"""


class ProfilingStore:
    """Persist profiling jobs, profiles, and pending DataPoints."""

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        self._database_url = database_url or str(settings.database.url)
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        if self._pool is None:
            dsn = self._normalize_postgres_url(self._database_url)
            self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
        await self._pool.execute(_CREATE_JOBS_TABLE)
        await self._pool.execute(_CREATE_PROFILES_TABLE)
        await self._pool.execute(_CREATE_PENDING_TABLE)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create_job(self, connection_id: UUID) -> ProfilingJob:
        self._ensure_pool()
        job = ProfilingJob(connection_id=connection_id)
        await self._pool.execute(
            """
            INSERT INTO profiling_jobs (
                job_id, connection_id, status, progress, error, profile_id, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            job.job_id,
            job.connection_id,
            job.status,
            job.progress.model_dump() if job.progress else None,
            job.error,
            job.profile_id,
            job.created_at,
            job.updated_at,
        )
        return job

    async def update_job(
        self,
        job_id: UUID,
        status: str | None = None,
        progress: ProfilingProgress | None = None,
        error: str | None = None,
        profile_id: UUID | None = None,
    ) -> ProfilingJob:
        self._ensure_pool()
        updated_at = datetime.now(UTC)
        await self._pool.execute(
            """
            UPDATE profiling_jobs
            SET status = COALESCE($2, status),
                progress = COALESCE($3, progress),
                error = COALESCE($4, error),
                profile_id = COALESCE($5, profile_id),
                updated_at = $6
            WHERE job_id = $1
            """,
            job_id,
            status,
            progress.model_dump() if progress else None,
            error,
            profile_id,
            updated_at,
        )
        return await self.get_job(job_id)

    async def get_job(self, job_id: UUID) -> ProfilingJob:
        self._ensure_pool()
        row = await self._pool.fetchrow(
            """
            SELECT job_id, connection_id, status, progress, error, profile_id, created_at, updated_at
            FROM profiling_jobs
            WHERE job_id = $1
            """,
            job_id,
        )
        if row is None:
            raise KeyError(f"Profiling job not found: {job_id}")
        progress = None
        if row["progress"]:
            progress = ProfilingProgress(**row["progress"])
        return ProfilingJob(
            job_id=row["job_id"],
            connection_id=row["connection_id"],
            status=row["status"],
            progress=progress,
            error=row["error"],
            profile_id=row["profile_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def save_profile(self, profile: DatabaseProfile) -> DatabaseProfile:
        self._ensure_pool()
        await self._pool.execute(
            """
            INSERT INTO profiling_profiles (profile_id, connection_id, payload, created_at)
            VALUES ($1, $2, $3, $4)
            """,
            profile.profile_id,
            profile.connection_id,
            profile.model_dump(mode="json"),
            profile.created_at,
        )
        return profile

    async def get_profile(self, profile_id: UUID) -> DatabaseProfile:
        self._ensure_pool()
        row = await self._pool.fetchrow(
            """
            SELECT payload
            FROM profiling_profiles
            WHERE profile_id = $1
            """,
            profile_id,
        )
        if row is None:
            raise KeyError(f"Profile not found: {profile_id}")
        return DatabaseProfile.model_validate(row["payload"])

    async def add_pending_datapoints(
        self, profile_id: UUID, datapoints: list[PendingDataPoint]
    ) -> list[PendingDataPoint]:
        self._ensure_pool()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for item in datapoints:
                    await conn.execute(
                        """
                        INSERT INTO pending_datapoints (
                            pending_id, profile_id, datapoint, confidence, status, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        item.pending_id,
                        profile_id,
                        item.datapoint,
                        item.confidence,
                        item.status,
                        item.created_at,
                    )
        return datapoints

    async def list_pending(self, status: str | None = None) -> list[PendingDataPoint]:
        self._ensure_pool()
        if status:
            rows = await self._pool.fetch(
                """
                SELECT pending_id, profile_id, datapoint, confidence, status,
                       created_at, reviewed_at, review_note
                FROM pending_datapoints
                WHERE status = $1
                ORDER BY created_at DESC
                """,
                status,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT pending_id, profile_id, datapoint, confidence, status,
                       created_at, reviewed_at, review_note
                FROM pending_datapoints
                ORDER BY created_at DESC
                """
            )
        return [self._row_to_pending(row) for row in rows]

    async def update_pending_status(
        self,
        pending_id: UUID,
        status: str,
        review_note: str | None = None,
    ) -> PendingDataPoint:
        self._ensure_pool()
        reviewed_at = datetime.now(UTC)
        row = await self._pool.fetchrow(
            """
            UPDATE pending_datapoints
            SET status = $2, reviewed_at = $3, review_note = $4
            WHERE pending_id = $1
            RETURNING pending_id, profile_id, datapoint, confidence, status,
                      created_at, reviewed_at, review_note
            """,
            pending_id,
            status,
            reviewed_at,
            review_note,
        )
        if row is None:
            raise KeyError(f"Pending DataPoint not found: {pending_id}")
        return self._row_to_pending(row)

    async def bulk_update_pending(self, status: str) -> list[PendingDataPoint]:
        self._ensure_pool()
        reviewed_at = datetime.now(UTC)
        rows = await self._pool.fetch(
            """
            UPDATE pending_datapoints
            SET status = $1, reviewed_at = $2
            WHERE status = 'pending'
            RETURNING pending_id, profile_id, datapoint, confidence, status,
                      created_at, reviewed_at, review_note
            """,
            status,
            reviewed_at,
        )
        return [self._row_to_pending(row) for row in rows]

    @staticmethod
    def _row_to_pending(row: asyncpg.Record) -> PendingDataPoint:
        return PendingDataPoint(
            pending_id=row["pending_id"],
            profile_id=row["profile_id"],
            datapoint=row["datapoint"],
            confidence=row["confidence"],
            status=row["status"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            review_note=row["review_note"],
        )

    def _ensure_pool(self) -> None:
        if self._pool is None:
            raise RuntimeError("ProfilingStore is not initialized")

    @staticmethod
    def _normalize_postgres_url(database_url: str) -> str:
        if database_url.startswith("postgresql+asyncpg://"):
            return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return database_url
