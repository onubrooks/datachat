"""Profiling and DataPoint generation routes."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.database.manager import DatabaseConnectionManager
from backend.profiling.generator import DataPointGenerator
from backend.profiling.models import PendingDataPoint, ProfilingProgress
from backend.profiling.profiler import SchemaProfiler
from backend.profiling.store import ProfilingStore

router = APIRouter()


class ProfilingRequest(BaseModel):
    sample_size: int = Field(default=100, gt=0, le=1000)
    tables: list[str] | None = None


class ProfilingJobResponse(BaseModel):
    job_id: UUID
    connection_id: UUID
    status: str
    progress: ProfilingProgress | None = None
    error: str | None = None
    profile_id: UUID | None = None


class GenerateDataPointsRequest(BaseModel):
    profile_id: UUID


class PendingDataPointResponse(BaseModel):
    pending_id: UUID
    profile_id: UUID
    datapoint: dict
    confidence: float
    status: str
    review_note: str | None = None


class PendingDataPointListResponse(BaseModel):
    pending: list[PendingDataPointResponse]


class ReviewNoteRequest(BaseModel):
    review_note: str | None = None


def _get_store() -> ProfilingStore:
    from backend.api.main import app_state

    store = app_state.get("profiling_store")
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profiling store unavailable",
        )
    return store


def _get_manager() -> DatabaseConnectionManager:
    from backend.api.main import app_state

    manager = app_state.get("database_manager")
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database registry unavailable",
        )
    return manager


def _get_vector_store():
    from backend.api.main import app_state

    store = app_state.get("vector_store")
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable",
        )
    return store


def _get_knowledge_graph():
    from backend.api.main import app_state

    graph = app_state.get("knowledge_graph")
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph unavailable",
        )
    return graph


def _to_pending_response(pending: PendingDataPoint) -> PendingDataPointResponse:
    return PendingDataPointResponse(
        pending_id=pending.pending_id,
        profile_id=pending.profile_id,
        datapoint=pending.datapoint,
        confidence=pending.confidence,
        status=pending.status,
        review_note=pending.review_note,
    )


@router.post(
    "/databases/{connection_id}/profile",
    response_model=ProfilingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_profiling_job(
    connection_id: UUID, payload: ProfilingRequest
) -> ProfilingJobResponse:
    store = _get_store()
    manager = _get_manager()

    job = await store.create_job(connection_id)

    async def run_job() -> None:
        profiler = SchemaProfiler(manager)

        async def progress_callback(total: int, completed: int) -> None:
            await store.update_job(
                job.job_id,
                progress=ProfilingProgress(total_tables=total, tables_completed=completed),
            )

        try:
            await store.update_job(job.job_id, status="running")
            profile = await profiler.profile_database(
                str(connection_id),
                sample_size=payload.sample_size,
                tables=payload.tables,
                progress_callback=progress_callback,
            )
            await store.save_profile(profile)
            await store.update_job(
                job.job_id,
                status="completed",
                profile_id=profile.profile_id,
                progress=ProfilingProgress(
                    total_tables=len(profile.tables),
                    tables_completed=len(profile.tables),
                ),
            )
        except Exception as exc:
            await store.update_job(job.job_id, status="failed", error=str(exc))

    asyncio.create_task(run_job())

    return ProfilingJobResponse(
        job_id=job.job_id,
        connection_id=job.connection_id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        profile_id=job.profile_id,
    )


@router.get("/profiling/jobs/{job_id}", response_model=ProfilingJobResponse)
async def get_profiling_job(job_id: UUID) -> ProfilingJobResponse:
    store = _get_store()
    try:
        job = await store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ProfilingJobResponse(
        job_id=job.job_id,
        connection_id=job.connection_id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        profile_id=job.profile_id,
    )


@router.post("/datapoints/generate", response_model=PendingDataPointListResponse)
async def generate_datapoints(payload: GenerateDataPointsRequest) -> PendingDataPointListResponse:
    store = _get_store()
    profile = await store.get_profile(payload.profile_id)

    generator = DataPointGenerator()
    generated = await generator.generate_from_profile(profile)

    pending_items = [
        PendingDataPoint(profile_id=profile.profile_id, datapoint=item.datapoint, confidence=item.confidence)
        for item in generated.schema_datapoints + generated.business_datapoints
    ]

    await store.add_pending_datapoints(profile.profile_id, pending_items)

    return PendingDataPointListResponse(
        pending=[_to_pending_response(item) for item in pending_items]
    )


@router.get("/datapoints/pending", response_model=PendingDataPointListResponse)
async def list_pending_datapoints() -> PendingDataPointListResponse:
    store = _get_store()
    pending = await store.list_pending(status="pending")
    return PendingDataPointListResponse(
        pending=[_to_pending_response(item) for item in pending]
    )


@router.post("/datapoints/pending/{pending_id}/approve", response_model=PendingDataPointResponse)
async def approve_datapoint(
    pending_id: UUID, payload: ReviewNoteRequest | None = None
) -> PendingDataPointResponse:
    store = _get_store()
    vector_store = _get_vector_store()
    graph = _get_knowledge_graph()

    pending = await store.update_pending_status(
        pending_id, status="approved", review_note=payload.review_note if payload else None
    )

    from backend.models.datapoint import DataPoint

    datapoint = DataPoint.model_validate(pending.datapoint)
    await vector_store.add_datapoints([datapoint])
    graph.add_datapoint(datapoint)

    return _to_pending_response(pending)


@router.post("/datapoints/pending/{pending_id}/reject", response_model=PendingDataPointResponse)
async def reject_datapoint(
    pending_id: UUID, payload: ReviewNoteRequest | None = None
) -> PendingDataPointResponse:
    store = _get_store()
    pending = await store.update_pending_status(
        pending_id, status="rejected", review_note=payload.review_note if payload else None
    )
    return _to_pending_response(pending)


@router.post("/datapoints/pending/bulk-approve", response_model=PendingDataPointListResponse)
async def bulk_approve_datapoints() -> PendingDataPointListResponse:
    store = _get_store()
    vector_store = _get_vector_store()
    graph = _get_knowledge_graph()

    approved = await store.bulk_update_pending(status="approved")

    from backend.models.datapoint import DataPoint

    datapoints = [DataPoint.model_validate(item.datapoint) for item in approved]
    if datapoints:
        await vector_store.add_datapoints(datapoints)
        for datapoint in datapoints:
            graph.add_datapoint(datapoint)

    return PendingDataPointListResponse(
        pending=[_to_pending_response(item) for item in approved]
    )
