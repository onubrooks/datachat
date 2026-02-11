"""DataPoint CRUD and sync routes."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.models.datapoint import DataPoint
from backend.sync.orchestrator import save_datapoint_to_disk

router = APIRouter()

DATA_ROOT = Path("datapoints")
DATA_DIR = DATA_ROOT / "managed"
SOURCE_PRIORITY = {
    "user": 4,
    "managed": 3,
    "custom": 2,
    "unknown": 2,
    "example": 1,
}


class SyncStatusResponse(BaseModel):
    status: str
    job_id: str | None
    sync_type: str | None
    started_at: str | None
    finished_at: str | None
    total_datapoints: int
    processed_datapoints: int
    error: str | None


class SyncTriggerResponse(BaseModel):
    job_id: UUID


class DataPointSummary(BaseModel):
    datapoint_id: str
    type: str
    name: str | None
    source_tier: str | None = None
    source_path: str | None = None


class DataPointListResponse(BaseModel):
    datapoints: list[DataPointSummary]


def _get_vector_store():
    from backend.api.main import app_state

    vector_store = app_state.get("vector_store")
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable",
        )
    return vector_store


def _get_orchestrator():
    from backend.api.main import app_state

    orchestrator = app_state.get("sync_orchestrator")
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sync orchestrator unavailable",
        )
    return orchestrator


def _file_path(datapoint_id: str) -> Path:
    return DATA_DIR / f"{datapoint_id}.json"


@router.post("/datapoints", status_code=status.HTTP_201_CREATED)
async def create_datapoint(payload: dict) -> dict:
    datapoint = DataPoint.model_validate(payload)
    path = _file_path(datapoint.datapoint_id)
    if path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Datapoint already exists",
        )
    save_datapoint_to_disk(datapoint.model_dump(mode="json", by_alias=True), path)

    orchestrator = _get_orchestrator()
    orchestrator.enqueue_sync_incremental([datapoint.datapoint_id])

    return datapoint.model_dump(mode="json", by_alias=True)


@router.put("/datapoints/{datapoint_id}")
async def update_datapoint(datapoint_id: str, payload: dict) -> dict:
    datapoint = DataPoint.model_validate(payload)
    if datapoint.datapoint_id != datapoint_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Datapoint ID mismatch",
        )
    path = _file_path(datapoint_id)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Datapoint not found",
        )
    save_datapoint_to_disk(datapoint.model_dump(mode="json", by_alias=True), path)

    orchestrator = _get_orchestrator()
    orchestrator.enqueue_sync_incremental([datapoint.datapoint_id])

    return datapoint.model_dump(mode="json", by_alias=True)


@router.delete("/datapoints/{datapoint_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_datapoint(datapoint_id: str) -> None:
    path = _file_path(datapoint_id)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Datapoint not found",
        )
    path.unlink(missing_ok=True)

    orchestrator = _get_orchestrator()
    orchestrator.enqueue_sync_incremental([datapoint_id])


@router.post("/sync", response_model=SyncTriggerResponse)
async def trigger_sync() -> SyncTriggerResponse:
    orchestrator = _get_orchestrator()
    job_id = orchestrator.enqueue_sync_all()
    return SyncTriggerResponse(job_id=job_id)


@router.get("/datapoints", response_model=DataPointListResponse)
async def list_datapoints() -> DataPointListResponse:
    vector_store = _get_vector_store()
    try:
        items = await vector_store.list_datapoints(limit=10000)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datapoints: {exc}",
        ) from exc

    deduped: dict[str, DataPointSummary] = {}
    for item in items:
        datapoint_id = str(item.get("datapoint_id", ""))
        if not datapoint_id:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_tier_raw = metadata.get("source_tier")
        source_tier = str(source_tier_raw) if source_tier_raw is not None else "unknown"
        source_path_raw = metadata.get("source_path")
        source_path = str(source_path_raw) if source_path_raw else None
        summary = DataPointSummary(
            datapoint_id=datapoint_id,
            type=str(metadata.get("type", "Unknown")),
            name=str(metadata["name"]) if metadata.get("name") is not None else None,
            source_tier=source_tier,
            source_path=source_path,
        )
        existing = deduped.get(datapoint_id)
        if existing is None:
            deduped[datapoint_id] = summary
            continue
        existing_priority = SOURCE_PRIORITY.get(existing.source_tier or "unknown", 0)
        candidate_priority = SOURCE_PRIORITY.get(summary.source_tier or "unknown", 0)
        if candidate_priority > existing_priority:
            deduped[datapoint_id] = summary

    datapoints = sorted(
        deduped.values(),
        key=lambda item: (
            -(SOURCE_PRIORITY.get(item.source_tier or "unknown", 0)),
            item.type,
            item.name or item.datapoint_id,
        ),
    )
    return DataPointListResponse(datapoints=datapoints)


@router.get("/sync/status", response_model=SyncStatusResponse)
async def get_sync_status() -> SyncStatusResponse:
    orchestrator = _get_orchestrator()
    status_payload = orchestrator.get_status()
    for key in ("started_at", "finished_at"):
        value = status_payload.get(key)
        if value is not None and hasattr(value, "isoformat"):
            status_payload[key] = value.isoformat()
    return SyncStatusResponse(**status_payload)
