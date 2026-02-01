"""DataPoint CRUD and sync routes."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.models.datapoint import DataPoint
from backend.sync.orchestrator import save_datapoint_to_disk

router = APIRouter()

DATA_DIR = Path("datapoints") / "managed"


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


class DataPointListResponse(BaseModel):
    datapoints: list[DataPointSummary]


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
    if not DATA_DIR.exists():
        return DataPointListResponse(datapoints=[])
    datapoints: list[DataPointSummary] = []
    for path in DATA_DIR.glob("*.json"):
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        datapoints.append(
            DataPointSummary(
                datapoint_id=str(payload.get("datapoint_id", path.stem)),
                type=str(payload.get("type", "Unknown")),
                name=payload.get("name"),
            )
        )
    datapoints.sort(key=lambda item: (item.type, item.name or item.datapoint_id))
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
