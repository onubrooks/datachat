"""Sync orchestrator for DataPoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from backend.knowledge.datapoints import DataPointLoader
from backend.models.datapoint import DataPoint


@dataclass
class SyncJob:
    """Status for a sync job."""

    job_id: UUID
    status: str
    sync_type: str
    started_at: datetime
    finished_at: datetime | None = None
    total_datapoints: int = 0
    processed_datapoints: int = 0
    error: str | None = None


class SyncOrchestrator:
    """Coordinate syncing DataPoints into vector store and knowledge graph."""

    def __init__(
        self,
        vector_store,
        knowledge_graph,
        datapoints_dir: str | Path = "datapoints",
        loader: DataPointLoader | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._knowledge_graph = knowledge_graph
        self._datapoints_dir = Path(datapoints_dir)
        self._loader = loader or DataPointLoader()
        self._loop = loop
        self._current_job: SyncJob | None = None
        self._lock = asyncio.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def get_status(self) -> dict:
        if not self._current_job:
            return {
                "status": "idle",
                "job_id": None,
                "sync_type": None,
                "started_at": None,
                "finished_at": None,
                "total_datapoints": 0,
                "processed_datapoints": 0,
                "error": None,
            }
        payload = asdict(self._current_job)
        payload["job_id"] = str(payload["job_id"])
        return payload

    def enqueue_sync_all(self) -> UUID:
        job_id = uuid4()
        self._schedule_job(job_id, "full", None)
        return job_id

    def enqueue_sync_incremental(self, datapoint_ids: Iterable[str]) -> UUID:
        job_id = uuid4()
        self._schedule_job(job_id, "incremental", list(datapoint_ids))
        return job_id

    async def sync_all(self) -> SyncJob:
        job = self._start_job(uuid4(), "full")
        await self._run_sync_all(job)
        return job

    async def sync_incremental(self, datapoint_ids: Iterable[str]) -> SyncJob:
        job = self._start_job(uuid4(), "incremental")
        await self._run_sync_incremental(job, list(datapoint_ids))
        return job

    def _schedule_job(self, job_id: UUID, sync_type: str, datapoint_ids: list[str] | None) -> None:
        if not self._loop:
            raise RuntimeError("Sync orchestrator requires an event loop")
        if sync_type == "full":
            coro = self._run_sync_all(self._start_job(job_id, sync_type))
        else:
            coro = self._run_sync_incremental(
                self._start_job(job_id, sync_type), datapoint_ids or []
            )
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _start_job(self, job_id: UUID, sync_type: str) -> SyncJob:
        job = SyncJob(
            job_id=job_id,
            status="running",
            sync_type=sync_type,
            started_at=datetime.now(UTC),
        )
        self._current_job = job
        return job

    async def _run_sync_all(self, job: SyncJob) -> None:
        async with self._lock:
            try:
                datapoints = self._load_all_datapoints()
                job.total_datapoints = len(datapoints)

                await self._vector_store.clear()
                self._knowledge_graph.clear()

                if datapoints:
                    await self._vector_store.add_datapoints(datapoints)
                    for datapoint in datapoints:
                        self._knowledge_graph.add_datapoint(datapoint)
                        job.processed_datapoints += 1

                job.status = "completed"
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = datetime.now(UTC)

    async def _run_sync_incremental(self, job: SyncJob, datapoint_ids: list[str]) -> None:
        async with self._lock:
            try:
                job.total_datapoints = len(datapoint_ids)

                if datapoint_ids:
                    await self._vector_store.delete(datapoint_ids)
                    for datapoint_id in datapoint_ids:
                        if hasattr(self._knowledge_graph, "remove_datapoint"):
                            self._knowledge_graph.remove_datapoint(datapoint_id)

                datapoints = self._load_datapoints_by_id(datapoint_ids)
                if datapoints:
                    await self._vector_store.add_datapoints(datapoints)
                    for datapoint in datapoints:
                        self._knowledge_graph.add_datapoint(datapoint)
                        job.processed_datapoints += 1

                job.status = "completed"
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = datetime.now(UTC)

    def _load_all_datapoints(self) -> list[DataPoint]:
        datapoint_files = self._collect_datapoint_files()
        datapoints: list[DataPoint] = []
        for file_path in datapoint_files:
            try:
                datapoints.append(self._loader.load_file(file_path))
            except Exception:
                continue
        return datapoints

    def _load_datapoints_by_id(self, datapoint_ids: Iterable[str]) -> list[DataPoint]:
        if not datapoint_ids:
            return []
        id_set = {str(item) for item in datapoint_ids}
        datapoint_files = self._collect_datapoint_files()
        datapoints: list[DataPoint] = []
        for file_path in datapoint_files:
            try:
                datapoint = self._loader.load_file(file_path)
            except Exception:
                continue
            if datapoint.datapoint_id in id_set:
                datapoints.append(datapoint)
        return datapoints

    def _collect_datapoint_files(self) -> list[Path]:
        if not self._datapoints_dir.exists():
            return []
        files = []
        for path in self._datapoints_dir.rglob("*.json"):
            if "schemas" in path.parts:
                continue
            files.append(path)
        return files


def save_datapoint_to_disk(datapoint: dict, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(datapoint, handle, indent=2)
