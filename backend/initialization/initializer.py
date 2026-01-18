"""
System Initialization

Provides initialization status checks and guided setup helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.connectors.base import ConnectionError
from backend.connectors.postgres import PostgresConnector
from backend.knowledge.graph import KnowledgeGraph
from backend.knowledge.retriever import Retriever
from backend.knowledge.vectors import VectorStore
from backend.pipeline.orchestrator import DataChatPipeline


@dataclass(frozen=True)
class SetupStep:
    """Represents a required setup step."""

    step: str
    title: str
    description: str
    action: str


@dataclass(frozen=True)
class SystemStatus:
    """System initialization status."""

    is_initialized: bool
    has_databases: bool
    has_datapoints: bool
    setup_required: list[SetupStep]


class SystemInitializer:
    """Initialization workflow for DataChat."""

    def __init__(self, app_state: dict[str, Any]) -> None:
        self._app_state = app_state

    async def _check_database(self) -> bool:
        connector = self._app_state.get("connector")
        if connector is None:
            return False
        try:
            await connector.connect()
        except ConnectionError:
            return False
        return True

    async def _check_datapoints(self) -> bool:
        vector_store: VectorStore | None = self._app_state.get("vector_store")
        if vector_store is None:
            return False
        try:
            count = await vector_store.get_count()
        except Exception:
            return False
        return count > 0

    async def status(self) -> SystemStatus:
        has_databases = await self._check_database()
        has_datapoints = await self._check_datapoints()
        setup_required: list[SetupStep] = []

        if not has_databases:
            setup_required.append(
                SetupStep(
                    step="database_connection",
                    title="Connect a database",
                    description="Configure the database connection used for queries.",
                    action="configure_database",
                )
            )

        if not has_datapoints:
            setup_required.append(
                SetupStep(
                    step="datapoints",
                    title="Load DataPoints",
                    description="Add DataPoints describing your schema and business logic.",
                    action="load_datapoints",
                )
            )

        return SystemStatus(
            is_initialized=has_databases and has_datapoints,
            has_databases=has_databases,
            has_datapoints=has_datapoints,
            setup_required=setup_required,
        )

    async def initialize(
        self, database_url: str | None, auto_profile: bool
    ) -> tuple[SystemStatus, str]:
        message = "Initialization completed."

        if database_url:
            from urllib.parse import urlparse

            parsed = urlparse(database_url)
            if not parsed.scheme or not parsed.hostname:
                raise ValueError("Invalid database URL.")
            if parsed.scheme not in {"postgres", "postgresql"}:
                raise ValueError("Only PostgreSQL database URLs are supported.")
            connector = PostgresConnector(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") if parsed.path else "datachat",
                user=parsed.username or "postgres",
                password=parsed.password or "",
            )
            await connector.connect()
            self._app_state["connector"] = connector

            vector_store: VectorStore | None = self._app_state.get("vector_store")
            knowledge_graph: KnowledgeGraph | None = self._app_state.get("knowledge_graph")
            if vector_store and knowledge_graph:
                retriever = Retriever(vector_store=vector_store, knowledge_graph=knowledge_graph)
                self._app_state["pipeline"] = DataChatPipeline(
                    retriever=retriever,
                    connector=connector,
                    max_retries=3,
                )

        if auto_profile and database_url:
            profiling_store = self._app_state.get("profiling_store")
            database_manager = self._app_state.get("database_manager")
            if profiling_store and database_manager:
                try:
                    existing = None
                    for connection in await database_manager.list_connections():
                        if connection.database_url.get_secret_value() == database_url:
                            existing = connection
                            break

                    if existing is None:
                        existing = await database_manager.add_connection(
                            name="Primary Database",
                            database_url=database_url,
                            database_type="postgresql",
                            tags=["auto-profiled"],
                            description="Auto-profiled during setup",
                            is_default=True,
                        )

                    from backend.profiling.profiler import SchemaProfiler
                    from backend.profiling.models import ProfilingProgress

                    job = await profiling_store.create_job(existing.connection_id)

                    async def run_profile_job() -> None:
                        profiler = SchemaProfiler(database_manager)

                        async def progress_callback(total: int, completed: int) -> None:
                            await profiling_store.update_job(
                                job.job_id,
                                progress=ProfilingProgress(
                                    total_tables=total, tables_completed=completed
                                ),
                            )

                        try:
                            await profiling_store.update_job(job.job_id, status="running")
                            profile = await profiler.profile_database(
                                str(existing.connection_id),
                                progress_callback=progress_callback,
                            )
                            await profiling_store.save_profile(profile)
                            await profiling_store.update_job(
                                job.job_id,
                                status="completed",
                                profile_id=profile.profile_id,
                                progress=ProfilingProgress(
                                    total_tables=len(profile.tables),
                                    tables_completed=len(profile.tables),
                                ),
                            )
                        except Exception as exc:
                            await profiling_store.update_job(
                                job.job_id, status="failed", error=str(exc)
                            )

                    import asyncio

                    asyncio.create_task(run_profile_job())
                    message = (
                        "Initialization completed. Auto-profiling started; "
                        f"job_id={job.job_id}."
                    )
                except Exception as exc:
                    message = (
                        "Initialization completed, but auto-profiling failed to start: "
                        f"{exc}."
                    )
            else:
                message = (
                    "Initialization completed, but auto-profiling is unavailable. "
                    "Configure DATABASE_CREDENTIALS_KEY to enable profiling."
                )

        return await self.status(), message
