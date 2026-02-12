"""Tool execution endpoints."""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from backend.connectors.factory import infer_database_type
from backend.models.api import ToolExecuteRequest, ToolExecuteResponse, ToolInfo
from backend.tools import ToolExecutor, ToolRegistry, initialize_tools
from backend.tools.base import ToolContext
from backend.tools.policy import ToolPolicyError

router = APIRouter()


async def _resolve_database_context(
    target_database: str | None,
) -> tuple[str | None, str | None]:
    from backend.api.main import app_state
    from backend.config import get_settings

    manager = app_state.get("database_manager")
    if target_database and manager is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "target_database requires an active database registry. "
                "Configure SYSTEM_DATABASE_URL and DATABASE_CREDENTIALS_KEY."
            ),
        )

    if manager is not None:
        try:
            if target_database:
                connection = await manager.get_connection(target_database)
            else:
                connection = await manager.get_default_connection()
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        if target_database and connection is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database connection not found: {target_database}",
            )
        if connection is not None:
            return connection.database_type, connection.database_url.get_secret_value()

    settings = get_settings()
    if settings.database.url:
        url = str(settings.database.url)
        try:
            return infer_database_type(url), url
        except Exception:
            return settings.database.db_type, url
    return None, None


async def _build_tool_metadata(payload: ToolExecuteRequest) -> dict:
    from backend.api.main import app_state

    pipeline = app_state.get("pipeline")
    retriever = getattr(pipeline, "retriever", None) if pipeline else None
    connector = app_state.get("connector")
    manager = app_state.get("database_manager")
    database_type, database_url = await _resolve_database_context(payload.target_database)

    metadata = {
        "requested_at": time.time(),
        "retriever": retriever,
        "connector": connector,
        "database_manager": manager,
        "database_type": database_type,
        "database_url": database_url,
        "target_database": payload.target_database,
    }
    return metadata


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools() -> list[ToolInfo]:
    initialize_tools()
    tools = []
    for tool in ToolRegistry.list_definitions():
        tools.append(
            ToolInfo(
                name=tool.name,
                description=tool.description,
                category=tool.category.value,
                requires_approval=tool.policy.requires_approval,
                enabled=tool.policy.enabled,
                parameters_schema=tool.parameters_schema,
            )
        )
    return tools


@router.post("/tools/execute", response_model=ToolExecuteResponse)
async def execute_tool(payload: ToolExecuteRequest) -> ToolExecuteResponse:
    initialize_tools()
    executor = ToolExecutor()
    correlation_id = payload.correlation_id or f"tool-{uuid4()}"
    metadata = await _build_tool_metadata(payload)
    ctx = ToolContext(
        user_id=payload.user_id or "api-user",
        correlation_id=correlation_id,
        approved=payload.approved,
        metadata=metadata,
    )
    try:
        result = await executor.execute(payload.name, payload.arguments, ctx)
        return ToolExecuteResponse(
            tool=payload.name, success=True, result=result.get("result")
        )
    except ToolPolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
