"""Tool execution endpoints."""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from backend.models.api import ToolExecuteRequest, ToolExecuteResponse, ToolInfo
from backend.tools import ToolExecutor, ToolRegistry, initialize_tools
from backend.tools.base import ToolContext
from backend.tools.policy import ToolPolicyError

router = APIRouter()


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
    ctx = ToolContext(
        user_id=payload.user_id or "api-user",
        correlation_id=correlation_id,
        approved=payload.approved,
        metadata={
            "requested_at": time.time(),
        },
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
