"""
System Routes

Initialization status and guided setup endpoints.
"""

from fastapi import APIRouter, HTTPException, Request, status

from backend.initialization.initializer import SystemInitializer
from backend.models.api import (
    SystemInitializeRequest,
    SystemInitializeResponse,
    SystemStatusResponse,
)

router = APIRouter()


@router.get("/system/status", response_model=SystemStatusResponse)
async def system_status(request: Request) -> SystemStatusResponse:
    """Return current initialization status."""
    from backend.api.main import app_state

    initializer = SystemInitializer(app_state)
    status_state = await initializer.status()
    return SystemStatusResponse(
        is_initialized=status_state.is_initialized,
        has_databases=status_state.has_databases,
        has_datapoints=status_state.has_datapoints,
        setup_required=[
            {
                "step": step.step,
                "title": step.title,
                "description": step.description,
                "action": step.action,
            }
            for step in status_state.setup_required
        ],
    )


@router.post("/system/initialize", response_model=SystemInitializeResponse)
async def system_initialize(
    request: Request, payload: SystemInitializeRequest
) -> SystemInitializeResponse:
    """Run guided initialization."""
    from backend.api.main import app_state

    initializer = SystemInitializer(app_state)

    try:
        status_state, message = await initializer.initialize(
            database_url=payload.database_url,
            auto_profile=payload.auto_profile,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return SystemInitializeResponse(
        message=message,
        is_initialized=status_state.is_initialized,
        has_databases=status_state.has_databases,
        has_datapoints=status_state.has_datapoints,
        setup_required=[
            {
                "step": step.step,
                "title": step.title,
                "description": step.description,
                "action": step.action,
            }
            for step in status_state.setup_required
        ],
    )
