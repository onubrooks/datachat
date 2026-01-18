"""Database registry routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.database.manager import DatabaseConnectionManager
from backend.models.database import (
    DatabaseConnection,
    DatabaseConnectionCreate,
    DatabaseConnectionUpdateDefault,
)

router = APIRouter()


def _get_manager() -> DatabaseConnectionManager:
    from backend.api.main import app_state

    manager = app_state.get("database_manager")
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database registry is unavailable. Ensure DATABASE_CREDENTIALS_KEY is set.",
        )
    return manager


@router.post("/databases", response_model=DatabaseConnection, status_code=status.HTTP_201_CREATED)
async def create_database_connection(
    payload: DatabaseConnectionCreate,
) -> DatabaseConnection:
    """Create a new database connection."""
    manager = _get_manager()
    try:
        return await manager.add_connection(
            name=payload.name,
            database_url=payload.database_url.get_secret_value(),
            database_type=payload.database_type,
            tags=payload.tags,
            description=payload.description,
            is_default=payload.is_default,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/databases", response_model=list[DatabaseConnection])
async def list_database_connections() -> list[DatabaseConnection]:
    """List all active database connections."""
    manager = _get_manager()
    return await manager.list_connections()


@router.get("/databases/{connection_id}", response_model=DatabaseConnection)
async def get_database_connection(connection_id: str) -> DatabaseConnection:
    """Retrieve a single connection by ID."""
    manager = _get_manager()
    try:
        return await manager.get_connection(connection_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.put("/databases/{connection_id}/default", status_code=status.HTTP_204_NO_CONTENT)
async def set_default_database(
    connection_id: str, payload: DatabaseConnectionUpdateDefault
) -> None:
    """Set the default database connection."""
    if not payload.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="is_default must be true to set default connection",
        )
    manager = _get_manager()
    try:
        await manager.set_default(connection_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.delete("/databases/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_database_connection(connection_id: str) -> None:
    """Delete a database connection."""
    manager = _get_manager()
    try:
        await manager.remove_connection(connection_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
