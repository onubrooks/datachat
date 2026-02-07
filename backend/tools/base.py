"""Tool system base types and decorator."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCategory(StrEnum):
    DATABASE = "database"
    PROFILING = "profiling"
    KNOWLEDGE = "knowledge"
    SYSTEM = "system"


class ToolPolicy(BaseModel):
    enabled: bool = True
    requires_approval: bool = False
    max_execution_time_seconds: int = Field(default=30, ge=1)
    allowed_users: list[str] | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    category: ToolCategory
    policy: ToolPolicy
    parameters_schema: dict[str, Any]
    return_schema: dict[str, Any]


class ToolContext(BaseModel):
    user_id: str
    correlation_id: str
    approved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] | None = None

    def log_action(self, action: str, metadata: dict[str, Any]) -> None:
        logger.info(
            "tool_action",
            extra={
                "user_id": self.user_id,
                "correlation_id": self.correlation_id,
                "action": action,
                "metadata": metadata,
            },
        )


def _extract_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in ("ctx", "context"):
            continue
        properties[name] = {"type": "string"}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True,
    }


def _extract_return_schema(func: Callable[..., Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": True}


def tool(
    name: str,
    description: str,
    category: ToolCategory,
    requires_approval: bool = False,
    **policy_kwargs: Any,
):
    def decorator(func: Callable[..., Any]):
        from backend.tools.registry import ToolRegistry

        tool_def = ToolDefinition(
            name=name,
            description=description,
            category=category,
            policy=ToolPolicy(requires_approval=requires_approval, **policy_kwargs),
            parameters_schema=_extract_parameters_schema(func),
            return_schema=_extract_return_schema(func),
        )
        ToolRegistry.register(tool_def, func)
        return func

    return decorator
