"""Tool execution engine."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from backend.tools.base import ToolContext
from backend.tools.policy import PolicyEngine, ToolPolicyError
from backend.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    pass


class ToolExecutor:
    def __init__(self, policy_engine: PolicyEngine | None = None) -> None:
        self.policy_engine = policy_engine or PolicyEngine()

    async def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        definition = ToolRegistry.get_definition(name)
        handler = ToolRegistry.get_handler(name)
        if not definition or not handler:
            raise ToolExecutionError(f"Unknown tool: {name}")

        self.policy_engine.enforce(definition, ctx)

        ctx.log_action("tool_invoked", {"tool": name, "args": list(args.keys())})

        try:
            if "ctx" in inspect.signature(handler).parameters:
                result = handler(**args, ctx=ctx)
            else:
                result = handler(**args)

            if inspect.isawaitable(result):
                result = await result

            ctx.log_action("tool_completed", {"tool": name})
            return {
                "tool": name,
                "success": True,
                "result": result,
            }
        except ToolPolicyError:
            raise
        except Exception as exc:
            logger.error(f"Tool execution failed: {name} - {exc}")
            raise ToolExecutionError(str(exc)) from exc

    async def execute_plan(
        self, tool_calls: list[dict[str, Any]], ctx: ToolContext
    ) -> list[dict[str, Any]]:
        results = []
        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments") or {}
            if not name:
                continue
            results.append(await self.execute(name, args, ctx))
        return results
