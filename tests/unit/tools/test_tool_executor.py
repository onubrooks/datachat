import pytest

from backend.tools.base import ToolCategory, ToolContext, tool
from backend.tools.executor import ToolExecutor
from backend.tools.policy import ToolPolicyError
from backend.tools.registry import ToolRegistry


@tool(
    name="test_requires_approval",
    description="Test tool requiring approval",
    category=ToolCategory.SYSTEM,
    requires_approval=True,
)
def _test_tool(value: str, ctx: ToolContext | None = None):
    return {"value": value}


@tool(
    name="test_typed_schema",
    description="Tool with typed arguments",
    category=ToolCategory.SYSTEM,
)
def _typed_schema_tool(
    limit: int = 5,
    include_stats: bool = False,
    threshold: float = 0.25,
    tags: list[str] | None = None,
    options: dict[str, int] | None = None,
    ctx: ToolContext | None = None,
):
    return {
        "limit": limit,
        "include_stats": include_stats,
        "threshold": threshold,
        "tags": tags or [],
        "options": options or {},
    }


@pytest.mark.asyncio
async def test_tool_executor_blocks_without_approval():
    executor = ToolExecutor()
    ctx = ToolContext(user_id="tester", correlation_id="test-1", approved=False)
    with pytest.raises(ToolPolicyError):
        await executor.execute("test_requires_approval", {"value": "hi"}, ctx)


@pytest.mark.asyncio
async def test_tool_executor_runs_with_approval():
    executor = ToolExecutor()
    ctx = ToolContext(user_id="tester", correlation_id="test-2", approved=True)
    result = await executor.execute("test_requires_approval", {"value": "hi"}, ctx)
    assert result["success"] is True
    assert result["result"]["value"] == "hi"


def test_tool_schema_uses_typed_parameter_definitions():
    definition = ToolRegistry.get_definition("test_typed_schema")
    assert definition is not None
    schema = definition.parameters_schema
    props = schema["properties"]
    assert props["limit"]["type"] == "integer"
    assert props["include_stats"]["type"] == "boolean"
    assert props["threshold"]["type"] == "number"
    assert any(
        item.get("type") == "array"
        for item in props["tags"].get("anyOf", [])
        if isinstance(item, dict)
    )
    assert any(
        item.get("type") == "object"
        for item in props["options"].get("anyOf", [])
        if isinstance(item, dict)
    )
    assert schema["additionalProperties"] is False
