import pytest

from backend.tools.base import ToolCategory, ToolContext, tool
from backend.tools.executor import ToolExecutor
from backend.tools.policy import ToolPolicyError


@tool(
    name="test_requires_approval",
    description="Test tool requiring approval",
    category=ToolCategory.SYSTEM,
    requires_approval=True,
)
def _test_tool(value: str, ctx: ToolContext | None = None):
    return {"value": value}


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
