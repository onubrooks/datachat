from backend.agents.tool_planner import ToolPlannerAgent


def test_tool_planner_parses_valid_plan():
    agent = ToolPlannerAgent.__new__(ToolPlannerAgent)
    content = """
    {
      "tool_calls": [
        {"name": "list_tables", "arguments": {"schema": "public"}}
      ],
      "rationale": "Need tables",
      "fallback": "pipeline"
    }
    """
    plan = agent._parse_plan(content)
    assert plan.tool_calls[0].name == "list_tables"


def test_tool_planner_falls_back_on_invalid():
    agent = ToolPlannerAgent.__new__(ToolPlannerAgent)
    plan = agent._parse_plan("not json")
    assert plan.fallback == "pipeline"
