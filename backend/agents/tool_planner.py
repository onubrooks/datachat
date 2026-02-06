"""ToolPlannerAgent: choose tools to execute for a query."""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.agents.base import BaseAgent
from backend.config import get_settings
from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMMessage, LLMRequest
from backend.models import ToolPlan, ToolPlannerAgentInput, ToolPlannerAgentOutput
from backend.prompts.loader import PromptLoader

logger = logging.getLogger(__name__)


class ToolPlannerAgent(BaseAgent):
    """Select tools for a given query."""

    def __init__(self, llm_provider=None) -> None:
        super().__init__(name="ToolPlannerAgent")
        self.config = get_settings()
        if llm_provider is None:
            self.llm = LLMProviderFactory.create_default_provider(
                self.config.llm, model_type="mini"
            )
        else:
            self.llm = llm_provider
        self.prompts = PromptLoader()

    async def execute(self, input: ToolPlannerAgentInput) -> ToolPlannerAgentOutput:
        logger.info(f"[{self.name}] Planning tools for query")

        tool_list = json.dumps(input.available_tools, indent=2)
        prompt = self.prompts.render(
            "agents/tool_planner.md",
            user_query=input.query,
            tool_list=tool_list,
        )
        response = await self.llm.generate(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=self.prompts.load("system/main.md")),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0.0,
                max_tokens=800,
            )
        )
        plan = self._parse_plan(response.content)

        metadata = self._create_metadata()
        metadata.llm_calls = 1
        return ToolPlannerAgentOutput(success=True, plan=plan, metadata=metadata)

    def _parse_plan(self, content: str) -> ToolPlan:
        payload = self._extract_json(content)
        if payload:
            try:
                return ToolPlan.model_validate(payload)
            except Exception:
                logger.debug("ToolPlannerAgent payload failed validation")

        return ToolPlan(tool_calls=[], rationale="Fallback to pipeline.", fallback="pipeline")

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any] | None:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end <= start:
                return None
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                return None
