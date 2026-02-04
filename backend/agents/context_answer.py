"""
ContextAnswerAgent

Synthesizes answers directly from DataPoints without SQL execution.
Uses LLM to compose a grounded answer and evidence list.
"""

import json
import logging
from typing import Any

from backend.agents.base import BaseAgent
from backend.config import get_settings
from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMMessage, LLMRequest
from backend.models.agent import (
    ContextAnswer,
    ContextAnswerAgentInput,
    ContextAnswerAgentOutput,
    EvidenceItem,
    LLMError,
)
from backend.prompts.loader import PromptLoader

logger = logging.getLogger(__name__)


class ContextAnswerAgent(BaseAgent):
    """Generate a context-only answer from DataPoints."""

    def __init__(self, llm_provider=None):
        super().__init__(name="ContextAnswerAgent")

        self.config = get_settings()
        if llm_provider is None:
            self.llm = LLMProviderFactory.create_default_provider(
                self.config.llm, model_type="mini"
            )
        else:
            self.llm = llm_provider
        self.prompts = PromptLoader()

    async def execute(self, input: ContextAnswerAgentInput) -> ContextAnswerAgentOutput:
        logger.info(f"[{self.name}] Generating context-only answer")

        try:
            context_summary = self._build_context_summary(input.investigation_memory)
            prompt = self.prompts.render(
                "agents/context_answer.md",
                user_query=input.query,
                context_summary=context_summary,
            )

            request = LLMRequest(
                messages=[
                    LLMMessage(role="system", content=self.prompts.load("system/main.md")),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            response = await self.llm.generate(request)

            context_answer = self._parse_response(response.content, input)

            metadata = self._create_metadata()
            metadata.llm_calls = 1

            return ContextAnswerAgentOutput(
                success=True,
                context_answer=context_answer,
                metadata=metadata,
                next_agent=None,
            )

        except Exception as exc:
            logger.error(f"[{self.name}] Failed to generate context answer: {exc}")
            raise LLMError(self.name, f"Context answer generation failed: {exc}") from exc

    def _build_context_summary(self, memory) -> str:
        lines = []
        for dp in memory.datapoints[:20]:
            metadata = dp.metadata if isinstance(dp.metadata, dict) else {}
            lines.append(
                f"- {dp.datapoint_type} | {dp.name} | id={dp.datapoint_id} | score={dp.score:.2f}"
            )
            if dp.datapoint_type == "Schema":
                table = metadata.get("table_name")
                schema = metadata.get("schema") or metadata.get("schema_name")
                full_name = f"{schema}.{table}" if schema and table else table
                if full_name:
                    lines.append(f"  Table: {full_name}")
                purpose = metadata.get("business_purpose")
                if purpose:
                    lines.append(f"  Purpose: {purpose}")
                columns = metadata.get("key_columns") or []
                if columns:
                    col_names = [col.get("name", "unknown") for col in columns[:8]]
                    lines.append(f"  Columns: {', '.join(col_names)}")
            elif dp.datapoint_type == "Business":
                calculation = metadata.get("calculation")
                if calculation:
                    lines.append(f"  Calculation: {calculation}")
                synonyms = metadata.get("synonyms") or []
                if synonyms:
                    lines.append(f"  Synonyms: {', '.join(synonyms[:5])}")

        return "\n".join(lines) if lines else "No DataPoints available."

    def _parse_response(
        self, content: str, input: ContextAnswerAgentInput
    ) -> ContextAnswer:
        payload = self._extract_json(content)
        if payload:
            return ContextAnswer(
                answer=str(payload.get("answer", "")).strip() or "No answer available.",
                confidence=float(payload.get("confidence", 0.5)),
                evidence=self._parse_evidence(payload.get("evidence", [])),
                needs_sql=bool(payload.get("needs_sql", False)),
                clarifying_questions=[
                    str(item) for item in payload.get("clarifying_questions", [])
                ],
            )

        return self._fallback_answer(input)

    def _extract_json(self, content: str) -> dict[str, Any] | None:
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

    def _parse_evidence(self, evidence: list[dict[str, Any]]) -> list[EvidenceItem]:
        items = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            items.append(
                EvidenceItem(
                    datapoint_id=str(item.get("datapoint_id", "unknown")),
                    name=item.get("name"),
                    type=item.get("type"),
                    reason=item.get("reason"),
                )
            )
        return items

    def _fallback_answer(self, input: ContextAnswerAgentInput) -> ContextAnswer:
        datapoints = input.investigation_memory.datapoints
        if not datapoints:
            return ContextAnswer(
                answer="I don't have enough context to answer that yet.",
                confidence=0.1,
                evidence=[],
                needs_sql=True,
                clarifying_questions=["Which table or metric should I use?"],
            )

        top = datapoints[0]
        evidence = [
            EvidenceItem(
                datapoint_id=top.datapoint_id,
                name=top.name,
                type=top.datapoint_type,
                reason="Top retrieved DataPoint",
            )
        ]
        return ContextAnswer(
            answer=f"Here is a relevant DataPoint: {top.name}.",
            confidence=0.4,
            evidence=evidence,
            needs_sql=False,
            clarifying_questions=[],
        )
