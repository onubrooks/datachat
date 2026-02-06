"""
ClassifierAgent: Intent detection and entity extraction.

This agent classifies user queries using GPT-4o-mini (fast and cheap):
- Intent classification: data_query, exploration, explanation, meta
- Entity extraction: tables, columns, metrics, time references
- Complexity assessment: simple, medium, complex
- Ambiguity detection and clarification

Uses LLM for intelligent natural language understanding.
"""

import json
import logging
from typing import Any

from backend.agents.base import BaseAgent
from backend.config import get_settings
from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMMessage, LLMRequest
from backend.models import (
    ClassifierAgentInput,
    ClassifierAgentOutput,
    ExtractedEntity,
    LLMError,
    QueryClassification,
)
from backend.prompts.loader import PromptLoader

logger = logging.getLogger(__name__)


class ClassifierAgent(BaseAgent):
    """
    Query classification and entity extraction agent.

    Uses LLM (GPT-4o-mini) to understand user intent and extract entities.
    Fast and cheap - optimized for speed over complex reasoning.
    """

    def __init__(self, llm_provider=None):
        """
        Initialize ClassifierAgent with LLM provider.

        Args:
            llm_provider: Optional LLM provider. If None, creates default provider.
        """
        super().__init__(name="ClassifierAgent")

        # Get configuration
        self.config = get_settings()

        # Get LLM provider (use mini model for speed/cost)
        if llm_provider is None:
            self.llm = LLMProviderFactory.create_default_provider(
                self.config.llm, model_type="mini"
            )
        else:
            self.llm = llm_provider
        self.prompts = PromptLoader()

    async def execute(self, input: ClassifierAgentInput) -> ClassifierAgentOutput:
        """
        Classify user query and extract entities.

        Args:
            input: ClassifierAgentInput with query and conversation history

        Returns:
            ClassifierAgentOutput with classification results

        Raises:
            LLMError: If LLM classification fails
        """
        logger.info(f"[{self.name}] Classifying query: {input.query}")

        try:
            # Build classification prompt
            system_prompt, user_prompt = self._build_classification_prompt(
                input.query, input.conversation_history
            )

            # Call LLM
            request = LLMRequest(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                temperature=0.3,
            )
            response = await self.llm.generate(request)

            # Parse structured response
            classification = self._parse_classification_response(response.content)
            extra_calls = 0

            if self._should_deep_classify(classification):
                deep_classification = await self._deep_classify(
                    input.query, input.conversation_history
                )
                if deep_classification:
                    classification = deep_classification
                    extra_calls = 1

            logger.info(
                f"[{self.name}] Classification complete: intent={classification.intent}, "
                f"complexity={classification.complexity}, entities={len(classification.entities)}"
            )

            # Create metadata with LLM call count
            metadata = self._create_metadata()
            metadata.llm_calls = 1 + extra_calls

            return ClassifierAgentOutput(
                success=True,
                classification=classification,
                metadata=metadata,
                next_agent="ContextAgent" if classification.intent == "data_query" else None,
            )

        except Exception as e:
            logger.error(f"[{self.name}] Classification failed: {e}")
            raise LLMError(self.name, f"Failed to classify query: {e}") from e

    def _build_classification_prompt(
        self, query: str, conversation_history: list[Any]
    ) -> tuple[str, str]:
        """
        Build prompt for query classification.

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages

        Returns:
            Formatted prompt string
        """
        system_prompt = self.prompts.load("system/main.md")

        context = ""
        if conversation_history:
            lines = []
            for msg in conversation_history[-3:]:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                lines.append(f"{role}: {content}")
            context = "\n".join(lines)

        user_prompt = self.prompts.render(
            "agents/classifier.md",
            user_query=query,
            conversation_history=context or "None",
        )

        return system_prompt, user_prompt

    def _should_deep_classify(self, classification: QueryClassification) -> bool:
        if classification.confidence < 0.6:
            return True
        if not classification.entities and classification.complexity != "simple":
            return True
        return False

    async def _deep_classify(
        self, query: str, conversation_history: list[Any]
    ) -> QueryClassification | None:
        system_prompt = self.prompts.load("system/main.md")
        context = ""
        if conversation_history:
            lines = []
            for msg in conversation_history[-3:]:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                lines.append(f"{role}: {content}")
            context = "\n".join(lines)
        user_prompt = self.prompts.render(
            "agents/classifier_deep.md",
            user_query=query,
            conversation_history=context or "None",
        )
        response = await self.llm.generate(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                temperature=0.2,
            )
        )
        try:
            return self._parse_classification_response(response.content)
        except Exception:
            return None

    def _parse_classification_response(self, response: str) -> QueryClassification:
        """
        Parse LLM response into QueryClassification.

        Args:
            response: LLM response text

        Returns:
            QueryClassification object

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Extract JSON from response (LLM might add explanation text)
            response_text = response.strip()

            # Try to find JSON in response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            # Parse entities
            entities = [
                ExtractedEntity(
                    entity_type=e.get("entity_type", "other"),
                    value=e.get("value", ""),
                    confidence=e.get("confidence", 1.0),
                    normalized_value=e.get("normalized_value"),
                )
                for e in data.get("entities", [])
            ]

            return QueryClassification(
                intent=data.get("intent", "data_query"),
                entities=entities,
                complexity=data.get("complexity", "simple"),
                clarification_needed=data.get("clarification_needed", False),
                clarifying_questions=data.get("clarifying_questions", []),
                confidence=data.get("confidence", 1.0),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse classification response: {e}\nResponse: {response}")
            # Return default classification on parse failure
            return QueryClassification(
                intent="data_query",
                entities=[],
                complexity="simple",
                clarification_needed=True,
                clarifying_questions=[
                    "I couldn't fully understand your query. Could you rephrase?"
                ],
                confidence=0.5,
            )
