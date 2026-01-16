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
                self.config.llm,
                model_type="mini"
            )
        else:
            self.llm = llm_provider

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

            logger.info(
                f"[{self.name}] Classification complete: intent={classification.intent}, "
                f"complexity={classification.complexity}, entities={len(classification.entities)}"
            )

            # Create metadata with LLM call count
            metadata = self._create_metadata()
            metadata.llm_calls = 1

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
        system_prompt = """You are a query classifier for a data assistant.
Your job is to analyze user queries and extract structured information.

**Intent Types:**
- data_query: User wants to retrieve/analyze data
- exploration: User wants to understand what data is available
- explanation: User wants to understand how something works
- meta: User has questions about the system itself

**Entity Types:**
- table: Database table names
- column: Column names
- metric: Business metrics (revenue, sales, users, etc.)
- time_reference: Time periods (last quarter, 2024, yesterday)
- filter: Filter conditions
- other: Other entities

**Complexity Levels:**
- simple: Single table, simple aggregation
- medium: Joins, multiple conditions
- complex: Multiple joins, subqueries, complex logic

**Response Format (JSON):**
```json
{
  "intent": "data_query|exploration|explanation|meta",
  "entities": [
    {
      "entity_type": "metric",
      "value": "total revenue",
      "confidence": 0.95,
      "normalized_value": "revenue"
    }
  ],
  "complexity": "simple|medium|complex",
  "clarification_needed": true/false,
  "clarifying_questions": ["What time period?"],
  "confidence": 0.92
}
```

Be generous with entity extraction - extract anything that might be relevant.
Mark clarification_needed=true if the query is ambiguous."""

        # Add conversation context if available
        context = ""
        if conversation_history:
            context = "\n\n**Previous Conversation:**\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.role
                content = msg.content
                context += f"{role}: {content}\n"

        user_prompt = f"""**User Query:** {query}{context}

Analyze this query and return the classification in JSON format."""

        return system_prompt, user_prompt

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
