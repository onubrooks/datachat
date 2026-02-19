"""
Unit tests for ClassifierAgent.

Tests query classification including:
- Intent detection
- Entity extraction (tables, metrics, time references)
- Complexity assessment
- Clarification detection
- Conversation history handling
"""

import pytest

from backend.agents.classifier import ClassifierAgent
from backend.models import ClassifierAgentInput


class TestClassifierAgent:
    """Test suite for ClassifierAgent."""

    @pytest.fixture
    def classifier_agent(self, mock_llm_provider, mock_openai_api_key):
        """Create ClassifierAgent with mocked LLM."""
        agent = ClassifierAgent()
        agent.llm = mock_llm_provider
        return agent

    @pytest.fixture
    def sample_input(self):
        """Sample ClassifierAgentInput."""
        return ClassifierAgentInput(
            query="What was total revenue last quarter?",
            conversation_history=[],
            context={},
        )

    # ============================================================================
    # Intent Detection Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_detects_data_query_intent(
        self, classifier_agent, sample_input, mock_llm_provider
    ):
        """Test that data queries are correctly classified."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "metric", "value": "revenue", "confidence": 0.95}
              ],
              "complexity": "simple",
              "clarification_needed": false,
              "clarifying_questions": [],
              "confidence": 0.92
            }
            """
        )

        result = await classifier_agent.execute(sample_input)

        assert result.success is True
        assert result.classification.intent == "data_query"
        assert result.next_agent == "ContextAgent"

    @pytest.mark.asyncio
    async def test_detects_exploration_intent(self, classifier_agent, mock_llm_provider):
        """Test exploration intent detection."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "exploration",
              "entities": [],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.88
            }
            """
        )

        input = ClassifierAgentInput(
            query="What data do you have about sales?",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert result.classification.intent == "exploration"
        assert result.next_agent is None  # No ContextAgent for exploration

    @pytest.mark.asyncio
    async def test_detects_explanation_intent(self, classifier_agent, mock_llm_provider):
        """Test explanation intent detection."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "explanation",
              "entities": [
                {"entity_type": "metric", "value": "revenue", "confidence": 0.9}
              ],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.85
            }
            """
        )

        input = ClassifierAgentInput(
            query="How is revenue calculated?",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert result.classification.intent == "explanation"

    @pytest.mark.asyncio
    async def test_detects_meta_intent(self, classifier_agent, mock_llm_provider):
        """Test meta/system intent detection."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "meta",
              "entities": [],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.95
            }
            """
        )

        input = ClassifierAgentInput(
            query="What can you help me with?",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert result.classification.intent == "meta"

    # ============================================================================
    # Entity Extraction Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_extracts_table_entities(self, classifier_agent, mock_llm_provider):
        """Test table name extraction."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "table", "value": "fact_sales", "confidence": 0.98, "normalized_value": "analytics.fact_sales"}
              ],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.92
            }
            """
        )

        input = ClassifierAgentInput(
            query="Show me data from fact_sales",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert len(result.classification.entities) == 1
        entity = result.classification.entities[0]
        assert entity.entity_type == "table"
        assert entity.value == "fact_sales"
        assert entity.normalized_value == "analytics.fact_sales"

    @pytest.mark.asyncio
    async def test_extracts_metric_entities(
        self, classifier_agent, sample_input, mock_llm_provider
    ):
        """Test metric extraction including synonyms."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "metric", "value": "total revenue", "confidence": 0.95, "normalized_value": "revenue"},
                {"entity_type": "metric", "value": "sales", "confidence": 0.90, "normalized_value": "revenue"}
              ],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.92
            }
            """
        )

        result = await classifier_agent.execute(sample_input)

        metrics = [e for e in result.classification.entities if e.entity_type == "metric"]
        assert len(metrics) == 2
        assert metrics[0].value == "total revenue"
        assert metrics[0].normalized_value == "revenue"

    @pytest.mark.asyncio
    async def test_extracts_time_references(
        self, classifier_agent, sample_input, mock_llm_provider
    ):
        """Test time reference extraction."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "time_reference", "value": "last quarter", "confidence": 0.92, "normalized_value": "Q4_2023"}
              ],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.90
            }
            """
        )

        result = await classifier_agent.execute(sample_input)

        time_refs = [e for e in result.classification.entities if e.entity_type == "time_reference"]
        assert len(time_refs) == 1
        assert time_refs[0].value == "last quarter"

    @pytest.mark.asyncio
    async def test_extracts_column_entities(self, classifier_agent, mock_llm_provider):
        """Test column name extraction."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "column", "value": "customer_id", "confidence": 0.95},
                {"entity_type": "column", "value": "amount", "confidence": 0.95}
              ],
              "complexity": "medium",
              "clarification_needed": false,
              "confidence": 0.88
            }
            """
        )

        input = ClassifierAgentInput(
            query="Group by customer_id and sum amount",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        columns = [e for e in result.classification.entities if e.entity_type == "column"]
        assert len(columns) == 2

    @pytest.mark.asyncio
    async def test_extracts_filter_entities(self, classifier_agent, mock_llm_provider):
        """Test filter extraction."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "filter", "value": "amount > 1000", "confidence": 0.90}
              ],
              "complexity": "medium",
              "clarification_needed": false,
              "confidence": 0.85
            }
            """
        )

        input = ClassifierAgentInput(
            query="Show sales where amount is over 1000",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        filters = [e for e in result.classification.entities if e.entity_type == "filter"]
        assert len(filters) >= 1

    # ============================================================================
    # Complexity Assessment Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_simple_complexity(self, classifier_agent, sample_input, mock_llm_provider):
        """Test simple query complexity."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.92
            }
            """
        )

        result = await classifier_agent.execute(sample_input)

        assert result.classification.complexity == "simple"

    @pytest.mark.asyncio
    async def test_medium_complexity(self, classifier_agent, mock_llm_provider):
        """Test medium query complexity."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [],
              "complexity": "medium",
              "clarification_needed": false,
              "confidence": 0.85
            }
            """
        )

        input = ClassifierAgentInput(
            query="What's the revenue by region and product category?",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert result.classification.complexity == "medium"

    @pytest.mark.asyncio
    async def test_complex_complexity(self, classifier_agent, mock_llm_provider):
        """Test complex query complexity."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [],
              "complexity": "complex",
              "clarification_needed": false,
              "confidence": 0.80
            }
            """
        )

        input = ClassifierAgentInput(
            query="Compare YoY revenue growth by region, excluding refunds, with rolling 3-month averages",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert result.classification.complexity == "complex"

    # ============================================================================
    # Clarification Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_flags_ambiguous_queries(self, classifier_agent, mock_llm_provider):
        """Test that ambiguous queries are flagged for clarification."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [],
              "complexity": "simple",
              "clarification_needed": true,
              "clarifying_questions": ["Which time period do you mean?"],
              "confidence": 0.60
            }
            """
        )

        input = ClassifierAgentInput(
            query="Show me sales",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert result.classification.clarification_needed is True
        assert len(result.classification.clarifying_questions) > 0
        assert "time period" in result.classification.clarifying_questions[0].lower()

    @pytest.mark.asyncio
    async def test_provides_clarifying_questions(self, classifier_agent, mock_llm_provider):
        """Test that clarifying questions are provided."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [],
              "complexity": "medium",
              "clarification_needed": true,
              "clarifying_questions": [
                "Do you mean total sales or average sales?",
                "Which region are you interested in?"
              ],
              "confidence": 0.55
            }
            """
        )

        input = ClassifierAgentInput(
            query="What about the numbers?",
            conversation_history=[],
        )

        result = await classifier_agent.execute(input)

        assert len(result.classification.clarifying_questions) == 2

    # ============================================================================
    # Conversation History Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_uses_conversation_history(self, classifier_agent, mock_llm_provider):
        """Test that conversation history is considered."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "metric", "value": "revenue", "confidence": 0.95}
              ],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.92
            }
            """
        )

        input = ClassifierAgentInput(
            query="What about last month?",
            conversation_history=[
                {"role": "user", "content": "Show me revenue"},
                {"role": "assistant", "content": "Here's the revenue data"},
            ],
        )

        result = await classifier_agent.execute(input)

        # Should resolve based on context
        assert result.classification.intent == "data_query"

    # ============================================================================
    # Edge Cases and Error Handling
    # ============================================================================

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self, classifier_agent, mock_llm_provider):
        """Test graceful handling of invalid JSON from LLM."""
        mock_llm_provider.set_response("This is not valid JSON at all")

        input = ClassifierAgentInput(query="test query", conversation_history=[])

        result = await classifier_agent.execute(input)

        # Should return default classification with clarification
        assert result.success is True
        assert result.classification.clarification_needed is True
        assert result.classification.confidence < 1.0

    @pytest.mark.asyncio
    async def test_handles_partial_json_response(self, classifier_agent, mock_llm_provider):
        """Test handling of partial/malformed JSON."""
        mock_llm_provider.set_response(
            """
            Here's the classification:
            {
              "intent": "data_query",
              "complexity": "simple"
            }
            Some extra text
            """
        )

        input = ClassifierAgentInput(query="test", conversation_history=[])

        result = await classifier_agent.execute(input)

        assert result.success is True
        assert result.classification.intent == "data_query"

    @pytest.mark.asyncio
    async def test_metadata_populated(self, classifier_agent, sample_input, mock_llm_provider):
        """Test that metadata is properly populated."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.92
            }
            """
        )

        result = await classifier_agent.execute(sample_input)

        assert result.metadata is not None
        assert result.metadata.agent_name == "ClassifierAgent"
        assert result.metadata.llm_calls == 1
        assert result.metadata.started_at is not None

    @pytest.mark.asyncio
    async def test_confidence_score_included(
        self, classifier_agent, sample_input, mock_llm_provider
    ):
        """Test that confidence scores are included."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "data_query",
              "entities": [
                {"entity_type": "metric", "value": "revenue", "confidence": 0.95}
              ],
              "complexity": "simple",
              "clarification_needed": false,
              "confidence": 0.87
            }
            """
        )

        result = await classifier_agent.execute(sample_input)

        assert result.classification.confidence == 0.87
        assert result.classification.entities[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, classifier_agent, mock_llm_provider):
        """Test handling of empty queries."""
        mock_llm_provider.set_response(
            """
            {
              "intent": "meta",
              "entities": [],
              "complexity": "simple",
              "clarification_needed": true,
              "clarifying_questions": ["What would you like to know?"],
              "confidence": 0.5
            }
            """
        )

        input = ClassifierAgentInput(query="", conversation_history=[])

        result = await classifier_agent.execute(input)

        assert result.classification.clarification_needed is True
