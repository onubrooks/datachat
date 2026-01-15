"""
Unit tests for SQLAgent.

Tests the SQL generation agent that creates SQL queries from natural language
with self-correction capabilities.
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

from backend.agents.sql import SQLAgent
from backend.llm.models import LLMResponse, LLMUsage
from backend.models.agent import (
    SQLAgentInput,
    SQLAgentOutput,
    GeneratedSQL,
    ValidationIssue,
    CorrectionAttempt,
    InvestigationMemory,
    RetrievedDataPoint,
    SQLGenerationError,
    LLMError,
)


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = Mock()
    provider.generate = AsyncMock()
    provider.provider = "openai"
    provider.model = "gpt-4o"
    return provider


@pytest.fixture
def sql_agent(mock_llm_provider):
    """Create SQLAgent with mock LLM provider."""
    # Mock get_settings to avoid API key validation
    mock_settings = Mock()
    mock_settings.llm = Mock()

    with patch("backend.agents.sql.get_settings", return_value=mock_settings), \
         patch("backend.agents.sql.LLMProviderFactory.create_agent_provider", return_value=mock_llm_provider):
        agent = SQLAgent()
    return agent


@pytest.fixture
def sample_investigation_memory():
    """Create sample investigation memory with schema context."""
    return InvestigationMemory(
        query="What were total sales last quarter?",
        datapoints=[
            RetrievedDataPoint(
                datapoint_id="table_fact_sales_001",
                datapoint_type="Schema",
                name="Fact Sales Table",
                score=0.95,
                source="hybrid",
                metadata={
                    "table_name": "analytics.fact_sales",
                    "schema": "analytics",
                    "business_purpose": "Central fact table for all sales transactions",
                    "key_columns": [
                        {
                            "name": "amount",
                            "type": "DECIMAL(18,2)",
                            "business_meaning": "Transaction value in USD",
                            "nullable": False
                        },
                        {
                            "name": "date",
                            "type": "DATE",
                            "business_meaning": "Transaction date",
                            "nullable": False
                        }
                    ],
                    "relationships": [],
                    "gotchas": ["Always filter by date for performance"]
                }
            ),
            RetrievedDataPoint(
                datapoint_id="metric_revenue_001",
                datapoint_type="Business",
                name="Revenue",
                score=0.88,
                source="vector",
                metadata={
                    "calculation": "SUM(fact_sales.amount) WHERE status = 'completed'",
                    "synonyms": ["sales", "income"],
                    "business_rules": [
                        "Exclude refunds (status != 'refunded')",
                        "Only completed transactions"
                    ]
                }
            )
        ],
        total_retrieved=2,
        retrieval_mode="hybrid",
        sources_used=["table_fact_sales_001", "metric_revenue_001"]
    )


@pytest.fixture
def sample_sql_agent_input(sample_investigation_memory):
    """Create sample SQLAgentInput."""
    return SQLAgentInput(
        query="What were total sales last quarter?",
        investigation_memory=sample_investigation_memory,
        max_correction_attempts=3
    )


@pytest.fixture
def sample_valid_llm_response():
    """Create sample valid LLM response with SQL."""
    response_json = {
        "sql": "SELECT SUM(amount) FROM analytics.fact_sales WHERE date >= '2024-07-01' AND date < '2024-10-01'",
        "explanation": "This query calculates total sales for Q3 2024",
        "used_datapoints": ["table_fact_sales_001", "metric_revenue_001"],
        "confidence": 0.95,
        "assumptions": ["'last quarter' refers to Q3 2024"],
        "clarifying_questions": []
    }

    return LLMResponse(
        content=f"```json\n{json.dumps(response_json)}\n```",
        model="gpt-4o",
        usage=LLMUsage(
            prompt_tokens=500,
            completion_tokens=150,
            total_tokens=650
        ),
        finish_reason="stop",
        provider="openai"
    )


class TestInitialization:
    """Test SQLAgent initialization."""

    def test_initialization_creates_llm_provider(self, sql_agent):
        """Test agent initializes with LLM provider."""
        assert sql_agent.name == "SQLAgent"
        assert sql_agent.llm is not None
        assert sql_agent.llm.provider == "openai"
        assert sql_agent.llm.model == "gpt-4o"


class TestExecution:
    """Test SQLAgent execution."""

    @pytest.mark.asyncio
    async def test_successful_sql_generation(
        self,
        sql_agent,
        sample_sql_agent_input,
        sample_valid_llm_response
    ):
        """Test successful SQL generation without corrections."""
        # Mock LLM response
        sql_agent.llm.generate.return_value = sample_valid_llm_response

        # Execute
        output = await sql_agent(sample_sql_agent_input)

        # Assertions
        assert isinstance(output, SQLAgentOutput)
        assert output.success is True
        assert output.generated_sql.sql.startswith("SELECT")
        assert "fact_sales" in output.generated_sql.sql.lower()
        assert output.generated_sql.confidence == 0.95
        assert len(output.correction_attempts) == 0
        assert output.needs_clarification is False
        assert output.metadata.llm_calls == 1
        assert output.metadata.tokens_used == 650

    @pytest.mark.asyncio
    async def test_tracks_used_datapoints(
        self,
        sql_agent,
        sample_sql_agent_input,
        sample_valid_llm_response
    ):
        """Test tracks which DataPoints were used in generation."""
        sql_agent.llm.generate.return_value = sample_valid_llm_response

        output = await sql_agent(sample_sql_agent_input)

        assert "table_fact_sales_001" in output.generated_sql.used_datapoints
        assert "metric_revenue_001" in output.generated_sql.used_datapoints

    @pytest.mark.asyncio
    async def test_handles_clarifying_questions(
        self,
        sql_agent,
        sample_sql_agent_input
    ):
        """Test handles ambiguous queries with clarifying questions."""
        # Create response with clarifying questions
        response_json = {
            "sql": "SELECT SUM(amount) FROM analytics.fact_sales WHERE date >= '2024-07-01'",
            "explanation": "Partial query - needs date range clarification",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.7,
            "assumptions": [],
            "clarifying_questions": [
                "Which quarter do you mean by 'last quarter'? Q3 2024 or Q2 2024?"
            ]
        }

        llm_response = LLMResponse(
            content=f"```json\n{json.dumps(response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        sql_agent.llm.generate.return_value = llm_response

        output = await sql_agent(sample_sql_agent_input)

        assert output.success is True
        assert output.needs_clarification is True
        assert len(output.generated_sql.clarifying_questions) == 1


class TestSelfCorrection:
    """Test SQLAgent self-correction capabilities."""

    @pytest.mark.asyncio
    async def test_self_corrects_missing_table(
        self,
        sql_agent,
        sample_sql_agent_input
    ):
        """Test self-corrects when referencing non-existent table."""
        # First response with wrong table name
        bad_response_json = {
            "sql": "SELECT SUM(amount) FROM wrong_table WHERE date >= '2024-07-01'",
            "explanation": "Query with wrong table",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": []
        }

        bad_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        # Corrected response
        good_response_json = {
            "sql": "SELECT SUM(amount) FROM analytics.fact_sales WHERE date >= '2024-07-01'",
            "explanation": "Corrected query with right table",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": []
        }

        good_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(good_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        # Mock: first call returns bad SQL, second call returns corrected SQL
        sql_agent.llm.generate.side_effect = [bad_llm_response, good_llm_response]

        output = await sql_agent(sample_sql_agent_input)

        # Should have made correction
        assert output.success is True
        assert len(output.correction_attempts) == 1
        assert output.correction_attempts[0].attempt_number == 1
        assert "wrong_table" in output.correction_attempts[0].original_sql.lower()
        assert "fact_sales" in output.correction_attempts[0].corrected_sql.lower()
        assert output.correction_attempts[0].success is True
        assert output.metadata.llm_calls == 2  # Initial + 1 correction

    @pytest.mark.asyncio
    async def test_self_corrects_syntax_error(
        self,
        sql_agent,
        sample_sql_agent_input
    ):
        """Test self-corrects syntax errors."""
        # Response missing FROM clause
        bad_response_json = {
            "sql": "SELECT SUM(amount) WHERE date >= '2024-07-01'",
            "explanation": "Query missing FROM",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": []
        }

        bad_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        # Corrected response
        good_response_json = {
            "sql": "SELECT SUM(amount) FROM analytics.fact_sales WHERE date >= '2024-07-01'",
            "explanation": "Corrected query with FROM clause",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": []
        }

        good_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(good_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        sql_agent.llm.generate.side_effect = [bad_llm_response, good_llm_response]

        output = await sql_agent(sample_sql_agent_input)

        assert output.success is True
        assert len(output.correction_attempts) == 1
        assert any(issue.issue_type == "syntax" for issue in output.correction_attempts[0].issues_found)

    @pytest.mark.asyncio
    async def test_respects_max_correction_attempts(
        self,
        sql_agent,
        sample_sql_agent_input
    ):
        """Test respects maximum correction attempts."""
        # Always return bad SQL
        bad_response_json = {
            "sql": "SELECT SUM(amount) WHERE date >= '2024-07-01'",  # Missing FROM
            "explanation": "Bad query",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": []
        }

        bad_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        sql_agent.llm.generate.return_value = bad_llm_response

        # Set max attempts to 2
        sample_sql_agent_input.max_correction_attempts = 2

        output = await sql_agent(sample_sql_agent_input)

        # Should try: initial + 2 corrections = 3 LLM calls
        assert output.metadata.llm_calls == 3
        assert len(output.correction_attempts) == 2
        assert output.needs_clarification is True  # Has unresolved issues


class TestValidation:
    """Test SQL validation logic."""

    def test_validates_select_statement(self, sql_agent, sample_sql_agent_input):
        """Test validates SQL starts with SELECT."""
        # Create invalid SQL (no SELECT)
        invalid_sql = GeneratedSQL(
            sql="UPDATE fact_sales SET amount = 100",
            explanation="Update query",
            used_datapoints=[],
            confidence=0.9,
            assumptions=[],
            clarifying_questions=[]
        )

        issues = sql_agent._validate_sql(invalid_sql, sample_sql_agent_input)

        assert len(issues) > 0
        assert any(issue.issue_type == "syntax" and "SELECT" in issue.message for issue in issues)

    def test_validates_from_clause(self, sql_agent, sample_sql_agent_input):
        """Test validates SQL has FROM clause."""
        invalid_sql = GeneratedSQL(
            sql="SELECT SUM(amount) WHERE date > '2024-01-01'",
            explanation="Query missing FROM",
            used_datapoints=[],
            confidence=0.9,
            assumptions=[],
            clarifying_questions=[]
        )

        issues = sql_agent._validate_sql(invalid_sql, sample_sql_agent_input)

        assert len(issues) > 0
        assert any(issue.issue_type == "syntax" and "FROM" in issue.message for issue in issues)

    def test_validates_table_names(self, sql_agent, sample_sql_agent_input):
        """Test validates table names exist in DataPoints."""
        invalid_sql = GeneratedSQL(
            sql="SELECT SUM(amount) FROM nonexistent_table WHERE date > '2024-01-01'",
            explanation="Query with wrong table",
            used_datapoints=[],
            confidence=0.9,
            assumptions=[],
            clarifying_questions=[]
        )

        issues = sql_agent._validate_sql(invalid_sql, sample_sql_agent_input)

        assert len(issues) > 0
        assert any(issue.issue_type == "missing_table" for issue in issues)

    def test_accepts_valid_sql(self, sql_agent, sample_sql_agent_input):
        """Test accepts valid SQL with no issues."""
        valid_sql = GeneratedSQL(
            sql="SELECT SUM(amount) FROM analytics.fact_sales WHERE date >= '2024-07-01'",
            explanation="Valid query",
            used_datapoints=["table_fact_sales_001"],
            confidence=0.95,
            assumptions=[],
            clarifying_questions=[]
        )

        issues = sql_agent._validate_sql(valid_sql, sample_sql_agent_input)

        assert len(issues) == 0


class TestPromptBuilding:
    """Test prompt construction logic."""

    def test_builds_generation_prompt_with_schema(self, sql_agent, sample_sql_agent_input):
        """Test generation prompt includes schema context."""
        prompt = sql_agent._build_generation_prompt(sample_sql_agent_input)

        assert "fact_sales" in prompt
        assert "amount" in prompt
        assert "date" in prompt
        assert sample_sql_agent_input.query in prompt

    def test_builds_generation_prompt_with_business_rules(self, sql_agent, sample_sql_agent_input):
        """Test generation prompt includes business rules."""
        prompt = sql_agent._build_generation_prompt(sample_sql_agent_input)

        assert "Revenue" in prompt
        assert "completed" in prompt or "refund" in prompt.lower()

    def test_builds_correction_prompt(self, sql_agent, sample_sql_agent_input):
        """Test correction prompt includes issues and original SQL."""
        generated_sql = GeneratedSQL(
            sql="SELECT amount FROM wrong_table",
            explanation="Wrong query",
            used_datapoints=[],
            confidence=0.8,
            assumptions=[],
            clarifying_questions=[]
        )

        issues = [
            ValidationIssue(
                issue_type="missing_table",
                message="Table 'wrong_table' not found",
                suggested_fix="Use analytics.fact_sales"
            )
        ]

        prompt = sql_agent._build_correction_prompt(generated_sql, issues, sample_sql_agent_input)

        assert "wrong_table" in prompt
        assert "not found" in prompt.lower()
        assert "analytics.fact_sales" in prompt or "fact_sales" in prompt


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self, sql_agent, sample_sql_agent_input):
        """Test handles LLM API failures."""
        sql_agent.llm.generate.side_effect = Exception("API Error")

        with pytest.raises(SQLGenerationError) as exc_info:
            await sql_agent(sample_sql_agent_input)

        assert "Failed to generate SQL" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self, sql_agent, sample_sql_agent_input):
        """Test handles invalid JSON in LLM response."""
        invalid_response = LLMResponse(
            content="This is not JSON",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        sql_agent.llm.generate.return_value = invalid_response

        with pytest.raises(SQLGenerationError):
            await sql_agent(sample_sql_agent_input)

    @pytest.mark.asyncio
    async def test_handles_missing_required_fields(self, sql_agent, sample_sql_agent_input):
        """Test handles JSON missing required fields."""
        # Response missing 'sql' field
        bad_response_json = {
            "explanation": "Query explanation",
            "confidence": 0.9
        }

        bad_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai"
        )

        sql_agent.llm.generate.return_value = bad_response

        with pytest.raises(SQLGenerationError):
            await sql_agent(sample_sql_agent_input)


class TestInputValidation:
    """Test input validation."""

    @pytest.mark.asyncio
    async def test_validates_input_type(self, sql_agent):
        """Test validates input is SQLAgentInput."""
        from backend.models.agent import AgentInput, ValidationError

        invalid_input = AgentInput(query="test")

        with pytest.raises(ValidationError) as exc_info:
            await sql_agent(invalid_input)

        assert "Expected SQLAgentInput" in str(exc_info.value)
