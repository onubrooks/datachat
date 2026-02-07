"""
Unit tests for SQLAgent.

Tests the SQL generation agent that creates SQL queries from natural language
with self-correction capabilities.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.agents.sql import SQLAgent
from backend.llm.models import LLMResponse, LLMUsage
from backend.models.agent import (
    GeneratedSQL,
    InvestigationMemory,
    RetrievedDataPoint,
    SQLAgentInput,
    SQLAgentOutput,
    SQLGenerationError,
    ValidationIssue,
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

    with (
        patch("backend.agents.sql.get_settings", return_value=mock_settings),
        patch(
            "backend.agents.sql.LLMProviderFactory.create_agent_provider",
            return_value=mock_llm_provider,
        ),
    ):
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
                            "nullable": False,
                        },
                        {
                            "name": "date",
                            "type": "DATE",
                            "business_meaning": "Transaction date",
                            "nullable": False,
                        },
                    ],
                    "relationships": [],
                    "gotchas": ["Always filter by date for performance"],
                },
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
                        "Only completed transactions",
                    ],
                },
            ),
        ],
        total_retrieved=2,
        retrieval_mode="hybrid",
        sources_used=["table_fact_sales_001", "metric_revenue_001"],
    )


@pytest.fixture
def sample_sql_agent_input(sample_investigation_memory):
    """Create sample SQLAgentInput."""
    return SQLAgentInput(
        query="What were total sales last quarter?",
        investigation_memory=sample_investigation_memory,
        max_correction_attempts=3,
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
        "clarifying_questions": [],
    }

    return LLMResponse(
        content=f"```json\n{json.dumps(response_json)}\n```",
        model="gpt-4o",
        usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
        finish_reason="stop",
        provider="openai",
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
        self, sql_agent, sample_sql_agent_input, sample_valid_llm_response
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
        self, sql_agent, sample_sql_agent_input, sample_valid_llm_response
    ):
        """Test tracks which DataPoints were used in generation."""
        sql_agent.llm.generate.return_value = sample_valid_llm_response

        output = await sql_agent(sample_sql_agent_input)

        assert "table_fact_sales_001" in output.generated_sql.used_datapoints
        assert "metric_revenue_001" in output.generated_sql.used_datapoints

    @pytest.mark.asyncio
    async def test_handles_clarifying_questions(self, sql_agent, sample_sql_agent_input):
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
            ],
        }

        llm_response = LLMResponse(
            content=f"```json\n{json.dumps(response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai",
        )

        sql_agent.llm.generate.return_value = llm_response

        output = await sql_agent(sample_sql_agent_input)

        assert output.success is True
        assert output.needs_clarification is True
        assert len(output.generated_sql.clarifying_questions) == 1

    @pytest.mark.asyncio
    async def test_uses_deterministic_catalog_for_table_list(
        self, sql_agent, sample_investigation_memory
    ):
        input_data = SQLAgentInput(
            query="list tables",
            investigation_memory=sample_investigation_memory,
            database_type="postgresql",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert output.generated_sql.sql.startswith("SELECT table_schema, table_name")
        assert output.metadata.llm_calls == 0
        sql_agent.llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_deterministic_catalog_for_list_columns(
        self, sql_agent, sample_investigation_memory
    ):
        input_data = SQLAgentInput(
            query="show columns in analytics.fact_sales",
            investigation_memory=sample_investigation_memory,
            database_type="postgresql",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert "information_schema.columns" in output.generated_sql.sql
        assert "table_name = 'fact_sales'" in output.generated_sql.sql
        assert output.metadata.llm_calls == 0

    @pytest.mark.asyncio
    async def test_requests_clarification_for_columns_without_table(
        self, sql_agent, sample_investigation_memory
    ):
        memory = InvestigationMemory(
            query="show columns",
            datapoints=[
                RetrievedDataPoint(
                    datapoint_id="table_sales_001",
                    datapoint_type="Schema",
                    name="Sales",
                    score=0.9,
                    source="hybrid",
                    metadata={"table_name": "public.sales", "key_columns": [{"name": "amount"}]},
                ),
                RetrievedDataPoint(
                    datapoint_id="table_orders_001",
                    datapoint_type="Schema",
                    name="Orders",
                    score=0.85,
                    source="hybrid",
                    metadata={"table_name": "public.orders", "key_columns": [{"name": "order_id"}]},
                ),
            ],
            total_retrieved=2,
            retrieval_mode="hybrid",
            sources_used=["table_sales_001", "table_orders_001"],
        )
        input_data = SQLAgentInput(
            query="show columns",
            investigation_memory=memory,
            database_type="postgresql",
        )

        output = await sql_agent(input_data)

        assert output.needs_clarification is True
        assert "Which table should I list columns for?" in output.generated_sql.clarifying_questions
        assert output.metadata.llm_calls == 0


class TestSelfCorrection:
    """Test SQLAgent self-correction capabilities."""

    @pytest.mark.asyncio
    async def test_self_corrects_missing_table(self, sql_agent, sample_sql_agent_input):
        """Test self-corrects when referencing non-existent table."""
        # First response with wrong table name
        bad_response_json = {
            "sql": "SELECT SUM(amount) FROM wrong_table WHERE date >= '2024-07-01'",
            "explanation": "Query with wrong table",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": [],
        }

        bad_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai",
        )

        # Corrected response
        good_response_json = {
            "sql": "SELECT SUM(amount) FROM analytics.fact_sales WHERE date >= '2024-07-01'",
            "explanation": "Corrected query with right table",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": [],
        }

        good_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(good_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai",
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
    async def test_self_corrects_syntax_error(self, sql_agent, sample_sql_agent_input):
        """Test self-corrects syntax errors."""
        # Response missing FROM clause
        bad_response_json = {
            "sql": "SELECT SUM(amount) WHERE date >= '2024-07-01'",
            "explanation": "Query missing FROM",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": [],
        }

        bad_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai",
        )

        # Corrected response
        good_response_json = {
            "sql": "SELECT SUM(amount) FROM analytics.fact_sales WHERE date >= '2024-07-01'",
            "explanation": "Corrected query with FROM clause",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": [],
        }

        good_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(good_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai",
        )

        sql_agent.llm.generate.side_effect = [bad_llm_response, good_llm_response]

        output = await sql_agent(sample_sql_agent_input)

        assert output.success is True
        assert len(output.correction_attempts) == 1
        assert any(
            issue.issue_type == "syntax" for issue in output.correction_attempts[0].issues_found
        )

    @pytest.mark.asyncio
    async def test_respects_max_correction_attempts(self, sql_agent, sample_sql_agent_input):
        """Test respects maximum correction attempts."""
        # Always return bad SQL
        bad_response_json = {
            "sql": "SELECT SUM(amount) WHERE date >= '2024-07-01'",  # Missing FROM
            "explanation": "Bad query",
            "used_datapoints": ["table_fact_sales_001"],
            "confidence": 0.95,
            "assumptions": [],
            "clarifying_questions": [],
        }

        bad_llm_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai",
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
            clarifying_questions=[],
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
            clarifying_questions=[],
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
            clarifying_questions=[],
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
            clarifying_questions=[],
        )

        issues = sql_agent._validate_sql(valid_sql, sample_sql_agent_input)

        assert len(issues) == 0

    def test_accepts_ctes(self, sql_agent, sample_sql_agent_input):
        """Test accepts CTEs (Common Table Expressions) without flagging as missing tables."""
        # SQL with CTE - the 'sales' CTE should not be flagged as missing_table
        cte_sql = GeneratedSQL(
            sql="""WITH sales AS (
                SELECT amount, date FROM analytics.fact_sales WHERE date >= '2024-07-01'
            )
            SELECT SUM(amount) FROM sales""",
            explanation="Query using CTE",
            used_datapoints=["table_fact_sales_001"],
            confidence=0.95,
            assumptions=[],
            clarifying_questions=[],
        )

        issues = sql_agent._validate_sql(cte_sql, sample_sql_agent_input)

        # Should have NO issues - 'sales' is a CTE, not a missing table
        assert len(issues) == 0

    def test_accepts_multiple_ctes(self, sql_agent, sample_sql_agent_input):
        """Test accepts multiple CTEs."""
        multi_cte_sql = GeneratedSQL(
            sql="""WITH
                sales AS (SELECT amount FROM analytics.fact_sales),
                filtered_sales AS (SELECT amount FROM sales WHERE amount > 100)
            SELECT SUM(amount) FROM filtered_sales""",
            explanation="Query with multiple CTEs",
            used_datapoints=["table_fact_sales_001"],
            confidence=0.95,
            assumptions=[],
            clarifying_questions=[],
        )

        issues = sql_agent._validate_sql(multi_cte_sql, sample_sql_agent_input)

        # Both 'sales' and 'filtered_sales' are CTEs, should not be flagged
        assert len(issues) == 0

    def test_skips_missing_table_without_schema_datapoints(self, sql_agent):
        """Test missing_table validation is skipped in credentials-only mode."""
        memory = InvestigationMemory(
            query="What is total sales?",
            datapoints=[],
            total_retrieved=0,
            retrieval_mode="hybrid",
            sources_used=[],
        )
        sql_input = SQLAgentInput(
            query="What is total sales?",
            investigation_memory=memory,
        )
        generated_sql = GeneratedSQL(
            sql="SELECT SUM(amount) FROM sales",
            explanation="Sum sales amount",
            used_datapoints=[],
            confidence=0.7,
            assumptions=[],
            clarifying_questions=[],
        )

        issues = sql_agent._validate_sql(generated_sql, sql_input)

        assert not any(issue.issue_type == "missing_table" for issue in issues)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT COUNT(*) FROM pg_tables",
            "SELECT COUNT(*) FROM information_schema.tables",
        ],
    )
    def test_accepts_catalog_tables(self, sql_agent, sample_sql_agent_input, sql):
        """Test accepts catalog tables without DataPoints."""
        catalog_sql = GeneratedSQL(
            sql=sql,
            explanation="Catalog query",
            used_datapoints=[],
            confidence=0.8,
            assumptions=[],
            clarifying_questions=[],
        )

        issues = sql_agent._validate_sql(catalog_sql, sample_sql_agent_input)

        assert len(issues) == 0

    def test_accepts_clickhouse_system_tables(self, sql_agent, sample_sql_agent_input):
        """Test accepts ClickHouse system tables even when DataPoints exist."""
        sql_input = sample_sql_agent_input.model_copy(update={"database_type": "clickhouse"})
        catalog_sql = GeneratedSQL(
            sql="SELECT name FROM system.tables",
            explanation="ClickHouse catalog query",
            used_datapoints=[],
            confidence=0.8,
            assumptions=[],
            clarifying_questions=[],
        )

        issues = sql_agent._validate_sql(catalog_sql, sql_input)

        assert len(issues) == 0

    def test_accepts_mysql_show_tables(self, sql_agent, sample_sql_agent_input):
        """Test accepts MySQL SHOW TABLES in validation."""
        sql_input = sample_sql_agent_input.model_copy(update={"database_type": "mysql"})
        catalog_sql = GeneratedSQL(
            sql="SHOW TABLES",
            explanation="MySQL catalog query",
            used_datapoints=[],
            confidence=0.8,
            assumptions=[],
            clarifying_questions=[],
        )

        issues = sql_agent._validate_sql(catalog_sql, sql_input)

        assert len(issues) == 0


class TestPromptBuilding:
    """Test prompt construction logic."""

    @pytest.mark.asyncio
    async def test_builds_generation_prompt_with_schema(
        self, sql_agent, sample_sql_agent_input
    ):
        """Test generation prompt includes schema context."""
        prompt = await sql_agent._build_generation_prompt(sample_sql_agent_input)

        assert "fact_sales" in prompt
        assert "amount" in prompt
        assert "date" in prompt
        assert sample_sql_agent_input.query in prompt

    @pytest.mark.asyncio
    async def test_builds_generation_prompt_with_business_rules(
        self, sql_agent, sample_sql_agent_input
    ):
        """Test generation prompt includes business rules."""
        prompt = await sql_agent._build_generation_prompt(sample_sql_agent_input)

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
            clarifying_questions=[],
        )

        issues = [
            ValidationIssue(
                issue_type="missing_table",
                message="Table 'wrong_table' not found",
                suggested_fix="Use analytics.fact_sales",
            )
        ]

        prompt = sql_agent._build_correction_prompt(generated_sql, issues, sample_sql_agent_input)

        assert "wrong_table" in prompt
        assert "not found" in prompt.lower()
        assert "analytics.fact_sales" in prompt or "fact_sales" in prompt


class TestDatabaseContext:
    """Test database context propagation into SQL generation."""

    def test_introspection_query_respects_database_type(self, sql_agent):
        postgres_query = sql_agent._build_introspection_query(
            "what tables are available?",
            database_type="postgresql",
        )
        mysql_query = sql_agent._build_introspection_query(
            "show tables",
            database_type="mysql",
        )
        clickhouse_query = sql_agent._build_introspection_query(
            "what tables are available?",
            database_type="clickhouse",
        )
        bigquery_query = sql_agent._build_introspection_query(
            "list tables",
            database_type="bigquery",
        )
        redshift_query = sql_agent._build_introspection_query(
            "list tables",
            database_type="redshift",
        )
        assert postgres_query is not None
        assert "information_schema.tables" in postgres_query
        assert mysql_query is not None
        assert "information_schema.tables" in mysql_query
        assert clickhouse_query is not None
        assert "system.tables" in clickhouse_query
        assert bigquery_query is not None
        assert "information_schema.tables" in bigquery_query.lower()
        assert redshift_query is not None
        assert "pg_table_def" in redshift_query.lower()

    def test_row_count_fallback_uses_explicit_table(self, sql_agent):
        memory = InvestigationMemory(
            query="How many rows are in pg_tables?",
            datapoints=[],
            total_retrieved=0,
            retrieval_mode="hybrid",
            sources_used=[],
        )
        sql_input = SQLAgentInput(
            query="How many rows are in pg_tables?",
            investigation_memory=memory,
        )
        sql = sql_agent._build_row_count_fallback(sql_input)
        assert sql == "SELECT COUNT(*) AS row_count FROM pg_tables"

    def test_row_count_fallback_schema_qualified_table(self, sql_agent):
        memory = InvestigationMemory(
            query="How many rows are in information_schema.tables?",
            datapoints=[],
            total_retrieved=0,
            retrieval_mode="hybrid",
            sources_used=[],
        )
        sql_input = SQLAgentInput(
            query="How many rows are in information_schema.tables?",
            investigation_memory=memory,
        )
        sql = sql_agent._build_row_count_fallback(sql_input)
        assert sql == "SELECT COUNT(*) AS row_count FROM information_schema.tables"

    def test_sample_rows_fallback_uses_explicit_table(self, sql_agent):
        memory = InvestigationMemory(
            query="Show me the first 2 rows from public.orders",
            datapoints=[],
            total_retrieved=0,
            retrieval_mode="hybrid",
            sources_used=[],
        )
        sql_input = SQLAgentInput(
            query="Show me the first 2 rows from public.orders",
            investigation_memory=memory,
        )
        sql = sql_agent._build_sample_rows_fallback(sql_input)
        assert sql == "SELECT * FROM public.orders LIMIT 2"

    def test_list_columns_fallback_uses_explicit_table(self, sql_agent):
        memory = InvestigationMemory(
            query="show columns in public.orders",
            datapoints=[],
            total_retrieved=0,
            retrieval_mode="hybrid",
            sources_used=[],
        )
        sql_input = SQLAgentInput(
            query="show columns in public.orders",
            investigation_memory=memory,
            database_type="postgresql",
        )
        sql = sql_agent._build_list_columns_fallback(sql_input)
        assert sql is not None
        assert "information_schema.columns" in sql
        assert "table_name = 'orders'" in sql

    @pytest.mark.asyncio
    async def test_build_prompt_uses_input_database_context(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(
            update={
                "database_type": "clickhouse",
                "database_url": "clickhouse://user:pass@click.example.com:8123/analytics",
            }
        )
        with patch.object(
            sql_agent,
            "_get_live_schema_context",
            new=AsyncMock(return_value=None),
        ) as mock_live_context:
            prompt = await sql_agent._build_generation_prompt(sql_input)

        assert "clickhouse" in prompt.lower()
        assert mock_live_context.await_count == 1
        assert mock_live_context.await_args.kwargs["database_type"] == "clickhouse"
        assert mock_live_context.await_args.kwargs["database_url"] == sql_input.database_url
        assert mock_live_context.await_args.kwargs["include_profile"] is False

    @pytest.mark.asyncio
    async def test_prompt_includes_conversation_context(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(
            update={
                "query": "sales",
                "conversation_history": [
                    {"role": "user", "content": "Show me the first 5 rows"},
                    {"role": "assistant", "content": "Which table should I use?"},
                ]
            }
        )
        prompt = await sql_agent._build_generation_prompt(sql_input)

        assert "conversation" in prompt.lower()
        assert "which table should i use" in prompt.lower()
        assert "show 5 rows from sales" in prompt.lower()

    @pytest.mark.asyncio
    async def test_live_schema_lookup_prefers_input_database_url(self, sql_agent):
        sql_agent.config.database.url = "postgresql://wrong:wrong@wrong-host:5432/wrong_db"

        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock()
        mock_connector.close = AsyncMock()

        with (
            patch("backend.agents.sql.PostgresConnector", return_value=mock_connector) as connector_cls,
            patch.object(
                sql_agent,
                "_fetch_live_schema_context",
                new=AsyncMock(return_value=("schema-context", ["public.sales"])),
            ),
        ):
            context = await sql_agent._get_live_schema_context(
                query="show tables",
                database_type="postgresql",
                database_url="postgresql://demo:demo@chosen-host:5432/chosen_db",
            )

        assert context == "schema-context"
        connector_cls.assert_called_once()
        called = connector_cls.call_args.kwargs
        assert called["host"] == "chosen-host"
        assert called["database"] == "chosen_db"

    def test_build_cached_profile_context_uses_matching_focus_tables(self, sql_agent):
        with patch(
            "backend.agents.sql.load_profile_cache",
            return_value={
                "tables": [
                    {
                        "name": "public.orders",
                        "status": "completed",
                        "row_count": 100,
                        "columns": [
                            {"name": "order_id", "data_type": "integer"},
                            {"name": "total_amount", "data_type": "numeric"},
                        ],
                    },
                    {
                        "name": "public.customers",
                        "status": "completed",
                        "row_count": 50,
                        "columns": [{"name": "id", "data_type": "integer"}],
                    },
                ]
            },
        ):
            context = sql_agent._build_cached_profile_context(
                db_type="postgresql",
                db_url="postgresql://demo:demo@localhost:5432/warehouse",
                focus_tables=["public.orders"],
            )

        assert "Auto-profile cache snapshot" in context
        assert "public.orders" in context
        assert "public.customers" not in context


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
            provider="openai",
        )

        sql_agent.llm.generate.return_value = invalid_response

        output = await sql_agent(sample_sql_agent_input)
        assert output.needs_clarification is True
        assert output.generated_sql.clarifying_questions

    @pytest.mark.asyncio
    async def test_handles_missing_required_fields(self, sql_agent, sample_sql_agent_input):
        """Test handles JSON missing required fields."""
        # Response missing 'sql' field
        bad_response_json = {"explanation": "Query explanation", "confidence": 0.9}

        bad_response = LLMResponse(
            content=f"```json\n{json.dumps(bad_response_json)}\n```",
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            finish_reason="stop",
            provider="openai",
        )

        sql_agent.llm.generate.return_value = bad_response

        output = await sql_agent(sample_sql_agent_input)
        assert output.needs_clarification is True
        assert output.generated_sql.clarifying_questions

    @pytest.mark.asyncio
    async def test_parses_sql_from_markdown_block_when_json_missing(
        self, sql_agent, sample_sql_agent_input
    ):
        llm_response = LLMResponse(
            content=(
                "I can run this query:\n\n"
                "```sql\nSELECT SUM(amount) FROM analytics.fact_sales;\n```"
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=400, completion_tokens=60, total_tokens=460),
            finish_reason="stop",
            provider="openai",
        )
        sql_agent.llm.generate.return_value = llm_response

        output = await sql_agent(sample_sql_agent_input)

        assert output.success is True
        assert output.needs_clarification is False
        assert output.generated_sql.sql == "SELECT SUM(amount) FROM analytics.fact_sales"

    def test_does_not_treat_natural_language_show_phrase_as_sql(self, sql_agent):
        text = (
            '{"explanation": "I can show you the first 5 rows if you provide a table.", '
            '"clarifying_questions": ["Which table?"]}'
        )
        extracted = sql_agent._extract_sql_statement(text)
        assert extracted is None

    def test_short_command_like_followup_is_not_treated_as_table_hint(self, sql_agent):
        assert sql_agent._looks_like_followup_hint("show columns") is False

    def test_merge_query_with_table_hint_replaces_old_table_reference(self, sql_agent):
        merged = sql_agent._merge_query_with_table_hint(
            "show 2 rows in public.sales",
            "petra_campuses",
        )
        assert merged == "Show 2 rows from petra_campuses"


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
