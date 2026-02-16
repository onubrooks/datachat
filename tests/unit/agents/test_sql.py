"""
Unit tests for SQLAgent.

Tests the SQL generation agent that creates SQL queries from natural language
with self-correction capabilities.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.agents.sql import SQLAgent, TableResolution
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
    mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)

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
    async def test_metric_bundle_query_uses_llm_generation(self, sql_agent):
        """Store-level revenue+margin+waste bundle should go through LLM SQL generation."""
        input_data = SQLAgentInput(
            query="What is total revenue, gross margin, and waste cost by store for the last 30 days?",
            investigation_memory=InvestigationMemory(
                query="metric bundle",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales Transactions",
                        score=0.9,
                        source="vector",
                        metadata={"table_name": "public.grocery_sales_transactions"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_products_001",
                        datapoint_type="Schema",
                        name="Products",
                        score=0.88,
                        source="vector",
                        metadata={"table_name": "public.grocery_products"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_waste_events_001",
                        datapoint_type="Schema",
                        name="Waste",
                        score=0.87,
                        source="vector",
                        metadata={"table_name": "public.grocery_waste_events"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_stores_001",
                        datapoint_type="Schema",
                        name="Stores",
                        score=0.86,
                        source="vector",
                        metadata={"table_name": "public.grocery_stores"},
                    ),
                ],
                total_retrieved=4,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT s.store_name, SUM(t.total_amount) AS revenue, "
                        "SUM(t.total_amount) - SUM(t.quantity * COALESCE(p.unit_cost, 0)) AS gross_margin, "
                        "SUM(w.waste_cost) AS waste_cost "
                        "FROM public.grocery_sales_transactions t "
                        "JOIN public.grocery_products p ON p.product_id = t.product_id "
                        "JOIN public.grocery_stores s ON s.store_id = t.store_id "
                        "LEFT JOIN public.grocery_waste_events w ON w.store_id = t.store_id "
                        "GROUP BY s.store_name"
                    ),
                    "explanation": "Revenue, margin, and waste by store.",
                    "used_datapoints": [],
                    "confidence": 0.92,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=320, completion_tokens=160, total_tokens=480),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)
        assert output.success is True
        assert output.needs_clarification is False
        assert "SELECT s.store_name" in output.generated_sql.sql
        assert "grocery_sales_transactions" in output.generated_sql.sql
        assert "grocery_waste_events" in output.generated_sql.sql
        assert sql_agent.llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_store_metric_bundle_query_uses_related_tables_context(self, sql_agent):
        """LLM output can use tables surfaced via related_tables context."""
        input_data = SQLAgentInput(
            query="What is total revenue, gross margin, and waste cost by store for the last 30 days?",
            investigation_memory=InvestigationMemory(
                query="metric bundle",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales Transactions",
                        score=0.9,
                        source="vector",
                        metadata={"table_name": "public.grocery_sales_transactions"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_waste_events_001",
                        datapoint_type="Schema",
                        name="Waste",
                        score=0.87,
                        source="vector",
                        metadata={"table_name": "public.grocery_waste_events"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_stores_001",
                        datapoint_type="Schema",
                        name="Stores",
                        score=0.86,
                        source="vector",
                        metadata={"table_name": "public.grocery_stores"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="metric_gross_margin_grocery_001",
                        datapoint_type="Business",
                        name="Gross Margin",
                        score=0.85,
                        source="vector",
                        metadata={
                            "related_tables": [
                                "public.grocery_sales_transactions",
                                "public.grocery_products",
                            ]
                        },
                    ),
                ],
                total_retrieved=4,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT SUM(t.total_amount) AS revenue, "
                        "SUM(t.total_amount) - SUM(t.quantity * p.unit_cost) AS gross_margin "
                        "FROM public.grocery_sales_transactions t "
                        "JOIN public.grocery_products p ON p.product_id = t.product_id"
                    ),
                    "explanation": "Uses product costs for margin.",
                    "used_datapoints": [],
                    "confidence": 0.9,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=110, total_tokens=410),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)
        assert output.success is True
        assert "grocery_products" in output.generated_sql.sql
        assert sql_agent.llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_stockout_risk_query_uses_llm_generation(self, sql_agent):
        """SKU stockout risk ranking should use the LLM SQL generation path."""
        input_data = SQLAgentInput(
            query=(
                "Which 5 SKUs have the highest stockout risk this week based on on-hand, "
                "reserved, and reorder level?"
            ),
            investigation_memory=InvestigationMemory(
                query="stockout risk ranking",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_inventory_snapshots_001",
                        datapoint_type="Schema",
                        name="Inventory Snapshots",
                        score=0.92,
                        source="vector",
                        metadata={"table_name": "public.grocery_inventory_snapshots"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_products_001",
                        datapoint_type="Schema",
                        name="Products",
                        score=0.91,
                        source="vector",
                        metadata={"table_name": "public.grocery_products"},
                    ),
                ],
                total_retrieved=2,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT p.sku, p.product_name, i.on_hand_qty, i.reserved_qty, p.reorder_level "
                        "FROM public.grocery_inventory_snapshots i "
                        "JOIN public.grocery_products p ON p.product_id = i.product_id "
                        "ORDER BY (p.reorder_level + i.reserved_qty - i.on_hand_qty) DESC LIMIT 5"
                    ),
                    "explanation": "Top stockout risk SKUs.",
                    "used_datapoints": [],
                    "confidence": 0.9,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=280, completion_tokens=120, total_tokens=400),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert output.needs_clarification is False
        assert "ORDER BY (p.reorder_level + i.reserved_qty - i.on_hand_qty) DESC" in output.generated_sql.sql
        assert "grocery_inventory_snapshots" in output.generated_sql.sql
        assert "grocery_products" in output.generated_sql.sql
        assert "LIMIT 5" in output.generated_sql.sql
        assert sql_agent.llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_weekend_weekday_sales_lift_query_uses_llm_generation(self, sql_agent):
        """Weekend vs weekday sales-lift query should use LLM SQL generation."""
        input_data = SQLAgentInput(
            query="Compare weekend vs weekday sales lift by category and store.",
            investigation_memory=InvestigationMemory(
                query="sales lift",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales Transactions",
                        score=0.92,
                        source="vector",
                        metadata={"table_name": "public.grocery_sales_transactions"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_products_001",
                        datapoint_type="Schema",
                        name="Products",
                        score=0.9,
                        source="vector",
                        metadata={"table_name": "public.grocery_products"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_stores_001",
                        datapoint_type="Schema",
                        name="Stores",
                        score=0.88,
                        source="vector",
                        metadata={"table_name": "public.grocery_stores"},
                    ),
                ],
                total_retrieved=3,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT s.store_name, p.category, "
                        "AVG(CASE WHEN t.is_weekend = TRUE THEN t.total_amount END) AS weekend_avg, "
                        "AVG(CASE WHEN t.is_weekend = FALSE THEN t.total_amount END) AS weekday_avg "
                        "FROM public.grocery_sales_transactions t "
                        "JOIN public.grocery_products p ON p.product_id = t.product_id "
                        "JOIN public.grocery_stores s ON s.store_id = t.store_id "
                        "GROUP BY s.store_name, p.category"
                    ),
                    "explanation": "Weekend vs weekday comparison.",
                    "used_datapoints": [],
                    "confidence": 0.9,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=320, completion_tokens=150, total_tokens=470),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert output.needs_clarification is False
        assert "weekend_avg" in output.generated_sql.sql
        assert "grocery_sales_transactions" in output.generated_sql.sql
        assert "grocery_products" in output.generated_sql.sql
        assert "grocery_stores" in output.generated_sql.sql
        assert sql_agent.llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_inventory_movement_vs_sales_gap_query_uses_llm_generation(self, sql_agent):
        """Inventory movement vs recorded sales gap should use LLM SQL generation."""
        input_data = SQLAgentInput(
            query="Which stores have the largest gap between inventory movement and recorded sales?",
            investigation_memory=InvestigationMemory(
                query="movement vs sales gap",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_inventory_snapshots_001",
                        datapoint_type="Schema",
                        name="Inventory Snapshots",
                        score=0.92,
                        source="vector",
                        metadata={"table_name": "public.grocery_inventory_snapshots"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales Transactions",
                        score=0.9,
                        source="vector",
                        metadata={"table_name": "public.grocery_sales_transactions"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_stores_001",
                        datapoint_type="Schema",
                        name="Stores",
                        score=0.88,
                        source="vector",
                        metadata={"table_name": "public.grocery_stores"},
                    ),
                ],
                total_retrieved=3,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "WITH movement AS ("
                        "SELECT store_id, product_id, snapshot_date, "
                        "GREATEST(on_hand_qty - LEAD(on_hand_qty) OVER (PARTITION BY store_id, product_id ORDER BY snapshot_date), 0) AS movement_qty "
                        "FROM public.grocery_inventory_snapshots) "
                        "SELECT s.store_name, SUM(m.movement_qty) AS inventory_movement, "
                        "SUM(t.quantity) AS recorded_sales, "
                        "SUM(m.movement_qty) - SUM(t.quantity) AS movement_sales_gap "
                        "FROM movement m "
                        "JOIN public.grocery_sales_transactions t ON t.store_id = m.store_id AND t.product_id = m.product_id "
                        "JOIN public.grocery_stores s ON s.store_id = m.store_id "
                        "GROUP BY s.store_name"
                    ),
                    "explanation": "Gap between inventory movement and sales by store.",
                    "used_datapoints": [],
                    "confidence": 0.89,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=360, completion_tokens=180, total_tokens=540),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert output.needs_clarification is False
        assert "movement AS (" in output.generated_sql.sql
        assert "movement_sales_gap" in output.generated_sql.sql
        assert "grocery_inventory_snapshots" in output.generated_sql.sql
        assert "grocery_sales_transactions" in output.generated_sql.sql
        assert "grocery_stores" in output.generated_sql.sql
        assert sql_agent.llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_stockout_risk_query_supports_mysql_sql_generation(self, sql_agent):
        """Stockout risk query should accept MySQL-safe SQL from the LLM path."""
        input_data = SQLAgentInput(
            query="Top 3 products by stockout risk this week using on hand reserved reorder level",
            investigation_memory=InvestigationMemory(
                query="stockout risk ranking mysql",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_inventory_snapshots_001",
                        datapoint_type="Schema",
                        name="Inventory Snapshots",
                        score=0.92,
                        source="vector",
                        metadata={"table_name": "grocery_inventory_snapshots"},
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_products_001",
                        datapoint_type="Schema",
                        name="Products",
                        score=0.91,
                        source="vector",
                        metadata={"table_name": "grocery_products"},
                    ),
                ],
                total_retrieved=2,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="mysql",
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT p.sku, p.product_name, i.on_hand_qty, i.reserved_qty, p.reorder_level "
                        "FROM grocery_inventory_snapshots i "
                        "JOIN grocery_products p ON p.product_id = i.product_id "
                        "WHERE i.snapshot_date >= DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY) "
                        "ORDER BY (p.reorder_level + i.reserved_qty - i.on_hand_qty) DESC LIMIT 3"
                    ),
                    "explanation": "MySQL stockout ranking.",
                    "used_datapoints": [],
                    "confidence": 0.88,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=260, completion_tokens=110, total_tokens=370),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert "DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY)" in output.generated_sql.sql
        assert "LIMIT 3" in output.generated_sql.sql
        assert sql_agent.llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_stockout_risk_query_uses_llm_when_datapoints_missing(self, sql_agent):
        """LLM SQL generation should still work with live schema mode and zero DataPoints."""
        input_data = SQLAgentInput(
            query=(
                "Which 5 SKUs have the highest stockout risk this week based on on-hand, "
                "reserved, and reorder level?"
            ),
            investigation_memory=InvestigationMemory(
                query="stockout risk ranking",
                datapoints=[],
                total_retrieved=0,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
            database_url="postgresql://postgres:@localhost:5432/datachat_grocery",
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT p.sku, p.product_name "
                        "FROM public.grocery_inventory_snapshots i "
                        "JOIN public.grocery_products p ON p.product_id = i.product_id "
                        "ORDER BY (p.reorder_level + i.reserved_qty - i.on_hand_qty) DESC LIMIT 5"
                    ),
                    "explanation": "Stockout ranking.",
                    "used_datapoints": [],
                    "confidence": 0.86,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=250, completion_tokens=90, total_tokens=340),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert "grocery_inventory_snapshots" in output.generated_sql.sql
        assert "grocery_products" in output.generated_sql.sql
        assert "LIMIT 5" in output.generated_sql.sql
        assert sql_agent.llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_stockout_risk_query_uses_llm_with_domain_agnostic_table_names(self, sql_agent):
        """LLM SQL generation should work with domain-agnostic table names."""
        input_data = SQLAgentInput(
            query=(
                "Which 5 SKUs have the highest stockout risk this week based on on-hand, "
                "reserved, and reorder level?"
            ),
            investigation_memory=InvestigationMemory(
                query="stockout risk ranking",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_inventory_daily_001",
                        datapoint_type="Schema",
                        name="Inventory Daily",
                        score=0.9,
                        source="vector",
                        metadata={
                            "table_name": "public.inventory_daily",
                            "key_columns": [
                                {"name": "snapshot_date"},
                                {"name": "store_id"},
                                {"name": "product_id"},
                                {"name": "on_hand_qty"},
                                {"name": "reserved_qty"},
                            ],
                        },
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_product_catalog_001",
                        datapoint_type="Schema",
                        name="Product Catalog",
                        score=0.88,
                        source="vector",
                        metadata={
                            "table_name": "public.product_catalog",
                            "key_columns": [
                                {"name": "product_id"},
                                {"name": "sku"},
                                {"name": "product_name"},
                                {"name": "reorder_level"},
                            ],
                        },
                    ),
                ],
                total_retrieved=2,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        sql_agent.llm.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT p.sku, p.product_name, i.on_hand_qty "
                        "FROM public.inventory_daily i "
                        "JOIN public.product_catalog p ON p.product_id = i.product_id "
                        "ORDER BY (p.reorder_level + i.reserved_qty - i.on_hand_qty) DESC LIMIT 5"
                    ),
                    "explanation": "Stockout ranking from generic tables.",
                    "used_datapoints": [],
                    "confidence": 0.9,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=240, completion_tokens=90, total_tokens=330),
            finish_reason="stop",
            provider="openai",
        )

        output = await sql_agent(input_data)

        assert output.success is True
        assert "FROM public.inventory_daily" in output.generated_sql.sql
        assert "JOIN public.product_catalog" in output.generated_sql.sql
        assert output.needs_clarification is False
        assert sql_agent.llm.generate.await_count == 1

    def test_table_resolver_low_confidence_triggers_clarification(self, sql_agent):
        """Low-confidence resolver output with no candidates should hard-clarify."""
        decision = sql_agent._should_force_table_clarification(
            query="Show rows from one of these tables",
            table_resolution=TableResolution(
                candidate_tables=[],
                column_hints=["total_amount"],
                confidence=0.31,
                needs_clarification=True,
                clarifying_question="Do you want sales transactions or inventory snapshots?",
            ),
        )

        assert decision is True

    @pytest.mark.asyncio
    async def test_table_resolver_low_confidence_with_candidates_continues_generation(self):
        """When resolver returns candidates, SQL generation should continue instead of hard-clarifying."""
        resolver_response = LLMResponse(
            content=json.dumps(
                {
                    "candidate_tables": ["public.grocery_sales_transactions"],
                    "column_hints": ["total_amount", "store_id"],
                    "confidence": 0.31,
                    "needs_clarification": True,
                    "clarifying_question": "Do you want sales transactions or inventory snapshots?",
                }
            ),
            model="gpt-4o-mini",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280),
            finish_reason="stop",
            provider="openai",
        )
        sql_response = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT store_id, SUM(total_amount) AS total_amount "
                        "FROM public.grocery_sales_transactions GROUP BY store_id"
                    ),
                    "explanation": "Revenue by store.",
                    "used_datapoints": [],
                    "confidence": 0.9,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=420, completion_tokens=140, total_tokens=560),
            finish_reason="stop",
            provider="openai",
        )

        fast_provider = Mock()
        fast_provider.generate = AsyncMock(return_value=resolver_response)
        fast_provider.provider = "openai"
        fast_provider.model = "gpt-4o-mini"

        main_provider = Mock()
        main_provider.generate = AsyncMock(return_value=sql_response)
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_table_resolver_enabled=True,
            sql_table_resolver_confidence_threshold=0.55,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
            sql_two_stage_enabled=True,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, fast_provider],
            ),
        ):
            agent = SQLAgent()

        input_data = SQLAgentInput(
            query="Which stores have the largest gap between inventory movement and recorded sales?",
            investigation_memory=InvestigationMemory(
                query="resolver",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales",
                        score=0.9,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_sales_transactions",
                            "key_columns": [{"name": "store_id"}, {"name": "total_amount"}],
                        },
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_inventory_snapshots_001",
                        datapoint_type="Schema",
                        name="Inventory",
                        score=0.88,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_inventory_snapshots",
                            "key_columns": [{"name": "store_id"}, {"name": "on_hand_qty"}],
                        },
                    ),
                ],
                total_retrieved=2,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        output = await agent(input_data)
        assert output.success is True
        assert output.needs_clarification is False
        assert "grocery_sales_transactions" in output.generated_sql.sql
        assert fast_provider.generate.await_count == 1
        assert main_provider.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_table_resolver_candidates_are_injected_into_sql_prompt(self):
        """Resolver-selected tables should be injected into SQL generation context."""
        resolver_response = LLMResponse(
            content=json.dumps(
                {
                    "candidate_tables": [
                        "public.grocery_sales_transactions",
                        "public.grocery_stores",
                    ],
                    "column_hints": ["total_amount", "store_id", "business_date"],
                    "confidence": 0.84,
                    "needs_clarification": False,
                    "clarifying_question": None,
                }
            ),
            model="gpt-4o-mini",
            usage=LLMUsage(prompt_tokens=220, completion_tokens=90, total_tokens=310),
            finish_reason="stop",
            provider="openai",
        )
        sql_response = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT s.store_name, SUM(t.total_amount) AS total_revenue "
                        "FROM public.grocery_sales_transactions t "
                        "JOIN public.grocery_stores s ON s.store_id = t.store_id "
                        "GROUP BY s.store_name ORDER BY total_revenue DESC"
                    ),
                    "explanation": "Revenue by store",
                    "used_datapoints": [],
                    "confidence": 0.9,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=600, completion_tokens=180, total_tokens=780),
            finish_reason="stop",
            provider="openai",
        )

        fast_provider = Mock()
        fast_provider.generate = AsyncMock(return_value=resolver_response)
        fast_provider.provider = "openai"
        fast_provider.model = "gpt-4o-mini"

        main_provider = Mock()
        main_provider.generate = AsyncMock(return_value=sql_response)
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_table_resolver_enabled=True,
            sql_table_resolver_confidence_threshold=0.55,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
            sql_two_stage_enabled=False,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, fast_provider],
            ),
        ):
            agent = SQLAgent()

        input_data = SQLAgentInput(
            query="What is total revenue by store?",
            investigation_memory=InvestigationMemory(
                query="resolver",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales",
                        score=0.9,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_sales_transactions",
                            "key_columns": [
                                {"name": "store_id"},
                                {"name": "total_amount"},
                                {"name": "business_date"},
                            ],
                        },
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_stores_001",
                        datapoint_type="Schema",
                        name="Stores",
                        score=0.88,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_stores",
                            "key_columns": [
                                {"name": "store_id"},
                                {"name": "store_name"},
                            ],
                        },
                    ),
                ],
                total_retrieved=2,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        output = await agent(input_data)
        assert output.success is True
        assert output.needs_clarification is False
        assert "grocery_sales_transactions" in output.generated_sql.sql
        assert fast_provider.generate.await_count == 1
        assert main_provider.generate.await_count == 1

        call_args = main_provider.generate.await_args[0][0]
        assert "Likely source tables (resolver)" in call_args.messages[1].content
        assert "public.grocery_stores" in call_args.messages[1].content

    @pytest.mark.asyncio
    async def test_table_resolver_falls_back_to_ranked_candidates_on_invalid_json(self):
        """Invalid resolver JSON should fall back to ranked candidates instead of clarifying."""
        resolver_response = LLMResponse(
            content="not-json",
            model="gpt-4o-mini",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            finish_reason="stop",
            provider="openai",
        )
        sql_response = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT s.store_name, SUM(t.quantity) AS sold_qty "
                        "FROM public.grocery_sales_transactions t "
                        "JOIN public.grocery_inventory_snapshots i ON i.store_id = t.store_id "
                        "JOIN public.grocery_stores s ON s.store_id = t.store_id "
                        "GROUP BY s.store_name"
                    ),
                    "explanation": "Inventory movement vs sales by store.",
                    "used_datapoints": [],
                    "confidence": 0.89,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=520, completion_tokens=140, total_tokens=660),
            finish_reason="stop",
            provider="openai",
        )

        fast_provider = Mock()
        fast_provider.generate = AsyncMock(return_value=resolver_response)
        fast_provider.provider = "openai"
        fast_provider.model = "gpt-4o-mini"

        main_provider = Mock()
        main_provider.generate = AsyncMock(return_value=sql_response)
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_table_resolver_enabled=True,
            sql_table_resolver_confidence_threshold=0.55,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
            sql_two_stage_enabled=False,
            sql_table_resolver_ranked_min_score=2,
            sql_table_resolver_ranked_max_tables=3,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, fast_provider],
            ),
        ):
            agent = SQLAgent()

        input_data = SQLAgentInput(
            query="Which stores have the largest gap between inventory movement and recorded sales?",
            investigation_memory=InvestigationMemory(
                query="resolver",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales",
                        score=0.9,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_sales_transactions",
                            "key_columns": [
                                {"name": "store_id"},
                                {"name": "product_id"},
                                {"name": "quantity"},
                                {"name": "business_date"},
                            ],
                        },
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_inventory_snapshots_001",
                        datapoint_type="Schema",
                        name="Inventory Snapshots",
                        score=0.89,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_inventory_snapshots",
                            "key_columns": [
                                {"name": "store_id"},
                                {"name": "product_id"},
                                {"name": "snapshot_date"},
                                {"name": "on_hand_qty"},
                            ],
                        },
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_stores_001",
                        datapoint_type="Schema",
                        name="Stores",
                        score=0.88,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_stores",
                            "key_columns": [{"name": "store_id"}, {"name": "store_name"}],
                        },
                    ),
                ],
                total_retrieved=3,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        output = await agent(input_data)
        assert output.success is True
        assert output.needs_clarification is False
        assert "grocery_sales_transactions" in output.generated_sql.sql
        assert "grocery_inventory_snapshots" in output.generated_sql.sql
        assert fast_provider.generate.await_count == 1
        assert main_provider.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_table_resolver_uses_candidates_when_clarification_flagged_but_confident(self):
        """When candidates are clear and confidence is high, proceed without clarification."""
        resolver_response = LLMResponse(
            content=json.dumps(
                {
                    "candidate_tables": ["public.grocery_sales_transactions"],
                    "column_hints": ["total_amount", "business_date"],
                    "confidence": 0.82,
                    "needs_clarification": True,
                    "clarifying_question": "Which sales table should I use?",
                }
            ),
            model="gpt-4o-mini",
            usage=LLMUsage(prompt_tokens=180, completion_tokens=70, total_tokens=250),
            finish_reason="stop",
            provider="openai",
        )
        sql_response = LLMResponse(
            content=json.dumps(
                {
                    "sql": (
                        "SELECT business_date, SUM(total_amount) AS total_sales "
                        "FROM public.grocery_sales_transactions "
                        "GROUP BY business_date ORDER BY business_date"
                    ),
                    "explanation": "Daily sales totals",
                    "used_datapoints": [],
                    "confidence": 0.87,
                    "assumptions": [],
                    "clarifying_questions": [],
                }
            ),
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=420, completion_tokens=120, total_tokens=540),
            finish_reason="stop",
            provider="openai",
        )

        fast_provider = Mock()
        fast_provider.generate = AsyncMock(return_value=resolver_response)
        fast_provider.provider = "openai"
        fast_provider.model = "gpt-4o-mini"

        main_provider = Mock()
        main_provider.generate = AsyncMock(return_value=sql_response)
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_table_resolver_enabled=True,
            sql_table_resolver_confidence_threshold=0.55,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
            sql_two_stage_enabled=False,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, fast_provider],
            ),
        ):
            agent = SQLAgent()

        input_data = SQLAgentInput(
            query="Show daily sales totals",
            investigation_memory=InvestigationMemory(
                query="resolver confident candidates",
                datapoints=[
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_sales_transactions_001",
                        datapoint_type="Schema",
                        name="Sales",
                        score=0.9,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_sales_transactions",
                            "key_columns": [
                                {"name": "business_date"},
                                {"name": "total_amount"},
                            ],
                        },
                    ),
                    RetrievedDataPoint(
                        datapoint_id="table_grocery_inventory_snapshots_001",
                        datapoint_type="Schema",
                        name="Inventory",
                        score=0.88,
                        source="vector",
                        metadata={
                            "table_name": "public.grocery_inventory_snapshots",
                            "key_columns": [
                                {"name": "snapshot_date"},
                                {"name": "on_hand_qty"},
                            ],
                        },
                    ),
                ],
                total_retrieved=2,
                retrieval_mode="hybrid",
                sources_used=[],
            ),
            database_type="postgresql",
        )

        output = await agent(input_data)

        assert output.success is True
        assert output.needs_clarification is False
        assert output.generated_sql.clarifying_questions == []
        assert "grocery_sales_transactions" in output.generated_sql.sql
        assert fast_provider.generate.await_count == 1
        assert main_provider.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_two_stage_sql_accepts_fast_model_when_confident(
        self, sample_sql_agent_input, sample_valid_llm_response
    ):
        fast_provider = Mock()
        fast_provider.generate = AsyncMock(return_value=sample_valid_llm_response)
        fast_provider.provider = "openai"
        fast_provider.model = "gpt-4o-mini"

        main_provider = Mock()
        main_provider.generate = AsyncMock(return_value=sample_valid_llm_response)
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_two_stage_enabled=True,
            sql_two_stage_confidence_threshold=0.7,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, fast_provider],
            ),
        ):
            agent = SQLAgent()

        output = await agent(sample_sql_agent_input)
        assert output.success is True
        assert output.metadata.llm_calls == 1
        assert fast_provider.generate.await_count == 1
        assert main_provider.generate.await_count == 0

    @pytest.mark.asyncio
    async def test_two_stage_sql_escalates_to_main_when_fast_low_confidence(
        self, sample_sql_agent_input, sample_valid_llm_response
    ):
        fast_response = LLMResponse(
            content=(
                "```json\n"
                + json.dumps(
                    {
                        "sql": "SELECT SUM(amount) FROM analytics.fact_sales",
                        "explanation": "Draft SQL",
                        "used_datapoints": ["table_fact_sales_001"],
                        "confidence": 0.4,
                        "assumptions": [],
                        "clarifying_questions": [],
                    }
                )
                + "\n```"
            ),
            model="gpt-4o-mini",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=120, total_tokens=620),
            finish_reason="stop",
            provider="openai",
        )

        fast_provider = Mock()
        fast_provider.generate = AsyncMock(return_value=fast_response)
        fast_provider.provider = "openai"
        fast_provider.model = "gpt-4o-mini"

        main_provider = Mock()
        main_provider.generate = AsyncMock(return_value=sample_valid_llm_response)
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_two_stage_enabled=True,
            sql_two_stage_confidence_threshold=0.7,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, fast_provider],
            ),
        ):
            agent = SQLAgent()

        output = await agent(sample_sql_agent_input)
        assert output.success is True
        assert output.metadata.llm_calls == 2
        assert fast_provider.generate.await_count == 1
        assert main_provider.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_two_stage_skips_when_providers_are_effectively_same(
        self, sample_sql_agent_input, sample_valid_llm_response
    ):
        fast_provider = Mock()
        fast_provider.generate = AsyncMock(return_value=sample_valid_llm_response)
        fast_provider.provider = "openai"
        fast_provider.model = "gpt-4o"

        main_provider = Mock()
        main_provider.generate = AsyncMock(return_value=sample_valid_llm_response)
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_two_stage_enabled=True,
            sql_two_stage_confidence_threshold=0.7,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, fast_provider],
            ),
        ):
            agent = SQLAgent()

        output = await agent(sample_sql_agent_input)
        assert output.success is True
        assert output.metadata.llm_calls == 1
        assert main_provider.generate.await_count == 1
        assert fast_provider.generate.await_count == 0

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

    def test_truncate_context_applies_budget(self, sql_agent):
        text = "x" * 100
        truncated = sql_agent._truncate_context(text, 40)
        assert len(truncated) > 40
        assert "Context truncated for latency budget" in truncated

    def test_columns_context_map_applies_focus_and_column_limits(self, sql_agent):
        sql_agent.config.pipeline = Mock(
            sql_prompt_budget_enabled=True,
            sql_prompt_focus_tables=2,
            sql_prompt_max_columns_per_table=1,
        )
        columns_by_table = {
            "public.orders": [("id", "integer"), ("total", "numeric")],
            "public.customers": [("id", "integer"), ("name", "text")],
            "public.items": [("id", "integer"), ("sku", "text")],
        }
        context, focus_tables = sql_agent._build_columns_context_from_map(
            query="show revenue by customer",
            qualified_tables=list(columns_by_table.keys()),
            columns_by_table=columns_by_table,
        )
        assert len(focus_tables) == 2
        table_lines = [line for line in context.splitlines() if line.startswith("- ")]
        assert len(table_lines) == 2
        assert table_lines[0].count("(") == 1
        assert table_lines[1].count("(") == 1


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
            patch("backend.agents.sql.create_connector", return_value=mock_connector) as connector_factory,
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
        connector_factory.assert_called_once_with(
            database_url="postgresql://demo:demo@chosen-host:5432/chosen_db",
            database_type="postgresql",
            pool_size=sql_agent.config.database.pool_size,
            timeout=10,
        )

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
    async def test_recovers_missing_sql_with_formatter_fallback(self, sample_sql_agent_input):
        """Formatter fallback should recover malformed SQL JSON output."""
        malformed_response = LLMResponse(
            content='{"explanation":"I can help with SQL","confidence":0.7}',
            model="gemini-2.5-flash",
            usage=LLMUsage(prompt_tokens=350, completion_tokens=40, total_tokens=390),
            finish_reason="stop",
            provider="google",
        )
        formatter_response = LLMResponse(
            content=(
                "```json\n"
                + json.dumps(
                    {
                        "sql": (
                            "SELECT SUM(amount) FROM analytics.fact_sales "
                            "WHERE date >= '2024-07-01' AND date < '2024-10-01'"
                        ),
                        "explanation": "Recovered SQL JSON.",
                        "used_datapoints": ["table_fact_sales_001"],
                        "confidence": 0.86,
                        "assumptions": [],
                        "clarifying_questions": [],
                    }
                )
                + "\n```"
            ),
            model="gemini-2.5-flash-lite",
            usage=LLMUsage(prompt_tokens=180, completion_tokens=60, total_tokens=240),
            finish_reason="stop",
            provider="google",
        )

        main_provider = Mock()
        main_provider.generate = AsyncMock(side_effect=[malformed_response, malformed_response])
        main_provider.provider = "google"
        main_provider.model = "gemini-2.5-flash"

        formatter_provider = Mock()
        formatter_provider.generate = AsyncMock(return_value=formatter_response)
        formatter_provider.provider = "google"
        formatter_provider.model = "gemini-2.5-flash-lite"

        mock_settings = Mock()
        mock_settings.llm = Mock(sql_formatter_model="gemini-2.5-flash-lite")
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_two_stage_enabled=False,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
            sql_formatter_fallback_enabled=True,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, formatter_provider],
            ),
        ):
            agent = SQLAgent()

        output = await agent(sample_sql_agent_input)

        assert output.success is True
        assert output.needs_clarification is False
        assert "SELECT SUM(amount)" in output.generated_sql.sql
        assert output.metadata.llm_calls == 3
        assert main_provider.generate.await_count == 2
        assert formatter_provider.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_formatter_fallback_respects_disable_flag(self, sample_sql_agent_input):
        """When disabled, formatter fallback should not run."""
        malformed_response = LLMResponse(
            content='{"explanation":"missing sql"}',
            model="gpt-4o",
            usage=LLMUsage(prompt_tokens=120, completion_tokens=20, total_tokens=140),
            finish_reason="stop",
            provider="openai",
        )

        main_provider = Mock()
        main_provider.generate = AsyncMock(side_effect=[malformed_response, malformed_response])
        main_provider.provider = "openai"
        main_provider.model = "gpt-4o"

        formatter_provider = Mock()
        formatter_provider.generate = AsyncMock()
        formatter_provider.provider = "openai"
        formatter_provider.model = "gpt-4o-mini"

        mock_settings = Mock()
        mock_settings.llm = Mock(sql_formatter_model=None)
        mock_settings.database = Mock(url=None, db_type="postgresql", pool_size=5)
        mock_settings.pipeline = Mock(
            sql_two_stage_enabled=False,
            sql_prompt_budget_enabled=False,
            schema_snapshot_cache_enabled=False,
            sql_formatter_fallback_enabled=False,
        )

        with (
            patch("backend.agents.sql.get_settings", return_value=mock_settings),
            patch(
                "backend.agents.sql.LLMProviderFactory.create_agent_provider",
                side_effect=[main_provider, formatter_provider],
            ),
        ):
            agent = SQLAgent()

        output = await agent(sample_sql_agent_input)

        assert output.needs_clarification is True
        assert main_provider.generate.await_count == 2
        assert formatter_provider.generate.await_count == 0

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

    def test_parse_llm_response_caps_implicit_limit_to_default(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(update={"query": "List all grocery stores"})
        content = json.dumps(
            {"sql": "SELECT * FROM public.grocery_stores LIMIT 10000", "confidence": 0.9}
        )

        generated = sql_agent._parse_llm_response(content, sql_input)

        assert generated.sql == "SELECT * FROM public.grocery_stores LIMIT 5"

    def test_parse_llm_response_caps_explicit_limit_to_ten(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(
            update={"query": "Show me the first 50 rows from public.grocery_stores"}
        )
        content = json.dumps(
            {"sql": "SELECT * FROM public.grocery_stores LIMIT 10000", "confidence": 0.9}
        )

        generated = sql_agent._parse_llm_response(content, sql_input)

        assert generated.sql == "SELECT * FROM public.grocery_stores LIMIT 10"

    def test_parse_llm_response_adds_default_limit_when_missing(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(
            update={"query": "Show rows from public.grocery_stores"}
        )
        content = json.dumps({"sql": "SELECT * FROM public.grocery_stores", "confidence": 0.9})

        generated = sql_agent._parse_llm_response(content, sql_input)

        assert generated.sql == "SELECT * FROM public.grocery_stores LIMIT 5"

    def test_parse_llm_response_adds_outer_limit_when_only_subquery_has_limit(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(update={"query": "Show orders"})
        content = json.dumps(
            {
                "sql": (
                    "SELECT * FROM public.orders o "
                    "WHERE EXISTS ("
                    "SELECT 1 FROM public.order_items oi "
                    "WHERE oi.order_id = o.id LIMIT 1"
                    ")"
                ),
                "confidence": 0.9,
            }
        )

        generated = sql_agent._parse_llm_response(content, sql_input)

        assert "LIMIT 1" in generated.sql
        assert generated.sql.endswith("LIMIT 5")

    def test_parse_llm_response_keeps_top_level_parameterized_limit(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(update={"query": "Show orders"})
        content = json.dumps(
            {
                "sql": (
                    "SELECT * FROM public.orders o "
                    "WHERE EXISTS ("
                    "SELECT 1 FROM public.order_items oi "
                    "WHERE oi.order_id = o.id LIMIT 1"
                    ") LIMIT $1"
                ),
                "confidence": 0.9,
            }
        )

        generated = sql_agent._parse_llm_response(content, sql_input)

        assert generated.sql.endswith("LIMIT $1")

    def test_parse_llm_response_does_not_force_limit_on_single_aggregate(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(update={"query": "What is total revenue?"})
        content = json.dumps(
            {
                "sql": (
                    "SELECT SUM(total_amount) AS total_revenue "
                    "FROM public.grocery_sales_transactions"
                ),
                "confidence": 0.9,
            }
        )

        generated = sql_agent._parse_llm_response(content, sql_input)

        assert generated.sql == (
            "SELECT SUM(total_amount) AS total_revenue FROM public.grocery_sales_transactions"
        )

    def test_parse_llm_response_handles_non_string_content(
        self, sql_agent, sample_sql_agent_input
    ):
        sql_input = sample_sql_agent_input.model_copy(update={"query": "Show tables"})

        with pytest.raises(ValueError, match="missing 'sql' field|empty content"):
            sql_agent._parse_llm_response(AsyncMock(), sql_input)


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
