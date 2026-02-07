"""Unit tests for deterministic catalog intelligence."""

from backend.database.catalog import CatalogIntelligence
from backend.models.agent import InvestigationMemory, RetrievedDataPoint


def _memory() -> InvestigationMemory:
    return InvestigationMemory(
        query="show tables",
        datapoints=[
            RetrievedDataPoint(
                datapoint_id="table_sales",
                datapoint_type="Schema",
                name="Sales",
                score=0.9,
                source="vector",
                metadata={
                    "table_name": "public.sales",
                    "key_columns": [{"name": "amount", "type": "numeric"}],
                },
            ),
            RetrievedDataPoint(
                datapoint_id="table_orders",
                datapoint_type="Schema",
                name="Orders",
                score=0.88,
                source="vector",
                metadata={
                    "table_name": "public.orders",
                    "key_columns": [{"name": "id", "type": "bigint"}],
                },
            ),
        ],
        total_retrieved=2,
        retrieval_mode="hybrid",
        sources_used=["table_sales", "table_orders"],
    )


def test_catalog_plan_list_tables_query():
    service = CatalogIntelligence()
    plan = service.plan_query(
        query="list tables",
        database_type="postgresql",
        investigation_memory=_memory(),
    )
    assert plan is not None
    assert plan.operation == "list_tables"
    assert "information_schema.tables" in (plan.sql or "")


def test_catalog_plan_row_count_requires_table():
    service = CatalogIntelligence()
    plan = service.plan_query(
        query="how many rows?",
        database_type="postgresql",
        investigation_memory=_memory(),
    )
    assert plan is not None
    assert plan.sql is None
    assert "Which table should I count rows for?" in plan.clarifying_questions


def test_catalog_ranked_schema_context_includes_columns():
    service = CatalogIntelligence()
    context = service.build_ranked_schema_context(
        query="show sales amount",
        investigation_memory=_memory(),
    )
    assert context is not None
    assert "public.sales" in context
    assert "amount (numeric)" in context


def test_extract_explicit_table_name_from_clarification_prefix():
    service = CatalogIntelligence()
    table = service.extract_explicit_table_name(
        'Regarding "Which table should I list columns for?": vbs_registrations'
    )
    assert table == "vbs_registrations"


def test_extract_limit_handles_show_n_rows_phrase():
    service = CatalogIntelligence()
    assert service.extract_limit("show 2 rows in public.events") == 2


def test_extract_limit_caps_at_ten():
    service = CatalogIntelligence()
    assert service.extract_limit("show 100 rows in public.events") == 10
