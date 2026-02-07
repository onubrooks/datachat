"""Unit tests for catalog query templates used in credentials-only mode."""

from backend.database.catalog_templates import (
    get_catalog_aliases,
    get_catalog_schemas,
    get_list_tables_query,
    normalize_database_type,
)


def test_normalize_database_type_maps_postgres_aliases():
    assert normalize_database_type("postgres") == "postgresql"
    assert normalize_database_type("postgresql") == "postgresql"


def test_list_tables_query_exists_for_supported_databases():
    for db_type in ("postgresql", "mysql", "clickhouse", "bigquery", "redshift"):
        query = get_list_tables_query(db_type)
        assert query is not None
        assert "table" in query.lower()


def test_catalog_schemas_include_common_system_namespaces():
    assert "information_schema" in get_catalog_schemas("postgresql")
    assert "system" in get_catalog_schemas("clickhouse")
    assert "pg_catalog" in get_catalog_schemas("redshift")


def test_catalog_aliases_cover_redshift_and_clickhouse_system_tables():
    redshift_aliases = get_catalog_aliases("redshift")
    clickhouse_aliases = get_catalog_aliases("clickhouse")
    assert "svv_tables" in redshift_aliases
    assert "system.tables" in clickhouse_aliases
