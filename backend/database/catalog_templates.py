"""Database catalog query templates for credentials-only discovery."""

from __future__ import annotations

from typing import NamedTuple


class CatalogTemplates(NamedTuple):
    """System-catalog templates for metadata discovery."""

    list_tables: str


_CATALOG_TEMPLATES: dict[str, CatalogTemplates] = {
    "postgresql": CatalogTemplates(
        list_tables=(
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY table_schema, table_name"
        ),
    ),
    "mysql": CatalogTemplates(
        list_tables=(
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('mysql', 'performance_schema', 'information_schema', 'sys') "
            "ORDER BY table_schema, table_name"
        ),
    ),
    "clickhouse": CatalogTemplates(
        list_tables=(
            "SELECT database, name "
            "FROM system.tables "
            "WHERE database NOT IN ('system', 'INFORMATION_SCHEMA', 'information_schema') "
            "ORDER BY database, name"
        ),
    ),
    "bigquery": CatalogTemplates(
        list_tables=(
            "SELECT table_schema, table_name "
            "FROM INFORMATION_SCHEMA.TABLES "
            "ORDER BY table_schema, table_name"
        ),
    ),
    "redshift": CatalogTemplates(
        list_tables=(
            "SELECT schemaname AS table_schema, tablename AS table_name "
            "FROM pg_table_def "
            "WHERE schemaname NOT IN ('pg_catalog', 'information_schema') "
            "GROUP BY schemaname, tablename "
            "ORDER BY schemaname, tablename"
        ),
    ),
}


_CATALOG_SCHEMAS: dict[str, set[str]] = {
    "postgresql": {"information_schema", "pg_catalog"},
    "mysql": {"information_schema", "mysql", "performance_schema", "sys"},
    "clickhouse": {"information_schema", "system"},
    "bigquery": {"information_schema"},
    "redshift": {"information_schema", "pg_catalog", "pg_internal"},
}


_CATALOG_ALIASES: dict[str, set[str]] = {
    "postgresql": {
        "pg_tables",
        "pg_class",
        "pg_namespace",
        "pg_attribute",
        "pg_stat_all_tables",
        "pg_stat_user_tables",
        "pg_indexes",
        "pg_constraint",
        "pg_description",
        "pg_type",
        "pg_roles",
        "pg_stat_activity",
        "pg_stat_database",
        "pg_locks",
        "pg_settings",
    },
    "mysql": {
        "information_schema.tables",
        "information_schema.columns",
        "information_schema.statistics",
        "information_schema.key_column_usage",
        "mysql.user",
        "mysql.db",
        "performance_schema.threads",
        "performance_schema.events_statements_summary_by_digest",
    },
    "clickhouse": {
        "system.tables",
        "system.columns",
        "system.databases",
        "system.parts",
        "system.settings",
    },
    "bigquery": {
        "information_schema.tables",
        "information_schema.columns",
        "information_schema.schemata",
        "information_schema.table_options",
    },
    "redshift": {
        "svv_tables",
        "svv_columns",
        "svv_table_info",
        "svv_views",
        "svv_schema",
        "stl_query",
        "stl_scan",
        "stl_wlm_query",
        "svl_qlog",
        "svl_query_report",
        "pg_table_def",
    },
}


def normalize_database_type(database_type: str | None) -> str:
    """Map aliases to canonical database type keys."""
    normalized = (database_type or "").strip().lower()
    if normalized in {"postgres", "postgresql"}:
        return "postgresql"
    return normalized


def get_list_tables_query(database_type: str | None) -> str | None:
    """Return catalog query used for table discovery."""
    db_type = normalize_database_type(database_type)
    templates = _CATALOG_TEMPLATES.get(db_type)
    return templates.list_tables if templates else None


def get_catalog_schemas(database_type: str | None) -> set[str]:
    """Return schema names treated as catalog/system schemas."""
    db_type = normalize_database_type(database_type)
    return _CATALOG_SCHEMAS.get(db_type, {"information_schema"})


def get_catalog_aliases(database_type: str | None) -> set[str]:
    """Return table aliases treated as catalog/system objects."""
    db_type = normalize_database_type(database_type)
    return _CATALOG_ALIASES.get(db_type, set())
