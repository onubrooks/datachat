# Credentials-Only Mode

Credentials-only mode lets DataChat answer questions with just database credentials,
without requiring DataPoints. This is also called "live schema mode."

## How It Works

1. Live schema snapshot (tables + columns).
2. Lightweight stats cache for top matched tables (Postgres only):
   - Row count estimates (pg_class)
   - Column stats (pg_stats)
3. Join hints (heuristic) based on *_id patterns (Postgres only).
4. SQL generation uses the live snapshot when DataPoints are missing.

## Capabilities vs. Limits

| Capability | Works? | Notes |
| --- | --- | --- |
| Basic table/column discovery | Yes | Live schema snapshot. |
| Row counts | Yes | Uses COUNT(*) fallback or pg_class estimates. |
| Simple aggregations (sum/avg) | Usually | Best when column names are descriptive. |
| Join inference | Partial | Heuristic via *_id columns (Postgres only). |
| Business logic (revenue, refunds) | No | Requires DataPoints or docs. |
| Metric consistency | No | Definitions vary without DataPoints. |
| Complex data modeling | Partial | Lacks dbt/ETL context. |
| System catalog queries | Yes | Postgres, MySQL, ClickHouse catalogs supported. |
| BigQuery/Redshift catalogs | Not yet | Planned; use Postgres mode for Redshift. |

## When to Use

- Quick exploration of a new database.
- Prototyping questions before defining DataPoints.
- Diagnostics against system catalogs.

## When Not to Use

- Production metrics that require precise definitions.
- KPI reporting with strict business rules.
- Queries that need ETL or semantic model awareness.

## Tips for Better Results

- Use explicit table names in the question when possible.
- Ask for samples first if you are unsure about column names.
- Add DataPoints for critical tables/metrics once patterns stabilize.
