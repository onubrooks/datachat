# DataChat Levels

This file is the source of truth for level maturity.

## Status Legend

- `Implemented`: available in current codebase.
- `Partial`: core pieces exist, but not complete as a full level.
- `Planned`: roadmap only.

## Level 1: Schema-Aware Querying

Status: `Implemented`

Delivered today:

- Credentials-only querying with live schema context.
- Deterministic catalog handling for:
  - list tables
  - list columns
  - row counts
  - sample rows
- NL -> SQL -> validation -> execution pipeline.
- Clarification flow for ambiguous prompts.

Database runtime support:

- PostgreSQL
- ClickHouse

Notes:

- MySQL/BigQuery/Redshift catalog templates exist, but runtime connector execution is pending.

## Level 2: Context Enhancement (DataPoints)

Status: `Partial`

Delivered today:

- DataPoint ingestion/sync.
- Retrieval from vectors/graph as query context.
- Context-only answers and evidence metadata.
- Auto-profiling to generate pending DataPoints (with review/approval workflow).

Still limited:

- Deep domain semantics still depend on DataPoint quality/coverage.
- No fully managed ontology lifecycle.

## Level 3: Executable Metrics

Status: `Planned`

Target outcome:

- Deterministic metric templates and parameterized execution path for stable KPI answers.

Not shipped yet as a dedicated level.

## Level 4: Performance Optimization

Status: `Planned`

Target outcome:

- Materialization/caching strategies, adaptive query routing, and refresh policies.

Not shipped yet.

## Level 5: Intelligence

Status: `Planned`

Target outcome:

- Automated anomaly detection, dependency-aware diagnosis, and remediation workflows.

Not shipped yet.

## Practical Guidance

For production use now:

1. Start with Level 1 credentials-only to validate connectivity and query quality.
2. Add Level 2 DataPoints for critical business metrics and terms.
3. Treat Levels 3-5 as roadmap items, not current guarantees.
