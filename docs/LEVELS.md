# DataChat Levels

This file is the source of truth for level maturity.

## Document Role

`LEVELS.md` defines capability maturity boundaries.

- It answers: "What qualifies as Level X?"
- It does not own delivery sequence or implementation status history.

For initiative status/dependencies, use `docs/ROADMAP.md`.

## Status Legend

- `Implemented`: available in current codebase.
- `Partial`: core pieces exist, but not complete as a full level.
- `Planned`: roadmap only.

## Level 1: Schema-Aware Querying

Status: `Implemented`
Roadmap references: `FND-001`, `FND-003`, `FND-005`

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
- MySQL

Notes:

- BigQuery/Redshift catalog templates exist, but runtime connector execution is pending.

## Level 1.5: MetadataOps Foundation (Cross-Level)

Status: `In Progress`
Roadmap references: `FND-001`..`FND-006`

Purpose:

- Treat metadata authoring quality and observability as a product surface.
- Reduce failures by improving source quality and traceability before adding more agent complexity.

Scope:

- metadata contracts and linting gates
- metadata authoring/review lifecycle
- CI evaluation gates (retrieval, qa, intent, catalog)
- retrieval and answer trace observability
- runtime telemetry loops for clarification/fallback failure modes

Delivery rule:

- major Level 3-5 expansion should follow this foundation lane, not bypass it

## Level 1.4: Simple Entry Layer (Thin Wrapper)

Status: `Planned`
Roadmap reference: `SMP-001`

Purpose:

- improve onboarding and first-query success with lower cognitive load
- provide convenience wrappers without changing core retrieval/routing behavior

Guardrails:

- must not introduce new answer semantics
- must not bypass metadata contracts, eval gates, or provenance traces
- should be reversible/configurable without changing runtime truth paths

Examples:

- quickstart/train-style helper commands over existing DataPoint sync flows
- guided setup wrappers for demo/bootstrap

## Level 1.6: Deterministic Simplicity Package

Status: `Planned`
Roadmap reference: `SMP-002`

Purpose:

- deliver a production-grade deterministic lane for repeated KPI/metric questions
- provide embed-ready integration while preserving governance guarantees

Scope:

- deterministic template/function execution path (before freeform SQL when applicable)
- integration-oriented SDK/embeddable surface
- operational checks for deterministic route safety, correctness, and regression gating

Sequencing rule:

- Level 1.6 should follow Level 1.5 KPI stability, except for thin Level 1.4 wrappers.

## Level 2: Context Enhancement (DataPoints)

Status: `Partial`
Roadmap references: `PLT-002`, `PLT-003`

Delivered today:

- DataPoint ingestion/sync.
- Retrieval from vectors/graph as query context.
- Context-only answers and evidence metadata.
- Auto-profiling to generate pending DataPoints (with review/approval workflow).

Still limited:

- Deep domain semantics still depend on DataPoint quality/coverage.
- No fully managed ontology lifecycle.

Planned enhancements:

- **Knowledge Graph column-level edges** (in progress)
  - DERIVES_FROM: column → column lineage
  - COMPUTES: metric → column relationships
  - HAS_GRAIN: table → granularity
- **Entity memory in session context** (planned)
  - Track tables/metrics/columns mentioned in conversation

## Level 2.5: QueryDataPoints (Reusable SQL Patterns)

Status: `Planned`
Roadmap references: `SMP-002`, `DYN-001`

Purpose:

- Store and retrieve pre-validated SQL templates for common queries
- Skip SQL generation for known query patterns (faster, more consistent)
- Enable parameterized execution with validation

Scope:

- QueryDataPoint model: `sql_template`, `parameters`, `backend_variants`
- CLI: `datachat dp add-query`, `datachat dp list --type Query`
- Pipeline integration: ContextAgent retrieves, SQLAgent uses template
- Contract validation: SQL syntax, parameter matching

## Level 3: Executable Metrics

Status: `Planned`
Roadmap references: `SMP-002`, `DYN-001`

Target outcome:

- Deterministic metric templates and parameterized execution path for stable KPI answers.

Not shipped yet as a dedicated level.

## Level 4: Performance Optimization

Status: `Planned`
Roadmap references: `DYN-003`, `DYN-004`

Target outcome:

- Materialization/caching strategies, adaptive query routing, and refresh policies.

Not shipped yet.

## Level 5: Intelligence

Status: `Planned`
Roadmap references: `DYN-006`, `DYN-007`

Target outcome:

- Automated anomaly detection, dependency-aware diagnosis, and remediation workflows.

Not shipped yet.

## Practical Guidance

For production use now:

1. Start with Level 1 credentials-only to validate connectivity and query quality.
2. Prioritize Level 1.5 foundation controls (contracts + eval + observability).
3. Add Level 2 DataPoints for critical business metrics and terms.
4. Treat Levels 3-5 as roadmap items, not current guarantees.
