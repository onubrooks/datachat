# Product Requirements Document: DataChat

Version: 1.1
Date: February 8, 2026
Status: Active (delivery-tracking)

## Product Vision

DataChat provides a natural-language interface for operational and analytics databases, with progressive enhancement:
- credentials-only querying first
- richer business accuracy via DataPoints
- future expansion into deeper intelligence and workspace awareness

## Current Product Scope (Shipped)

### Core Query Experience

- Ask questions in natural language via CLI, UI, or API.
- Produce SQL + validated execution results.
- Provide clarifying questions when prompts are ambiguous.
- Bound clarification loops with configurable limits.

### Credentials-Only First-Class Path

- Works with only DB credentials and LLM key.
- Deterministic system-catalog queries for table/column/row-shape intents.
- Live schema context used for SQL generation.
- Explicit response hint when running without DataPoints.

### Multi-Database Routing

- Registry-backed connections with encrypted URLs.
- Per-request `target_database` routing in chat and tools endpoints.
- Fail-fast behavior when explicit target routing cannot be honored.

### Tooling Reliability

- Tool registry + policy enforcement.
- Typed tool parameter schemas.
- `/tools/execute` context injection for built-in tools.

## In Scope Next (Near-Term)

1. Connector expansion beyond PostgreSQL/ClickHouse runtime.
2. Improved deterministic handling for additional catalog intents.
3. Better answer quality for semantic metrics in low-context environments.

## Planned, Not Yet Shipped

### Workspace/Code Understanding

Planned but not released as a product feature:
- indexing arbitrary folders/workspaces
- code-aware retrieval across dbt/SQL/docs/code assets
- workspace search/status API and CLI workflows

### Advanced Levels (3-5)

Planned but not released:
- executable metric template layer (Level 3)
- performance/materialization automation (Level 4)
- anomaly + root-cause automation (Level 5)

## Product Principles

1. Truth over aspiration: docs must separate shipped vs planned.
2. Safe by default: avoid silent fallback to unintended databases.
3. Determinism first: use catalog/system queries before expensive LLM reasoning when possible.
4. Progressive enhancement: DataPoints improve quality but are not required to start.

## Success Metrics (Current)

- Credentials-only queries succeed for schema-shape intents without DataPoints.
- Clarification flow prevents obvious misfires for ambiguous prompts.
- `target_database` routing is deterministic and test-covered.
- Tool execution fails safely when context/approval requirements are not met.

## Documentation Contract

A feature can only be marked "supported" when:
- code path exists,
- behavior is test-covered,
- user docs explain operational preconditions.
