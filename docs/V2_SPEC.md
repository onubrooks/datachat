# DataChat v2 Spec and Status

This document distinguishes what is implemented from what is roadmap.

## v2 Goals

- Improve routing and reliability for database-first workflows.
- Make tool execution safe, typed, and policy-controlled.
- Keep credentials-only mode viable as a first-class path.
- Add workspace/code understanding in a later phase.

## Implemented in Current Codebase

### Routing and Intent Handling

- Intent gate before main pipeline.
- Clarification loops with max-turn controls.
- Fast-path handling for obvious non-data intents (`help`, exit-like prompts, etc.).

### SQL and Credentials-Only Improvements

- Deterministic catalog intelligence for schema-shape questions.
- Live schema snapshots for SQL context.
- Confidence/clarification safeguards to avoid overconfident wrong answers.

### Multi-Database and Context Propagation

- `target_database` support in chat API.
- Per-request DB type/URL propagation through orchestration.
- Strict failure behavior when explicit overrides cannot be resolved.

### Tooling Reliability

- `/api/v1/tools` exposes typed parameter schemas.
- `/api/v1/tools/execute` supports `target_database`.
- Runtime metadata injection for built-in tools (`database_type`, `database_url`, retriever/connector handles).
- Planner argument coercion to expected schema types.

## Planned (Not Yet Implemented)

### Workspace Features

Planned only:
- workspace index/status/search endpoints and commands
- filesystem/code ingestion as first-class retrieval context
- robust folder-level policies for workspace operations

### Connector Expansion

Planned only:
- runtime MySQL connector
- runtime BigQuery connector
- runtime Redshift connector

Note:
- Query templates for these engines may exist in profiling/catalog modules, but templates are not the same as connector execution support.

## Non-Goals for Current v2 Increment

- Automatic write operations on user files/workspaces.
- Multi-tenant authz model.
- Full incident automation (anomaly -> remediation loop).

## Acceptance Criteria for "Implemented" Claims

A feature is considered implemented only if:
1. Runtime path exists in backend code.
2. API/CLI behavior is test-covered.
3. Docs do not require speculative components to function.
