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

### Priority 0: MetadataOps Foundation (Ship before major Level 3-5 expansion)

1. **Metadata contracts + linting (authoring quality gate)**
   - Enforce required metadata fields (grain, units, freshness, owner, exclusions, confidence notes).
   - Fail DataPoint ingest/sync when contracts are violated.
2. **Metadata authoring lifecycle controls**
   - Add review/versioning conventions and ownership workflow for managed/user DataPoints.
   - Prevent silent overrides and ambiguous duplicate definitions.
3. **AI-readiness evaluation in CI**
   - Expand eval gates for intent/retrieval/sql/source/clarification behavior.
   - Require thresholded eval pass before merge for retrieval/prompt changes.
4. **Governance metadata APIs**
   - Expose lineage/freshness/quality metadata as queryable API surfaces.
   - Ensure agents can consume governance context instead of inferring it.
5. **RAG observability and traceability**
   - Persist retrieval traces (source path/tier/version, score, fallback route).
   - Add deterministic debug views for why a source was selected.
6. **Runtime telemetry loops**
   - Track clarification churn, wrong-table selections, fallback rates, and low-confidence hotspots.
   - Use telemetry to prioritize metadata improvements over prompt tweaks.

### Phase Sequencing Decision: Simplicity Package vs Foundation

Decision:

- Ship the full "Vanna-style simplicity package" in **Phase 1.6**, after MetadataOps foundation KPIs are green.
- Allow only a thin, low-risk onboarding wrapper in **Phase 1.4**.

Phase 1.4 (allowed scope):

- simple entry UX/wrappers (for example, quickstart or train-style helper commands)
- no new retrieval/routing semantics
- no bypass of metadata/eval/trace controls

Phase 1.6 (full package scope):

- deterministic template/function execution lane for repeated business questions
- embed-ready SDK/surface for app integration
- operational safety checks and route/eval coverage for deterministic-first execution

Rationale:

- preserve correctness and governance while improving onboarding
- avoid introducing convenience paths that drift from metadata truth

### Priority 1: Platform expansion after foundation KPIs are green

1. Connector expansion beyond current runtime engines.
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

## Success Metrics (Foundation Lane)

- Metadata contract lint pass rate: >= 95% on managed/user DataPoints.
- Eval pass rate: retrieval/qa/intent/catalog suites pass configured thresholds in CI.
- Retrieval trace coverage: >= 95% of responses include inspectable source provenance.
- Clarification churn: repeated clarification loops reduced release-over-release.
- Wrong-table regressions: no uncaught critical regressions in deterministic intent tests.

## Delivery Gate for Advanced Levels

Before major Level 3-5 feature expansion, foundation lane metrics must be green for at least one full release cycle.

## Documentation Contract

A feature can only be marked "supported" when:

- code path exists,
- behavior is test-covered,
- user docs explain operational preconditions.
