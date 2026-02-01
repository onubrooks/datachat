# DataChat v2 Spec (Draft)

## Goals

- Expand beyond SQL-only flows with a routed, tool-aware architecture.
- Keep analytics-first focus while enabling safe workspace ingestion.
- Provide pluggable tools with policy controls and auditability.
- Improve context quality via WorkspaceDataPoints and incremental indexing.
- Deliver a seamless UI + CLI experience with consistent workflows.

## Product Context

### User Personas (from PRD)

- **Sarah (Analyst):** needs fast ad-hoc answers, minimal setup, consistent metrics.
- **Marcus (Data Engineer):** wants standardized metric definitions and fewer repeat questions.
- **Priya (Data Platform Lead):** cares about governance, auditability, and root-cause analysis.
- **James (Executive):** wants trustworthy numbers with provenance and easy language.

#### Persona → v2 UX Requirements

- **Sarah (Analyst)**
  - One-command workspace index and quick search with clear snippets.
  - Fast path to first answer with minimal setup and guided errors.
- **Marcus (Data Engineer)**
  - Tool policies as config, audit logs, and deterministic routing.
  - Workspace indexing that highlights schema, dbt models, and SQL definitions.
- **Priya (Data Platform Lead)**
  - Correlation IDs and traceability across tools, agents, and pipelines.
  - Clear status reporting and explicit sync controls for multi-node environments.
- **James (Executive)**
  - Plain-language summaries with evidence links and confidence cues.
  - UX that favors answers over setup, with minimal technical exposure.

### Levels Progression (from LEVELS)

- **Level 1: Schema-Aware Querying** - auto-profiling, ManagedDataPoints, immediate NL→SQL.
- **Level 2: Context Enhancement** - user DataPoints add business definitions, filters, joins.
- **Level 3: Executable Metrics** - SQL templates with parameters and backend variants.
- **Level 4: Performance Optimization** - materialized/pre-aggregated metrics and caching.
- **Level 5: Intelligence** - knowledge graph dependencies, anomaly detection, root-cause hooks.

## Scope

### In

- Agent Router that selects between SQL pipeline and Tool pipeline.
- Tool registry with allowlist, policies, and audit logs.
- Read-only filesystem tools (list/read/search/metadata).
- Workspace indexing into WorkspaceDataPoints.
- API + CLI for workspace indexing and search.

### Out (v2)

- Automatic write operations to filesystem (requires explicit approval).
- Full IDE/terminal execution harness beyond approved tools.
- Multi-tenant auth/permissions (still single-tenant).

## Architecture

### Routed, Tool-Aware Pipeline

1. Router evaluates intent and context needs.
2. SQL pipeline for data questions.
3. Tool pipeline for workspace tasks (file search, doc lookup).
4. Tool outputs become retrieval context for final response.

### WorkspaceDataPoint

- file_path, language, symbols, docstrings, domain_tags
- extracted_entities (tables, models, configs)
- last_modified, checksum

## UX: Drop-In Workspace Flow

- Users can drop a folder into a workspace and run one command.
- Default ignore rules (node_modules, .git, build artifacts).
- Automatic grouping by domain (dbt, migrations, API schemas, docs).
- Clear index summary with what was ingested vs skipped.
- Same workflow in UI and CLI (status, search, reindex).

## UX Acceptance Criteria

### CLI

- `datachat workspace index --root ./` completes with a summary table.
- `datachat workspace status` shows last run, files indexed, files skipped.
- `datachat workspace search "query"` returns relevant hits with file paths.
- `datachat workspace index --dry-run` shows what will be indexed.
- Errors explain missing permissions or blocked paths.

### UI

- Workspace page allows selecting a folder and starting indexing.
- Live progress shows files scanned, indexed, skipped, and errors.
- Search UI returns results with file path + snippet + tags.
- Users can reindex and see when the last index ran.
- Errors are actionable and link to docs/troubleshooting.

### Persona Acceptance Criteria

- **Sarah (Analyst)**
  - `datachat workspace index --root ./` runs without extra config.
  - Search results include file path + snippet + tags in < 2 seconds on typical repos.
  - Error messages include a next step (setup or missing permissions).
- **Marcus (Data Engineer)**
  - Tool policies load from config and changes apply without code edits.
  - Audit logs show tool name, args, user/session, and correlation ID.
  - Workspace indexing groups dbt/models/migrations for quick review.
- **Priya (Data Platform Lead)**
  - Correlation IDs span API → pipeline → agents → tools in logs.
  - `/api/v1/workspace/status` reports last run, counts, and errors.
  - Multi-node guidance is documented and referenced in errors.
- **James (Executive)**
  - Responses include confidence cues and sources when available.
  - UI exposes only high-level actions (ask, search, reindex).

## API Additions

- `POST /api/v1/workspace/index`
  - Body: `{ "root": ".", "include": ["**/*.py"], "exclude": ["**/node_modules/**"] }`
  - Starts incremental indexing job.
- `GET /api/v1/workspace/status`
  - Returns indexing progress, last run, errors.
- `GET /api/v1/workspace/search?q=...`
  - Semantic search across WorkspaceDataPoints.

## CLI Additions

- `datachat workspace index --root ./`
- `datachat workspace search "auth flow"`

## Risks

- Security: tool access must be sandboxed and allowlisted.
- Cost: large file ingestion increases token usage.
- Performance: indexing needs incremental scanning to avoid slow rescans.
- UX: routing must be deterministic and explainable to users.

## Milestones

1. Read-only tools + WorkspaceDataPoint ingestion.
2. Router + tool planner/executor with policy checks.
3. Optional write tools gated by approval.

## RAG Upgrade Roadmap

**Near-term (v2 foundation)**
- Chunking strategy for WorkspaceDataPoints (size + overlap).
- Simple query normalization (lowercasing, stopword trim).
- Metadata filters for DataPoint type/tags.
- Optional lightweight reranker (disabled by default).

**Mid-term (quality gains)**
- BM25 keyword retrieval alongside vectors.
- Query rewriting/decomposition for multi-part questions.
- Context compression (dedupe, truncate, relevance caps).

**Later (advanced)**
- Retrieval evaluation set + regression gating.
- Embedding fine-tuning for schema/metric language.
- Adaptive retrieval routing based on intent/confidence.

## v2 Milestone Checklist

### Milestone 1: Read-Only Tools + Workspace Ingestion

- [ ] Tool registry + policy config loader
- [ ] Read-only filesystem tools (list/read/search/metadata)
- [ ] Audit logging with correlation IDs
- [ ] WorkspaceIndexer with incremental checksum/mtime
- [ ] WorkspaceDataPoint ingestion into vectors + graph
- [ ] CLI: `workspace index/status/search`
- [ ] API: `/workspace/index`, `/workspace/status`, `/workspace/search`

### Milestone 2: Router + Tool Planner/Executor

- [ ] Router selects SQL vs tool pipeline with explainable rationale
- [ ] Tool planner outputs structured calls
- [ ] Policy checks enforced at execution time
- [ ] Streaming updates show tool steps + outcomes

### Milestone 3: Controlled Write Tools (Gated)

- [ ] Explicit user approval flow for write actions
- [ ] Policy levels for read/write tools
- [ ] Audit log includes approvals and diff summaries
