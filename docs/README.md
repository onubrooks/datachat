# DataChat Docs

This folder contains the product, architecture, and operations documentation for DataChat.

## Feature Snapshot (Combined)

- Levels 1-5 progression: schema-aware querying, business context, executable metrics, performance optimization, intelligence.
- Prompt system: versioned prompt files, PromptLoader, regression tests, agent-specific prompts.
- v2 tooling: agent router, tool registry with policy-as-config, read-only tools, audit logging.
- Workspace ingestion: WorkspaceDataPoints, incremental indexing, semantic search, UI/CLI parity.
- RAG roadmap: chunking, hybrid retrieval upgrades, reranking, evaluation, and routing.
- API/CLI surface: system init, chat, databases, profiling, datapoints lifecycle, sync, workspace.
- Ops + multi-db: encrypted registry, default/target routing, multi-node sync guidance.
- CLI: run `datachat --version`; install with `pip install -e .` if missing.
- System DB: set `SYSTEM_DATABASE_URL` for registry/profiling/demo data.
- Setup persistence: URLs are stored in `~/.datachat/config.json`.

## Document Map

- `docs/PRD.md` - product vision, personas, milestones.
- `docs/LEVELS.md` - Levels 1-5 progression and behavior.
- `docs/DATAPOINT_SCHEMA.md` - DataPoint schema across all levels.
- `docs/V2_SPEC.md` - v2 scope and tool/workspace roadmap.
- `docs/PROMPTS.md` - prompt architecture and best practices.
- `docs/API.md` - API endpoints and request bodies.
- `docs/MULTI_DATABASE.md` - multiple connection routing and setup.
- `docs/CREDENTIALS_ONLY_MODE.md` - capabilities, limits, and support matrix for credentials-only mode.
- `docs/OPERATIONS.md` - deployment and sync guidance.
- `docs/DEMO_PLAYBOOK.md` - persona-based demo environments and scripts.
- `docs/RAG_EVAL.md` - retrieval + end-to-end evaluation plan.
- `docs/CLAUDE.md` - architecture overview and knowledge model.
- `docs/PLAYBOOK.md` - dev patterns and coding workflows.

## Prompts

Prompts live in `prompts/` with versioning and tests. See `prompts/README.md` and `docs/PROMPTS.md` for structure and guidelines.
