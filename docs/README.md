# DataChat Docs

This directory contains product and engineering documentation for DataChat.

## Implementation Snapshot (February 2026)

Implemented now:

- credentials-only querying with deterministic catalog intelligence
- DataPoint-enhanced retrieval and answer synthesis
- multi-database registry with `target_database` routing
- tool registry/execution with policy checks and typed parameter schemas
- profiling pipeline that generates pending DataPoints for review

Planned (not yet implemented as full product features):

- workspace/folder indexing and codebase understanding workflows
- runtime connectors for BigQuery/Redshift
- Levels 3-5 automation features

## Document Map

- `../GETTING_STARTED.md` - setup paths and first-run flow.
- `ARCHITECTURE.md` - system architecture and engineering design guide.
- `API.md` - API endpoints and payloads.
- `CREDENTIALS_ONLY_MODE.md` - capabilities/limits for credentials-only mode.
- `MULTI_DATABASE.md` - connection registry and per-request routing.
- `LEVELS.md` - maturity model with implementation status.
- `PRD.md` - delivery-tracking PRD (shipped vs planned).
- `CAPABILITY_CONTRACT.md` - shipped runtime capability matrix (supported/partial/planned).
- `METADATAOPS_FOUNDATION.md` - philosophy, priorities, and contributor alignment for metadata quality + observability.
- `PHASE1_KPI_GATES.md` - CI and release KPI gates for Phase 1 operational hardening.
- `MANUAL_EVAL_SCORECARD.md` - manual scoring rubric + runner commands for UI/CLI comparison.
- `OPERATIONS.md` - deployment and operational guidance.
- `DEMO_PLAYBOOK.md` - demo setup and persona flows.
- `DATAPOINT_SCHEMA.md` - DataPoint model and conventions.
- `DATAPOINT_CONTRACTS.md` - lintable metadata contracts for DataPoint quality gates.
- `PLAYBOOK.md` - development workflows.
- `PROMPTS.md` - prompt architecture and guardrails.
- `DATAPOINT_EXAMPLES_TESTING.md` - end-to-end DataPoint-driven manual test playbook for grocery + fintech examples.

## Prompt Files

Prompt sources live in `prompts/`. See `prompts/README.md` and `PROMPTS.md`.
