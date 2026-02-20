# Dynamic Data Agent Harness Architecture

**Status:** Accepted direction (canonical target-design reference)  
**Date:** February 20, 2026  
**Audience:** Product, backend, frontend, platform, and data engineering teams

---

## 1. Intent

Build DataChat into a **dynamic data agent harness** with Claude-Code/Codex-level runtime quality, but specialized for:

- Databases and datastores
- Business logic in docs/files
- Organizational knowledge and operating context

This is **not** a coding agent architecture. It is a **data + business knowledge operator** architecture.

Roadmap linkage:

- `DYN-001`..`DYN-007` in `docs/ROADMAP.md` track implementation sequencing for this proposal.

---

## 2. Product Goals

### Primary goals

- Handle multi-step data requests dynamically, not via fixed agent pipelines.
- Combine structured data context and unstructured business context in one decision loop.
- Safely execute data actions with policy/approval controls.
- Preserve reasoning traceability, reproducibility, and auditability.
- Reach harness quality comparable to top coding agents in:
  - action selection quality
  - context gathering quality
  - verification and recovery behavior
  - tool orchestration quality

### Non-goals

- General software engineering/copilot workflows (file edits, build systems, code refactors).
- Unbounded autonomous write operations in production systems.
- Replacing BI governance or data platform ownership.

---

## 3. High-Level System Shape

```text
User Request
   ↓
Session + Workspace Context Loader
   ↓
Goal Decomposer + Planner
   ↓
Dynamic Action Loop (plan → act → verify → adapt)
   ↓
Evidence-backed Answer + Action Artifacts + Trace
```

Architecture layers:

1. **Control Plane**
   - Planner, loop controller, policy engine, approvals, checkpoints.
2. **Execution Plane**
   - Tool harness for queries, profiling, metadata, documentation retrieval, lineage, telemetry.
3. **Knowledge Plane**
   - Unified retrieval over DataPoints, schema metadata, docs, runbooks, and org artifacts.
4. **Experience Plane**
   - UI/CLI session continuity, stateful workspaces, explainability, incident-safe controls.

---

## 4. Dynamic Agentic Loop (Core Runtime)

The fixed sequence pipeline is replaced by a bounded loop:

```text
Gather Context → Choose Next Action → Execute → Verify → Update State → Repeat
```

Loop invariants:

- Every action must declare:
  - purpose
  - required inputs
  - expected output contract
  - safety class (`read`, `write`, `destructive`, `external`)
- Every action result must be verified with deterministic checks where possible.
- Loop termination occurs on:
  - confidence threshold met
  - explicit user confirmation
  - max loop budget (time/actions/cost)
  - policy violation or unresolved ambiguity

Execution budgets:

- `max_actions_per_turn`
- `max_total_latency_ms`
- `max_llm_tokens`
- `max_query_cost`
- `max_write_operations` (default `0` unless approved)

---

## 5. Unified Data Workspace Model

The runtime workspace should unify three knowledge classes:

1. **Data systems**
   - SQL DBs, warehouses, lakehouse engines, selected NoSQL/search stores.
   - Schemas, statistics, lineage, quality metadata.
2. **Business logic corpus**
   - Metric definitions, PRDs, runbooks, SOPs, decision docs, policy docs.
3. **Organizational context**
   - Teams, owners, data stewardship, SLAs, escalation paths, glossary terms.

Suggested workspace state:

```yaml
workspace_state:
  session:
    session_id: "..."
    goals: []
    open_questions: []
    assumptions: []
  data_targets:
    selected_connections: []
    default_connection: "..."
    schema_snapshots: {}
  knowledge:
    datapoints: []
    docs_index_refs: []
    org_entities: []
  runtime:
    recent_actions: []
    verification_history: []
    risk_flags: []
```

---

## 6. Tool Harness (Data-First)

Tool categories:

1. **Data read tools**
   - list schemas/tables/columns
   - sample rows
   - run read-only SQL
   - explain plans
2. **Metadata and health tools**
   - profiling, stats refresh, freshness checks
   - quality summary, anomaly scan
3. **Knowledge retrieval tools**
   - semantic doc retrieval
   - keyword retrieval
   - entity/owner lookup
4. **Governed write tools (approval-gated)**
   - data fix suggestions
   - migration/runbook draft actions
   - controlled updates in scoped environments
5. **Operational tools**
   - monitor query/job status
   - audit report generation
   - incident context collection

Tool contract requirements:

- Typed inputs/outputs with versioned schemas.
- Deterministic failure classes (`validation`, `auth`, `timeout`, `semantic_mismatch`, etc.).
- Cost/risk estimate before execution.
- Full trace logging with correlation IDs.

---

## 7. Knowledge Fabric Design

Unify retrieval from:

- DataPoints (schema/business/process/query)
- live schema metadata
- profiling outputs
- docs and runbooks
- glossary and ownership maps

Retrieval architecture:

1. **Candidate generation**
   - vector retrieval + keyword retrieval + graph expansion.
2. **Context assembly**
   - dedupe, relevance caps, source balancing (schema vs business vs org).
3. **Grounding constraints**
   - enforce source attribution and confidence intervals.

Outcome:

- Answers can reference both:
  - technical truth (`table/column/query evidence`)
  - business truth (`policy/definition/owner evidence`)

---

## 8. Planning and Verification

### Planner responsibilities

- Decompose ambiguous requests into bounded sub-goals.
- Choose next best action based on uncertainty and expected value.
- Decide when to ask clarifying questions.

### Verifier responsibilities

- Validate result against:
  - schema compatibility
  - business rule consistency
  - expected output shape
  - contradiction with previously accepted facts

### Recovery behavior

- On failed verification:
  - revise plan
  - gather missing context
  - retry with explicit rationale
  - escalate to clarification when uncertainty remains high

---

## 9. Memory and Session Model

Memory layers:

1. **Turn memory**
   - action outputs and verification for current request.
2. **Session memory**
   - user goals, selected data domains, accepted assumptions.
3. **Workspace memory**
   - durable state: snapshots, cached profiles, approved definitions.

Key properties:

- Deterministic truncation policies.
- Snapshot checkpointing for replay/debug.
- Cross-device continuity through backend persistence.

---

## 10. Safety, Governance, and Permissions

Default posture:

- Read-only operations by default.
- Any write-like action requires explicit policy and approval.

Policy engine dimensions:

- environment (`dev`, `staging`, `prod`)
- tool category and risk level
- data sensitivity class (PII/financial/restricted)
- actor role and approval state

Required controls:

- preflight risk summaries
- approval checkpoints
- immutable audit logs
- rollback playbooks for governed writes

---

## 11. Subagents and Skills (Data Domain)

Subagents should be domain-specialized, for example:

- `QueryInvestigationSubagent`
- `MetricDefinitionSubagent`
- `DataQualitySubagent`
- `LineageImpactSubagent`
- `IncidentTriageSubagent`

Skills should encode reusable workflows, for example:

- KPI discrepancy investigation
- freshness incident diagnosis
- metric definition reconciliation
- source-of-truth conflict resolution

Each skill provides:

- trigger patterns
- action templates
- verification rules
- escalation criteria

---

## 12. Interface and UX Requirements

UI/CLI must expose harness state clearly:

- current goal and sub-goals
- actions taken and why
- verification outcomes
- pending approvals
- confidence and unresolved assumptions

Critical UX requirements:

- one-click trace inspection
- deterministic retry from checkpoint
- safe interruption/cancellation
- explicit “what changed” between retries

---

## 13. Observability and Evaluation

Metrics to operationalize:

- task completion rate (single and multi-step)
- verification pass rate
- clarification necessity rate
- policy violation prevention rate
- time-to-correct-answer
- source grounding quality
- business-definition consistency score

Evaluation framework:

- golden task suites by domain
- replayable traces
- regression gates for routing/planning/verification
- failure taxonomy dashboards

---

## 14. Implementation Roadmap

### Phase 0: Foundation hardening

- Standardize tool contracts and trace IDs.
- Unify session/workspace state schema.
- Introduce loop budgets and stop conditions.

### Phase 1: Dynamic loop MVP

- Add planner + verifier around existing tools.
- Replace fixed sequencing with bounded action loop.
- Ship action trace UI.

### Phase 2: Unified knowledge plane

- Merge docs/runbooks/org context retrieval with DataPoints.
- Add source balancing and contradiction detection.

### Phase 3: Policy and governed operations

- Full approval workflows and risk classes.
- Scoped write-capable tools for approved environments.

### Phase 4: Domain subagents and skills

- Ship reusable investigation skills.
- Add subagent orchestration and specialized evaluators.

---

## 15. Risks and Mitigations

Risk: runaway agent loops  
Mitigation: strict budgets, hard stop conditions, and bounded retries.

Risk: hallucinated business logic  
Mitigation: source-required assertions and business-definition verification checks.

Risk: unsafe actions in production  
Mitigation: read-only defaults, policy gates, and mandatory approvals.

Risk: context overload and latency  
Mitigation: multi-stage retrieval, compression, and caching.

---

## 16. Architecture Decision Summary

This direction is valid and recommended.

DataChat should evolve into a **dynamic data-agent harness** with:

- dynamic planning and action loops
- multi-source grounding (data + business docs + org context)
- strong policy controls
- high-quality verification/recovery behavior

This achieves harness-level capability parity with leading coding agents while staying true to DataChat’s product identity: **data intelligence + business logic + organizational knowledge**.
