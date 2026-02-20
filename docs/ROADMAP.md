# DataChat Unified Roadmap

**Version:** 1.0  
**Last Updated:** February 20, 2026

This document is the single source of truth for **delivery status, sequencing, and initiative tracking**.

---

## Must-Win Workflow Charter (Finance)

Current product wedge:

- **Workflow:** Revenue variance and liquidity risk investigation.
- **Buyer persona:** Head of Finance / CFO delegate.
- **Operator persona:** FP&A lead, finance manager, treasury/risk analyst.
- **Business problem:** slow, low-trust cross-system reconciliation for decision-grade finance answers.

Workflow steps to optimize:

1. Ask: submit finance question with period/segment context.
2. Ground: resolve canonical metric definitions + ownership.
3. Retrieve: gather evidence across DataPoints, docs, and governed sources.
4. Verify: run checks, surface caveats, and produce confidence.
5. Decide: return answer package with drill-down and audit trail.

Hard prioritization rule:

- roadmap items must map to at least one workflow step above and one charter KPI in `docs/PRD.md`.
- if no direct mapping, item moves to backlog.

---

## Ownership Model

- `docs/PRD.md` owns product intent (`what` and `why`).
- `docs/ARCHITECTURE.md` owns technical design (`how`).
- `docs/LEVELS.md` owns maturity definitions (`which capabilities define each level`).
- `docs/ROADMAP.md` owns delivery plan (`when`, `status`, `dependency`).

Conflict resolution:

1. Product scope conflict -> `PRD.md` wins.
2. Design conflict -> `ARCHITECTURE.md` wins.
3. Level classification conflict -> `LEVELS.md` wins.
4. Status/timeline conflict -> `ROADMAP.md` wins.

---

## Status Legend

- `Done` - shipped and validated.
- `In Progress` - actively being implemented.
- `Planned` - accepted but not started.
- `Blocked` - cannot proceed due to unresolved dependency.

## Execution Model (Solo)

Current operator model:

- single maintainer workflow (Onuh)
- no required owner column per initiative
- no issue/epic dependency on GitHub for roadmap operation

---

## Initiative Index

| ID | Initiative | Area | Status | Depends On | Spec |
|----|------------|------|--------|------------|------|
| FND-001 | Metadata contracts + linting gates | MetadataOps | In Progress | - | [FND-001](specs/FND-001.md) |
| FND-002 | Metadata authoring lifecycle controls | MetadataOps | Planned | FND-001 | [FND-002](specs/FND-002.md) |
| FND-003 | AI-readiness eval gates in CI | MetadataOps | In Progress | FND-001 | [FND-003](specs/FND-003.md) |
| FND-004 | Governance metadata APIs | MetadataOps | Planned | FND-001 | [FND-004](specs/FND-004.md) |
| FND-005 | Retrieval/source trace observability | MetadataOps | In Progress | FND-001 | [FND-005](specs/FND-005.md) |
| FND-006 | Runtime telemetry loops (clarification/fallback hotspots) | MetadataOps | Planned | FND-003, FND-005 | [FND-006](specs/FND-006.md) |
| SMP-001 | Thin simple entry layer wrappers (non-semantic) | UX | Planned | FND-001 | [SMP-001](specs/SMP-001.md) |
| SMP-002 | Deterministic simplicity package (template/function lane) | Runtime | Planned | FND-001..FND-006 stable | [SMP-002](specs/SMP-002.md) |
| PLT-001 | Runtime connector expansion | Platform | Planned | FND foundation stable | [PLT-001](specs/PLT-001.md) |
| PLT-002 | Deterministic coverage for additional catalog intents | Runtime | Planned | FND-003 | [PLT-002](specs/PLT-002.md) |
| PLT-003 | Semantic accuracy improvements in low-context mode | Runtime | Planned | FND-005, FND-006 | [PLT-003](specs/PLT-003.md) |
| DYN-001 | Dynamic planner + verifier loop foundation | Dynamic Agent | Planned | FND foundation stable | [DYN-001](specs/DYN-001.md) |
| DYN-002 | Unified workspace state (data + docs + org context) | Dynamic Agent | Planned | DYN-001 | [DYN-002](specs/DYN-002.md) |
| DYN-003 | Data-first tool harness with policy/approval classes | Dynamic Agent | Planned | DYN-001 | [DYN-003](specs/DYN-003.md) |
| DYN-004 | Unified knowledge fabric (DataPoints + docs + org retrieval) | Dynamic Agent | Planned | DYN-002 | [DYN-004](specs/DYN-004.md) |
| DYN-005 | Checkpoints, memory layers, replayable traces | Dynamic Agent | Planned | DYN-001 | [DYN-005](specs/DYN-005.md) |
| DYN-006 | Domain subagents + skills | Dynamic Agent | Planned | DYN-001..DYN-005 | [DYN-006](specs/DYN-006.md) |
| DYN-007 | Dynamic-agent eval gates and operational scorecards | Dynamic Agent | Planned | DYN-001..DYN-005 | [DYN-007](specs/DYN-007.md) |
| WDG-001 | Finance wedge workflow v1 (revenue variance + liquidity risk) | Product | In Progress | FND-001, FND-005 | [WDG-001](specs/WDG-001.md) |

---

## Initiative-to-Workflow Mapping (Finance Wedge)

| Initiative IDs | Workflow Step(s) | Expected KPI Movement |
|----------------|------------------|-----------------------|
| FND-001, FND-002, FND-004 | Ground | reduce wrong-definition incidents, reduce rework |
| FND-003, FND-005, FND-006 | Verify | increase attribution coverage, reduce clarification loops |
| SMP-001, SMP-002 | Ask, Decide | reduce time-to-answer package |
| PLT-001, PLT-002 | Retrieve | improve coverage/speed across systems |
| PLT-003 | Verify, Decide | improve answer quality in low-context runs |
| DYN-001, DYN-002, DYN-003 | Ask, Retrieve, Verify | reduce latency + operator effort |
| DYN-004, DYN-005 | Ground, Verify | improve trust, auditability, replayability |
| DYN-006, DYN-007 | Decide | stabilize outcome quality and operational scorecards |
| WDG-001 | Ask, Ground, Retrieve, Verify, Decide | reduce time-to-trusted-answer and rework |

---

## Prioritization Scoring Rubric

Use this rubric before moving any initiative from `Planned` to `In Progress`.

Score each dimension from 1 to 5:

- **Workflow Impact:** expected improvement to finance wedge workflow outcome.
- **Trust/Risk Reduction:** improvement to provenance, governance, or wrong-answer prevention.
- **Speed/Operator Efficiency:** reduction in time-to-trusted-answer.
- **Feasibility (Solo):** realistic implementation/test burden for a single maintainer.
- **Reusability:** value of capability beyond first wedge without diluting wedge focus.

Composite score:

- `Priority Score = 0.35*Workflow Impact + 0.25*Trust + 0.20*Speed + 0.15*Feasibility + 0.05*Reusability`

Promotion threshold:

- `>= 3.8`: eligible for next active slot.
- `3.0-3.7`: keep planned, needs tighter scope.
- `< 3.0`: backlog.

Mandatory gate:

- any item with Trust/Risk Reduction `< 3` cannot be promoted, regardless of composite score.

---

## Sequencing Plan

### Phase A: Foundation Stabilization

Scope:

- FND-001..FND-006
- SMP-001 (only thin wrappers, no semantic/routing changes)

Exit criteria:

- foundation KPIs green for one full release cycle
- deterministic intent regressions blocked by CI
- provenance/traces inspectable in production workflows

### Phase B: Simplicity + Platform Expansion

Scope:

- SMP-002
- PLT-001..PLT-003

Gate:

- Phase A exit criteria complete

### Phase C: Dynamic Data Agent Foundation

Scope:

- DYN-001..DYN-005

Gate:

- Phase B stable operations

### Phase D: Dynamic Data Agent Productization

Scope:

- DYN-006..DYN-007

Gate:

- replayable and auditable dynamic loop with safety controls in place

---

## Level Mapping

| Level | Initiative IDs |
|-------|----------------|
| Level 1 | FND-001, FND-003, FND-005 |
| Level 1.4 | SMP-001 |
| Level 1.5 | FND-001..FND-006 |
| Level 1.6 | SMP-002 |
| Level 2 | PLT-002, PLT-003 |
| Level 3 | SMP-002 + DYN-001 |
| Level 4 | DYN-003, DYN-004 |
| Level 5 | DYN-006, DYN-007 |

---

## Maintenance Rules

When adding/changing work:

1. Create or update initiative ID here first.
2. Create/update a spec in `docs/specs/<ID>.md`.
3. Do not move an initiative to `In Progress` unless its spec has no placeholders and includes concrete acceptance criteria + test plan.
4. Reference the ID in PRD/Architecture/Levels instead of duplicating status prose.
5. Update dependencies and phase placement.
6. Mark status changes only in this file.
