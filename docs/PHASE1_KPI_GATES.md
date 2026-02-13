# Phase 1 KPI Gates

This page defines the operational hardening gates for Phase 1 (core runtime).

## Goals

- enforce stability before merge
- enforce credentials-only intent/catalog quality before release
- track simple SLOs (latency + model-call budget) for release sign-off

## Configuration

All thresholds and checks are defined in:

- `config/phase1_kpi.json`

## Runner

Use:

```bash
python scripts/phase1_kpi_gate.py --mode ci
```

CI mode executes deterministic local checks configured in `ci_checks`.

Release mode (requires running API and representative environment):

```bash
python scripts/phase1_kpi_gate.py --mode release --api-base http://localhost:8000
```

Release mode runs configured eval suites and enforces:

- catalog thresholds (`sql/source/clarification` match rates)
- intent average latency ceiling
- intent average LLM-call ceiling

## Recommended Release Sign-off

1. Run `python scripts/phase1_kpi_gate.py --mode ci`.
2. Run `python scripts/phase1_kpi_gate.py --mode release --api-base ...` against staging.
3. Confirm no threshold failures and attach outputs to release notes.
