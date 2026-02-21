# Finance Workflow v1 Manual Test

Use this runbook to validate the finance wedge workflow with existing demo data.

## Goal

Confirm DataChat can produce decision-grade finance answers (summary + drivers + caveats + sources) with reproducible prompts.

## Prerequisites

- backend API running
- frontend running (optional for UI pass)
- target database reachable
- optional system DB configured for registry/profiling features

## Track A: Fast Local Validation (global scope)

1. Seed fintech demo data:

```bash
datachat quickstart \
  --database-url <TARGET_DATABASE_URL> \
  --dataset fintech \
  --demo-reset \
  --non-interactive
```

2. Sync fintech DataPoints globally:

```bash
datachat dp sync \
  --datapoints-dir datapoints/examples/fintech_bank \
  --global-scope
```

3. Run prompts in CLI:

```bash
datachat ask "Show total deposits and net flow trend by segment for the last 8 weeks."
datachat ask "Which segments contributed most to week-over-week decline in net flow?"
datachat ask "Show failed transaction rate by day for the last 30 days and top driving transaction types."
datachat ask "Which loans are at highest risk based on days past due and recent payment status?"
```

## Track B: Scoped Validation (recommended)

Use when validating multi-database routing behavior.

1. Seed fintech and grocery datasets:

```bash
datachat demo --dataset fintech --reset --no-workspace
datachat demo --dataset grocery --reset --no-workspace
```

2. Sync DataPoints scoped to each connection:

```bash
datachat dp sync --datapoints-dir datapoints/examples/fintech_bank --connection-id <FINTECH_CONNECTION_ID>
datachat dp sync --datapoints-dir datapoints/examples/grocery_store --connection-id <GROCERY_CONNECTION_ID>
```

3. UI test:

- select fintech connection and run finance prompts from Track A
- verify no grocery entities appear in sources
- switch to grocery connection and ask grocery-only prompt:
  - `Which 5 SKUs have the highest stockout risk this week based on on-hand, reserved, and reorder level?`

## Pass/Fail Checklist

Mark pass only if all are true:

- answer includes direct summary (not only SQL)
- key driver breakdown is present
- caveats/assumptions are explicit
- source evidence is visible (tables/datapoints)
- follow-up question can continue the same investigation context
- multi-database run respects selected connection scope

## KPI Capture Sheet (manual)

For each prompt, capture:

- time-to-answer (start to final response)
- clarification count
- has_source_attribution (`yes/no`)
- reviewer confidence (`high/medium/low`)
- rework needed (`yes/no`)

## Quality Bar (release gate for workflow-mode contract)

Pass this gate before enabling workflow-mode request contract broadly:

- source coverage >= 95% (`has_source_attribution=yes` and >= 2 sources for each passed prompt)
- average clarifications <= 0.5 per prompt
- driver quality pass rate >= 80% (top drivers are directional and materially explain variance)
- consistency pass rate >= 95% for prompts that include deposits/withdrawals/net flow arithmetic
- reproducibility pass rate >= 90% across two reruns on unchanged data

Suggested minimum sample:

- 10 fintech prompts
- 5 cross-check prompts with scoped routing

## Suggested Prompt Set (Finance Wedge)

1. What is total deposits, withdrawals, and net flow by segment for the last 8 weeks?
2. Which segment had the sharpest week-over-week drop in net flow?
3. Show failed transaction rate trend by day and top 3 txn types driving failures.
4. Which countries have highest volume-adjusted decline rates?
5. What is loan default rate (90+ DPD) by segment and loan type?
6. Which loans are most likely to migrate to non-performing next cycle?
7. What share of fee income comes from card purchases vs transfers vs withdrawals?
8. If declined transfers recovered at 30%, how much monthly value is added?
9. Show concentration risk: top customers by balance and share of total deposits.
10. Summarize top 3 liquidity risk signals from the last 30 days with caveats.
