# Finance Workflow Value Proof (One-Pager)

Use this in demo calls and buyer conversations to prove business value fast.

## What DataChat Solves

Finance teams lose hours reconciling definitions, checking source trust, and explaining variance before a decision can be made.

DataChat turns that into a repeatable workflow that outputs a decision-ready finance brief with sources, drivers, caveats, and follow-ups.

## Ideal Buyer and User

- Buyer: Head of Finance, CFO delegate, or VP Finance.
- Daily users: FP&A manager, treasury analyst, risk analyst, finance operations lead.

## Must-Win Workflow

- Weekly liquidity and variance review:
  - total deposits/withdrawals/net flow by segment
  - week-over-week decline drivers
  - caveats and confidence
  - explicit source evidence

## Baseline vs Target Outcome

- Baseline (today in many teams):
  - 60-120 minutes to build a trusted answer pack
  - 1-3 back-and-forth clarification loops
  - frequent rework when leadership asks "where did this number come from?"
- Target with DataChat:
  - <= 15-30 minutes to trusted brief
  - <= 1 clarification loop
  - source-attributed outputs by default

## Demo Proof Script (20-30 mins)

1. Seed fintech data (`scripts/fintech_seed.sql`).
2. Sync finance datapoints (`datapoints/examples/fintech_bank`).
3. Run prompts from `docs/finance/FINANCE_PROMPT_PACK_V1.md`.
4. Show:
   - direct answer
   - top 2-3 drivers
   - caveats/assumptions
   - source evidence
5. Score with `docs/templates/finance_workflow_scorecard.csv`.
6. Gate results via `scripts/finance_workflow_gate.py`.

Reference runbooks:

- `docs/finance/FINANCE_END_USER_QUICKSTART.md`
- `docs/finance/FINANCE_WORKFLOW_V1_MANUAL_TEST.md`

## What "Success" Looks Like for a Paid Pilot

- Scope: one finance workflow, one business unit, one dataset family.
- Duration: 2-4 weeks.
- Exit criteria:
  - >= 95% answers with source attribution
  - <= 0.5 average clarifications per prompt in the test pack
  - >= 80% driver-quality pass rate
  - demonstrable time-to-answer reduction versus current process

## Commercial Packaging (Practical Start)

- Paid pilot:
  - fixed-scope workflow implementation and quality bar sign-off
  - includes onboarding, datapoint setup, and scorecard reporting
- Expansion:
  - add adjacent workflows (risk, cost variance, forecast accuracy)
  - expand persona coverage after pilot KPI pass

## Objections and Straight Answers

- "Why not use a generic BI chatbot?"
  - Generic chat often lacks explicit provenance and workflow-grade packaging.
  - DataChat is optimized for decision workflows, not just ad hoc Q&A.
- "Can we trust the output?"
  - Trust is enforced through source attribution, caveats, and scorecarded quality gates.
- "Will this be shelfware?"
  - Pilot is scoped to one recurring finance meeting workflow with measurable KPI targets.

## Next Sales Asset to Build

Create a 5-slide companion deck:

1. workflow pain
2. live output example
3. KPI scorecard before/after
4. pilot scope and timeline
5. pricing and expansion path
