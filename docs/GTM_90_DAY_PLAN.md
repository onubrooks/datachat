# DataChat 90-Day GTM, Pricing, and Release Plan

**Version:** 1.0  
**Last Updated:** 2026-02-20  
**Owner:** Solo founder (Onuh)

## 1. Blunt Assessment

DataChat has real value, but the market is crowded and unforgiving.

If positioned as generic "chat with your data," it will likely underperform.  
If positioned as a trusted, auditable, data decision copilot for messy business logic, it can win a narrow but valuable segment.

Current strengths (already shipped):

- credentials-only path with deterministic schema/catalog flows
- DataPoint-enhanced retrieval
- multi-database target routing
- UI/CLI/API parity on core ask flow

Current gaps (important for paid adoption):

- limited runtime connector breadth (BigQuery/Redshift not shipped)
- metadata quality controls still being hardened (`FND-001`, `FND-003`, `FND-005`)
- dynamic data-agent harness is accepted direction but not yet runtime reality

## 2. Positioning and Wedge

Positioning statement:

- DataChat is a **decision workflow system for finance teams** now.
- DataChat becomes an **AI platform for business decision makers** after finance workflow outcomes are repeatable.
- Product proof is anchored on correctness, provenance, and business-logic grounding, not generic "chat with data."

Wedge use case (initial):

- weekly finance operating reviews (liquidity/variance/risk) where teams reconcile definitions across systems and need decision-ready, source-backed output

Initial ICP (Ideal Customer Profile):

- fintech, digital banking, and finance-heavy mid-market teams
- organizations with recurring leadership finance review cycles
- teams with high cost of metric disputes, reconciliation lag, and low trust in ad hoc BI/chat answers
- environments where governance/provenance matter more than exploratory novelty

Do not target first:

- large regulated enterprises requiring full compliance stack on day one
- teams needing broad warehouse connector coverage immediately
- "replace all BI" buyers

## 3. Monetization Model

Use open-core + hosted commercial packaging.

### 3.1 Near-term Revenue (Days 1-90)

Offer design partner plans immediately:

- **Design Partner Lite:** $1,000/month
- **Design Partner Core:** $2,500/month
- **Design Partner Plus:** $5,000/month

What partners get:

- prioritized support and onboarding
- guided metadata/DataPoint setup
- weekly reliability review (traces, bad answers, fixes)
- roadmap influence

What they commit:

- named operator(s)
- weekly usage and feedback
- permission to anonymize outcomes for case studies

### 3.2 Post-90 Day Packaging (Target)

- **Community (OSS/self-host):** free
- **Pro (hosted):** $99-$299/workspace + usage
- **Team (hosted):** $499-$1,499/workspace + usage + governance features
- **Enterprise:** custom (SSO, audit extensions, deployment options, SLA)

### 3.3 Licensing and Source-Code Reality

Current state:

- repository is Apache 2.0 licensed
- customers can legally self-host and run the code

Implication:

- paid hosted plans must win on operational value (speed, support, reliability, integrations), not code secrecy

Commercial protection options (future decision):

1. Keep Apache open-core and monetize hosting/support/SLA only.
2. Move to dual-license model (open + commercial terms for specific uses).
3. Keep core OSS but ship key enterprise modules as closed-source extensions.
4. Adopt a source-available business license for non-community use.

Decision criteria:

- desired community adoption vs revenue protection
- enterprise procurement requirements
- legal/operational overhead acceptable for a solo founder

## 4. Release Strategy (90 Days)

## 4.1 Release Principles

- sell outcomes, not feature counts
- no broad launch until reliability baseline is demonstrably strong
- every release must improve trust metrics (traceability, consistency, false-answer rate)

## 4.2 Milestones

### Days 1-30: Foundation + Pipeline Build

Goals:

- complete `FND-001` implementation and ship
- prove repeatable onboarding for one target persona
- build design-partner pipeline

Execution:

1. Product hardening:
   - ship `FND-001` contracts/lint gate
   - progress `FND-003` + `FND-005` enough to produce reliability evidence
2. Sales prep:
   - create one clear offer page and one 5-slide sales deck
   - create one 20-minute live demo flow using grocery + one real prospect dataset
3. Outbound:
   - contact 60 qualified prospects (founder-led)
   - target 12 discovery calls
   - target 4 technical deep-dive calls

Exit criteria:

- 2 design partner verbal commitments or 1 signed paid partner
- `FND-001` running and validated in CI/local workflows

### Days 31-60: Paid Pilots + Evidence Capture

Goals:

- onboard 2-3 design partners
- get measurable weekly usage and reliability improvements
- produce first case-study-grade evidence

Execution:

1. Onboarding playbook:
   - week 1 setup + schema mapping + top 10 recurring questions
   - week 2 metadata refinement + answer trace reviews
   - week 3 KPI question package and operator training
2. Reliability loop:
   - weekly review of clarification churn, fallback hotspots, wrong-answer incidents
   - patch high-frequency failures rapidly
3. Proof assets:
   - collect baseline vs current outcomes (time-to-answer, trust rate, rework rate)

Exit criteria:

- 2 paying partners active weekly
- at least 1 documented win ("replaced/manual workflow reduced")

### Days 61-90: Convert, Narrow, and Public Launch

Goals:

- convert pilot users into retained paying customers
- define repeatable narrow GTM motion
- launch publicly with proof, not hype

Execution:

1. Commercial:
   - convert active partners to 3- or 6-month commitments
   - formalize pricing bands and support boundaries
2. Messaging:
   - publish 2 case studies with measurable outcomes
   - publish reliability scorecard and known-limitations page
3. Launch:
   - soft launch to targeted communities (analytics engineering, data leads)
   - run 2 live demos/webinars focused on trust + provenance

Exit criteria:

- 3 paying customers
- clear retained ICP with repeatable onboarding under 10 business days

## 5. KPI Framework

Track three layers: commercial, adoption, reliability.

### 5.1 Commercial KPIs

- discovery-to-paid conversion rate
- number of paying design partners
- monthly recurring revenue (MRR)
- gross churn and expansion potential

90-day targets:

- 3 paying customers
- $4k-$10k MRR range

### 5.2 Adoption KPIs

- weekly active workspaces
- weekly active operators per workspace
- queries per active workspace per week
- repeat usage on top recurring question sets

90-day targets:

- >= 70% weekly activity among paying workspaces
- >= 2 active operators per paying workspace

### 5.3 Reliability/Trust KPIs

- metadata lint pass rate (`FND-001`): target >= 95%
- retrieval trace coverage (`FND-005`): target >= 95%
- clarification churn trend: down release-over-release
- high-severity wrong-answer incidents: near zero on top recurring flows

90-day targets:

- publish weekly reliability report for all paying partners

## 6. Marketing Plan (Founder-Led)

Primary channels:

- direct founder outreach (LinkedIn/email) to data leaders
- technical content on deterministic + auditable data AI
- live demos in targeted communities

Content cadence:

- 1 deep technical post/week
- 1 practical "before/after" walkthrough/week
- 1 public reliability update every 2 weeks

Message architecture:

- Problem: "Your team does not trust AI answers on business metrics."
- Mechanism: "Deterministic routing + metadata contracts + evidence traces."
- Outcome: "Faster answers with lower rework and fewer metric disputes."

## 7. Go/No-Go Rules

At day 45:

- if no paid partner, narrow ICP further and simplify offer (single use case only)

At day 90:

- if fewer than 2 paying retained customers, pause broad roadmap expansion and revalidate problem/segment fit
- if trust/reliability metrics are not improving, focus exclusively on foundation lane (`FND-*`) before adding advanced features

## 8. Weekly Operating Cadence (Solo)

Every week:

1. Monday: pipeline review (leads, calls, deal stage)
2. Tuesday: product reliability review (`FND` metrics + incidents)
3. Wednesday: build day (highest-impact fix)
4. Thursday: partner onboarding/support
5. Friday: publish update (learning + metrics + next week focus)

Time split recommendation:

- 40% product/reliability
- 40% sales/onboarding
- 20% content/ops

## 9. Immediate Next Actions (Next 14 Days)

1. Complete `FND-001` and validate against real metadata sets.
2. Build one "design partner package" page and outbound script.
3. Create target list of 60 prospects matching ICP.
4. Run 12 discovery calls.
5. Close first paid design partner.
