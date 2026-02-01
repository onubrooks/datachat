# Product Requirements Document: DataChat

**Version:** 1.0  
**Date:** January 30, 2026  
**Status:** Draft  
**Owner:** Onu Abah

---

## Executive Summary

DataChat is a data-native agent operating system that enables natural language interaction with databases, business metrics, and code repositories. It serves users across all sophistication levels through a progressive enhancement architecture, differentiating itself as the only AI assistant that understands database schemas, business semantics, AND implementation code in a unified system.

**Target Release:** v1.0 in 4 weeks (internal testing at Moniepoint + tech friends)  
**Open Source Launch:** Q2-Q3 2026

---

## Product Vision

### Problem Statement

Data teams face three disconnected challenges:

1. **Access Barrier:** Non-technical users can't query databases without SQL knowledge
2. **Consistency Problem:** Same metric calculated differently across teams (5 definitions of "revenue")
3. **Investigation Gap:** When metrics look wrong, debugging requires manual SQL sleuthing across code/data

**Current "Solutions" Fall Short:**
- Text-to-SQL tools: Generate SQL but don't understand business context (hallucinate metrics)
- Semantic layers (Cube, dbt): Require upfront configuration, don't handle natural language
- BI tools: Lock insights behind pre-built dashboards, can't answer novel questions

### Our Solution

DataChat combines three capabilities that no other tool offers:

1. **Natural Language Interface:** Query any database in English, zero SQL required
2. **Business Context Layer:** Optionally define metrics once, everyone gets consistent answers
3. **Code Understanding:** Read dbt models, Airflow DAGs, and docs to explain metric implementations

**Progressive Enhancement:** Users get value on day 1 (no setup), and sophistication scales as needed.

---

## User Personas

### Persona 1: Sarah - Data Analyst at E-commerce Startup

**Background:**
- 2 years experience, comfortable with Excel, basic SQL
- Needs to answer ad-hoc business questions quickly
- Currently: Bothers data engineers for every query

**Goals:**
- Query production database without waiting for engineers
- Understand what "conversion rate" means at this company
- Generate charts for weekly exec reviews

**DataChat Value:**
- **Level 1:** Query database in English immediately
- **Level 2:** Learn company metric definitions
- **Level 3:** Get consistent metrics without SQL memorization

**Success Metric:** Reduces time-to-insight from 2 days (waiting for eng) to 5 minutes

---

### Persona 2: Marcus - Data Engineer at SaaS Company

**Background:**
- 5 years experience, expert in SQL/Python/dbt
- Owns data warehouse and metric definitions
- Currently: Answers same questions repeatedly from analysts

**Goals:**
- Standardize metric definitions across organization
- Reduce analyst support burden
- Ensure dashboards and ad-hoc queries use same logic

**DataChat Value:**
- **Level 3:** Define executable DataPoints (SQL templates)
- **Level 4:** Pre-compute expensive aggregations automatically
- **Workspace:** DataChat reads dbt models, understands data transformations

**Success Metric:** 80% reduction in "how do I calculate X?" Slack messages

---

### Persona 3: Priya - Data Platform Lead at Fintech

**Background:**
- 10 years experience, deep expertise in data systems
- Responsible for data governance, quality, and performance
- Currently: Manual root cause analysis when metrics look wrong

**Goals:**
- Detect anomalies before business notices
- Automatically diagnose metric deviations
- Full audit trail for compliance

**DataChat Value:**
- **Level 5:** Knowledge graph tracks metric dependencies
- **Intelligence:** AI-powered anomaly detection and root cause analysis
- **Security:** Full audit logging, role-based access control

**Success Metric:** Mean time to resolution for metric issues reduced from 4 hours to 15 minutes

---

### Persona 4: James - Executive (Non-Technical)

**Background:**
- CFO with no technical background
- Needs accurate numbers for board presentations
- Currently: Relies on analysts, worries about data accuracy

**Goals:**
- Ask questions in plain English, get trustworthy answers
- Understand where numbers come from
- Confidence that metrics match what finance team uses

**DataChat Value:**
- **Level 1-2:** Natural language interface, no SQL needed
- **Level 3:** Consistent metrics (finance team controls definitions)
- **Audit:** Full lineage showing where data comes from

**Success Metric:** Increases usage of data for decision-making (vs gut feel)

---

## User Stories

### Epic 1: Natural Language Querying (Level 1)

**As a data analyst, I want to query my database in plain English so that I can get insights without knowing SQL.**

**Acceptance Criteria:**
- [ ] Connect to database (Postgres/ClickHouse/BigQuery) via credentials
- [ ] System auto-profiles schema and generates ManagedDataPoints
- [ ] Ask question in natural language: "Show me top 10 customers by revenue"
- [ ] Receive accurate SQL and query results within 5 seconds
- [ ] See generated SQL for learning/verification
- [ ] Export results as CSV or share link

**Technical Requirements:**
- Multi-agent pipeline (Classifier → Context → SQL → Validator → Executor)
- Schema profiler runs on first connection, caches metadata
- LLM generates SQL with >95% success rate
- Query timeout after 30 seconds with helpful error message

---

### Epic 2: Business Context (Level 2)

**As a data engineer, I want to define metric meanings so that everyone in the company uses the same calculations.**

**Acceptance Criteria:**
- [ ] Create DataPoint YAML file with metric definition
- [ ] System validates DataPoint schema on load
- [ ] DataChat uses context when generating SQL
- [ ] Users can search DataPoints: "What metrics are related to revenue?"
- [ ] Git tracks changes to DataPoint definitions

**Technical Requirements:**
- DataPoint schema supports context-only (Type 1) definitions
- Validator checks YAML syntax and required fields
- Context merging combines ManagedDataPoint + UserDataPoint
- Git integration for version control

---

### Epic 3: Executable Metrics (Level 3)

**As a data engineer, I want to provide SQL templates for common metrics so queries are fast and consistent.**

**Acceptance Criteria:**
- [ ] Upgrade DataPoint to include SQL template
- [ ] System detects when query matches known metric
- [ ] Uses template instead of LLM generation (2-5x faster)
- [ ] Template supports parameterization (time ranges, filters)
- [ ] Backend-specific variants (ClickHouse vs BigQuery syntax)

**Technical Requirements:**
- DataPoint schema supports execution block (Type 2)
- Template compiler substitutes parameters safely
- Metric identification via keyword matching + LLM disambiguation
- Fallback to LLM generation if no match

---

### Epic 4: Performance Optimization (Level 4)

**As a data platform lead, I want expensive queries to be pre-computed so users get fast results.**

**Acceptance Criteria:**
- [ ] System monitors query patterns and execution times
- [ ] Suggests materialization for slow, frequent queries
- [ ] One-click enable (CLI or UI)
- [ ] Creates materialized view/table automatically
- [ ] Routes queries to pre-aggregated data transparently
- [ ] Manages incremental refresh schedules

**Technical Requirements:**
- Query pattern analyzer tracks frequency + execution time
- Materialization manager creates backend-specific views (ClickHouse SummingMergeTree, BigQuery partitioned tables)
- Incremental refresh (lookback window, partition pruning)
- Query router checks materialization freshness before routing

---

### Epic 5: AI Intelligence (Level 5)

**As a data platform lead, I want AI to detect anomalies and explain root causes so I can fix issues faster.**

**Acceptance Criteria:**
- [ ] System monitors metrics automatically
- [ ] Detects anomalies using ensemble algorithms (SPC + Prophet + ML)
- [ ] Sends alerts when thresholds breached
- [ ] Provides root cause analysis via knowledge graph traversal
- [ ] Can auto-trigger remediation (with approval)
- [ ] Shows impact chain (which downstream metrics affected)

**Technical Requirements:**
- Knowledge graph (Neo4j) stores DataPoint relationships
- Anomaly detector runs on schedule (every 10 minutes)
- Root cause analyzer traverses dependency graph
- Auto-remediation framework triggers Airflow DAGs or sends alerts
- Confidence scoring for root cause hypotheses

---

### Epic 6: Filesystem Integration

**As a data engineer, I want DataChat to read my dbt models so it can explain how metrics are calculated.**

**Acceptance Criteria:**
- [ ] Index workspace directory (dbt/models, sql/, docs/)
- [ ] Extract metadata: tables, models, columns, docstrings
- [ ] Link WorkspaceDataPoints to DataPoints
- [ ] Answer: "Show me the code that calculates revenue"
- [ ] Answer: "Why is customer_ltv NULL for 30% of customers?"
- [ ] Incremental updates when files change

**Technical Requirements:**
- Filesystem watcher with debouncing (500ms)
- Language-specific parsers (SQL, Python, YAML)
- Symbol extraction (functions, models, variables)
- Checksum-based change detection
- WorkspaceDataPoint schema with metadata

---

### Epic 7: Tool System

**As a data platform lead, I want to control which operations DataChat can perform so my data stays secure.**

**Acceptance Criteria:**
- [ ] Policy-as-config (YAML) defines allowed tools
- [ ] Read-only database queries by default
- [ ] Write operations require approval (CLI prompt or Slack)
- [ ] Users can install custom tools (Python plugins or YAML configs)
- [ ] All tool executions logged for audit

**Technical Requirements:**
- Tool registry with allowlist
- Policy enforcement (check before execution)
- Approval workflow (CLI prompt, timeout after 5 minutes)
- Plugin system supports both Python and HTTP API tools
- Audit logger records all tool calls

---

## Non-Functional Requirements

### Performance

| Metric | Target | Rationale |
|--------|--------|-----------|
| Query generation time | <2s | LLM latency + processing |
| Query execution time | <5s (P95) | Depends on database/complexity |
| Schema profiling time | <10s for 100 tables | First-time setup cost |
| Cache hit rate | >40% | Reduce LLM costs |
| Concurrent users | 50+ | Internal testing scale |
| Uptime | >99.5% | Production-grade reliability |

### Security

| Requirement | Implementation |
|-------------|----------------|
| SQL injection prevention | Parameterized queries only, injection detector |
| Authentication | API keys, JWT tokens |
| Authorization | Row-level security, database-specific policies |
| Audit logging | All queries logged (user, SQL, timestamp, results hash) |
| PII masking | Auto-detect and mask sensitive columns |
| Network isolation | VPN required for production databases |
| Data at rest | Encrypted storage for credentials |
| Data in transit | TLS 1.3 required |

### Scalability

| Component | Scaling Strategy |
|-----------|------------------|
| API server | Horizontal (stateless, load balanced) |
| Database connections | Connection pooling (max 10 per DB) |
| LLM calls | Rate limiting (30/min per user) |
| Cache | Redis cluster (multi-node) |
| Knowledge graph | Neo4j sharding (if >100K DataPoints) |
| Filesystem indexing | Asynchronous, batched updates |

---

## Success Metrics

### v1.0 (Internal Testing)

**Technical Metrics:**
- [ ] Query success rate: >95% (valid SQL generated)
- [ ] Query generation time: <2s (P95)
- [ ] Cache hit rate: >30%
- [ ] Zero critical security vulnerabilities

**User Metrics:**
- [ ] 10+ active users at Moniepoint
- [ ] 20+ queries per day per active user
- [ ] User satisfaction: >4/5 in surveys
- [ ] Time to first successful query: <5 minutes

**Business Metrics:**
- [ ] 5+ use cases validated (ad-hoc analysis, metric standardization, debugging)
- [ ] 50% reduction in "how do I calculate X?" questions in Slack

---

### v2.0 (Open Source Launch)

**Community Metrics:**
- [ ] GitHub stars: 1000+ within 3 months
- [ ] Contributors: 10+ within 6 months
- [ ] Production deployments: 50+ organizations

**Technical Metrics:**
- [ ] 50K+ DataPoints created globally
- [ ] 1M+ queries processed per month
- [ ] Query success rate: >98%

**Business Metrics:**
- [ ] 5+ enterprise inquiries per month
- [ ] 3+ conference talks/blog posts by users
- [ ] Featured in major data newsletters (Data Engineering Weekly, etc.)

---

## Release Roadmap

### v1.0 (4 weeks) - Core Experience

**Week 1-2: Foundation**
- Multi-agent pipeline
- Schema profiler
- ManagedDataPoint generation
- Basic security (SQL injection prevention)

**Week 3: Enhancement**
- UserDataPoint support (Level 2)
- Context merging
- Query caching
- Error handling improvements

**Week 4: Polish**
- Workspace indexing (basic SQL/dbt)
- Tool system foundation
- Testing at Moniepoint
- Documentation

**Deliverables:**
- Working product for internal testing
- User guide + API documentation
- Security review completed

---

### v1.1 (2 months) - Power Features

**Month 1:**
- Executable DataPoints (Level 3)
- Template compilation + routing
- Manual materialization (Level 4)
- Advanced workspace parsers (Python, Airflow)

**Month 2:**
- Tool approval workflows
- Custom tool plugins
- Query optimizer
- Performance dashboard

**Deliverables:**
- Production-ready for Moniepoint use cases
- Beta testing with 5+ external tech friends

---

### v2.0 (Q2-Q3 2026) - Intelligence & Open Source

**Q2:**
- Knowledge graph (Neo4j)
- Anomaly detection (ensemble algorithms)
- Root cause analysis
- Adaptive materialization

**Q3:**
- Open source preparation (docs, examples, CI/CD)
- Community features (GitHub issues, Discord)
- Launch content (blog posts, demos, talks)

**Deliverables:**
- Open source launch
- 1000+ GitHub stars
- Enterprise support tier ready

---

## Out of Scope (v1.0)

Explicitly not included in v1.0 to maintain focus:

- ❌ Multi-tenant SaaS (future: v2.0+)
- ❌ Managed cloud offering (future: v2.0+)
- ❌ Advanced visualizations (users export to BI tools)
- ❌ Real-time streaming data (batch queries only)
- ❌ Data quality testing framework (separate product)
- ❌ Data catalog integration (Collibra, Alation, etc.)
- ❌ Mobile apps (web/CLI only)

---

## Dependencies

### External Dependencies

| Dependency | Purpose | License | Risk |
|------------|---------|---------|------|
| OpenAI API | LLM for SQL generation | Proprietary | Rate limits, cost |
| Anthropic API | Alternative LLM | Proprietary | Rate limits, cost |
| Neo4j | Knowledge graph (v2.0) | GPL/Commercial | Deployment complexity |
| ChromaDB | Vector store | Apache 2.0 | None |
| sqlglot | SQL parsing/validation | Apache 2.0 | None |
| FastAPI | API framework | MIT | None |

**Mitigation Strategies:**
- LLM: Support multiple providers (OpenAI, Anthropic, local models)
- Neo4j: Start with NetworkX (in-memory), migrate later
- ChromaDB: Could swap for alternatives (Weaviate, Qdrant)

---

## Open Questions

### Technical

1. **NetworkX vs Neo4j for v1.0?**
   - **Decision:** NetworkX for v1.0 (simpler), Neo4j for v2.0 (scale)
   - **Rationale:** Avoid infrastructure complexity during initial testing

2. **How to handle database credentials securely?**
   - **Options:** Environment variables, HashiCorp Vault, KMS encryption
   - **Decision Needed:** Week 1

3. **Should materialization be opt-in or automatic?**
   - **Decision:** Manual opt-in for v1.1, adaptive in v2.0
   - **Rationale:** Users learn system before automation

### Product

1. **Pricing model for managed cloud offering?**
   - **Options:** Per-user, per-query, per-database
   - **Decision Needed:** Post v2.0 launch

2. **How to balance open source vs commercial features?**
   - **Current Thinking:** Core open, managed service + enterprise features paid
   - **Decision Needed:** Q2 2026

3. **Community governance model?**
   - **Options:** BDFL (Onu), committee, foundation
   - **Decision Needed:** Before open source launch

---

## Risks & Mitigation

### High Priority Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLM hallucinations generate incorrect SQL | High | Medium | Multi-agent validation, confidence scoring, user review for low confidence |
| Performance degrades at scale (100+ concurrent users) | High | Medium | Horizontal scaling, caching, rate limiting, load testing |
| Security vulnerability discovered | Critical | Low | Penetration testing, audit logging, bug bounty program |
| Competitor launches similar product | Medium | Medium | Speed to market, unique differentiators (Levels 4-5), community building |

### Medium Priority Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLM API costs exceed budget | Medium | High | Caching, local models, usage limits |
| Database connector compatibility issues | Medium | Medium | Thorough testing, community contributions for new connectors |
| User adoption slower than expected | Medium | Medium | User research, rapid iteration, dogfooding at Moniepoint |

---

## Appendix: Technical Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│              (Web UI + CLI + API)                       │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│                  Agent Router                           │
│       (Route to DB / Investigation / Tool)              │
└────────────────────┬────────────────────────────────────┘
                     ↓
         ┌───────────┼────────────┐
         ↓           ↓            ↓
┌───────────┐ ┌──────────┐ ┌───────────┐
│ Database  │ │ Investig-│ │   Tool    │
│ Pipeline  │ │  ation   │ │ Executor  │
└─────┬─────┘ └────┬─────┘ └─────┬─────┘
      ↓            ↓             ↓
┌─────────────────────────────────────────────────────────┐
│            Knowledge System (Levels 1-5)                │
│  • ManagedDataPoints (schema profiling)                │
│  • UserDataPoints (business context)                   │
│  • ExecutableDataPoints (SQL templates)                │
│  • MaterializedDataPoints (pre-aggregations)           │
│  • GraphDataPoints (relationships + intelligence)      │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│              APEX Engine (Semantic Layer)               │
│  • SQL Compiler • Materializer • Optimizer              │
│  • Anomaly Detector • Root Cause Analyzer               │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│                 Data Sources                            │
│  Databases | Filesystem | External Tools                │
└─────────────────────────────────────────────────────────┘
```

---

## Sign-Off

**Approved By:**  
Onu Abah - Product Owner & Lead Engineer

**Date:** January 30, 2026

**Next Review:** After v1.0 internal testing (4 weeks)

---

*This PRD is a living document. Update as requirements evolve based on user feedback and technical discoveries.*
