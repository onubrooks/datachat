# DataChat Levels: Progressive Enhancement Guide

This document explains how DataChat evolves from simple query tool (Level 1) to intelligent data assistant (Level 5).

---

## Overview: The Value Ladder

```
Level 5: AI Intelligence
         ↑ Knowledge graph, anomaly detection, root cause analysis
         
Level 4: Performance Optimization  
         ↑ Materialized views, adaptive caching
         
Level 3: Executable Metrics
         ↑ SQL templates, consistent calculations
         
Level 2: Context Enhancement
         ↑ Business definitions, user-provided semantics
         
Level 1: Schema-Aware Querying
         ↑ Auto-profiling, immediate natural language access
```

**Key Principle:** Each level adds value without requiring the previous level. Users can skip directly to Level 3 if they want, or progress naturally 1→2→3→4→5.

---

## Level 1: Schema-Aware Querying

### What You Get

**Zero-setup data access.** Connect your database, DataChat profiles it automatically, you can immediately query in plain English.

### How It Works

```
1. User connects database
   ↓
2. Schema profiler runs
   - Discovers tables, columns, types
   - Samples data for statistics
   - Infers relationships (foreign keys)
   ↓
3. ManagedDataPoints generated
   - Stored in datapoints/managed/
   - Contains schema metadata
   ↓
4. User asks question
   ↓
5. LLM generates SQL using schema context
   ↓
6. Query executes, results returned
```

### Technical Details

**ManagedDataPoint Contents:**
- Table names and schemas
- Column names, types, nullable
- Sample values (first 10 rows)
- Statistics: min, max, avg, cardinality
- Inferred relationships (FK detection)

**Deterministic Catalog Intelligence (Credentials-Only):**
- Runs before SQL-generation LLM calls for schema-shape intents
- Handles table/column discovery, row counts, and sample-row requests via system catalog queries
- Generates targeted clarifications when table selection is ambiguous
- Injects compact ranked schema context into SQL prompts when LLM generation is needed

**Confidence Gating and Clarification Flow (Credentials-Only):**
- Low-confidence semantic responses are gated behind clarifying questions instead of guessed answers
- Clarifications are table-targeted when candidate tables are available
- Clarification loops are bounded (`max_clarifications`) to avoid infinite back-and-forth
- Responses consistently carry `answer_source` and `answer_confidence` metadata for UI/CLI rendering

**Supported Catalog Query Templates:**
- PostgreSQL
- MySQL
- ClickHouse
- BigQuery
- Redshift
  Connector execution support remains dependent on available runtime connectors.

**Query Generation:**
- Multi-agent pipeline (Classifier → Context → SQL → Validator → Executor)
- Context Agent loads relevant ManagedDataPoints
- SQL Agent generates query with schema awareness
- Validator checks syntax, safety, cost
- Executor runs query with timeout

**Storage:**
- ManagedDataPoints stored as read-only YAML in `datapoints/managed/`
- Loaded into memory on startup (fast retrieval)
- Re-profiled when schema changes detected

### User Experience

```bash
# Connect database
$ datachat database add prod postgres://user:pass@host/db
Profiling schema... ✓ (47 tables, 312 columns discovered)

# Immediate querying
$ datachat query "Show me top 10 customers by revenue"
Generating SQL... ✓
Executing query... ✓

| customer_name      | total_revenue |
|--------------------|---------------|
| Acme Corp          | $1,245,678    |
| Global Industries  | $987,543      |
...

Query completed in 2.3s
```

### When to Use Level 1

✅ **Good for:**
- Quick exploratory analysis
- Ad-hoc questions
- New databases you're learning
- Prototyping before formalizing metrics

❌ **Not sufficient for:**
- Consistent metric definitions (need Level 2+)
- Frequently-run expensive queries (need Level 4)
- Complex business logic (need Level 2+)

### Limitations

- No business context (LLM doesn't know "revenue" = completed sales only)
- Regenerates SQL every time (slower, potentially inconsistent)
- No performance optimization (queries raw tables)
- No understanding of metric relationships
- For engines without runtime connectors, catalog templates are available but execution is deferred until connector support is enabled

---

## Level 2: Context Enhancement

### What You Get

**Business-aware query generation.** Add context DataPoints to teach DataChat what metrics mean in your organization.

### How It Works

```
1. User creates DataPoint YAML
   - Defines "revenue" = sum(completed sales)
   - Specifies filters, joins, business rules
   ↓
2. DataPoint loaded into system
   ↓
3. User asks question about revenue
   ↓
4. Context Agent merges:
   - ManagedDataPoint (schema)
   - UserDataPoint (business logic)
   ↓
5. SQL Agent generates query using BOTH
   - Schema metadata (which tables/columns)
   - Business rules (which filters to apply)
   ↓
6. More accurate SQL generated
```

### Technical Details

**Context Merging:**
```python
merged_context = {
    # From ManagedDataPoint (Level 1)
    "schema": {
        "table": "transactions",
        "columns": ["id", "amount", "status", "type"],
        "types": {"amount": "decimal", "status": "varchar"}
    },
    
    # From UserDataPoint (Level 2)
    "business_rules": {
        "filters": ["status = 'completed'", "type = 'sale'"],
        "definition": "Total sales excluding refunds",
        "owner": "finance-team"
    }
}
```

**DataPoint Type 1 Schema:**
```yaml
datapoint:
  id: revenue
  name: Revenue
  type: metric
  definition: "Business explanation"
  owner: team-name
  
  # Level 2 additions
  data_sources:
    - table: transactions
      columns: [amount, status, type]
      filters:
        - "status = 'completed'"
        - "type = 'sale'"
```

### User Experience

```bash
# Create context DataPoint
$ cat > datapoints/user/sales/revenue.yaml <<EOF
datapoint:
  id: revenue
  name: Revenue
  type: metric
  definition: Completed sales transactions (excludes refunds)
  owner: finance-team
  data_sources:
    - table: transactions
      columns: [amount, status, type]
      filters:
        - "status = 'completed'"
        - "type = 'sale'"
EOF

# Validate
$ datachat datapoint validate datapoints/user/sales/revenue.yaml
✓ Schema valid
✓ References valid tables

# Query with context
$ datachat query "What was revenue last month?"
Using context from 'revenue' DataPoint... ✓

SELECT SUM(amount) 
FROM transactions
WHERE status = 'completed'     ← From DataPoint
  AND type = 'sale'            ← From DataPoint
  AND transaction_time >= '2026-01-01'
  AND transaction_time < '2026-02-01'

Result: $1,234,567
```

**Before Level 2 (might generate):**
```sql
-- LLM guesses what "revenue" means
SELECT SUM(amount) FROM transactions
-- Missing business filters!
```

**After Level 2 (generates correctly):**
```sql
SELECT SUM(amount) FROM transactions
WHERE status = 'completed' AND type = 'sale'
-- Applies business logic from DataPoint
```

### When to Use Level 2

✅ **Good for:**
- Standardizing metric definitions
- Educating new team members
- Documenting tribal knowledge
- Improving query accuracy

❌ **Not sufficient for:**
- Guaranteed consistency (use Level 3)
- Performance optimization (use Level 4)
- Complex dependencies (use Level 5)

### Migration Path

**Week 1:** Use Level 1 (no setup)  
**Week 2:** Identify 5-10 commonly asked metrics  
**Week 3:** Create Level 2 DataPoints for those metrics  
**Week 4:** Monitor query accuracy improvement

---

## Level 3: Executable Metrics

### What You Get

**Consistent, fast metric calculations.** Define SQL templates once, DataChat uses them every time = same answer for everyone.

### How It Works

```
1. User upgrades DataPoint to Type 2
   - Adds execution block with SQL template
   ↓
2. User asks question about metric
   ↓
3. Context Agent identifies matching metric
   ↓
4. Router decision:
   - Match found → Use template (fast)
   - No match → Generate with LLM (flexible)
   ↓
5. If using template:
   - Template compiler substitutes parameters
   - Validator checks (safety, not syntax—template pre-validated)
   - Executor runs query
   ↓
6. Results + metadata returned
   - "Used template: revenue (v1.2.0)"
   - "Definition owner: finance-team"
```

### Technical Details

**Template Structure:**
```yaml
execution:
  sql_template: |
    SELECT 
      SUM(amount) as value,
      DATE_TRUNC('{granularity}', transaction_time) as period
    FROM transactions
    WHERE status = 'completed'
      AND type = 'sale'
      AND transaction_time >= {start_time}
      AND transaction_time < {end_time}
    GROUP BY period
    ORDER BY period
  
  parameters:
    granularity: {type: enum, values: [day, week, month]}
    start_time: {type: timestamp, required: true}
    end_time: {type: timestamp, required: true}
```

**Parameter Extraction:**
```python
# User query: "What was revenue last month?"
parameters = {
    "granularity": "day",           # Inferred from "last month"
    "start_time": "2026-01-01",     # Computed from "last month"
    "end_time": "2026-02-01"
}

# Compile template
sql = template.format(**parameters)
```

**Backend Variants:**
```yaml
backend_variants:
  clickhouse: |
    SELECT SUM(amount), toStartOfDay(transaction_time)
    FROM transactions WHERE ...
  
  bigquery: |
    SELECT SUM(amount), DATE_TRUNC(transaction_time, DAY)
    FROM transactions WHERE ...
```

### User Experience

```bash
# Upgrade DataPoint to Level 3
$ datachat datapoint upgrade revenue --to-level 3

# Provide SQL template
$ cat > revenue_template.sql <<EOF
SELECT 
  SUM(amount) as value,
  DATE({time_column}) as date
FROM transactions
WHERE status = 'completed'
  AND type = 'sale'
  AND {time_column} >= {start_time}
  AND {time_column} < {end_time}
GROUP BY date
ORDER BY date
EOF

$ datachat datapoint set-template revenue revenue_template.sql

# Query with template
$ datachat query "What was revenue last month?"
Using template: revenue (v1.2.0) ✓
Query completed in 0.4s (was 2.1s with LLM generation)

Result: $1,234,567

# Same query, different user, same day
$ datachat query "revenue last month"
Using template: revenue (v1.2.0) ✓
Result: $1,234,567  ← Exact same answer
```

**Performance Comparison:**

| Method | Query Time | Consistency |
|--------|-----------|-------------|
| Level 1 (LLM) | 2.1s | ❌ Varies |
| Level 2 (LLM + context) | 1.8s | ⚠️ Mostly consistent |
| Level 3 (Template) | 0.4s | ✅ 100% consistent |

### When to Use Level 3

✅ **Good for:**
- Metrics used by multiple people
- Board presentations (need exact reproducibility)
- Compliance reporting (audit requirements)
- Reducing "why do I get different numbers?" questions

❌ **Not sufficient for:**
- Expensive queries (use Level 4 for caching)
- Understanding metric relationships (use Level 5)

### Best Practices

**Start with most critical metrics:**
1. Revenue, profit, margin (financial)
2. Active users, churn, conversion (product)
3. SLA metrics (operations)

**Version your templates:**
```yaml
datapoint:
  id: revenue
  version: "1.2.0"  # Semantic versioning
  
  # Include changelog
  changelog:
    - version: "1.2.0"
      date: "2026-01-15"
      changes: "Added tax exclusion logic"
    - version: "1.1.0"
      date: "2025-12-01"
      changes: "Initial executable template"
```

---

## Level 4: Performance Optimization

### What You Get

**10-50x faster queries through pre-computation.** DataChat creates materialized views for expensive queries, manages refreshes automatically.

### How It Works

```
1. System monitors query patterns
   - Tracks frequency, execution time
   ↓
2. Identifies materialization candidates
   - High frequency (10+ queries/day)
   - Slow execution (>1s average)
   - Consistent pattern (not ad-hoc)
   ↓
3. Recommends materialization
   - "Revenue queried 47x/day, 3.2s avg"
   - "Enable materialization for 20x speedup?"
   ↓
4. User enables (one command)
   ↓
5. System creates materialized view
   - Backend-specific (ClickHouse SummingMergeTree, etc.)
   - Sets up refresh schedule
   ↓
6. Query router updated
   - Check: Is materialization fresh?
   - Yes → Query pre-agg table (fast)
   - No → Query raw table (slow but correct)
```

### Technical Details

**Materialization Strategy:**

**Adaptive (Recommended):**
```yaml
materialization:
  enabled: true
  strategy: adaptive  # System decides
  
  # System auto-configures based on:
  # - Query patterns (hourly, daily, monthly?)
  # - Data freshness requirements
  # - Storage vs compute tradeoff
```

**Manual (Advanced):**
```yaml
materialization:
  enabled: true
  strategy: manual
  
  granularity: day           # Pre-aggregate by day
  partition_by: "DATE(ts)"   # Partition for efficient refresh
  refresh_interval: "1 hour" # Update every hour
  incremental: true          # Only refresh recent data
  lookback_window: "7 days"  # Re-process last week
```

**Backend Implementation:**

**ClickHouse:**
```sql
CREATE MATERIALIZED VIEW datachat_mat_revenue
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY date
POPULATE AS
  SELECT 
    DATE(transaction_time) as date,
    SUM(amount) as revenue
  FROM transactions
  WHERE status = 'completed'
  GROUP BY date
```

**BigQuery:**
```sql
CREATE OR REPLACE TABLE datachat_mat_revenue
PARTITION BY date
CLUSTER BY date
AS
  SELECT 
    DATE(transaction_time) as date,
    SUM(amount) as revenue
  FROM transactions
  WHERE status = 'completed'
  GROUP BY date
```

**Refresh Logic:**
```python
# Incremental refresh (fast)
DELETE FROM datachat_mat_revenue
WHERE date >= CURRENT_DATE - INTERVAL 7 DAY

INSERT INTO datachat_mat_revenue
SELECT ...
FROM transactions
WHERE transaction_time >= CURRENT_DATE - INTERVAL 7 DAY
```

### User Experience

```bash
# System detects pattern
$ datachat query "daily revenue last 30 days"
Query completed in 3.2s

# After 10+ queries...
$ datachat materialize suggest

📊 Materialization Recommendation

Metric: daily_revenue
Usage: 47 queries/day
Avg time: 3.2s
Estimated speedup: 20x (3.2s → 160ms)
Storage cost: ~50MB

Enable? [Y/n] y

Creating materialized view... ✓
Setting up 1-hour refresh schedule... ✓

# Subsequent queries
$ datachat query "daily revenue last 30 days"
Query completed in 160ms (from materialized view) ⚡

# CLI users
$ datachat materialize enable daily_revenue
Materialized view created ✓
```

### When to Use Level 4

✅ **Good for:**
- Dashboards (queried frequently)
- Executive reports (need fast loading)
- Real-time monitoring (sub-second response)
- Expensive aggregations (multi-table joins)

❌ **Not needed for:**
- Rarely-queried metrics (<5 queries/week)
- Fast raw queries (<500ms)
- Ad-hoc exploration

### Cost-Benefit Analysis

**Benefits:**
- 10-50x faster queries
- Reduced database load
- Better user experience

**Costs:**
- Storage space (typically 50MB-500MB per metric)
- Refresh compute (runs periodically)
- Maintenance overhead (monitor freshness)

**Rule of Thumb:**
- Queries/day > 10 AND Execution time > 1s → Materialize
- Otherwise, use Level 3 templates (fast enough)

### Monitoring

```bash
$ datachat materialize status

┌────────────────────┬─────────────┬──────────────┬────────────┐
│ Metric             │ Queries/Day │ Speedup      │ Freshness  │
├────────────────────┼─────────────┼──────────────┼────────────┤
│ daily_revenue      │ 47          │ 20x          │ 10 min ago │
│ customer_churn     │ 23          │ 15x          │ 5 min ago  │
│ product_views      │ 156         │ 35x          │ 2 min ago  │
└────────────────────┴─────────────┴──────────────┴────────────┘

$ datachat materialize refresh daily_revenue
Refreshing materialized view (incremental)... ✓
Processed 47,283 new rows
Completed in 12s
```

---

## Level 5: AI Intelligence

### What You Get

**AI that monitors, explains, and fixes.** Knowledge graph tracks metric relationships, detects anomalies, provides root cause analysis, and can auto-trigger fixes.

### How It Works

```
1. User defines Level 5 DataPoint
   - SLA targets, thresholds
   - Anomaly detection config
   - Dependency relationships
   - Auto-remediation rules
   ↓
2. System builds knowledge graph
   - Nodes: DataPoints
   - Edges: Relationships (depends_on, impacts)
   ↓
3. Monitoring loop runs (every 10 minutes)
   - Query current metric values
   - Run anomaly detection (ensemble)
   - Compare to SLA targets
   ↓
4. If anomaly detected:
   - Traverse dependency graph
   - Find likely root causes
   - Generate investigation report
   - Trigger alerts/remediation
   ↓
5. User receives diagnosis
   - "Revenue down 15% because..."
   - "Root cause: Payment gateway outage"
   - "Impact: $150K lost revenue"
   - "Suggested action: Check incident #4521"
```

### Technical Details

**Knowledge Graph (Neo4j):**
```cypher
// DataPoint nodes
(:DataPoint {id: "revenue", owner: "finance", sla_target: 1000000})
(:DataPoint {id: "transaction_count", owner: "ops"})
(:DataPoint {id: "payment_gateway_health", owner: "platform"})

// Relationships
(revenue)-[:DEPENDS_ON {impact: 0.6}]->(transaction_count)
(transaction_count)-[:DEPENDS_ON {impact: 0.9}]->(payment_gateway_health)

// Systems
(revenue)-[:USES_SYSTEM]->(payment_gateway:System {name: "Stripe"})
```

**Anomaly Detection (Ensemble):**
```python
def detect_anomalies(values: List[float]) -> List[Anomaly]:
    # Algorithm 1: Statistical Process Control (fast, interpretable)
    spc_anomalies = spc_detection(values, threshold=3_sigma)
    
    # Algorithm 2: Prophet (trend-aware)
    prophet_anomalies = prophet_forecasting(values, forecast_horizon=7)
    
    # Algorithm 3: Isolation Forest (ML, unsupervised)
    ml_anomalies = isolation_forest(values, contamination=0.05)
    
    # Ensemble: Require 2+ algorithms to agree
    confirmed = []
    for anomaly in spc_anomalies:
        agreement = sum([
            anomaly in spc_anomalies,
            anomaly in prophet_anomalies,
            anomaly in ml_anomalies
        ])
        if agreement >= 2:
            confirmed.append(anomaly)
    
    return confirmed
```

**Root Cause Analysis:**
```python
def find_root_causes(metric_id: str, current_value: float) -> RootCauseReport:
    # Get metric's dependencies from graph
    dependencies = knowledge_graph.get_dependencies(metric_id)
    
    # Query current value of each dependency
    causes = []
    for dep in dependencies:
        dep_value = query_current_value(dep.id)
        dep_expected = get_expected_value(dep.id)
        deviation = abs(dep_value - dep_expected) / dep_expected
        
        # If dependency is also anomalous, it's a likely cause
        if deviation > 0.05:  # 5% threshold
            causes.append({
                "metric": dep.name,
                "expected": dep_expected,
                "actual": dep_value,
                "deviation": deviation,
                "impact": dep.impact_coefficient,
                "priority": deviation * dep.impact_coefficient  # Weighted score
            })
    
    # Sort by priority (highest impact causes first)
    causes.sort(key=lambda x: x["priority"], reverse=True)
    
    return RootCauseReport(
        metric=metric_id,
        current_value=current_value,
        likely_causes=causes[:5],  # Top 5 causes
        recommended_actions=generate_actions(causes)
    )
```

**Auto-Remediation:**
```yaml
auto_remediation:
  - condition: "value < 0.92 AND pricing_coverage < 0.95"
    action: trigger_dag
    config:
      dag_id: refresh_pricing_data
    reason: "Stale pricing often causes match failures"
  
  - condition: "value < 0.88"
    action: page_oncall
    config:
      service: finance-ops
    reason: "Critical SLA breach"
```

```python
def handle_anomaly(anomaly: Anomaly):
    datapoint = get_datapoint(anomaly.metric_id)
    
    for rule in datapoint.auto_remediation:
        # Evaluate condition
        context = {
            "value": anomaly.current_value,
            "target": datapoint.sla.target,
            "pricing_coverage": query_current_value("pricing_coverage")
        }
        
        if eval(rule.condition, context):
            execute_action(rule.action, rule.config)
            log_remediation(anomaly, rule)
```

### User Experience

```bash
# Define Level 5 DataPoint
$ cat > datapoints/user/ops/pool_match_rate.yaml <<EOF
datapoint:
  id: pool_match_rate
  name: Pool Match Rate
  type: metric
  
  execution: {...}
  materialization: {...}
  
  intelligence:
    sla:
      target: 0.95
      warning_threshold: 0.92
      critical_threshold: 0.88
      alert_channel: "#finance-alerts"
    
    anomaly_detection:
      enabled: true
      algorithm: ensemble
      sensitivity: 0.02
  
  relationships:
    depends_on:
      - id: decimal_precision_status
        impact_coefficient: 0.80
      - id: pricing_coverage
        impact_coefficient: 0.10
    
    impacts:
      - id: unrecovered_revenue
        quantified_impact: "1% drop = $5K-8K daily"
EOF

# Enable monitoring
$ datachat monitor enable pool_match_rate
✓ Monitoring started (checks every 10 minutes)
✓ Knowledge graph updated
✓ Alerts configured

# Anomaly detected (example)
📉 Anomaly Alert: pool_match_rate

Current: 88.2% (6.8% below SLA target of 95%)
Severity: CRITICAL

Root Cause Analysis:
1. decimal_precision_status: 85.1% (expected 98%, 13% deviation)
   Impact: Accounts for 80% of pool match failures
   Priority: HIGH
   
2. pricing_coverage: 89.4% (expected 95%, 6% deviation)
   Impact: Accounts for 10% of pool match failures
   Priority: MEDIUM

Recommended Actions:
1. Review decimal standardization in transformation layer
2. Triggered: refresh_pricing_data DAG (ETA: 15 minutes)

Affected Downstream Metrics:
- unrecovered_revenue: Estimated $7.2K daily loss
- manual_investigation_hours: +6 engineer-hours

View details: datachat diagnose pool_match_rate

# User investigates
$ datachat diagnose pool_match_rate --detail

📊 Diagnostic Report: pool_match_rate

Timeline (Last 24 hours):
  00:00 - 08:00: 94.2% (normal)
  08:00 - 12:00: 91.8% (warning threshold breached)
  12:00 - 16:00: 88.2% (critical threshold breached)

Dependency Health:
  ✓ cba_ingestion_lag: Normal (2.1 minutes)
  ✗ decimal_precision_status: DEGRADED (85.1%)
  ✗ pricing_coverage: DEGRADED (89.4%)

System Health:
  ✓ CBA: Operational
  ✓ NetSuite: Operational
  ✓ Pricing Service: 2 errors (last hour)

Code References:
  - Transformation: dbt/models/staging/stg_transactions.sql (lines 45-67)
  - Pricing Logic: airflow/dags/pricing_refresh.py (lines 123-145)

Suggested Investigation:
1. Check pricing service logs: kubectl logs pricing-svc-7b8c9d
2. Review recent decimal precision changes: git log -- stg_transactions.sql
3. Validate sample mismatches: datachat query "show 10 unmatched transactions"
```

### When to Use Level 5

✅ **Good for:**
- Mission-critical metrics (revenue, SLAs, compliance)
- Complex systems (many dependencies)
- Ops teams (reduce MTTR)
- Proactive monitoring (catch issues early)

❌ **Overkill for:**
- Non-critical metrics
- Simple calculations (no dependencies)
- Infrequently checked metrics

### Best Practices

**Start with high-impact metrics:**
1. Revenue/profit (financial impact)
2. System SLAs (customer impact)
3. Data quality scores (operational impact)

**Define clear SLA targets:**
```yaml
# Bad: Vague
sla:
  target: "high"  # What does "high" mean?

# Good: Specific
sla:
  target: 0.95    # 95% match rate
  warning_threshold: 0.92  # Alert at 92%
  critical_threshold: 0.88 # Escalate at 88%
```

**Map dependencies accurately:**
```yaml
# Bad: Missing impact coefficients
relationships:
  depends_on:
    - id: upstream_metric

# Good: Quantified relationships
relationships:
  depends_on:
    - id: upstream_metric
      impact_coefficient: 0.75  # Explains 75% of variance
      description: "Primary driver of this metric"
```

**Test remediation actions in staging:**
```yaml
# Development/staging
auto_remediation:
  - condition: "value < 0.92"
    action: send_alert  # Just notify
    config:
      channel: "#dev-alerts"

# Production (after testing)
auto_remediation:
  - condition: "value < 0.92"
    action: trigger_dag  # Auto-fix
    config:
      dag_id: refresh_data
```

---

## Level Comparison Matrix

| Feature | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---------|---------|---------|---------|---------|---------|
| **Natural Language Queries** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Schema Auto-Discovery** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Business Context** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Consistent Calculations** | ⚠️ | ⚠️ | ✅ | ✅ | ✅ |
| **SQL Templates** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Performance Optimization** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Anomaly Detection** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Root Cause Analysis** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Knowledge Graph** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Auto-Remediation** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Setup Time** | 0 min | 5-30 min | 15-60 min | 30-120 min | 2-4 hours |
| **Maintenance Effort** | None | Low | Medium | Medium-High | High |

---

## Migration Paths

### Path 1: Rapid Iteration (Most Common)

```
Week 1: Level 1 only
  - Install DataChat
  - Connect databases
  - Query freely, learn system

Week 2-3: Add Level 2
  - Document 5-10 key metrics
  - Create context DataPoints
  - Improve query accuracy

Month 2: Upgrade to Level 3
  - Convert top 5 metrics to templates
  - Measure consistency improvement
  - Roll out to team

Month 3: Enable Level 4
  - Identify slow queries
  - Enable materialization for top 3
  - Monitor performance gains

Quarter 2: Implement Level 5
  - Pick 1-2 critical metrics
  - Define SLAs and dependencies
  - Enable monitoring and alerting
```

### Path 2: Enterprise Rollout (Structured)

```
Month 1: Foundation
  - Level 1 deployed to staging
  - Security review completed
  - 5 pilot users testing

Month 2: Standardization
  - Finance team defines Level 2 DataPoints for all board metrics
  - Sales team defines Level 2 for pipeline metrics
  - Engineering defines Level 2 for system SLAs

Month 3: Optimization
  - Convert all board metrics to Level 3 templates
  - Enable Level 4 for dashboard queries
  - Deploy to 50+ users

Month 4+: Intelligence
  - Level 5 for revenue, SLAs, data quality
  - Incident response automation
  - Full production rollout
```

### Path 3: Power User Direct (Skip Levels)

```
Day 1: Install
Day 2: Level 3 immediately
  - Define 20 core metrics with templates
  - Migrate from dbt metrics
  
Week 2: Level 4
  - Enable materialization for all dashboard metrics
  
Month 1: Level 5
  - Implement full knowledge graph
  - Deploy monitoring to production
```

---

## FAQ

**Q: Can I skip directly to Level 5?**  
A: Technically yes, but not recommended. Each level builds understanding. Start simple, add complexity as you learn what works.

**Q: Do I need all 5 levels?**  
A: No! Most users stop at Level 3 (templates) or Level 4 (materialization). Level 5 is for mission-critical metrics only.

**Q: How do I know which level is right for a metric?**  
A:
- Used once/month → Level 1 (no setup)
- Used weekly by multiple people → Level 2 (context)
- Used daily, needs consistency → Level 3 (template)
- Used 10+ times/day, expensive → Level 4 (materialize)
- Critical business metric, complex → Level 5 (intelligence)

**Q: Can I have different metrics at different levels?**  
A: Yes! That's the point. Each metric evolves independently based on its needs.

**Q: What if my query doesn't match any DataPoint?**  
A: DataChat falls back to Level 1 (LLM generation). You always get an answer, it just might not have business context.

**Q: How does performance compare across levels?**  
A:
- Level 1: ~2s (LLM generation + query)
- Level 2: ~1.8s (LLM with context + query)
- Level 3: ~0.5s (template compilation + query)
- Level 4: ~0.1s (query pre-aggregated table)

**Q: Does Level 5 require Neo4j?**  
A: For v1.0, no (uses NetworkX in-memory). For v2.0 at scale (100K+ DataPoints), yes (Neo4j recommended).

---

## Next Steps

**New to DataChat?**
→ Start with [CLAUDE.md](./CLAUDE.md) for architecture overview

**Ready to build?**
→ See [PLAYBOOK.md](./PLAYBOOK.md) for implementation patterns

**Defining metrics?**
→ See [DATAPOINT_SCHEMA.md](./DATAPOINT_SCHEMA.md) for schema details

**Planning rollout?**
→ See [PRD.md](./PRD.md) for product roadmap

---

*Questions? File an issue or ask in #datachat Slack*
