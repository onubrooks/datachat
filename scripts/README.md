# DataChat Scripts

This directory contains utility scripts for testing and development.

## Available Scripts

### 0. `demo_seed.sql` - Demo Database Seed

**Purpose**: Create demo tables (`users`, `orders`) with sample data for onboarding.

**Usage**:

```bash
psql "$SYSTEM_DATABASE_URL" -f scripts/demo_seed.sql
```

**Related DataPoints**: `datapoints/demo/*.json`

### 0b. `grocery_seed.sql` - Grocery Business Seed

**Purpose**: Create grocery operations tables with realistic sample data for DataPoint-driven evaluation.

Tables created:
- `grocery_stores`
- `grocery_suppliers`
- `grocery_products`
- `grocery_inventory_snapshots`
- `grocery_sales_transactions`
- `grocery_purchase_orders`
- `grocery_waste_events`

**Usage**:

```bash
createdb datachat_grocery
psql "postgresql://postgres:@localhost:5432/datachat_grocery" -f scripts/grocery_seed.sql
```

**Related DataPoints**: `datapoints/examples/grocery_store/*.json`
**Related eval datasets**: `eval/grocery/*.json`

### 1. `test_sql_agent.py` - Comprehensive SQLAgent Testing

**Purpose**: Test SQLAgent with predefined sample queries to verify SQL generation, self-correction, and metadata.

**Requirements**:

- Valid OpenAI API key (set in `.env` or as environment variable)
- Installed dependencies: `openai`, `tiktoken`, `pydantic`

**Usage**:

```bash
# Make sure you're in the project root
cd /Users/onuh/Documents/Work/Open\ Source/datachat

# Set your OpenAI API key (if not in .env)
export OPENAI_API_KEY=sk-...

# Run the test script
python scripts/test_sql_agent.py
```

**What it does**:

- Creates sample DataPoints mimicking ContextAgent output
- Tests multiple query types:
  - Simple aggregation: "What was the total sales amount?"
  - Date filtering: "Show sales from last quarter"
  - Joins: "Revenue by region and product category"
  - Business rules: "Calculate monthly recurring revenue"
  - CTEs: "Top 10 customers by sales with their average order value"
- Displays for each query:
  - Generated SQL
  - Explanation
  - Confidence score
  - Used DataPoints (for citations)
  - Assumptions made
  - Self-correction attempts (if any)
  - Timing and token usage

**Expected Output**:

```text
Testing SQLAgent with sample data
=================================

Test 1: Simple aggregation query
Query: What was the total sales amount?
Generated SQL:
  SELECT SUM(amount) AS total_sales
  FROM analytics.fact_sales
Explanation: Calculates total sales by summing all amounts
Confidence: 0.95
Used DataPoints: ['table_fact_sales_001']
...
```

---

### 2. `sql_agent_demo.py` - Interactive SQLAgent Demo

**Purpose**: Interactive REPL for testing SQLAgent with custom queries in real-time.

**Requirements**:

- Valid OpenAI API key (set in `.env` or as environment variable)
- Installed dependencies: `openai`, `tiktoken`, `pydantic`

**Usage**:

```bash
# Make sure you're in the project root
cd /Users/onuh/Documents/Work/Open\ Source/datachat

# Set your OpenAI API key (if not in .env)
export OPENAI_API_KEY=sk-...

# Run the interactive demo
python scripts/sql_agent_demo.py
```

**What it does**:

- Starts an interactive prompt
- You type natural language queries
- SQLAgent generates SQL in real-time
- Shows SQL, explanation, confidence, timing, and tokens
- Press Ctrl+C to exit

**Example Session**:

```text
SQLAgent Interactive Demo
========================

Sample Context:
- Table: analytics.fact_sales
- Columns: customer_id, amount, date

Type your queries (Ctrl+C to exit):

Query: Show me sales over $100
Processing...

Generated SQL:
  SELECT customer_id, amount, date
  FROM analytics.fact_sales
  WHERE amount > 100

Explanation: Filters sales transactions exceeding $100
Confidence: 0.92
Execution Time: 1.2s
Tokens Used: 450

Query: _
```

---

### 3. `eval_runner.py` - Minimal RAG Evaluation

**Purpose**: Run basic retrieval + end-to-end checks against the local API.

**Usage**:

```bash
python scripts/eval_runner.py --mode retrieval --dataset eval/retrieval.json
python scripts/eval_runner.py --mode qa --dataset eval/qa.json
python scripts/eval_runner.py --mode retrieval --dataset eval/grocery/retrieval.json --min-hit-rate 0.6 --min-recall 0.5 --min-mrr 0.4
python scripts/eval_runner.py --mode qa --dataset eval/grocery/qa.json --min-sql-match-rate 0.6 --min-answer-type-rate 0.6
```

**Notes**:

- Retrieval mode uses `sources` from `/api/v1/chat` as proxies for retrieved DataPoints.
- Answer types support both API columnar payloads and row-oriented payloads.
- Optional thresholds return non-zero exit codes to support CI gating.

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'openai'`

**Solution**:

```bash
pip install openai tiktoken pydantic
```

### Issue: `OpenAI API key not found`

**Solution**:

```bash
# Option 1: Set environment variable
export OPENAI_API_KEY=sk-...

# Option 2: Add to .env file in project root
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Issue: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: Make sure you run scripts from the project root:

```bash
cd /Users/onuh/Documents/Work/Open\ Source/datachat
python scripts/test_sql_agent.py  # ✅ Correct
```

Not from the scripts directory:

```bash
cd scripts
python test_sql_agent.py  # ❌ Incorrect - import paths will break
```

---

## Development Notes

### Modifying Sample Data

Both scripts use minimal sample DataPoints for testing. To add more context:

**Edit the `create_sample_context()` function in either script:**

```python
def create_sample_context():
    # Add more tables
    fact_orders = SchemaDataPoint(
        datapoint_id="table_fact_orders_001",
        type="Schema",
        name="Fact Orders Table",
        table_name="analytics.fact_orders",
        # ... more fields
    )

    # Add more business definitions
    churn_rate = BusinessDataPoint(
        datapoint_id="metric_churn_001",
        type="Business",
        name="Customer Churn Rate",
        calculation="COUNT(churned) / COUNT(total)",
        # ... more fields
    )
```

### Testing Different Providers

To test with different LLM providers (Claude, Gemini, local models), modify the config:

```python
# In the script, find where SQLAgent is initialized
sql_agent = SQLAgent()  # Uses default from config

# To override, modify backend/config.py or set environment variables:
# ANTHROPIC_API_KEY=... for Claude
# GOOGLE_API_KEY=... for Gemini
```

---

## Next Steps

After manual testing, consider:

1. Adding your successful queries to unit tests
2. Creating integration tests for end-to-end pipeline
3. Building a FastAPI endpoint wrapping these agents
4. Creating a simple web UI for non-technical users

---

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Full development guide
- [Backend Agents](../backend/agents/) - Agent implementations
- [Tests](../tests/) - Unit and integration tests
