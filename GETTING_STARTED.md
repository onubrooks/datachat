# Getting Started with DataChat

Complete guide for setting up and using DataChat with your database.

---

## Understanding DataChat Architecture

DataChat consists of two databases:

1. **DataChat System Database** - Stores DataChat's own metadata (currently minimal usage)
2. **Your Target Database** - The database you want to query with natural language

**Current Limitation:** Both use the same `DATABASE_URL`. This means DataChat queries the same database it uses for its operations.

---

## Initial Setup Steps

### Step 1: Install DataChat

**Option A: Docker Compose (Recommended)**
```bash
git clone https://github.com/onubrooks/datachat.git
cd datachat
cp .env.example .env
```

**Option B: Manual Installation**
```bash
# Backend (from repo root)
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Frontend
cd frontend
npm install
```

### Step 2: Configure Your Database

Edit `.env` file:

```env
# Your database connection (this is what DataChat will query)
DATABASE_URL=postgresql://user:password@host:5432/your_database

# Required: OpenAI API key for LLM
LLM_OPENAI_API_KEY=sk-...
```

**Important:** Replace `your_database` with your actual database name that contains the data you want to query.

### Step 3: Initialize DataPoints (REQUIRED)

⚠️ **Without DataPoints, DataChat cannot understand your database schema.**

DataPoints are JSON files that describe:
- Database tables and columns
- Business logic and metrics
- Data relationships

#### Option A: Auto-Generate from Database Schema

Use the setup wizard to auto-profile your database and generate draft DataPoints.
This requires the backend running (see Step 5).

**Web UI:**
1. Open <http://localhost:3000>
2. Follow the setup prompt and enable **Auto-profile**
3. Review pending DataPoints in **Database Management**
4. Approve the DataPoints you want to activate

**CLI:**
```bash
datachat setup
```

When prompted, enable auto-profiling. Approved DataPoints are loaded into the
vector store and knowledge graph immediately.

#### Option B: Manually Create DataPoints

Create a DataPoint file for each important table:

**Example: `datapoints/tables/users.json`**
```json
{
  "datapoint_id": "table_users_001",
  "type": "Schema",
  "name": "Users Table",
  "table_name": "public.users",
  "schema": "public",
  "business_purpose": "Stores user account information and authentication data",
  "key_columns": [
    {
      "name": "id",
      "type": "INTEGER",
      "business_meaning": "Unique user identifier, auto-incremented",
      "nullable": false
    },
    {
      "name": "email",
      "type": "VARCHAR(255)",
      "business_meaning": "User's email address, used for login",
      "nullable": false
    },
    {
      "name": "created_at",
      "type": "TIMESTAMP",
      "business_meaning": "When the user account was created",
      "nullable": false
    },
    {
      "name": "is_active",
      "type": "BOOLEAN",
      "business_meaning": "Whether the user account is active",
      "nullable": false
    }
  ],
  "relationships": [
    {
      "target_table": "orders",
      "join_column": "user_id",
      "cardinality": "1:N",
      "description": "Each user can have multiple orders"
    }
  ],
  "common_queries": [
    "SELECT * FROM users WHERE email = ?",
    "SELECT COUNT(*) FROM users WHERE is_active = true"
  ],
  "gotchas": [
    "Always filter by is_active for production queries",
    "Email field is case-insensitive"
  ],
  "freshness": "Real-time",
  "owner": "engineering@company.com"
}
```

**Example: `datapoints/tables/orders.json`**
```json
{
  "datapoint_id": "table_orders_001",
  "type": "Schema",
  "name": "Orders Table",
  "table_name": "public.orders",
  "schema": "public",
  "business_purpose": "Stores customer orders and purchase information",
  "key_columns": [
    {
      "name": "id",
      "type": "INTEGER",
      "business_meaning": "Unique order identifier",
      "nullable": false
    },
    {
      "name": "user_id",
      "type": "INTEGER",
      "business_meaning": "Foreign key to users table",
      "nullable": false
    },
    {
      "name": "total_amount",
      "type": "DECIMAL(10,2)",
      "business_meaning": "Total order amount in USD",
      "nullable": false
    },
    {
      "name": "status",
      "type": "VARCHAR(50)",
      "business_meaning": "Order status: pending, completed, cancelled",
      "nullable": false
    },
    {
      "name": "created_at",
      "type": "TIMESTAMP",
      "business_meaning": "When the order was placed",
      "nullable": false
    }
  ],
  "relationships": [
    {
      "target_table": "users",
      "join_column": "user_id",
      "cardinality": "N:1",
      "description": "Each order belongs to one user"
    }
  ],
  "common_queries": [
    "SELECT SUM(total_amount) FROM orders WHERE status = 'completed'",
    "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC"
  ],
  "gotchas": [
    "Filter by status for accurate revenue calculations",
    "Use created_at for date range queries, not updated_at"
  ],
  "freshness": "T-1 (updated daily at midnight)",
  "owner": "sales@company.com"
}
```

**Example: Business Metric `datapoints/metrics/revenue.json`**
```json
{
  "datapoint_id": "metric_revenue_001",
  "type": "Business",
  "name": "Total Revenue",
  "calculation": "SUM(orders.total_amount) WHERE orders.status = 'completed'",
  "synonyms": [
    "revenue",
    "sales",
    "income",
    "total sales",
    "earnings"
  ],
  "business_rules": [
    "Only include completed orders",
    "Exclude cancelled and refunded orders",
    "Convert to USD if multi-currency"
  ],
  "related_tables": ["orders", "payments"],
  "owner": "finance@company.com"
}
```

### Step 4: Load DataPoints into DataChat

```bash
# Using CLI (manual flow)
datachat dp sync --datapoints-dir ./datapoints

# Or add individually
datachat dp add schema ./datapoints/tables/users.json
datachat dp add schema ./datapoints/tables/orders.json
datachat dp add business ./datapoints/metrics/revenue.json

# Verify they're loaded
datachat dp list
```

If you used auto-profiling, you can skip this step after approving DataPoints.

**Expected Output:**
```
                    DataPoints
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Type      ┃ Name            ┃ Owner          ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Schema    │ Users Table     │ engineering... │
│ Schema    │ Orders Table    │ sales...       │
│ Business  │ Total Revenue   │ finance...     │
└───────────┴─────────────────┴────────────────┘

3 DataPoint(s) found
```

### Step 5: Start DataChat

```bash
# With Docker
docker-compose up

# Or manually
# Terminal 1: Backend (from repo root)
uvicorn backend.api.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Step 6: Verify Setup

```bash
# Check system status
datachat status

# Expected output shows:
# ✓ Database - Connected
# ✓ Vector Store - X datapoints
# ✓ Knowledge Graph - X nodes, X edges
```

### Step 7: Test with Sample Queries

Now you can query your database with natural language:

**Via Web UI (http://localhost:3000):**
- "How many users do we have?"
- "What was the total revenue last month?"
- "Show me the most recent orders"
- "Which users have placed more than 5 orders?"

**Via CLI:**
```bash
datachat ask "How many active users are there?"
datachat ask "What is the average order value?"
datachat ask "List all tables in the database"
```

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many orders were placed today?"}'
```

---

## What Happens Without DataPoints?

If you skip Step 3 and don't create DataPoints:

❌ **What Fails:**
1. ContextAgent returns no schema information
2. SQLAgent tries to generate SQL without knowing table/column names
3. Queries will likely fail or return errors
4. System has no understanding of business logic

✅ **What Still Works:**
- Health checks
- API endpoints
- General conversation (non-database questions)

**Example Without DataPoints:**
```
User: "How many users do we have?"

Response: "I don't have information about your database schema.
Please add DataPoints describing your tables so I can help query your data."
```

**Example With DataPoints:**
```
User: "How many users do we have?"

Generated SQL: SELECT COUNT(*) FROM users WHERE is_active = true

Response: "You currently have 1,247 active users in the database."
```

---

## Quick Start Checklist

Before your first query:

- [ ] Database connected (`DATABASE_URL` in `.env`)
- [ ] OpenAI API key configured
- [ ] System initialized (Web UI or `datachat setup`)
- [ ] DataPoints approved or created for key tables
- [ ] DataPoints loaded (`datachat dp sync` for manual flow)
- [ ] System status shows healthy (`datachat status`)
- [ ] Test with simple query

---

## Minimum DataPoint Requirements

For basic functionality, create DataPoints for:

1. **Core Tables** - Main business entities (users, orders, products)
2. **Important Metrics** - Key business KPIs (revenue, active users)
3. **Relationships** - How tables join together

**Minimum viable DataPoint** (for testing):
```json
{
  "datapoint_id": "table_test_001",
  "type": "Schema",
  "name": "Test Table",
  "table_name": "public.test",
  "schema": "public",
  "business_purpose": "Testing DataChat",
  "key_columns": [
    {
      "name": "id",
      "type": "INTEGER",
      "business_meaning": "ID",
      "nullable": false
    }
  ],
  "relationships": [],
  "common_queries": [],
  "gotchas": [],
  "freshness": "Real-time",
  "owner": "test@example.com"
}
```

---

## Common Scenarios

### Scenario 1: Testing DataChat with Sample Data

**Goal:** Try DataChat without setting up your production database

**Steps:**
1. Use the provided PostgreSQL Docker container
2. Create a simple test table:
   ```sql
   CREATE TABLE users (
     id SERIAL PRIMARY KEY,
     name VARCHAR(100),
     email VARCHAR(255),
     created_at TIMESTAMP DEFAULT NOW()
   );

   INSERT INTO users (name, email) VALUES
     ('Alice', 'alice@example.com'),
     ('Bob', 'bob@example.com');
   ```
3. Create one DataPoint describing the `users` table
4. Ask: "How many users are there?"

### Scenario 2: Production Database

**Goal:** Connect DataChat to your production analytics database

**Steps:**
1. Set `DATABASE_URL` to your production database (read-only user recommended)
2. Create DataPoints for 5-10 most important tables
3. Add Business DataPoints for key metrics
4. Test with your team's common questions
5. Iterate and add more DataPoints based on usage

### Scenario 3: Multiple Databases

**Current Limitation:** DataChat currently supports querying only the database specified in `DATABASE_URL`.

**Workaround:** Run separate DataChat instances for each database, or manually switch `DATABASE_URL` and restart.

**Planned:** Multi-database support in future versions.

---

## Best Practices

### 1. Start Small
- Begin with 3-5 core tables
- Add DataPoints incrementally
- Test each addition

### 2. Descriptive DataPoints
- Use clear `business_purpose` descriptions
- Include common query patterns
- Document gotchas and edge cases

### 3. Team Collaboration
- Involve domain experts in writing DataPoints
- Use `owner` field to track responsibility
- Version control DataPoints in git

### 4. Iterative Improvement
- Monitor which queries fail
- Add DataPoints for frequently queried tables
- Update business rules based on feedback

### 5. Security
- Use read-only database user for `DATABASE_URL`
- Don't include sensitive data in DataPoint examples
- Review generated SQL before execution (coming soon: approval workflow)

---

## Troubleshooting

### "No schema information found"
**Cause:** No DataPoints loaded
**Solution:** Run `datachat dp sync --datapoints-dir ./datapoints`

### "Table 'X' does not exist"
**Cause:** DataPoint references wrong table name or schema
**Solution:** Verify table name with `SELECT * FROM information_schema.tables`

### "No relevant DataPoints found for query"
**Cause:** Vector search didn't match query to DataPoints
**Solution:** Add more descriptive content to DataPoint `business_purpose` and `synonyms`

### Queries are slow
**Cause:** Database queries taking time
**Solution:** Add indexes, optimize queries, or configure timeout settings

---

## What's Next?

After setup:

1. **Add More DataPoints** - Cover more of your schema
2. **Create Business Metrics** - Define KPIs and calculations
3. **Document Processes** - Add ProcessDataPoints for ETL jobs
4. **Share with Team** - Onboard colleagues
5. **Monitor Usage** - Track common queries and add DataPoints accordingly

---

## Future Enhancements

Planned features:

- [ ] **Auto-generate DataPoints** from database introspection
- [ ] **Improve auto-profiling** coverage and customization
- [ ] **Cross-database querying** - Join results across connections
- [ ] **DataPoint templates** - Quick-start templates for common schemas
- [ ] **Visual DataPoint editor** - Web UI for creating DataPoints
- [ ] **Query approval workflow** - Review SQL before execution
- [ ] **Cached query results** - Faster responses for common questions

---

## Need Help?

- **Documentation:** Check [README.md](README.md) and [TESTING.md](TESTING.md)
- **Issues:** [GitHub Issues](https://github.com/onubrooks/datachat/issues)
- **Examples:** See `datapoints/examples/` directory

---

**Remember:** DataPoints are the key to DataChat's intelligence. The more comprehensive your DataPoints, the better DataChat understands your data and generates accurate SQL queries.
