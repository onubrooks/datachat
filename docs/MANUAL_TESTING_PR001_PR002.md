# Manual Testing: PR-001 + PR-002

This guide validates:
- PR-001: credentials-only mode (database required, DataPoints optional)
- PR-002: `target_database` propagates into SQL generation context and execution

## 1) Prerequisites

- Backend running at `http://localhost:8000`
- A reachable target PostgreSQL database
- Optional but recommended for multi-db tests:
  - `SYSTEM_DATABASE_URL` configured
  - `DATABASE_CREDENTIALS_KEY` configured

## 2) Automated Verification

Run the focused unit/integration suite used for these changes:

```bash
pytest \
  tests/unit/agents/test_sql.py \
  tests/unit/pipeline/test_orchestrator.py \
  tests/unit/api/test_chat.py \
  tests/unit/api/test_system.py \
  tests/integration/test_websocket.py --run-integration
```

Expected: all tests pass.

## 3) Manual Test A: Credentials-Only Mode

### Step A1: Initialize with only a target DB URL

```bash
curl -X POST http://localhost:8000/api/v1/system/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "database_url": "postgresql://postgres:postgres@localhost:5432/postgres",
    "auto_profile": false
  }'
```

Expected:
- `is_initialized` is `true`
- `has_databases` is `true`
- `has_datapoints` can be `false`

### Step A2: Confirm system status

```bash
curl http://localhost:8000/api/v1/system/status
```

Expected:
- `is_initialized=true` even when `has_datapoints=false`
- `setup_required` may include DataPoints as a recommended step

### Step A3: Run chat without DataPoints

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How many rows are in pg_tables?"
  }'
```

Expected:
- HTTP 200
- Response includes an answer (or safe SQL-related failure if the query itself is invalid)
- Answer includes:
  `Live schema mode: DataPoints are not loaded yet. Answers are generated from database metadata and query results only.`

## 4) Manual Test B: target_database Affects Generation and Execution

This validates that selected connection context is used for SQL generation (schema + dialect context), not only execution.

### Step B1: Prepare two databases with different column names

In DB A:

```sql
CREATE TABLE IF NOT EXISTS sales (amount numeric);
TRUNCATE TABLE sales;
INSERT INTO sales(amount) VALUES (10), (20);
```

In DB B:

```sql
CREATE TABLE IF NOT EXISTS sales (total_amount numeric);
TRUNCATE TABLE sales;
INSERT INTO sales(total_amount) VALUES (100), (200);
```

### Step B2: Register both DBs

```bash
curl -X POST http://localhost:8000/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DB A",
    "database_url": "postgresql://postgres:postgres@localhost:5432/db_a",
    "database_type": "postgresql",
    "is_default": true
  }'
```

```bash
curl -X POST http://localhost:8000/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DB B",
    "database_url": "postgresql://postgres:postgres@localhost:5432/db_b",
    "database_type": "postgresql"
  }'
```

Capture DB B `connection_id` from:

```bash
curl http://localhost:8000/api/v1/databases
```

### Step B3: Query DB B via target_database

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is total sales?",
    "target_database": "<DB_B_CONNECTION_ID>"
  }'
```

Expected:
- SQL uses `sales.total_amount` (DB B column) rather than `sales.amount`
- Query succeeds and returns sum from DB B values

If SQL still references `sales.amount` and fails on DB B, target database context is not being applied correctly.

## 5) Optional Manual Test C: WebSocket target_database

Use any WebSocket client to connect to:
- `ws://localhost:8000/api/v1/ws/chat`

Send payload:

```json
{
  "message": "What is total sales?",
  "target_database": "<DB_B_CONNECTION_ID>",
  "conversation_history": []
}
```

Expected:
- Stream completes with `event="complete"`
- Returned SQL/answer aligns with DB B schema (`total_amount`)

