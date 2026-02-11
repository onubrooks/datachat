# DataPoint-Driven Testing: Grocery Sample

This guide validates DataPoint-driven quality (not credentials-only fallback only).

## 1) Prepare a test database

Create a fresh PostgreSQL database and seed it:

```bash
createdb datachat_grocery
psql "postgresql://postgres:@localhost:5432/datachat_grocery" -f scripts/grocery_seed.sql
```

Shortcut (uses `DATABASE_URL` target + loads grocery DataPoints):

```bash
datachat demo --dataset grocery --reset
```

Set target DB in `.env` (or via CLI):

```env
DATABASE_URL=postgresql://postgres:@localhost:5432/datachat_grocery
```

Important:

- `DATABASE_URL` is the query target used by `datachat ask/chat`.
- `SYSTEM_DATABASE_URL` is only for registry/profiling metadata and does not change query execution target.

Quick verification:

```bash
datachat status
```

Expected:

- `Connection` shows `.../datachat_grocery`

## 2) Start backend (and optional frontend)

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Optional UI:

```bash
cd frontend
npm run dev
```

## 3) Exercise DataPoint add flow (single file)

```bash
datachat dp add schema datapoints/examples/grocery_store/table_grocery_stores_001.json
datachat dp list
```

Expected:

- one new Schema DataPoint appears
- no validation errors

## 4) Exercise DataPoint sync flow (bulk)

```bash
datachat dp sync --datapoints-dir datapoints/examples/grocery_store
datachat dp list
```

Expected:

- all grocery DataPoints load (schema + business + process)
- vector store count increases
- no failed files

## 5) CLI quality smoke checks

```bash
datachat ask "List all grocery stores"
datachat ask "What is total grocery revenue?"
datachat ask "Show gross margin by category"
datachat ask "Daily waste cost trend"
```

Expected:

- SQL references grocery tables
- metric prompts use metric-related tables/columns
- answer source is mostly `sql` or grounded `context` with evidence

## 6) Run retrieval eval

```bash
python scripts/eval_runner.py \
  --mode retrieval \
  --dataset eval/grocery/retrieval.json \
  --min-hit-rate 0.60 \
  --min-recall 0.50 \
  --min-mrr 0.40
```

Expected:

- exit code `0`
- summary prints Hit rate, Recall@K, MRR, Coverage

## 7) Run end-to-end QA eval

```bash
python scripts/eval_runner.py \
  --mode qa \
  --dataset eval/grocery/qa.json \
  --min-sql-match-rate 0.60 \
  --min-answer-type-rate 0.60
```

Expected:

- exit code `0`
- summary prints SQL and answer-type match rates

## 8) UI manual checks

In chat UI, run:

1. `What is total grocery revenue this week?`
2. `How do we compute gross margin?`
3. `Show stockout rate by store`
4. `What waste reasons are most common?`

Validate:

- Responses use grocery-specific vocabulary
- SQL references grocery tables
- Follow-up clarifications preserve intent
- No hallucinated table names
- If both example and auto-profiled DataPoints exist for the same table, managed/user DataPoints should win over examples.

Also validate DataPoint visibility in `Manage DataPoints`:

1. Open `Manage DataPoints`.
2. Confirm approved DataPoints include grocery entries loaded from `datapoints/examples/grocery_store`.
3. Confirm the list is populated even if you have not generated pending DataPoints from profiling yet.

Verify vector-store resilience (UI path):

1. Keep backend running.
2. In another terminal, run:

```bash
datachat reset --yes
datachat dp sync --datapoints-dir datapoints/examples/grocery_store
```

3. Return to UI and ask `List all grocery stores`.

Expected:

- No `hnsw`/`Nothing found on disk` retrieval error appears in the answer.
- Request still resolves (SQL answer or safe clarification).

## 9) Reset between runs (optional)

```bash
datachat reset --yes --keep-config --keep-vectors
```

Then rerun `dp sync` if needed.

Useful reset options:

- Keep user datapoints: `--keep-user-datapoints`
- Keep example/demo datapoints: `--keep-example-datapoints`
- Fully clear example/demo datapoints too: `--clear-example-datapoints`
