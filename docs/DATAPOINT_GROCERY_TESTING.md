# DataPoint-Driven Testing: Grocery Sample

This guide validates DataPoint-driven quality (not credentials-only fallback only).

## 1) Prepare a test database

Create a fresh PostgreSQL database and seed it:

```bash
createdb datachat_grocery
psql "postgresql://postgres:@localhost:5432/datachat_grocery" -f scripts/grocery_seed.sql
```

Set target DB in `.env` (or via CLI):

```env
DATABASE_URL=postgresql://postgres:@localhost:5432/datachat_grocery
```

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

## 9) Reset between runs (optional)

```bash
datachat reset --yes --keep-config --keep-vectors
```

Then rerun `dp sync` if needed.
