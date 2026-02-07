# CLI Quickstart

Use the CLI to initialize DataChat, profile a database, and ask questions from the terminal.

## Prerequisites

- Python 3.10+
- Backend running (`uvicorn backend.api.main:app --reload --port 8000`)
- CLI installed in editable mode from the repo root:

```bash
pip install -e .
datachat --help
```

## 1) Configure Databases

Set a target DB (your data) and a system DB (registry/profiling).

Interactive setup:

```bash
datachat setup
```

Non-interactive setup (recommended for repeatable tests):

```bash
datachat setup \
  --target-db postgresql://postgres:@localhost:5432/postgres \
  --system-db postgresql://postgres:@localhost:5432/postgres \
  --auto-profile \
  --max-tables 10 \
  --non-interactive
```

## 2) Generate DataPoints

If you already profiled, you can generate DataPoints from the latest profile without
specifying `--profile-id`:

```bash
datachat dp generate --max-tables 10 --depth metrics_basic --batch-size 10
```

If you know the profile ID, pass it explicitly:

```bash
datachat dp generate --profile-id <uuid> --depth metrics_full --batch-size 10
```

## 3) Review and Approve DataPoints

```bash
datachat dp pending list
datachat dp pending approve-all --latest
```

Note: `--max-tables` limits the number of tables, but each table can generate multiple
DataPoints (schema + metrics), so the pending count can exceed the table limit.

## 4) Ask Questions

```bash
datachat ask "What tables are available in the database?"
```

Control clarification prompts:

```bash
datachat ask --max-clarifications 3 "Show me the first 5 rows"
```

Quick tutorial (interactive):

```bash
datachat chat
```

Example flow:

1. Ask: `Show me the first 5 rows`
2. When prompted, answer with a table: `sales`
3. Follow up with a column or limit if asked (for example: `amount`, `first 2 rows`)

For long outputs, use a pager to keep the answer at the top and scroll as needed:

```bash
datachat ask --pager "Describe the public.events table"
```

Start an interactive session:

```bash
datachat chat --pager
```

## 5) Troubleshooting

- **Auto-profiling unavailable**
  - Ensure `SYSTEM_DATABASE_URL` and `DATABASE_CREDENTIALS_KEY` are set.
- **No DataPoints loaded**
  - Run `datachat dp sync --datapoints-dir datapoints/managed` after adding files.
