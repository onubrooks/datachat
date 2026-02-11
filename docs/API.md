# DataChat API

Base URL: `http://localhost:8000/api/v1`

## System Initialization

- `GET /system/status` - Returns initialization status and setup steps.
- `POST /system/initialize` - Initialize with a database URL and optional auto-profiling.

Request body:

```json
{
  "database_url": "postgresql://user:pass@host:5432/db",
  "system_database_url": "postgresql://user:pass@host:5432/datachat",
  "auto_profile": true
}
```

Notes:

- When provided, `database_url` and `system_database_url` are persisted to `~/.datachat/config.json`.
- `is_initialized=true` means a target database is connected and chat can run.
- DataPoints are optional enrichment; when absent, chat runs in live schema mode.

## Chat

- `POST /chat` - Submit a query.
- `WS /ws/chat` - WebSocket streaming for agent updates and answer chunks.

Request body:

```json
{
  "message": "What was revenue last quarter?",
  "conversation_id": "conv_123",
  "target_database": "optional-connection-id"
}
```

Notes:

- If `target_database` is provided, SQL generation and execution both use that
  connection's database type and URL.
- If `target_database` is omitted, the default registry connection is used when set.
- When no DataPoints are loaded, responses include a live schema mode notice.

## Database Connections

- `POST /databases` - Create a connection.
- `GET /databases` - List connections.
- `GET /databases/{id}` - Fetch a single connection.
- `PUT /databases/{id}/default` - Set default connection.
- `DELETE /databases/{id}` - Remove a connection.

## Profiling and DataPoint Generation

- `POST /databases/{id}/profile` - Start profiling a database.
- `GET /profiling/jobs/{id}` - Check profiling job status.
- `POST /datapoints/generate` - Generate DataPoints from a profile.
- `GET /datapoints/pending` - List pending DataPoints.
- `POST /datapoints/pending/{id}/approve` - Approve a DataPoint.
- `POST /datapoints/pending/{id}/reject` - Reject a DataPoint.
- `POST /datapoints/pending/bulk-approve` - Approve all pending DataPoints.

Profiling request payload (bounded/safe by default):

```json
{
  "sample_size": 100,
  "max_tables": 50,
  "max_columns_per_table": 100,
  "query_timeout_seconds": 5,
  "per_table_timeout_seconds": 20,
  "total_timeout_seconds": 180,
  "fail_fast": false,
  "tables": ["orders", "customers"]
}
```

Profiling progress now includes partial coverage metadata:

- `total_tables`
- `tables_completed`
- `tables_failed`
- `tables_skipped`

Notes:

- Profiling is resilient to per-table failures and timeouts; a job can complete with partial coverage.
- Lightweight profiling snapshots are cached locally and used to enrich credentials-only SQL prompts.
- Query templates are available for `postgresql`, `mysql`, `bigquery`, `clickhouse`, and `redshift`; runtime execution is currently PostgreSQL-only until additional connectors land.

Approve payload supports optional edits:

```json
{
  "review_note": "optional",
  "datapoint": { "datapoint_id": "table_users_001", "...": "..." }
}
```

## DataPoint Sync

- `POST /sync` - Trigger a full sync.
- `GET /sync/status` - Get sync job status.
- `GET /datapoints` - List locally available DataPoints.
- `POST /datapoints` - Create a DataPoint.
- `PUT /datapoints/{id}` - Update a DataPoint.
- `DELETE /datapoints/{id}` - Delete a DataPoint.

`GET /datapoints` returns DataPoints currently loaded in the vector store
(the same effective set used during retrieval/chat), deduplicated by
`datapoint_id` with priority:

- `user` > `managed` > `custom`/`unknown` > `example`

List item shape includes:

- `datapoint_id`
- `type`
- `name`
- `source_tier` (for example `managed`, `example`, `custom`)
- `source_path` (source file path when available)

## Tools

- `GET /tools` - List available tools and typed parameter schemas.
- `POST /tools/execute` - Execute a tool call.

Tool execute request:

```json
{
  "name": "get_table_sample",
  "arguments": {
    "table": "orders",
    "schema": "public",
    "limit": 5
  },
  "target_database": "optional-connection-id",
  "approved": false,
  "user_id": "optional-user-id",
  "correlation_id": "optional-correlation-id"
}
```

Notes:

- `target_database` is optional; when provided, tool execution uses that connection's database type/URL context.
- If `target_database` is omitted, the default connection context is used when available.
- `/tools/execute` injects runtime metadata for built-ins (`retriever`, `database_type`, `database_url`, registry/connector handles) so `context_answer`, `run_sql`, `list_tables`, `list_columns`, and `get_table_sample` work consistently from API calls.
- Tool parameter schemas are typed from Python annotations (for example `integer`, `boolean`, `number`, `array`, `object`) instead of all-string placeholders.

## Health

- `GET /health` - Health check.
- `GET /ready` - Readiness check.
