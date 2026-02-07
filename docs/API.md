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
- `POST /datapoints` - Create a DataPoint.
- `PUT /datapoints/{id}` - Update a DataPoint.
- `DELETE /datapoints/{id}` - Delete a DataPoint.

## Health

- `GET /health` - Health check.
- `GET /ready` - Readiness check.
