# DataChat API

Base URL: `http://localhost:8000/api/v1`

## System Initialization

- `GET /system/status` - Returns initialization status and setup steps.
- `POST /system/initialize` - Initialize with a database URL and optional auto-profiling.

Request body:
```json
{
  "database_url": "postgresql://user:pass@host:5432/db",
  "auto_profile": true
}
```

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

## DataPoint Sync

- `POST /sync` - Trigger a full sync.
- `GET /sync/status` - Get sync job status.
- `POST /datapoints` - Create a DataPoint.
- `PUT /datapoints/{id}` - Update a DataPoint.
- `DELETE /datapoints/{id}` - Delete a DataPoint.

## Health

- `GET /health` - Health check.
- `GET /ready` - Readiness check.
