# Multi-Database Guide

DataChat can manage multiple database connections and route queries to a specific
target database. Connections are stored in the system database with encrypted
credentials.

## Prerequisites

- Set `DATABASE_CREDENTIALS_KEY` in your environment (32 url-safe base64 bytes).
- Ensure the system database is running (PostgreSQL for registry storage).

Example:

```bash
python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

```env
DATABASE_CREDENTIALS_KEY=your-generated-key
```

## Add a Connection

### Web UI

Use the Database Management page to add a connection and optionally set it as
default.

### API

```bash
curl -X POST http://localhost:8000/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Analytics Warehouse",
    "database_url": "postgresql://user:pass@host:5432/analytics",
    "database_type": "postgresql",
    "tags": ["prod"],
    "is_default": true
  }'
```

## List, Set Default, and Remove

```bash
# List all connections
curl http://localhost:8000/api/v1/databases

# Set a default connection
curl -X PUT http://localhost:8000/api/v1/databases/<id>/default \
  -H "Content-Type: application/json" \
  -d '{"is_default": true}'

# Remove a connection
curl -X DELETE http://localhost:8000/api/v1/databases/<id>
```

## Query a Specific Database

Send `target_database` in your chat request to override the default connection:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Top 10 customers by revenue",
    "target_database": "<connection-id>"
  }'
```

If `target_database` is omitted, DataChat uses the default connection.
