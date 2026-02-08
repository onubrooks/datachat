# Multi-Database Guide

This document covers currently implemented multi-database behavior.

## What Is Supported Now

- Store multiple DB connections in the system database.
- Mark one default connection.
- Route chat requests to a specific connection with `target_database`.
- Route tool execution to a specific connection with `target_database`.
- Encrypted URL storage using `DATABASE_CREDENTIALS_KEY`.

## Prerequisites

Set both:

```env
SYSTEM_DATABASE_URL=postgresql://user:password@host:5432/datachat
DATABASE_CREDENTIALS_KEY=<fernet-key>
```

Generate key:

```bash
python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

## Database Types (Registry Validation)

Accepted today:
- `postgresql`
- `clickhouse`

Rejected today:
- `mysql` (returns validation error: not supported yet)

Notes:
- Credentials-only catalog templates exist for MySQL/BigQuery/Redshift, but runtime connectors are not yet wired for those engines.

## Add / List / Set Default / Delete

Add:

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

List:

```bash
curl http://localhost:8000/api/v1/databases
```

Set default:

```bash
curl -X PUT http://localhost:8000/api/v1/databases/<id>/default \
  -H "Content-Type: application/json" \
  -d '{"is_default": true}'
```

Delete:

```bash
curl -X DELETE http://localhost:8000/api/v1/databases/<id>
```

## Per-Request Routing

### Chat endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Top 10 customers by revenue",
    "target_database": "<connection-id>"
  }'
```

### Tools endpoint

```bash
curl -X POST http://localhost:8000/api/v1/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "name": "list_tables",
    "arguments": {"schema": "public"},
    "target_database": "<connection-id>"
  }'
```

## Behavior Guarantees

- If `target_database` is present and valid, that connection is used.
- If `target_database` is omitted, the default connection is used when available.
- If `target_database` is provided but registry is unavailable, request fails (no silent fallback).
- If `target_database` is invalid/unknown, request fails with `400/404`.

## Operational Recommendation

Use registry mode for team environments and any setup where accidental default-db execution is risky.
