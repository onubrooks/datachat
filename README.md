# DataChat

**AI-powered natural language interface for your data warehouse.**

Ask questions in plain English and get SQL queries, data visualizations, and insights - powered by multi-agent LLM pipeline with knowledge graph context.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/next.js-15-black.svg)](https://nextjs.org/)

---

## Features

- **Natural Language Queries** - Ask questions like "What was our revenue last quarter?"
- **Multi-Agent Pipeline** - 5 specialized agents (Classifier, Context, SQL, Validator, Executor)
- **Knowledge Graph** - Semantic understanding of your database schema and business logic
- **Vector Search** - RAG-powered context retrieval for accurate SQL generation
- **Real-Time Updates** - WebSocket streaming of agent execution status
- **Multiple LLM Providers** - OpenAI, Anthropic (Claude), Google (Gemini), or local models
- **SQL Validation** - Security checks, syntax validation, performance warnings
- **Web UI** - Modern React interface with chat, SQL display, and data tables
- **CLI** - Command-line interface for power users
- **Self-Correction** - Automatic retry loop for SQL validation failures
- **Initialization Wizard** - Guided setup for first-time installs
- **Auto-Profiling** - Generate draft DataPoints from your schema
- **Multi-Database Registry** - Store multiple connections with per-query routing
- **Auto-Sync** - Keep vectors and graph in sync with DataPoint changes

---

## Quick Start

### Using Docker Compose (Recommended)

The fastest way to get started:

```bash
# 1. Clone the repository
git clone https://github.com/onubrooks/datachat.git
cd datachat

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Start all services
docker-compose up

# 4. Open your browser
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

If `datachat --version` returns "command not found", install the CLI:

```bash
pip install -e .
```

The CLI/UI setup flow saves database URLs to `~/.datachat/config.json` for reuse.

That's it! DataChat is now running with:

- **Frontend** on port 3000
- **Backend API** on port 8000
- **PostgreSQL** on port 5432
- **ChromaDB** for vector storage

> **Next:** Complete initialization before running queries.
>
> **Option A: Use the setup wizard**
>
> - Open <http://localhost:3000> and follow the setup prompt, or run:
>   `docker-compose exec backend datachat setup`
> - Enable auto-profiling to generate draft DataPoints, then review them in the
>   Database Management page.
>
> **Option B: Manual DataPoints**
>
> - Create DataPoint files (examples in [GETTING_STARTED.md](GETTING_STARTED.md))
> - Load them: `docker-compose exec backend datachat dp sync`
>
> **Without DataPoints, queries will fail.**

**AWS RDS note:** Use SSL if required by your instance:
```
postgresql://user:password@host:5432/dbname?sslmode=require
```

**Credentials:** The URL must include username/password. Setup does not prompt
for password separately.

---

## Manual Installation

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- OpenAI API key (or other LLM provider)

### Backend Setup

```bash
# 1. Create virtual environment (run from repo root)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -e .

# 3. Set up environment
cp .env.example .env
# Edit .env with your configuration
# Generate encryption key for saved DB credentials:
python -c "import secrets; print(secrets.token_hex(32))"
# Set DATABASE_CREDENTIALS_KEY in .env

# 4. Start the server
uvicorn backend.api.main:app --reload --port 8000
```

Verify the CLI is available:

```bash
datachat --version
```

Or run both servers together (requires frontend deps installed):

```bash
datachat dev
```

Backend will be available at <http://localhost:8000>

### Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Set up environment
cp .env.example .env.local
# Edit .env.local if backend is not on localhost:8000

# 4. Start development server
npm run dev
```

Frontend will be available at <http://localhost:3000>

---

## Demo Data (Optional)

Want to try DataChat without wiring your own schema? Use the demo dataset:

```bash
# Quick setup (recommended)
datachat demo

# Or manual steps
# 1. Seed demo tables
psql "$SYSTEM_DATABASE_URL" -f scripts/demo_seed.sql

# 2. Load demo DataPoints
datachat dp sync --datapoints-dir datapoints/demo

# 3. Try a query
datachat ask "How many users are active?"
```

### Database Setup

```bash
# Create PostgreSQL database
createdb datachat

# Or using psql
psql -U postgres
CREATE DATABASE datachat;
\q
```

---

## Architecture

### Multi-Agent Pipeline

```text
User Query
    ↓
┌─────────────────┐
│ ClassifierAgent │ ← Intent, entities, complexity
└────────┬────────┘
         ↓
┌─────────────────┐
│  ContextAgent   │ ← Knowledge graph + vector search
└────────┬────────┘
         ↓
┌─────────────────┐
│    SQLAgent     │ ← Generate SQL query
└────────┬────────┘
         ↓
┌─────────────────┐
│ ValidatorAgent  │ ← Security & syntax check
└────────┬────────┘
         ↓ (retry if invalid)
┌─────────────────┐
│ ExecutorAgent   │ ← Run query & format results
└────────┬────────┘
         ↓
Response + SQL + Data
```

### Tech Stack

**Backend:**

- FastAPI - API framework
- LangGraph - Agent orchestration
- Pydantic v2 - Data validation
- ChromaDB - Vector store
- NetworkX - Knowledge graph
- asyncpg - PostgreSQL driver
- OpenAI/Anthropic/Gemini - LLM providers

**Frontend:**

- Next.js 15 - React framework
- TypeScript - Type safety
- Tailwind CSS - Styling
- shadcn/ui - Components
- Zustand - State management
- WebSockets - Real-time updates

---

## Configuration

### Environment Variables

See [`.env.example`](.env.example) for all available options.

**Required:**

```env
# LLM Provider
LLM_OPENAI_API_KEY=sk-...

# Target Database
DATABASE_URL=postgresql://user:pass@localhost:5432/target_db

# System Database (registry/profiling/demo)
SYSTEM_DATABASE_URL=postgresql://user:pass@localhost:5432/datachat

DATABASE_CREDENTIALS_KEY=replace_with_fernet_key
```

`DATABASE_CREDENTIALS_KEY` is required to store database registry credentials. Generate one with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Database Registry

DataChat stores database connections in the system PostgreSQL database with encrypted URLs. Set
`SYSTEM_DATABASE_URL` and `DATABASE_CREDENTIALS_KEY` in your environment and use the API endpoints under `/api/v1/databases`
to add connections and set a default. Chat requests can target a specific connection by passing
`target_database` (connection ID) in the chat request.

**Auto-profiling prerequisites:** `SYSTEM_DATABASE_URL` + `DATABASE_CREDENTIALS_KEY`.

### System vs Target Database

DataChat separates:

- **Target database**: the database you query (`DATABASE_URL`)
- **System database**: registry/profiling/demo storage (`SYSTEM_DATABASE_URL`)

If you want a quick first run, use `datachat demo` to seed sample tables and
DataPoints into the system database.

**Optional:**

```env
# Application
ENVIRONMENT=development
LOG_LEVEL=INFO

# Alternative LLM providers
LLM_ANTHROPIC_API_KEY=sk-ant-...
LLM_GOOGLE_API_KEY=...

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_data
```

### LLM Provider Selection

DataChat supports multiple LLM providers:

**OpenAI (Default):**

```env
LLM_DEFAULT_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-...
LLM_OPENAI_MODEL=gpt-4o
```

**Anthropic (Claude):**

```env
LLM_DEFAULT_PROVIDER=anthropic
LLM_ANTHROPIC_API_KEY=sk-ant-...
LLM_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**Google (Gemini):**

```env
LLM_DEFAULT_PROVIDER=google
LLM_GOOGLE_API_KEY=...
LLM_GOOGLE_MODEL=gemini-1.5-pro
```

**Per-Agent Overrides:**

```env
LLM_CLASSIFIER_PROVIDER=openai    # Fast, cheap model
LLM_SQL_PROVIDER=anthropic        # Powerful model
LLM_FALLBACK_PROVIDER=google      # Backup if primary fails
```

---

## Usage

### Web UI

1. Open <http://localhost:3000>
2. Type your question in natural language
3. Watch agents process your query in real-time
4. View SQL, results, and sources

### CLI

```bash
# Interactive chat mode
datachat chat

# Single query
datachat ask "What is the total revenue?"

# Set database connection
datachat connect postgresql://localhost/mydb

# Check system status
datachat status

# Manage DataPoints
datachat dp list
datachat dp add schema ./datapoints/tables/fact_sales.json
datachat dp sync --datapoints-dir ./datapoints
```

### API

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "What was the revenue last quarter?",
        "conversation_id": "conv_123"
    }
)

result = response.json()
print(result["answer"])
print(result["sql"])
print(result["data"])
```

---

## Documentation

- [Getting Started](GETTING_STARTED.md)
- [API Reference](docs/API.md)
- [Multi-Database Guide](docs/MULTI_DATABASE.md)
- [Operations Guide](docs/OPERATIONS.md)

---

## DataPoints

DataPoints are JSON files that describe your data warehouse schema, business logic, and processes.

### Types of DataPoints

1. **Schema DataPoints** - Database tables and columns
2. **Business DataPoints** - Metrics, KPIs, calculations
3. **Process DataPoints** - ETL jobs, data freshness

### Example Schema DataPoint

```json
{
  "datapoint_id": "table_fact_sales_001",
  "type": "Schema",
  "name": "Fact Sales Table",
  "table_name": "analytics.fact_sales",
  "schema": "analytics",
  "business_purpose": "Central fact table for all sales transactions",
  "key_columns": [
    {
      "name": "amount",
      "type": "DECIMAL(18,2)",
      "business_meaning": "Transaction value in USD",
      "nullable": false
    }
  ],
  "relationships": [
    {
      "target_table": "dim_region",
      "join_column": "region_id",
      "cardinality": "N:1"
    }
  ],
  "common_queries": ["SUM(amount)", "GROUP BY region"],
  "gotchas": ["Always filter by date for performance"],
  "freshness": "T-1",
  "owner": "data-team@company.com"
}
```

### Adding DataPoints

```bash
# Via CLI
datachat dp add schema ./datapoints/tables/fact_sales.json

# Or sync entire directory
datachat dp sync --datapoints-dir ./datapoints
```

---

## Development

### Project Structure

```text
datachat/
├── backend/                # Python backend
│   ├── agents/            # Multi-agent pipeline
│   ├── api/               # FastAPI application
│   ├── connectors/        # Database connectors
│   ├── knowledge/         # Vector store & graph
│   ├── llm/               # LLM provider integrations
│   ├── models/            # Pydantic models
│   └── pipeline/          # LangGraph orchestrator
├── frontend/              # Next.js frontend
│   ├── src/app/          # Pages
│   ├── src/components/   # React components
│   └── src/lib/          # API client, stores
├── datapoints/            # Knowledge base
├── tests/                 # Test suite
└── docker-compose.yml     # Docker orchestration
```

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# With coverage
pytest tests/ --cov=backend --cov-report=html

# Specific test file
pytest tests/unit/agents/test_sql.py -v
```

### Linting

```bash
# Check code
ruff check backend/

# Fix issues
ruff check backend/ --fix

# Format code
ruff format backend/
```

---

## Deployment

### Docker Production

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment-Specific Configs

```bash
# Development
docker-compose up

# Production
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

See [`k8s/`](k8s/) directory for Kubernetes manifests (coming soon).

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Starts

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Run linter (`ruff check backend/ --fix`)
6. Commit changes (`git commit -m 'feat: add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance

---

## Roadmap

- [x] Multi-agent pipeline with LangGraph
- [x] Knowledge graph + vector search
- [x] SQL validation and security
- [x] Web UI with real-time updates
- [x] CLI interface
- [x] Docker deployment
- [ ] Data visualization components
- [ ] Query caching
- [ ] Multi-database support (ClickHouse, Snowflake, BigQuery)
- [ ] Authentication & authorization
- [ ] Query history and favorites
- [ ] Scheduled reports
- [ ] Slack/Teams integration

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/onubrooks/datachat/issues)
- **Discussions:** [GitHub Discussions](https://github.com/onubrooks/datachat/discussions)

---

## Acknowledgments

Built with:

- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Next.js](https://nextjs.org/) - Frontend framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [shadcn/ui](https://ui.shadcn.com/) - UI components

---

## **Made with ❤️ by the DataChat team**
