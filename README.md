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

That's it! DataChat is now running with:

- **Frontend** on port 3000
- **Backend API** on port 8000
- **PostgreSQL** on port 5432
- **ChromaDB** for vector storage

> **⚠️ Important:** Before you can query your data, you must create **DataPoints** (JSON files describing your database schema). DataPoints tell DataChat about your tables, columns, and business logic.
>
> **Quick Setup:**
>
> 1. Create DataPoint files in `datapoints/tables/` (see examples in [GETTING_STARTED.md](GETTING_STARTED.md))
> 2. Load them: `docker-compose exec backend datachat dp sync`
> 3. Verify: `docker-compose exec backend datachat dp list`
>
> **Without DataPoints, queries will fail.** See [GETTING_STARTED.md](GETTING_STARTED.md) for complete instructions.

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

# 4. Start the server
uvicorn backend.api.main:app --reload --port 8000
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

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/datachat
```

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
