# DataChat Development Guide

## Project Overview

Open-source AI data assistant framework. Converts natural language to SQL using a multi-agent pipeline with knowledge graph context.

**Vision:** "ChatGPT for your data warehouse"  
**License:** Apache 2.0  
**Target:** November 2026 launch

---

## Tech Stack

### Backend

- **Python 3.11+** - Core language
- **FastAPI** - API framework
- **LangGraph** - Agent orchestration (NOT LangChain)
- **Pydantic v2** - Data validation
- **OpenAI/Claude/Gemini/Local** - LLM provider (GPT-4o, GPT-4o-mini)
- **Chroma** - Vector store (dev)
- **NetworkX** - Knowledge graph (in-memory)
- **SQLAlchemy** - Database ORM
- **asyncio** - Async throughout

### Frontend

- **Next.js 14** - React framework (App Router)
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **React Query** - Server state
- **Zustand** - Client state

### Infrastructure

- **PostgreSQL** - Primary database
- **Docker Compose** - Local development
- **pytest** - Testing
- **Ruff** - Linting/formatting

---

## Architecture

### Multi-Agent Pipeline

```text
User Query
    ↓
┌─────────────────┐
│ ClassifierAgent │ ← Intent, entities, complexity (lightweight model)
└────────┬────────┘
         ↓
┌─────────────────┐
│  ContextAgent   │ ← Retrieval only, NO LLM calls
└────────┬────────┘
         ↓
┌─────────────────┐
│    SQLAgent     │ ← Query generation (powerful model)
└────────┬────────┘
         ↓
┌─────────────────┐
│ ValidatorAgent  │ ← Syntax, security, schema check
└────────┬────────┘
         ↓ (retry loop if validation fails, max 3)
┌─────────────────┐
│ ExecutorAgent   │ ← Run query, format response
└────────┬────────┘
         ↓
Response + Citations
```

### LLM Provider Architecture

```text
┌──────────────────────────────────────────────┐
│           LLM Provider Registry              │
├──────────────────────────────────────────────┤
│  - OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5)    │
│  - Anthropic (Claude 3.5 Sonnet, Haiku)     │
│  - Google (Gemini Pro, Gemini Flash)        │
│  - Local (Ollama, vLLM, llama.cpp)          │
│  - Other (Cohere, Mistral AI, etc.)         │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│         BaseLLMProvider (ABC)                │
├──────────────────────────────────────────────┤
│  - generate(prompt) → response               │
│  - stream(prompt) → AsyncIterator            │
│  - count_tokens(text) → int                  │
│  - get_model_info() → dict                   │
└──────────────────────────────────────────────┘
                    ↓
         Agent uses provider via dependency injection
```

**Provider Selection:**

- **Classifier Agent**: Lightweight, fast models (GPT-4o-mini, Haiku, Gemini Flash)
- **SQL Agent**: Powerful, accurate models (GPT-4o, Sonnet, Gemini Pro)
- **Configurable per agent**: Each agent can use different providers
- **Fallback support**: Automatic failover to backup provider
- **Cost optimization**: Mix cheap and expensive models based on task complexity

### Knowledge System

```text
DataPoints (JSON files)
    ↓
┌──────────────────────────────────────┐
│         Knowledge Processor          │
├──────────────────────────────────────┤
│  Vector Store    │  Knowledge Graph  │
│  (Chroma)        │  (NetworkX)       │
│  - Embeddings    │  - Tables         │
│  - Semantic      │  - Columns        │
│    search        │  - Relationships  │
└──────────────────────────────────────┘
```

---

## Project Structure

```text
datachat/
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent ABC
│   │   ├── classifier.py        # ClassifierAgent
│   │   ├── context.py           # ContextAgent
│   │   ├── sql.py               # SQLAgent
│   │   ├── validator.py         # ValidatorAgent
│   │   └── executor.py          # ExecutorAgent
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseLLMProvider ABC
│   │   ├── factory.py           # Provider factory and registry
│   │   ├── openai.py            # OpenAI provider
│   │   ├── anthropic.py         # Anthropic (Claude) provider
│   │   ├── google.py            # Google (Gemini) provider
│   │   ├── local.py             # Local models (Ollama, vLLM)
│   │   └── models.py            # LLM request/response models
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── datapoints.py        # DataPoint loader/validator
│   │   ├── graph.py             # NetworkX knowledge graph
│   │   ├── vectors.py           # Chroma vector store
│   │   └── retriever.py         # Combined retrieval
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseConnector ABC
│   │   ├── postgres.py          # PostgreSQL connector
│   │   └── clickhouse.py        # ClickHouse connector
│   ├── models/
│   │   ├── __init__.py
│   │   ├── datapoint.py         # DataPoint Pydantic models
│   │   ├── agent.py             # Agent I/O models
│   │   └── api.py               # API request/response models
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── chat.py          # Chat endpoints
│   │   │   ├── datapoints.py    # DataPoint CRUD
│   │   │   └── health.py        # Health checks
│   │   └── websocket.py         # Streaming
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py      # LangGraph pipeline
│   ├── config.py                # Settings (pydantic-settings)
│   └── __init__.py
├── frontend/
│   ├── app/                     # Next.js App Router
│   ├── components/
│   │   ├── ui/                  # shadcn components
│   │   ├── chat/                # Chat interface
│   │   └── agents/              # Agent status display
│   └── lib/
│       ├── api.ts               # API client
│       └── stores/              # Zustand stores
├── datapoints/
│   ├── schemas/                 # JSON Schema definitions
│   └── examples/                # Example DataPoints
├── tests/
│   ├── unit/
│   │   ├── agents/
│   │   ├── knowledge/
│   │   └── connectors/
│   ├── integration/
│   └── conftest.py              # Fixtures
├── scripts/
│   ├── seed_datapoints.py
│   └── benchmark.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── CLAUDE.md                    # This file
├── README.md
└── .env.example
```

---

## Key Patterns & Conventions

### Agent Pattern

```python
# All agents inherit from BaseAgent
class BaseAgent(ABC):
    name: str
    
    @abstractmethod
    async def execute(self, input: AgentInput) -> AgentOutput:
        pass
    
    async def __call__(self, input: AgentInput) -> AgentOutput:
        # Timing, logging, error handling wrapper
        return await self.execute(input)
```

### Pydantic Models Everywhere

```python
# Every agent has typed input/output
class ClassifierInput(BaseModel):
    query: str
    conversation_history: list[Message] = []

class ClassifierOutput(BaseModel):
    intent: Literal["data_query", "exploration", "explanation", "meta"]
    entities: ExtractedEntities
    complexity: Literal["simple", "medium", "complex"]
```

### Async by Default

```python
# All I/O operations are async
async def retrieve_context(query: str) -> RetrievalResult:
    vector_results = await vector_store.search(query)
    graph_results = await knowledge_graph.traverse(entities)
    return combine_results(vector_results, graph_results)
```

### Error Handling

```python
# Custom exceptions with context
class AgentError(Exception):
    def __init__(self, agent: str, message: str, recoverable: bool = True):
        self.agent = agent
        self.recoverable = recoverable
        super().__init__(f"[{agent}] {message}")
```

---

## Commands

### Development

```bash
# Backend
cd backend
uv run uvicorn api.main:app --reload --port 8000

# Frontend
cd frontend
npm run dev

# Both (with docker)
docker-compose up
```

### Testing

```bash
# All tests
pytest tests/ -v

# Unit only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=backend --cov-report=html

# Single file
pytest tests/unit/agents/test_context.py -v
```

### Linting

```bash
# Check
ruff check backend/

# Fix
ruff check backend/ --fix

# Format
ruff format backend/
```

### Database

```bash
# Migrations (alembic)
alembic upgrade head
alembic revision --autogenerate -m "description"
```

---

## Git Workflow

### Branch Naming

```text
feature/agent-classifier
feature/knowledge-graph
fix/sql-injection-prevention
refactor/retrieval-pipeline
docs/api-documentation
```

### Commit Messages

```text
feat(agents): implement ClassifierAgent with intent detection
fix(sql): prevent SQL injection in ValidatorAgent
test(context): add unit tests for vector retrieval
docs(readme): update installation instructions
refactor(knowledge): extract common retrieval logic
```

### PR Process

1. Create feature branch from `main`
2. Implement feature with tests
3. Run full test suite locally
4. Create PR with description
5. Self-review diff
6. Merge after CI passes

---

## Current Development Focus

<!-- Update this section as you progress -->

### Active Sprint

- [ ] Implement ClassifierAgent (intent detection and entity extraction)
- [ ] Set up knowledge system (DataPoints, vector store, knowledge graph)

### Completed

- [x] Repository setup
- [x] Project scaffolding
- [x] Base agent framework with 100% test coverage (PR #1)
  - BaseAgent ABC with execute() method, timing, logging, retry logic
  - Pydantic models for AgentInput, AgentOutput, AgentMetadata, errors
  - 28 unit tests with 100% coverage
  - pytest-asyncio configuration and shared fixtures
- [x] Configuration system with pydantic-settings (PR #2)
  - Nested settings (LLM, Database, Chroma, Logging)
  - Multi-provider LLM support (OpenAI, Anthropic, Google, Local)
  - Per-agent provider selection and fallback support
  - Comprehensive validation and type safety
  - Cached factory function (singleton pattern)
  - 32 unit tests with 97% coverage
  - Auto-resource creation and environment detection
- [x] DataPoint Pydantic models (PR #3)
  - BaseDataPoint with common fields and validators
  - SchemaDataPoint, BusinessDataPoint, ProcessDataPoint
  - Discriminated unions using 'type' field
  - Custom validators for datapoint_id format and email
  - 31 unit tests
- [x] Multi-provider LLM abstraction layer (PR #4)
  - BaseLLMProvider ABC with generate(), stream(), count_tokens(), get_model_info()
  - OpenAI provider with tiktoken support
  - Anthropic (Claude) provider
  - Google (Gemini) provider
  - Local model provider (Ollama, vLLM, llama.cpp)
  - LLMProviderFactory with agent-specific overrides
  - Provider-agnostic request/response models
  - 68 unit tests with comprehensive coverage
- [x] DataPoint loader and validator (PR #5)
  - DataPointLoader class with single file and directory loading
  - Validates JSON against Pydantic DataPoint models
  - Graceful error handling with detailed error messages
  - Loading statistics tracking
  - Supports recursive directory traversal
  - 27 unit tests with 91% coverage
  - Sample DataPoint fixtures for testing

### Up Next

- [ ] Vector store integration (knowledge/vectors.py with Chroma)
- [ ] Knowledge graph implementation (knowledge/graph.py with NetworkX)
- [ ] ContextAgent (retrieval without LLM calls)
- [ ] SQLAgent (query generation with GPT-4o)
- [ ] FastAPI application setup with configuration integration

---

## Development Rules

### DO

- Write tests before or alongside code
- Use Pydantic models for all data structures
- Keep agents single-responsibility
- Use async/await for all I/O
- Log agent inputs/outputs for debugging
- Handle errors gracefully with retries
- Use type hints everywhere
- this is an open source project, everything you develop should be to the highest standards and use industry best practices where no directive is provided in this file.

### DON'T

- Don't use LangChain (too heavy, use LangGraph only)
- Don't add Neo4j yet (NetworkX is sufficient for MVP)
- Don't optimize prematurely
- Don't build auth until v1.2
- Don't add features not in current sprint
- Don't skip tests to "save time"
- Don't use `Any` type - be specific

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/datachat

# Optional
CHROMA_PERSIST_DIR=./chroma_data
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## DataPoint Schema Reference

### Schema DataPoint

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

### Business DataPoint

```json
{
  "datapoint_id": "metric_revenue_001",
  "type": "Business",
  "name": "Revenue",
  "calculation": "SUM(fact_sales.amount) WHERE status = 'completed'",
  "synonyms": ["sales", "income", "earnings", "total sales"],
  "business_rules": [
    "Exclude refunds (status != 'refunded')",
    "Convert to USD using daily rate"
  ],
  "related_tables": ["fact_sales", "dim_currency"],
  "owner": "finance@company.com"
}
```

### Process DataPoint

```json
{
  "datapoint_id": "proc_daily_etl_001",
  "type": "Process",
  "name": "Daily Sales ETL",
  "schedule": "0 2 * * *",
  "data_freshness": "T-1 (yesterday's data available by 3am UTC)",
  "target_tables": ["analytics.fact_sales"],
  "dependencies": ["raw.sales_events"],
  "owner": "data-eng@company.com"
}
```

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Response (p50) | < 1.5s | End-to-end query |
| API Response (p95) | < 3s | End-to-end query |
| Vector Search | < 50ms | Chroma query |
| Graph Traverse | < 50ms | NetworkX query |
| LLM Latency | < 1s | Per agent call |

---

## Debugging Tips

### Agent Issues

```python
# Enable verbose logging
import logging
logging.getLogger("datachat.agents").setLevel(logging.DEBUG)
```

### LLM Debugging

```python
# Log full prompts/responses
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # If using LangSmith
```

### Vector Search Issues

```python
# Check what's being retrieved
results = await vector_store.search(query, k=10)
for r in results:
    print(f"Score: {r.score:.3f} | {r.content[:100]}")
```

---

## Quick Reference

### Create New Agent

1. Create `backend/agents/{name}.py`
2. Define input/output models in `backend/models/agent.py`
3. Inherit from `BaseAgent`
4. Add to pipeline in `backend/pipeline/orchestrator.py`
5. Write tests in `tests/unit/agents/test_{name}.py`

### Add New DataPoint Type

1. Define Pydantic model in `backend/models/datapoint.py`
2. Add JSON Schema in `datapoints/schemas/`
3. Update loader in `backend/knowledge/datapoints.py`
4. Update embedder to handle new type
5. Add example in `datapoints/examples/`

### Add New Database Connector

1. Create `backend/connectors/{db}.py`
2. Inherit from `BaseConnector`
3. Implement `connect()`, `execute()`, `get_schema()`
4. Add to connector registry
5. Write integration tests

---

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Pydantic v2 Docs](https://docs.pydantic.dev/latest/)
- [Chroma Docs](https://docs.trychroma.com/)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

**Last Updated: January 2026*
