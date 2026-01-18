# DataChat Development Guide

## Project Overview

Open-source AI data assistant framework. Converts natural language to SQL using a multi-agent pipeline with knowledge graph context.

**Vision:** "ChatGPT for your data warehouse"  
**License:** Apache 2.0  
**Target:** November 2026 launch

---

## Recent Updates

- Added system initialization endpoints (`/api/v1/system/status`, `/api/v1/system/initialize`) and empty-state handling for chat.
- Added frontend setup wizard and CLI safeguards when DataPoints or database setup are missing.

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
- [ ] Implement ExecutorAgent (query execution and result formatting)

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
- [x] Vector store integration (PR #6)
  - VectorStore class wrapping ChromaDB PersistentClient
  - Async interface using asyncio.to_thread() for Chroma calls
  - Methods: initialize(), add_datapoints(), search(), delete(), get_count(), clear()
  - Semantic search with metadata filtering and distance scores
  - **Properly wired OpenAI embedding function** (fix from code review)
  - Configurable embedding model (default: text-embedding-3-small)
  - Loads OpenAI API key from config.llm.openai_api_key
  - Metadata storage: datapoint_id, type, name, owner, type-specific fields
  - Document creation from DataPoint fields for search optimization
  - Persistence to configured directory
  - Batch processing (default batch_size=100)
  - Custom VectorStoreError with context
  - 31 unit tests with 82% coverage (with MockEmbeddingFunction)
  - 222 total tests passing, 83% overall backend coverage
  - **Important fix:** Collection now uses OpenAIEmbeddingFunction, not default embedder
- [x] Knowledge graph implementation (PR #7)
  - KnowledgeGraph class using NetworkX DiGraph
  - Node types: TABLE, COLUMN, METRIC, PROCESS, GLOSSARY
  - Edge types: belongs_to, joins_with, calculates, uses, synonymous, depends_on
  - Methods: add_datapoint(), get_related(), find_path(), get_stats(), save/load
  - Automatic graph construction from Schema/Business/Process DataPoints
  - BFS traversal for related nodes (max_depth, edge_type filters)
  - Shortest path finding for join discovery
  - JSON serialization with NetworkX node_link format
  - Metadata preservation (names, types, business purposes, relationships)
  - Edge weighting for relationship strength
  - 26 unit tests with 89% coverage
  - 248 total tests passing, 84% overall backend coverage
- [x] Unified retrieval system (PR #9)
  - Retriever class combining VectorStore and KnowledgeGraph
  - Three retrieval modes: LOCAL (vector only), GLOBAL (graph only), HYBRID (RRF fusion)
  - **RRF Algorithm:** Reciprocal Rank Fusion with k=60 for combining rankings
  - Formula: `score(d) = Σ 1 / (k + rank(d))` where k=60 (standard from literature)
  - Configurable top_k for each source (vector_top_k, graph_top_k)
  - Set-based deduplication to prevent duplicate datapoint_ids in results
  - Score conversion: distance → similarity for consistency
  - **Hybrid mode resilience:** Tries all vector results as graph seed candidates
  - Stops on first successful graph traversal (no silent degradation)
  - Only falls back to vector-only if ALL seeds fail
  - Handles partial graph rebuilds and out-of-sync stores gracefully
  - Pydantic models: RetrievalMode, RetrievedItem, RetrievalResult
  - Source tracking: Items tagged as "vector", "graph", or "hybrid"
  - Metadata and content preservation from both sources
  - Custom RetrieverError with context
  - 24 unit tests with 99% coverage (includes fallback seed test)
  - 274 total tests passing, 85% overall backend coverage
- [x] Git workflow improvement (PR #8)
  - Untracked CLAUDE.md from repository (was causing stash issues)
  - File remains local for development notes
  - Clean git pull without stashing required
- [x] ContextAgent - Pure retrieval agent (PR #10)
  - NO LLM calls - pure knowledge retrieval only
  - Three modes: LOCAL (vector), GLOBAL (graph), HYBRID (RRF fusion)
  - Uses Retriever for semantic + structural search
  - **Models:** ContextAgentInput, ContextAgentOutput, InvestigationMemory
  - **ExtractedEntity:** Optional entities from ClassifierAgent (entity_type, value, confidence)
  - **RetrievedDataPoint:** Single DataPoint with score, source, and full metadata
  - **InvestigationMemory:** Query, ranked DataPoints, total count, sources for citations
  - Input validation before execution
  - Configurable retrieval mode and max_datapoints (1-50, default: 10)
  - Source deduplication for citation tracking
  - Metadata preservation from vector + graph sources
  - Context building for downstream agents (next_agent="SQLAgent")
  - Full error handling with RetrievalError
  - 20 unit tests with 100% coverage
  - 11 integration tests with real DataPoints
  - 294 total tests passing, 86% overall backend coverage
- [x] Database Connectors (PR #11)
  - **BaseConnector ABC:** Abstract interface for all database connectors
  - Async interface with connect(), execute(), get_schema(), close()
  - Models: ColumnInfo, TableInfo, QueryResult
  - Exceptions: ConnectionError, QueryError, SchemaError
  - **PostgresConnector:** asyncpg-based PostgreSQL connector
    - Connection pooling (min/max size configuration)
    - Schema introspection (information_schema + pg_catalog)
    - Discovers tables, columns, types, PKs, FKs, row counts
    - Parameterized queries ($1, $2 syntax)
    - Statement timeout enforcement
  - **ClickHouseConnector:** clickhouse-connect for OLAP
    - Wrapped with asyncio.to_thread for async
    - ClickHouse parameter syntax ({param:Type})
    - System table introspection
    - Engine type detection (MergeTree, etc.)
  - Async context manager support
  - Query timeout configuration
  - Connection pooling
  - 28 unit tests passing
  - Dependencies: asyncpg>=0.29.0, clickhouse-connect>=0.7.0
  - 327 total tests in suite
- [x] SQLAgent - SQL generation with self-correction (PR #12)
  - **LLM-powered SQL generation:** Converts natural language to SQL using configurable providers
  - Takes InvestigationMemory from ContextAgent as rich context
  - **Self-correction loop:** Max 3 attempts to fix validation issues
    - Validates: SELECT statement, FROM clause, table names in DataPoints
    - Automatic correction for syntax errors, missing tables, etc.
    - Tracks all correction attempts with CorrectionAttempt model
  - **Context-rich prompts:**
    - Schema context: tables, columns, types, relationships, gotchas
    - Business rules: calculations, synonyms, filters, business logic
    - Example queries from DataPoints
  - **Structured output (GeneratedSQL):**
    - SQL query with explanation
    - Confidence score (0-1)
    - Used DataPoint IDs for citation
    - Assumptions made during generation
    - Clarifying questions if query is ambiguous
  - **Models:** SQLAgentInput, SQLAgentOutput, GeneratedSQL, ValidationIssue, CorrectionAttempt
  - **Prompt engineering:**
    - System prompt with SQL best practices
    - JSON-structured output for reliable parsing
    - Correction prompts with validation issues + suggested fixes
  - **Error handling:** SQLGenerationError with context
  - **LLM provider support:** Uses LLMProviderFactory (OpenAI, Anthropic, Google, Local)
  - Temperature=0.0 for deterministic SQL generation
  - 18 unit tests with 100% pass rate
  - 5 integration tests (require --run-integration flag and API keys)
  - 345 total unit tests passing
- [x] ValidatorAgent - SQL validation with security and performance checks (PR #13)
  - **Rule-based validation (NO LLM calls)** for speed and deterministic results
  - **Security validation:**
    - 22 SQL injection patterns detected (DROP, UNION injection, always-true conditions)
    - Dangerous function detection (LOAD_FILE, INTO OUTFILE, SHELL, etc.)
    - Multiple statement prevention (no stacked queries)
    - Only SELECT and WITH (CTE) allowed
  - **Syntax validation:**
    - Uses sqlparse for SQL parsing
    - Validates SELECT/WITH statements
    - Supports CTEs with proper name extraction
    - Database-specific syntax checks (PostgreSQL, ClickHouse, MySQL)
  - **Schema validation:**
    - Table/column existence checks against DataPoints
    - CTE name exclusion from validation
  - **Performance validation:**
    - SELECT * warnings (encourages column specification)
    - Missing WHERE clause detection (potential full table scans)
    - Missing LIMIT warnings (prevents large result sets)
    - Excessive JOIN detection (4+ joins flagged)
    - DISTINCT without LIMIT warnings
    - Performance score calculation (0-1 scale)
  - **Features:**
    - Strict mode: treats warnings as errors for production
    - Actionable suggestions for every warning
    - Multi-database support (PostgreSQL, ClickHouse, MySQL, generic)
  - **Models:** SQLValidationError, ValidationWarning, ValidatedSQL, ValidatorAgentInput, ValidatorAgentOutput
  - **Implementation:** 500+ lines of validation logic in backend/agents/validator.py
  - 22 unit tests with 100% pass rate
  - 369 total unit tests passing
  - Dependencies: sqlparse>=0.5.0
- [x] ClassifierAgent - Intent detection and entity extraction (PR #15)
  - **LLM-powered classification:** Uses GPT-4o-mini for speed/cost optimization
  - **Intent types:** data_query, exploration, explanation, meta
  - **Entity extraction:** tables, columns, metrics, time_references, filters, other
  - **Complexity assessment:** simple, medium, complex
  - **Clarification detection:** Identifies ambiguous queries with suggested questions
  - **Conversation history:** Considers previous messages for context
  - **Models:** ClassifierAgentInput, ClassifierAgentOutput, ExtractedEntity, QueryClassification
  - **Graceful JSON parsing:** Handles imperfect LLM responses with fallback
  - **Error handling:** LLMError with agent context
  - Temperature=0.3 for balanced consistency and creativity
  - 20 comprehensive unit tests with 100% pass rate
  - Tests: intent detection, entity extraction, complexity, clarification, edge cases
- [x] ExecutorAgent - Query execution and result formatting (PR #15)
  - **SQL execution:** Runs validated queries with timeout protection (asyncio.wait_for)
  - **Database support:** PostgreSQL and ClickHouse connectors
  - **Natural language summarization:** Uses GPT-4o-mini to explain results
  - **Visualization hints:** Intelligently suggests chart types:
    - Time series detection (line_chart for date columns)
    - Category + value pairs (bar_chart for ≤10 rows, line_chart for ≤20 rows)
    - Large datasets (table for >20 rows)
    - Multi-dimensional (scatter for ≤100 rows with 3+ columns)
    - Single values (none)
  - **Result truncation:** Configurable max_rows (default: 1000) with truncation flag
  - **Timeout protection:** Configurable timeout (default: 30s) prevents runaway queries
  - **Source citations:** Tracks DataPoint IDs through pipeline
  - **Graceful degradation:** Fallback summaries if LLM fails
  - **Resource cleanup:** Always closes connector (finally block)
  - **Models:** ExecutorAgentInput, ExecutorAgentOutput, QueryResult, ExecutedQuery
  - **Error handling:** QueryError, ConnectionError with proper context
  - 27 comprehensive unit tests with 100% pass rate
  - Tests: execution, summarization, truncation, timeout, visualization hints, citations, errors
  - 417 total unit tests passing (99.8% pass rate)
- [x] Schema bug fix (PR #15)
  - Fixed P2 Codex issue: double-qualifying schema names in sql.py
  - Prevented "analytics.analytics.fact_sales" when table_name already includes schema
  - Added check: only prefix schema if table_name doesn't contain "."
- [x] Test infrastructure improvements (PR #15)
  - Added mock_llm_provider fixture to conftest.py
  - Added mock_postgres_connector fixture
  - Settings cache clearing for proper test isolation
  - Updated .env with valid test API key (20+ characters)
- [x] Pipeline Orchestrator - LangGraph multi-agent workflow (PR #16)
  - **LangGraph integration:** StateGraph-based workflow orchestration
  - **Complete agent pipeline:** classifier → context → sql → validator → executor
  - **Conditional routing:** Validator can retry SQL generation (max 3 attempts)
  - **State management:** Comprehensive TypedDict schema tracking all agent outputs
  - **Streaming support:** Real-time agent status updates via async iteration
  - **Error recovery:** Graceful error handling with error_handler node
  - **Metadata tracking:** Agent timings, LLM call counts, total latency
  - **Helper function:** create_pipeline() for easy initialization
  - **Models:** PipelineState (TypedDict with 25+ fields)
  - **Features:**
    - Self-correction loop: validator failures trigger SQL regeneration
    - Max retry enforcement: prevents infinite correction loops
    - Async streaming: yields node updates during execution
    - Error context: detailed error messages with agent information
    - Performance metrics: per-agent timing and aggregate statistics
  - **Dependency injection:** Agents accept optional llm_provider for testability
  - **Agent refactoring:** ClassifierAgent, SQLAgent, ExecutorAgent support DI
  - **Safe logging:** getattr() for mock-compatible attribute access
  - **Database URL parsing:** Handles Pydantic PostgresDsn objects
  - **Implementation:** 700+ lines in backend/pipeline/orchestrator.py
  - **Unit tests:** 8 tests in tests/unit/pipeline/test_orchestrator.py (test mocking needs refinement)
  - **Integration tests:** 5 E2E tests in tests/integration/test_pipeline_e2e.py (require database setup)
  - Dependencies: langgraph>=1.0.6
  - Note: Integration tests require PostgreSQL database with "datachat" database created
- [x] FastAPI Application Layer (PR #18)
  - **FastAPI app** (`backend/api/main.py` - 220+ lines)
    - Lifespan context manager for startup/shutdown
    - Initializes vector store, knowledge graph, database connector, pipeline on startup
    - CORS middleware for frontend integration (localhost:3000, localhost:3001)
    - Global exception handlers for AgentError, ConnectionError, QueryError
    - Root endpoint with API information
    - Helper functions: `get_pipeline()`, `get_connector()`
  - **API Models** (`backend/models/api.py` - 160+ lines)
    - `ChatRequest`: message, conversation_id, conversation_history
    - `ChatResponse`: answer, sql, data, visualization_hint, sources, metrics, conversation_id
    - `DataSource`: datapoint_id, type, name, relevance_score
    - `ChatMetrics`: total_latency_ms, agent_timings, llm_calls, retry_count
    - `HealthResponse`: status, version, timestamp
    - `ReadinessResponse`: status, version, timestamp, checks
    - `ErrorResponse`: error, message, agent, recoverable
  - **Chat Endpoint** (`backend/api/routes/chat.py`)
    - `POST /api/v1/chat`: Process natural language queries
    - Accepts message with optional conversation context
    - Returns structured response with answer, SQL, data, sources, metrics
    - Automatic conversation ID generation
    - Fallback answers on pipeline errors
    - Comprehensive error handling (500, 503 status codes)
  - **Health Endpoints** (`backend/api/routes/health.py`)
    - `GET /api/v1/health`: Liveness check (always returns 200)
    - `GET /api/v1/ready`: Readiness check with dependency verification
    - Checks: database connection, vector store, pipeline initialization
    - Returns 200 if ready, 503 if not ready
    - Detailed check results in response
  - **Comprehensive Testing** (21 tests, 100% pass rate)
    - Chat endpoint tests (`tests/unit/api/test_chat.py` - 9 tests):
      - Valid query returns 200
      - Structured response validation
      - Conversation history handling
      - Error handling (pipeline errors, not initialized)
      - Request body validation
      - Fallback answers
      - Conversation ID generation and preservation
    - Health endpoint tests (`tests/unit/api/test_health.py` - 12 tests):
      - Liveness check always succeeds
      - Readiness check with all dependencies healthy
      - Individual dependency checks (database, vector store, pipeline)
      - Failure handling for each dependency
      - Proper status codes (200, 503)
  - **Features:**
    - Async request handling throughout
    - Structured logging with context
    - Type-safe request/response models
    - Graceful degradation on errors
    - CORS support for frontend
    - Health checks for Kubernetes/Docker
    - Conversation tracking with IDs
  - Dependencies: `fastapi>=0.128.0`
  - All ruff checks passing
  - All 21 unit tests passing
- [x] WebSocket Streaming Support (PR #19)
  - **WebSocket Endpoint** (`backend/api/websocket.py` - 230+ lines)
    - `WS /ws/chat`: Real-time streaming chat endpoint
    - Event types: agent_start, agent_complete, complete, error
    - Accepts message with optional conversation_id and conversation_history
    - Streams agent execution events in real-time
    - Final "complete" event contains full response (answer, SQL, data, sources, metrics)
    - Graceful disconnect handling
    - Comprehensive error handling (validation, pipeline errors, JSON errors)
    - Automatic conversation ID generation
  - **Pipeline Streaming** (`backend/pipeline/orchestrator.py`)
    - Added `run_with_streaming()` method with callback support
    - Emits agent_start event when agent begins execution
    - Emits agent_complete event when agent finishes
    - Includes agent-specific data in events:
      - ClassifierAgent: intent, entities, complexity
      - ContextAgent: datapoints_found
      - SQLAgent: sql_generated, confidence
      - ValidatorAgent: validation_passed, issues_found
      - ExecutorAgent: rows_returned, visualization_hint
    - Tracks agent execution times (duration_ms)
    - Returns final pipeline state after all events
  - **Event Format:**
    - Agent start: `{"event": "agent_start", "agent": "ClassifierAgent", "timestamp": "..."}`
    - Agent complete: `{"event": "agent_complete", "agent": "ClassifierAgent", "data": {...}, "duration_ms": 234.5, "timestamp": "..."}`
    - Complete: `{"event": "complete", "answer": "...", "sql": "...", "data": {...}, "sources": [...], "metrics": {...}, "conversation_id": "..."}`
    - Error: `{"event": "error", "error": "error_type", "message": "..."}`
  - **Integration Tests** (8 tests, 100% pass rate)
    - WebSocket connects successfully
    - Receives agent_start and agent_complete events
    - Final message contains complete response
    - Handles client disconnect gracefully
    - Validates request message
    - Handles pipeline not initialized
    - Supports conversation_id preservation
    - Generates conversation_id if not provided
  - **Features:**
    - Real-time streaming for responsive UX
    - Agent-level progress visibility
    - Per-agent timing information
    - Graceful error recovery
    - Connection state management
    - Type-safe event structures
  - All ruff checks passing
  - All 8 integration tests passing
- [x] Command-Line Interface (CLI) with Click (PR #TBD)
  - **Comprehensive CLI:** 7 main commands using Click framework
  - **Interactive REPL:** `datachat chat` - conversational mode with history
  - **Single query mode:** `datachat ask "query"` - one-shot questions
  - **Connection management:** `datachat connect <url>` - save database connection
  - **Status checking:** `datachat status` - view system health and configuration
  - **DataPoint management:** `datachat dp` command group:
    - `dp list` - List all DataPoints in knowledge base
    - `dp add <type> <file>` - Add DataPoint from JSON (schema, business, process)
    - `dp sync` - Rebuild vector store and knowledge graph from directory
  - **State persistence:** CLIState class manages config in ~/.datachat/config.json
  - **Beautiful output:** Rich library for colored panels, tables, progress bars, markdown
  - **Features:**
    - Async pipeline integration (asyncio.run wrapper)
    - Connection string validation (urllib.parse.urlparse)
    - Formatted answer display with SQL syntax highlighting
    - Data tables for query results
    - Performance metrics display (latency, LLM calls, retries)
    - Progress bars for sync operations
    - Conversation history tracking
    - Error handling with user-friendly messages
  - **Entry point:** `datachat` command via setuptools console_scripts
  - **Implementation:** backend/cli.py (600+ lines)
  - **Dependencies:** click>=8.1.0, rich>=13.0.0
  - **Testing:** 30 unit tests with 100% pass rate
  - **Test classes:**
    - TestCLIBasics - Command existence and help text
    - TestConnectCommand - Connection string persistence
    - TestAskCommand - Single query execution
    - TestDataPointCommands - DataPoint CRUD operations
    - TestStatusCommand - Health checks
    - TestCLIState - Configuration management
    - TestCLIErrorHandling - Error scenarios
    - TestCLIIntegration - End-to-end workflows
  - All ruff checks passing
  - All formatting consistent with ruff format
- [x] Frontend Web UI with Next.js (PR #TBD)
  - **Next.js 15 Application:** Full-stack React framework with App Router
  - **TypeScript:** Complete type safety across frontend codebase
  - **Tailwind CSS:** Utility-first styling with custom design system
  - **shadcn/ui Components:** Accessible, customizable UI components (Button, Input, Card)
  - **Zustand State Management:** Lightweight store for chat state, messages, and agent status
  - **Real-time WebSocket:** Live agent status updates during query processing
  - **Component Architecture:**
    - `ChatInterface` - Main chat UI with message list, input, and auto-scroll
    - `Message` - Individual message with SQL, data tables, sources, metrics
    - `AgentStatus` - Pipeline progress indicator with agent execution history
  - **API Integration:**
    - `apiClient` - REST API client for /api/v1/chat endpoint
    - `wsClient` - WebSocket client with auto-reconnect for real-time updates
  - **Features:**
    - Chat message history with conversation tracking
    - SQL syntax highlighting in code blocks
    - Data table display (first 10 rows with total count)
    - Source citations with relevance scores
    - Performance metrics (latency, LLM calls, retries)
    - Agent pipeline visualization (Classifier → Context → SQL → Validator → Executor)
    - Connection status indicator
    - Clear conversation button
    - Responsive design (mobile-friendly)
    - Dark mode support with CSS variables
  - **State Management:**
    - Messages array with user/assistant roles
    - Conversation ID persistence
    - Agent status (current agent, status, message, error)
    - Agent execution history
    - Loading states
    - WebSocket connection status
  - **Project Structure:**
    - `src/app/` - Next.js App Router pages
    - `src/components/ui/` - shadcn/ui components
    - `src/components/chat/` - Chat-specific components
    - `src/components/agents/` - Agent status components
    - `src/lib/api.ts` - API client (300+ lines)
    - `src/lib/stores/chat.ts` - Zustand store (180+ lines)
  - **Dependencies:**
    - next@^15.5.9 (latest, security patches applied)
    - react@^19.0.0
    - zustand@^5.0.2
    - lucide-react@^0.469.0 (icons)
    - @tanstack/react-query@^5.62.11
    - tailwindcss@^3.4.1
    - tailwindcss-animate@^1.0.7
  - **Build Status:**
    - ✅ TypeScript compilation passing
    - ✅ Next.js build successful
    - ✅ ESLint checks passing
    - ✅ Production-ready build (118 kB First Load JS)
  - **Documentation:** Comprehensive README.md with setup, usage, and deployment instructions
- [x] Docker Deployment (PR #TBD)
  - **Backend Dockerfile:** Multi-stage build with Python 3.11-slim
    - Stage 1 (builder): Install dependencies with uv for speed
    - Stage 2 (runtime): Copy venv and application code
    - Non-root user (datachat:1000) for security
    - Health check via curl to /api/v1/health
    - Uvicorn server on port 8000
    - Persistent volumes for chroma_data and logs
  - **Frontend Dockerfile:** Multi-stage build with Node.js 20-alpine
    - Stage 1 (deps): Install production dependencies
    - Stage 2 (builder): Build Next.js with standalone output
    - Stage 3 (runner): Minimal runtime image
    - Non-root user (nextjs:1001) for security
    - Health check via node http request
    - Production server on port 3000
  - **docker-compose.yml:** Complete orchestration setup
    - PostgreSQL 16-alpine with health checks
    - Backend service with environment variables
    - Frontend service with Next.js standalone
    - Named volumes for data persistence
    - Custom bridge network (datachat-network)
    - Service dependencies with health conditions
    - Development volume mounts for hot reload
  - **.dockerignore files:** Optimized build contexts
    - Backend: Exclude venv, tests, **pycache**, logs
    - Frontend: Exclude node_modules, .next, build artifacts
  - **Features:**
    - Multi-stage builds for minimal image size
    - Health checks for all services (30s interval, 40s start period)
    - Auto-restart policies (unless-stopped)
    - Environment variable configuration
    - Volume mounts for development
    - CORS configuration for frontend
    - Non-root users for security
  - **Volumes:**
    - postgres_data - Database persistence
    - chroma_data - Vector store persistence
    - backend_logs - Application logs
  - **Documentation:**
    - Comprehensive README.md (500+ lines)
    - Quick start with docker-compose
    - Manual installation steps
    - Configuration options
    - Usage examples (Web UI, CLI, API)
    - DataPoint management guide
    - Development guidelines
    - Deployment instructions
    - Contributing guidelines
  - All ruff checks passing (1 error fixed, 2 files formatted)
- [x] Comprehensive Documentation & Architecture V2 Proposal
  - **GETTING_STARTED.md:** Complete setup guide addressing DataPoints requirement
    - Explains two-database architecture (system DB vs target DB)
    - 7-step setup process with examples
    - Complete DataPoint examples for Schema, Business, Process types
    - "What Happens Without DataPoints" section
    - Minimum viable DataPoint structure
    - Common scenarios (testing, production, multi-database workaround)
  - **TESTING.md:** Comprehensive testing procedures (800+ lines)
    - Prerequisites and environment setup
    - Option 1: Docker Compose testing (recommended)
    - Option 2: Manual testing for each component
    - Integration testing scenarios
    - Performance testing with hey tool
    - Troubleshooting guide
    - Success criteria checklists
  - **ARCHITECTURE_V2.md:** Future architecture proposal (700+ lines)
    - System initialization workflow with guided setup
    - Multi-database support architecture
    - Auto-profiling pipeline (SchemaProfiler component)
    - LLM-powered DataPoint generation (DataPointGenerator)
    - Intelligent query routing (QueryRouter)
    - Empty state handling with helpful errors
    - New API endpoints for database management
    - 4-phase implementation plan
    - Migration path for existing users
  - **README.md updates:**
    - Added prominent DataPoints warning in quick start
    - Links to GETTING_STARTED.md for setup instructions
    - Addresses critical UX gap where users couldn't query without DataPoints

### Up Next

- [ ] Data visualization component integration (charts for query results)
- [ ] Frontend integration tests with backend API
- [ ] Kubernetes deployment manifests

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

## Recent Updates (Classifier/Executor Agents)

- Added `backend/agents/classifier.py` for intent classification and entity extraction using a lightweight LLM.
- Added `backend/agents/executor.py` to execute validated SQL, summarize results, and suggest visualizations.
- Added classifier/executor input/output models in `backend/models/agent.py` and re-exported them in `backend/models/__init__.py`.
- Exported the new agents in `backend/agents/__init__.py`.
- Added unit tests in `tests/unit/agents/test_classifier.py` and `tests/unit/agents/test_executor.py`.
- Added shared test fixtures in `tests/conftest.py` for LLM and connector mocking.
- Fixed schema double-qualification in `backend/agents/sql.py` when building table names.

---

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Pydantic v2 Docs](https://docs.pydantic.dev/latest/)
- [Chroma Docs](https://docs.trychroma.com/)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

**Last Updated: January 2026*
