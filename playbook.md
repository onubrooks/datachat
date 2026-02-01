# DataChat Development Playbook

## Complete Step-by-Step Guide from Zero to Launch

**Duration:** 8 months (32 weeks)  
**Tool:** Claude Code  
**Methodology:** Vertical slices, test-first, small PRs

---

## Current Product Scope Snapshot

- Levels 1-5 ladder: schema profiling, business context, executable metrics, optimization, intelligence.
- Prompt system: versioned prompts, PromptLoader, regression tests.
- v2 tooling: router, tool registry with policy-as-config, read-only tools, audit logging.
- Workspace ingestion: WorkspaceDataPoints, incremental indexing, semantic search, UI/CLI parity.
- Multi-db + ops: encrypted registry, per-request routing, explicit sync in multi-node setups.

---

## Pre-Development Setup (Day 0)

### 1. Create Repository

```textbash
# Create and clone repo
gh repo create datachat --public --description "Open-source AI data assistant framework"
cd datachat

# Initialize git
git init
echo "# DataChat" > README.md
git add README.md
git commit -m "chore: initial commit"
git branch -M main
git push -u origin main
```text

### 2. Initialize Python Project

```textbash
# Using uv (recommended - faster than poetry)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv add fastapi uvicorn pydantic pydantic-settings
uv add openai langchain-openai langgraph
uv add chromadb networkx sqlalchemy asyncpg
uv add python-dotenv httpx

# Dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff
```text

### 3. Create Essential Files

```textbash
# Create directory structure
mkdir -p backend/{agents,knowledge,connectors,models,api/routes,pipeline}
mkdir -p frontend tests/{unit/{agents,knowledge,connectors},integration}
mkdir -p datapoints/{schemas,examples} scripts

# Create __init__.py files
find backend -type d -exec touch {}/__init__.py \;

# Create environment files
cat > .env.example << 'EOF'
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/datachat
CHROMA_PERSIST_DIR=./chroma_data
LOG_LEVEL=INFO
ENVIRONMENT=development
EOF

cp .env.example .env

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.env
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
chroma_data/
node_modules/
.next/
EOF
```text

### 4. Add CLAUDE.md

```textbash
# Copy CLAUDE.md content (from the file I created above)
# This is your primary reference for Claude Code
```text

### 5. Create Docker Compose

```textbash
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: datachat
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/datachat
    depends_on:
      - postgres
    volumes:
      - ./backend:/app/backend
      - ./datapoints:/app/datapoints

volumes:
  postgres_data:
EOF
```text

### 6. Initial Commit

```textbash
git add .
git commit -m "chore: project scaffolding and configuration"
git push
```text

---

## Phase 1: v1.0 MVP (Weeks 1-8)

### Week 1: Foundation & Base Classes

#### Day 1-2: Base Agent Framework

**Claude Code Command:**

```text
Read CLAUDE.md thoroughly. Then create the base agent framework:
1. backend/agents/base.py - BaseAgent ABC with execute() method, timing, logging
2. backend/models/agent.py - Pydantic models for AgentInput, AgentOutput, AgentError
3. Include proper async support, error handling, and type hints
4. Add docstrings explaining the pattern
```text

**Test Command:**

```text
Create tests/unit/agents/test_base.py with tests for:
1. BaseAgent cannot be instantiated directly
2. Subclass must implement execute()
3. Timing wrapper works correctly
4. Error handling captures agent name
```text

**Manual Testing:**

- Review generated code structure
- Verify type hints are complete
- Check docstrings are clear

**PR:** `feature/base-agent-framework`

---

#### Day 3-4: Configuration & Settings

**Claude Code Command:**

```text
Create backend/config.py using pydantic-settings:
1. Settings class with all env vars from .env.example
2. Nested settings for OpenAI, Database, Chroma
3. Validation for required fields
4. Factory function get_settings() with caching
```text

**Test Command:**

```text
Create tests/unit/test_config.py testing:
1. Settings loads from environment
2. Validation fails for missing required fields
3. Default values work correctly
4. Settings is cached (singleton pattern)
```text

**PR:** `feature/configuration`

---

#### Day 5: DataPoint Models

**Claude Code Command:**

```text
Create backend/models/datapoint.py with Pydantic models:
1. BaseDataPoint with common fields (datapoint_id, type, name)
2. SchemaDataPoint for table/column metadata
3. BusinessDataPoint for metrics/glossary
4. ProcessDataPoint for ETL/freshness info
5. Use discriminated unions for type field
6. Add validators for datapoint_id format
Reference the DataPoint schemas in CLAUDE.md
```text

**Test Command:**

```text
Create tests/unit/models/test_datapoint.py testing:
1. Each DataPoint type validates correctly
2. Invalid datapoint_id is rejected
3. Discriminated union works (type field)
4. Optional fields have correct defaults
```text

**PR:** `feature/datapoint-models`

---

### Week 2: Knowledge System

#### Day 1-2: DataPoint Loader

**Claude Code Command:**

```text
Create backend/knowledge/datapoints.py:
1. DataPointLoader class that reads JSON files from a directory
2. Validates against Pydantic models
3. Returns typed DataPoint objects
4. Handles file errors gracefully
5. Supports both single file and directory loading
6. Add logging for loaded/failed files
```text

**Test Command:**

```text
Create tests/unit/knowledge/test_datapoints.py:
1. Loads valid JSON file correctly
2. Raises error for invalid JSON
3. Raises error for schema violation
4. Loads all files from directory
5. Skips non-JSON files
Create tests/fixtures/ with sample DataPoint JSON files
```text

**PR:** `feature/datapoint-loader`

---

#### Day 3-4: Vector Store

**Claude Code Command:**

```text
Create backend/knowledge/vectors.py:
1. VectorStore class wrapping Chroma
2. Methods: add_datapoints(), search(), delete()
3. Async interface using asyncio.to_thread for Chroma calls
4. Configurable embedding model (default: OpenAI text-embedding-3-small)
5. Metadata storage for datapoint_id, type, source
6. Persistence to configured directory
```text

**Test Command:**

```text
Create tests/unit/knowledge/test_vectors.py:
1. Adds datapoints and retrieves by similarity
2. Search returns correct metadata
3. Delete removes from store
4. Persistence works across restarts
Use pytest fixtures for test Chroma instance
```text

**PR:** `feature/vector-store`

---

#### Day 5: Knowledge Graph

**Claude Code Command:**

```text
Create backend/knowledge/graph.py:
1. KnowledgeGraph class using NetworkX
2. Node types: TABLE, COLUMN, METRIC, GLOSSARY
3. Edge types: belongs_to, joins_with, calculates, synonymous
4. Methods: add_datapoint(), get_related(), find_path()
5. Build graph from DataPoints automatically
6. Serialize/deserialize to JSON for persistence
```text

**Test Command:**

```text
Create tests/unit/knowledge/test_graph.py:
1. Adds table nodes correctly
2. Creates edges from relationships
3. get_related() returns connected nodes
4. find_path() finds join paths between tables
5. Serialization roundtrip works
```text

**PR:** `feature/knowledge-graph`

---

### Week 3: Retrieval System

#### Day 1-2: Combined Retriever

**Claude Code Command:**

```text
Create backend/knowledge/retriever.py:
1. Retriever class combining VectorStore and KnowledgeGraph
2. Three modes: local (vector), global (graph), hybrid (both)
3. Rank fusion for combining results (RRF algorithm)
4. Configurable top_k for each source
5. Returns RetrievalResult with sources and scores
6. Deduplication of results
```text

**Test Command:**

```text
Create tests/unit/knowledge/test_retriever.py:
1. Local mode uses only vector search
2. Global mode uses only graph traversal
3. Hybrid mode combines both
4. RRF ranking works correctly
5. Deduplication removes duplicates
```text

**PR:** `feature/retriever`

---

#### Day 3-5: ContextAgent (First Real Agent!)

**Claude Code Command:**

```text
Create backend/agents/context.py:
1. ContextAgent inheriting from BaseAgent
2. Input: query string, extracted entities (optional)
3. Output: InvestigationMemory with relevant DataPoints
4. Uses Retriever in hybrid mode
5. NO LLM calls - pure retrieval
6. Formats context for downstream agents
7. Tracks which DataPoints were used (for citations)
```text

**Test Command:**

```text
Create tests/unit/agents/test_context.py:
1. Returns relevant DataPoints for query
2. Works with and without extracted entities
3. Respects retrieval mode configuration
4. InvestigationMemory has correct structure
5. Sources are tracked for citations
```text

**Integration Test Command:**

```text
Create tests/integration/test_context_integration.py:
1. End-to-end test with real DataPoints
2. Test with sample sales schema
3. Verify retrieval quality manually
```text

**PR:** `feature/context-agent`

---

### Week 4: SQL Generation

#### Day 1-2: Database Connectors

**Claude Code Command:**

```text
Create backend/connectors/base.py:
1. BaseConnector ABC with connect(), execute(), get_schema(), close()
2. Async interface throughout
3. Connection pooling support
4. Query timeout configuration

Create backend/connectors/postgres.py:
1. PostgresConnector implementing BaseConnector
2. Uses asyncpg for async operations
3. Schema introspection (tables, columns, types)
4. Parameterized query execution
5. Connection pool management
```text

**Test Command:**

```text
Create tests/unit/connectors/test_postgres.py:
1. Connects successfully
2. Executes simple query
3. Schema introspection returns tables
4. Parameterized queries work
5. Timeout is respected

Create tests/integration/test_postgres_integration.py:
1. Test against real PostgreSQL (docker-compose)
```text

**PR:** `feature/postgres-connector`

---

#### Day 3-5: SQLAgent

**Claude Code Command:**

```text
Create backend/agents/sql.py:
1. SQLAgent inheriting from BaseAgent
2. Input: query, InvestigationMemory from ContextAgent
3. Output: generated SQL, explanation, used_datapoints
4. Uses GPT-4o for generation
5. Prompt includes:
   - Schema context from DataPoints
   - Business rules
   - Example queries
   - User question
6. Structured output using Pydantic
7. Handles ambiguity by asking clarifying questions
```text

**Test Command:**

```text
Create tests/unit/agents/test_sql.py:
1. Generates valid SQL for simple query
2. Applies business rules from DataPoints
3. Uses correct table/column names
4. Explanation is coherent
5. Tracks used DataPoints
Mock OpenAI calls for unit tests
```text

**Integration Test Command:**

```text
Create tests/integration/test_sql_integration.py:
1. Real LLM call with sample context
2. Generated SQL is syntactically valid
3. Test various query types (aggregation, joins, filters)
```text

**PR:** `feature/sql-agent`

---

### Week 5: Validation & Execution

#### Day 1-2: ValidatorAgent

**Claude Code Command:**

```text
Create backend/agents/validator.py:
1. ValidatorAgent inheriting from BaseAgent
2. Input: generated SQL, target database schema
3. Output: is_valid, errors[], warnings[], suggestions[]
4. Validation checks:
   - SQL syntax (using sqlparse)
   - Table/column existence
   - SQL injection patterns (blocklist)
   - Performance concerns (SELECT *, missing WHERE)
5. NO LLM calls - rule-based validation
6. Returns structured feedback for retry
```text

**Test Command:**

```text
Create tests/unit/agents/test_validator.py:
1. Valid SQL passes
2. Syntax errors detected
3. Non-existent tables rejected
4. SQL injection patterns blocked
5. Performance warnings generated
6. Feedback is actionable
```text

**PR:** `feature/validator-agent`

---

#### Day 3-4: ExecutorAgent

**Claude Code Command:**

```text
Create backend/agents/executor.py:
1. ExecutorAgent inheriting from BaseAgent
2. Input: validated SQL, database connector
3. Output: data rows, natural language answer, visualization_hint
4. Executes query with timeout
5. Uses GPT-4o-mini to generate natural language summary
6. Suggests visualization type based on data shape
7. Formats large results (pagination/truncation)
8. Includes source citations from pipeline
```text

**Test Command:**

```text
Create tests/unit/agents/test_executor.py:
1. Executes query and returns data
2. Generates readable summary
3. Handles empty results gracefully
4. Timeout prevents runaway queries
5. Citations are included
```text

**PR:** `feature/executor-agent`

---

#### Day 5: ClassifierAgent

**Claude Code Command:**

```text
Create backend/agents/classifier.py:
1. ClassifierAgent inheriting from BaseAgent
2. Input: user query, conversation history (optional)
3. Output: intent, entities, complexity, clarification_needed
4. Uses GPT-4o-mini (fast, cheap)
5. Intent types: data_query, exploration, explanation, meta
6. Entity extraction: tables, columns, metrics, time_refs
7. Complexity: simple, medium, complex
8. Identifies ambiguities needing clarification
```text

**Test Command:**

```text
Create tests/unit/agents/test_classifier.py:
1. Detects data_query intent correctly
2. Extracts table names from query
3. Extracts metric names (including synonyms)
4. Identifies time references
5. Flags ambiguous queries
```text

**PR:** `feature/classifier-agent`

---

### Week 6: Pipeline Integration

#### Day 1-3: LangGraph Orchestrator

**Claude Code Command:**

```text
Create backend/pipeline/orchestrator.py:
1. DataChatPipeline class using LangGraph
2. Define state schema with all agent outputs
3. Create graph with conditional edges:
   - classifier → context → sql → validator
   - validator (fail) → sql (retry, max 3)
   - validator (pass) → executor
4. Streaming support for agent status
5. Error recovery and graceful degradation
6. Total cost/latency tracking
```text

**Test Command:**

```text
Create tests/unit/pipeline/test_orchestrator.py:
1. Pipeline executes in correct order
2. Retry loop works on validation failure
3. Max retries is respected
4. Streaming emits status updates
5. Errors are captured and returned
```text

**Integration Test Command:**

```text
Create tests/integration/test_pipeline_e2e.py:
1. Full end-to-end query execution
2. Test with real database and DataPoints
3. Verify answer quality manually
```text

**PR:** `feature/pipeline-orchestrator`

---

#### Day 4-5: API Layer

**Claude Code Command:**

```text
Create backend/api/main.py:
1. FastAPI application with lifespan for startup/shutdown
2. Initialize pipeline, connectors, knowledge on startup
3. CORS middleware for frontend
4. Exception handlers for AgentError

Create backend/api/routes/chat.py:
1. POST /api/v1/chat - synchronous chat endpoint
2. Request: message, conversation_id (optional)
3. Response: answer, sql, data, sources, metrics

Create backend/api/routes/health.py:
1. GET /health - liveness check
2. GET /ready - readiness check (DB connected, etc.)
```text

**Test Command:**

```text
Create tests/unit/api/test_chat.py:
1. Chat endpoint returns 200 for valid query
2. Returns structured response
3. Handles errors gracefully
4. Validates request body
```text

**PR:** `feature/api-layer`

---

### Week 7: Streaming & CLI

#### Day 1-2: WebSocket Streaming

**Claude Code Command:**

```text
Create backend/api/websocket.py:
1. WebSocket endpoint at /ws/chat
2. Streams agent status updates in real-time
3. Event types: agent_start, agent_complete, data_chunk, answer_chunk, complete
4. Handles disconnection gracefully
5. Supports conversation_id for context

Update backend/pipeline/orchestrator.py:
1. Add callback hooks for streaming
2. Emit events at each agent boundary
```text

**Test Command:**

```text
Create tests/integration/test_websocket.py:
1. WebSocket connects successfully
2. Receives agent status events
3. Final message contains complete response
4. Handles client disconnect
```text

**PR:** `feature/websocket-streaming`

---

#### Day 3-4: CLI Interface

**Claude Code Command:**

```text
Create backend/cli.py using click:
1. datachat chat - interactive REPL mode
2. datachat ask "query" - single query mode
3. datachat connect "connection_string" - set database
4. datachat dp list - list DataPoints
5. datachat dp add <type> <file> - add DataPoint
6. datachat dp sync - rebuild vectors and graph
7. datachat status - show connection status
Add entry point in pyproject.toml
```text

**Test Command:**

```text
Create tests/unit/test_cli.py:
1. Commands parse arguments correctly
2. Help text is shown
3. Error handling works
```text

**PR:** `feature/cli`

---

#### Day 5: ClickHouse Connector

**Claude Code Command:**

```text
Create backend/connectors/clickhouse.py:
1. ClickHouseConnector implementing BaseConnector
2. Uses clickhouse-connect library
3. Schema introspection for ClickHouse
4. Handle ClickHouse-specific SQL dialect
5. Connection pooling
```text

**Test Command:**

```text
Create tests/unit/connectors/test_clickhouse.py:
(Similar structure to postgres tests)
```text

**PR:** `feature/clickhouse-connector`

---

### Week 8: Frontend & Docker

#### Day 1-3: Web UI

**Claude Code Command:**

```text
Initialize Next.js frontend:
cd frontend && npx create-next-app@latest . --typescript --tailwind --eslint --app

Install dependencies:
npm install @tanstack/react-query zustand lucide-react
npx shadcn-ui@latest init
npx shadcn-ui@latest add button input card

Create the chat interface:
1. app/page.tsx - main chat page
2. components/chat/ChatInterface.tsx - message list and input
3. components/chat/Message.tsx - individual message
4. components/agents/AgentStatus.tsx - pipeline status display
5. lib/api.ts - API client with fetch
6. lib/stores/chat.ts - Zustand store for messages

The UI should:
- Show chat messages
- Display agent progress during processing
- Show SQL query and results
- Display source citations
- Support streaming responses via WebSocket
```text

**Manual Testing:**

- Test full flow: query → agent status → response
- Verify WebSocket connection
- Check responsive design

**PR:** `feature/web-ui`

---

#### Day 4-5: Docker & Documentation

**Claude Code Command:**

```text
Create Dockerfile for backend:
1. Multi-stage build
2. Python 3.11 slim base
3. Install dependencies with uv
4. Copy application code
5. Run with uvicorn

Update docker-compose.yml:
1. Add backend service
2. Add frontend service (optional)
3. Volume mounts for development
4. Health checks

Create README.md with:
1. Project overview
2. Quick start (docker-compose up)
3. Manual installation steps
4. Configuration options
5. Usage examples
6. Contributing guidelines
```text

**PR:** `feature/docker-deployment`

---

### Week 8.5-9: Critical UX Features (MUST HAVE for v1.0)

> **Note:** These features address critical UX gaps discovered during testing. They are REQUIRED for v1.0 release to ensure good user experience.

#### Day 1-2: Empty State Handling & System Initialization

**Context:** Users installing DataChat cannot query without DataPoints, but system doesn't guide them. Need initialization wizard and helpful errors.

**Claude Code Command:**

```text
Create backend/api/routes/system.py:
1. GET /api/v1/system/status - Check initialization state
   - Returns: is_initialized, has_databases, has_datapoints, setup_required
   - Lists missing setup steps with actions
2. POST /api/v1/system/initialize - Guided initialization endpoint
   - Accepts: database connection info, auto_profile flag
   - Returns: initialization progress and results

Update backend/api/routes/chat.py:
1. Add empty state check before processing query
2. Return structured error response when not initialized:
   - error: "system_not_initialized"
   - message: "DataChat requires setup. Please initialize first."
   - setup_steps: array of required actions
3. Include helpful error messages throughout

Create backend/initialization/initializer.py:
1. SystemInitializer class with initialization workflow
2. Check for database connections
3. Check for DataPoints
4. Provide clear status and next steps
```

**Test Command:**

```text
Create tests/unit/api/test_system.py:
1. Status endpoint returns correct initialization state
2. Returns setup steps when not initialized
3. Chat endpoint returns helpful error when empty state
4. Initialize endpoint validates input

Create tests/integration/test_initialization.py:
1. Full initialization workflow end-to-end
2. Error handling for invalid database URLs
```

**Frontend Update:**

```text
Update frontend to handle empty state:
1. Detect system_not_initialized error from API
2. Display setup wizard UI with clear steps
3. Guide user through database connection
4. Show progress during initialization
```

**CLI Update:**
do a similar detection and onboarding for cli and add tests

**PR:** `feature/empty-state-handling`

---

#### Day 3-4: Multi-Database Support Foundation

**Context:** Users need to query multiple databases. Implement foundation for database registry and selection.

**Claude Code Command:**

```text
Create backend/models/database.py:
1. DatabaseConnection Pydantic model:
   - connection_id (UUID)
   - name (user-friendly)
   - database_url (SecretStr)
   - database_type (postgresql, clickhouse, mysql, etc.)
   - is_active, is_default (booleans)
   - tags (list[str])
   - description (optional)
   - created_at, last_profiled (datetime)
   - datapoint_count (int)

Create backend/database/manager.py:
1. DatabaseConnectionManager class:
   - add_connection(name, url, type, tags) -> DatabaseConnection
   - list_connections() -> list[DatabaseConnection]
   - get_connection(connection_id) -> DatabaseConnection
   - set_default(connection_id) -> None
   - remove_connection(connection_id) -> None
2. Store connections in system database (PostgreSQL)
3. Secure credential storage (encrypted database_url)
4. Connection validation before saving

Create backend/api/routes/databases.py:
1. POST /api/v1/databases - Add new database connection
2. GET /api/v1/databases - List all connections
3. GET /api/v1/databases/{id} - Get single connection
4. PUT /api/v1/databases/{id}/default - Set as default
5. DELETE /api/v1/databases/{id} - Remove connection

Update backend/api/routes/chat.py:
1. Accept optional target_database parameter in request
2. Route query to specified database
3. Use default database if not specified
```

**Test Command:**

```text
Create tests/unit/database/test_manager.py:
1. Add connection validates URL and type
2. List connections returns all active
3. Set default works correctly
4. Remove connection cascades properly

Create tests/unit/api/test_databases.py:
1. POST creates connection successfully
2. GET lists all connections
3. PUT sets default correctly
4. DELETE removes connection
5. Validates credentials before saving
```

**PR:** `feature/multi-database-foundation`

---

#### Day 5-6: Auto-Profiling & DataPoint Generation

**Context:** Users shouldn't need to manually create DataPoints. System should auto-profile database and generate DataPoints using LLM.

**Claude Code Command:**

```text
Create backend/profiling/profiler.py:
1. SchemaProfiler class:
   - profile_database(connection_id, sample_size=100) -> DatabaseProfile
   - Introspect schema via information_schema
   - Collect table metadata (row counts, column types)
   - Sample actual data for each column
   - Discover relationships (FKs + heuristics)
   - Calculate statistics (nulls, cardinality, min/max)

Create backend/profiling/generator.py:
1. DataPointGenerator class:
   - generate_from_profile(profile, llm_provider) -> GeneratedDataPoints
   - Use LLM to understand schema and suggest:
     * Business purpose for each table
     * Business meaning for each column
     * Common metrics and KPIs
     * Potential business rules
     * Query patterns
   - Generate Schema DataPoints for all tables
   - Generate Business DataPoints for detected metrics
   - Return with confidence scores for review

Create backend/api/routes/profiling.py:
1. POST /api/v1/databases/{id}/profile - Trigger profiling
   - Accepts: sample_size, tables (optional filter)
   - Returns: profiling job ID
2. GET /api/v1/profiling/jobs/{id} - Check profiling status
3. POST /api/v1/datapoints/generate - Generate from profile
   - Uses LLM to create DataPoints
   - Returns pending DataPoints for approval
4. GET /api/v1/datapoints/pending - List pending for review
5. POST /api/v1/datapoints/pending/{id}/approve - Approve and activate
6. POST /api/v1/datapoints/pending/{id}/reject - Reject
7. POST /api/v1/datapoints/pending/bulk-approve - Approve all
```

**Test Command:**

```text
Create tests/unit/profiling/test_profiler.py:
1. Profiles PostgreSQL schema correctly
2. Samples data with correct size
3. Discovers foreign key relationships
4. Calculates statistics accurately

Create tests/unit/profiling/test_generator.py:
1. Generates Schema DataPoints from profile
2. Suggests metrics from numeric columns
3. Identifies time series patterns
4. Returns confidence scores

Create tests/integration/test_auto_profiling.py:
1. End-to-end profiling with real database
2. LLM-generated DataPoints are valid
3. Approval workflow works correctly
```

**Frontend Update:**

```text
Add profiling UI to frontend:
1. Trigger profiling from database management page
2. Show profiling progress (tables processed)
3. Review generated DataPoints with confidence scores
4. Bulk approve/reject/edit functionality
```

**PR:** `feature/auto-profiling`

---

#### Day 7: Auto-Sync Mechanism

**Context:** When DataPoints are added/updated, vector store and knowledge graph should automatically sync.

**Claude Code Command:**

```text
Create backend/sync/watcher.py:
1. DataPointWatcher class using watchdog library
2. Monitor datapoints/ directory for file changes
3. Trigger sync on: create, modify, delete
4. Debounce to avoid excessive syncs (5 second delay)

Update backend/api/routes/datapoints.py:
1. After POST (create) -> trigger sync automatically
2. After PUT (update) -> trigger sync automatically
3. After DELETE -> trigger sync automatically
4. POST /api/v1/sync - Manual sync endpoint
5. GET /api/v1/sync/status - Check sync status

Create backend/sync/orchestrator.py:
1. SyncOrchestrator class:
   - sync_all() - Full rebuild of vectors + graph
   - sync_incremental(datapoint_ids) - Update specific DataPoints
   - Background job support with status tracking
   - Progress callbacks for UI
```

**Test Command:**

```text
Create tests/unit/sync/test_watcher.py:
1. Detects file creation
2. Detects file modification
3. Detects file deletion
4. Debouncing works correctly

Create tests/unit/sync/test_orchestrator.py:
1. Full sync rebuilds everything
2. Incremental sync updates only changed
3. Status tracking works
```

**PR:** `feature/auto-sync`

---

#### Day 8: Integration & Documentation

**Claude Code Command:**

```text
Update documentation:
1. Update README.md with new initialization flow
2. Update GETTING_STARTED.md with auto-profiling steps
3. Create docs/MULTI_DATABASE.md guide
4. Update API documentation with new endpoints

Integration testing:
1. Test complete flow: install → initialize → profile → query
2. Test multi-database setup and querying
3. Test auto-sync after DataPoint changes
4. Verify empty state handling in UI

Update frontend:
1. Add initialization wizard component
2. Add database management page
3. Add profiling workflow UI
4. Add sync status indicator
```

**Final Testing:**

```text
Full user journey testing:
1. Fresh install with no configuration
2. Empty state shows helpful message
3. Initialize system with database URL
4. Auto-profiling generates DataPoints
5. Review and approve DataPoints
6. Successfully query data
7. Add second database
8. Query both databases
9. Modify DataPoint, verify auto-sync
```

**PR:** `feature/ux-integration`

---

## v1.0 Release Checklist (Updated)

```textbash
# Create release branch
git checkout -b release/v1.0.0

# Verify critical UX features are complete
echo "✓ Empty state handling with helpful errors"
echo "✓ Multi-database support foundation"
echo "✓ Auto-profiling and DataPoint generation"
echo "✓ Auto-sync mechanism"
echo "✓ System initialization wizard"

# Final testing
pytest tests/ -v --cov=backend
cd frontend && npm run build

# Test complete user journey
echo "Testing: Fresh install → Initialize → Profile → Query"
# 1. Start with clean database
# 2. Hit chat endpoint, verify empty state error
# 3. Initialize system with database URL
# 4. Trigger auto-profiling
# 5. Review and approve generated DataPoints
# 6. Successfully execute query
# 7. Add second database
# 8. Query both databases
# 9. Verify auto-sync on DataPoint change

# Update version
# pyproject.toml: version = "1.0.0"

# Update CHANGELOG.md with v1.0 features
cat > CHANGELOG.md << 'EOF'
# v1.0.0 - Multi-Agent Genesis with Smart Initialization

## Major Features

### Core Pipeline
- Multi-agent pipeline (Classifier, Context, SQL, Validator, Executor)
- LangGraph orchestration with retry logic
- Streaming responses via WebSocket
- Real-time agent status updates

### Knowledge System
- DataPoint framework (Schema, Business, Process)
- Vector embeddings with ChromaDB
- Knowledge graph with NetworkX
- Hybrid retrieval (RRF fusion)

### User Experience (NEW in v1.0)
- **System initialization wizard** - Guided setup for first-time users
- **Empty state handling** - Helpful error messages when not configured
- **Auto-profiling** - LLM-powered database schema analysis
- **DataPoint auto-generation** - No manual DataPoint creation required
- **Multi-database support** - Query multiple databases
- **Auto-sync** - Automatic knowledge base updates

### Interfaces
- Web UI (Next.js 15 with shadcn/ui)
- CLI (interactive REPL and single commands)
- REST API (/api/v1/chat, /api/v1/databases, /api/v1/profiling)
- WebSocket streaming (/ws/chat)

### Database Support
- PostgreSQL (primary)
- ClickHouse (OLAP)

### LLM Providers
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude)
- Google (Gemini)
- Local models (Ollama)

### Deployment
- Docker Compose orchestration
- Multi-stage Docker builds
- Health checks and auto-restart
- Production-ready configuration

## Breaking Changes
None (initial release)

## Known Limitations
- Single tenant only (no auth yet)
- English queries only
- No chart rendering (visualization hints only)

## Contributors
Built with Claude Code
EOF

# Create tag
git tag -a v1.0.0 -m "v1.0.0 - Multi-Agent Genesis with Smart Initialization"
git push origin v1.0.0

# GitHub release
gh release create v1.0.0 \
  --title "v1.0.0 - Multi-Agent Genesis with Smart Initialization" \
  --notes-file CHANGELOG.md
```text

---

## v1.0 Hardening: Trust, Accuracy, Onboarding

**Context:** Improve adoption for execs/analysts and new team members by boosting trust, correctness, and time-to-first-insight.

### Task A: Confidence + Evidence Layer

**Claude Code Command:**

```text
Update response models and output formatting:
1. Add confidence_score and evidence_summary to ChatResponse
2. Include datapoint_count, datapoint_ids, and freshness (if present)
3. Surface evidence in CLI and UI responses
4. Provide a short "why this answer" block
```

**Test Command:**

```text
Create tests/unit/api/test_chat_evidence.py:
1. Response includes confidence_score
2. Evidence summary includes datapoint_count and freshness when available
3. Missing freshness handled gracefully
```

**PR:** `feature/confidence-evidence`

### Task B: Deterministic Fallback Mode

**Claude Code Command:**

```text
Add a safe fallback path for SQL generation failures:
1. Create backend/agents/sql_fallback.py with rule-based templates
2. When SQLAgent fails, use fallback templates with strict validation
3. Return "fallback_used" flag in response metrics
```

**Test Command:**

```text
Create tests/unit/agents/test_sql_fallback.py:
1. Fallback produces valid SQL for simple aggregation
2. Fallback rejects unsafe patterns
3. Pipeline returns fallback_used flag when triggered
```

**PR:** `feature/sql-fallback`

### Task C: End-to-End Tracing + Metrics

**Claude Code Command:**

```text
Add correlation IDs and structured metrics:
1. Create RequestContext model with correlation_id
2. Propagate correlation_id across API → pipeline → agents → tools
3. Log structured metrics (latency, retries, tool calls) per request
4. Include correlation_id in ChatResponse and WebSocket events
```

**Test Command:**

```text
Create tests/unit/api/test_tracing.py:
1. correlation_id is generated and preserved
2. correlation_id appears in logs and responses
```

**PR:** `feature/tracing-metrics`

### Task D: Evaluation Suite + Regression Dataset

**Claude Code Command:**

```text
Create evaluation harness:
1. backend/evaluation/metrics.py for retrieval + SQL accuracy
2. backend/evaluation/runner.py to run a dataset
3. Add sample eval dataset in tests/fixtures/eval/
4. Add CLI: datachat eval run ./eval_dataset.json
```

**Test Command:**

```text
Create tests/unit/evaluation/test_metrics.py:
1. Retrieval metrics compute correctly
2. SQL correctness scoring handles failures
```

**PR:** `feature/evaluation-suite`

### Task E: Fast Demo Path (Time to First Insight)

**Claude Code Command:**

```text
Create a one-command demo workflow:
1. Add scripts/bootstrap_demo.sh to seed sample DB + DataPoints
2. Add CLI shortcut: datachat demo
3. Document the demo path in README.md
```

**Test Command:**

```text
Create tests/unit/cli/test_demo.py:
1. demo command exists
2. demo command prints next steps
```

**PR:** `feature/demo-path`

---

## Distribution Plan

### Primary (Recommended)

**PyPI + pipx (CLI):**

```text
1. Publish to PyPI: pip install datachat
2. Recommend pipx for isolated CLI installs
3. Include README + long description + entry points
```

**Docker Images (Backend + Frontend):**

```text
1. Publish images to GHCR or Docker Hub
2. Tag with semantic versions (v1.0.0, latest)
3. Provide docker-compose.yml examples
```

### Secondary

**Homebrew (CLI):**
```text
1. Publish a brew formula that installs the CLI
2. Use GitHub Releases as the artifact source
```

**Conda:**
```text
1. Provide conda package for data teams on Anaconda
```

**GitHub Releases:**
```text
1. Attach wheel + sdist
2. Optional: prebuilt CLI binaries via PyInstaller
```

---

## Commercial Distribution (Paid)

**Context:** Plan for paid tiers with feature gating and subscription enforcement.

### Options

**Hosted SaaS (Recommended):**
```text
1. Run DataChat as a hosted service
2. Sell subscriptions (per seat or usage-based)
3. Enforce feature access via server-side policies
```

**Private Docker Registry:**
```text
1. Publish images to a private registry (GHCR/ECR)
2. Require authenticated pulls for paid customers
3. Enforce license checks at startup
```

**Private Package Registry (CLI):**
```text
1. Publish to a private PyPI (Gemfury/Artifactory)
2. Distribute access tokens to paid users
3. Gate premium commands/features in code
```

### Feature Gating Strategy

```text
1. Define a feature flag matrix (free vs paid)
2. Enforce on both API and UI (server-side source of truth)
3. Add entitlement checks in CLI, API, and UI
4. Provide a graceful upgrade flow with clear prompts
```

### Starter Feature Matrix Template

```text
Feature Area | Free Tier | Paid Tier | Notes
-------------|----------|-----------|------
Core Chat    | ✅        | ✅         | Base NLQ + SQL
DataPoints   | ✅        | ✅         | Manual + auto-profiling
Workspace    | ✅        | ✅         | Read-only indexing/search
Tooling      | ✅        | ✅         | Read-only tools
Advanced     | ❌        | ✅         | Reranking, eval suite, custom tools
Security     | ❌        | ✅         | SSO, RBAC, audit exports
Scale        | ❌        | ✅         | Multi-DB routing, HA, quotas
Support      | ❌        | ✅         | SLA, onboarding support
```

**PR:** `docs/commercial-distribution`

---

## v2 Planning: Data-Native Agent OS

### Week 9-10: Tooling + Workspace Ingestion (Read-Only)

**Context:** Extend DataChat for BI workflows with safe filesystem ingestion and tool execution. Keep analytics-first focus while enabling workspace context.
**UX Goal:** Seamless UI + CLI workflows (same actions, same status, same guidance).

**Claude Code Command:**

```text
Create backend/tools/registry.py:
1. ToolSpec model (name, description, input_schema, output_schema, safety)
2. ToolRegistry with allowlist and per-tool policy
3. Policy-as-config loader (YAML/JSON) for allowlist + routing rules
4. Execution wrapper with audit logging

Create backend/tools/filesystem.py:
1. read_file(path) - read-only, workspace-rooted
2. list_dir(path) - read-only, workspace-rooted
3. search_text(root, pattern) - use rg if available
4. file_metadata(path) - size, mtime, checksum
5. Enforce sandbox + allowlist

Create backend/workspace/indexer.py:
1. WorkspaceIndexer class
2. Incremental scan (checksum + last_modified)
3. Chunking + summarization for large files
4. Emits WorkspaceDataPoints

Update backend/models/datapoint.py:
1. WorkspaceDataPoint type
2. file_path, language, symbols, docstrings, domain_tags
3. extracted_entities (tables, models, configs)
4. last_modified, checksum

Update backend/knowledge/datapoints.py:
1. Load WorkspaceDataPoints
2. Store in vectors + graph with source metadata

Add API endpoints:
1. POST /api/v1/workspace/index - run indexer
2. GET /api/v1/workspace/status - indexing status
3. GET /api/v1/workspace/search - semantic search

Add CLI commands:
1. datachat workspace index --root ./
2. datachat workspace search "auth flow"
```

**Test Command:**

```text
Create tests/unit/tools/test_registry.py:
1. Tool allowlist enforced
2. Policy blocks write tools by default
3. Audit log recorded
4. Policies load from config

Create tests/unit/tools/test_filesystem.py:
1. Read/list/search are workspace-rooted
2. Disallows path traversal
3. Handles missing files gracefully

Create tests/unit/workspace/test_indexer.py:
1. Incremental indexing works
2. Chunking boundaries enforced
3. WorkspaceDataPoint schema valid
```

**PR:** `feature/workspace-ingestion`

**UX Acceptance Criteria (CLI):**
```text
- datachat workspace index --root ./ completes with a summary table
- datachat workspace status shows last run, files indexed, files skipped
- datachat workspace search "query" returns relevant hits with file paths
- datachat workspace index --dry-run shows what will be indexed
- Errors explain missing permissions or blocked paths
```

**UX Acceptance Criteria (UI):**
```text
- Workspace page allows selecting a folder and starting indexing
- Live progress shows files scanned, indexed, skipped, and errors
- Search UI returns results with file path + snippet + tags
- Users can reindex and see when the last index ran
- Errors are actionable and link to docs/troubleshooting
```

---

### Persona Acceptance Criteria

```text
Sarah (Analyst)
- datachat workspace index --root ./ runs without extra config
- Search results include file path + snippet + tags in < 2 seconds on typical repos
- Errors include a clear next step for setup or permissions

Marcus (Data Engineer)
- Tool policies load from config without code edits
- Audit logs capture tool name, args, user/session, and correlation ID
- Workspace indexing groups dbt/models/migrations for quick review

Priya (Data Platform Lead)
- Correlation IDs span API → pipeline → agents → tools in logs
- GET /api/v1/workspace/status reports last run, counts, and errors
- Multi-node sync guidance is referenced in operational errors

James (Executive)
- Responses include confidence cues and sources when available
- UI exposes only high-level actions (ask, search, reindex)
```

### v2 Milestone Checklist

```text
Milestone 1: Read-Only Tools + Workspace Ingestion
- Tool registry + policy config loader
- Read-only filesystem tools (list/read/search/metadata)
- Audit logging with correlation IDs
- WorkspaceIndexer with incremental checksum/mtime
- WorkspaceDataPoint ingestion into vectors + graph
- CLI: workspace index/status/search
- API: /workspace/index, /workspace/status, /workspace/search

Milestone 2: Router + Tool Planner/Executor
- Router selects SQL vs tool pipeline with explainable rationale
- Tool planner outputs structured calls
- Policy checks enforced at execution time
- Streaming updates show tool steps + outcomes

Milestone 3: Controlled Write Tools (Gated)
- Explicit user approval flow for write actions
- Policy levels for read/write tools
- Audit log includes approvals and diff summaries
```

---

### Week 11-12: Agent Router + Tool Planner

**Claude Code Command:**

```text
Create backend/agents/router.py:
1. AgentRouter selects DB pipeline vs Tool pipeline
2. Uses intent + tool availability
3. Adds routing metadata to response

Create backend/agents/tools.py:
1. ToolPlannerAgent (decides tool calls)
2. ToolExecutorAgent (runs tool calls via ToolRegistry)
3. Captures tool outputs as context

Update backend/pipeline/orchestrator.py:
1. Add router stage
2. Add tool branch before SQL pipeline
```

**Test Command:**

```text
Create tests/unit/agents/test_router.py:
1. Routes DB queries to SQL pipeline
2. Routes workspace queries to tool pipeline

Create tests/unit/agents/test_tools.py:
1. Tool plans validated against schema
2. Tool execution uses registry policies
```

**PR:** `feature/tooling-router`

---

### RAG Upgrade Roadmap (Post-v2 Foundation)

**Near-term**
```text
- Chunking for WorkspaceDataPoints (size + overlap)
- Query normalization (lowercasing, stopword trim)
- Metadata filters by DataPoint type/tags
- Optional lightweight reranker (default off)
```

**Mid-term**
```text
- BM25 keyword retrieval alongside vectors
- Query rewriting/decomposition for multi-part questions
- Context compression (dedupe, truncate, relevance caps)
```

**Later**
```text
- Retrieval evaluation set + regression gating
- Embedding fine-tuning for schema/metric language
- Adaptive retrieval routing by intent/confidence
```

---

## Phase 2: v1.1 Precision Retrieval (Weeks 9-16)

### Week 9-10: Enhanced Retrieval

#### Contextual Embeddings

**Claude Code Command:**

```text
Update backend/knowledge/vectors.py:
1. Add ContextualEmbedder class
2. Generate context prefix based on DataPoint type:
   - Schema: "Table {name} in {schema}. Purpose: {purpose}. Columns: {cols}"
   - Business: "Metric {name}. Synonyms: {syns}. Related: {tables}"
   - Process: "Process {name}. Freshness: {fresh}. Tables: {targets}"
3. Prepend context before embedding
4. Configuration toggle: embeddings.contextual.enabled
5. Max context tokens configurable

Reference Anthropic's Contextual Retrieval blog for implementation details.
```text

**Test Command:**

```text
Create tests/unit/knowledge/test_contextual.py:
1. Context is prepended correctly for each type
2. Context respects max token limit
3. Toggle enables/disables feature
4. Embeddings are different with context
```text

**PR:** `feature/contextual-embeddings`

---

#### BM25 Search

**Claude Code Command:**

```text
Update backend/knowledge/vectors.py:
1. Add BM25Index class using rank_bm25 library
2. Build index from DataPoint text content
3. Search method returns scored results
4. Incremental update support

Update backend/knowledge/retriever.py:
1. Add BM25 as third retrieval source
2. Update RRF to handle three sources
3. Configuration: retrieval.bm25.enabled (default: true)
```text

**Test Command:**

```text
Create tests/unit/knowledge/test_bm25.py:
1. Index builds correctly
2. Exact matches score highest
3. Partial matches work
4. Incremental updates work
```text

**PR:** `feature/bm25-search`

---

#### Reranking

**Claude Code Command:**

```text
Create backend/knowledge/reranker.py:
1. BaseReranker ABC
2. CohereReranker implementation
3. VoyageReranker implementation
4. Rerank method: (query, chunks) → reranked chunks
5. Configurable top_k_input and top_k_output

Update backend/knowledge/retriever.py:
1. Add optional reranking step after fusion
2. Configuration: retrieval.reranker.enabled (default: false)
3. Graceful fallback if reranker fails
```text

**Test Command:**

```text
Create tests/unit/knowledge/test_reranker.py:
1. Reranking changes order
2. Top-k filtering works
3. Fallback on API error
```text

**PR:** `feature/reranking`

---

### Week 11-12: RAG Evaluation Suite

#### Evaluation Framework

**Claude Code Command:**

```text
Create backend/evaluation/ directory:

backend/evaluation/metrics.py:
1. RetrievalMetrics class:
   - precision_at_k(retrieved, relevant, k)
   - recall_at_k(retrieved, relevant, k)
   - hit_rate(retrieved, relevant)
   - mrr(retrieved, relevant)
   - ndcg_at_k(retrieved, relevant, k)

2. GenerationMetrics class:
   - faithfulness(answer, context) - LLM judge
   - answer_relevancy(answer, question) - LLM judge
   - sql_correctness(generated_sql, expected_sql, db) - execution comparison

backend/evaluation/runner.py:
1. EvaluationRunner class
2. Load test dataset (JSON)
3. Run pipeline on each test case
4. Compute all metrics
5. Generate report
```text

**Test Command:**

```text
Create tests/unit/evaluation/test_metrics.py:
1. Each metric computes correctly
2. Edge cases handled (empty results)
3. LLM judge prompts are correct
```text

**PR:** `feature/evaluation-framework`

---

#### Synthetic Data Generator

**Claude Code Command:**

```text
Create backend/evaluation/generator.py:
1. SyntheticGenerator class
2. Generate questions from DataPoints using LLM
3. For each DataPoint, generate:
   - Simple question (single fact lookup)
   - Medium question (aggregation/filter)
   - Complex question (joins/multi-step)
4. Output includes: question, expected_datapoints, expected_tables
5. Configurable count and difficulty distribution
```text

**CLI Command:**

```text
Add to backend/cli.py:
datachat eval generate --from-datapoints ./datapoints/ --count 100 --output ./eval_dataset.json
datachat eval run ./eval_dataset.json --output ./results.json
datachat eval compare --baseline ./v1.json --candidate ./v2.json
```text

**PR:** `feature/synthetic-generator`

---

### Week 13-14: Additional Integrations

#### Anthropic Claude Support

**Claude Code Command:**

```text
Create backend/llm/providers.py:
1. BaseLLMProvider ABC
2. OpenAIProvider (existing logic extracted)
3. AnthropicProvider using anthropic SDK
4. Provider factory based on config

Update agents to use provider abstraction.
Configuration: llm.provider = "openai" | "anthropic"
```text

**PR:** `feature/anthropic-provider`

---

#### MySQL & BigQuery Connectors

**Claude Code Command:**

```text
Create backend/connectors/mysql.py:
(Similar pattern to postgres)

Create backend/connectors/bigquery.py:
1. Uses google-cloud-bigquery
2. Service account authentication
3. Schema introspection
4. Query execution with pagination
```text

**PR:** `feature/mysql-connector`  
**PR:** `feature/bigquery-connector`

---

### Week 15-16: Polish & Release

#### DataPoint UI Manager

**Claude Code Command:**

```text
Create frontend pages:
1. app/datapoints/page.tsx - list all DataPoints
2. app/datapoints/[id]/page.tsx - view/edit single DataPoint
3. components/datapoints/DataPointForm.tsx - create/edit form
4. components/datapoints/DataPointList.tsx - table with search/filter

Create API routes:
backend/api/routes/datapoints.py:
1. GET /api/v1/datapoints - list
2. GET /api/v1/datapoints/{id} - get single
3. POST /api/v1/datapoints - create
4. PUT /api/v1/datapoints/{id} - update
5. DELETE /api/v1/datapoints/{id} - delete
```text

**PR:** `feature/datapoint-ui`

---

#### Performance Benchmarks

**Claude Code Command:**

```text
Create scripts/benchmark.py:
1. Run N queries against pipeline
2. Measure latency percentiles (p50, p95, p99)
3. Measure retrieval quality metrics
4. Output report with charts

Create docs/BENCHMARKS.md with results
```text

**PR:** `docs/benchmarks`

---

## v1.1 Release Checklist

```textbash
git checkout -b release/v1.1.0
pytest tests/ -v --cov=backend
git tag -a v1.1.0 -m "v1.1.0 - Precision Retrieval"
git push origin v1.1.0
gh release create v1.1.0 --title "v1.1.0 - Precision Retrieval"
```text

---

## Phase 3: v1.2 Connected World (Weeks 17-24)

### Week 17-20: Integrations

**Features to implement:**

1. Slack Bot (`feature/slack-bot`)
2. Microsoft Teams Bot (`feature/teams-bot`)
3. Python SDK (`feature/python-sdk`)
4. TypeScript SDK (`feature/typescript-sdk`)
5. Ollama local LLM support (`feature/ollama-provider`)
6. Snowflake connector (`feature/snowflake-connector`)

**Claude Code Command Pattern:**

```text
For each integration:
1. Research API/SDK documentation
2. Create integration in backend/integrations/{name}/
3. Write comprehensive tests
4. Add documentation
5. Create example usage
```text

### Week 21-24: Developer Experience

**Features:**

1. Graph visualization UI (`feature/graph-viz`)
2. Evaluation dashboard (`feature/eval-dashboard`)
3. Webhook endpoints (`feature/webhooks`)
4. Plugin system (`feature/plugins`)

---

## Phase 4: v1.3 Enterprise Ready (Weeks 25-32)

### Week 25-28: Security

**Features:**

1. SAML SSO (`feature/saml-sso`)
2. Advanced RBAC (`feature/rbac`)
3. PII detection (`feature/pii-detection`)
4. Data masking (`feature/data-masking`)
5. Audit logging (`feature/audit-logging`)

### Week 29-32: Scale & Polish

**Features:**

1. High availability deployment (`feature/ha-deployment`)
2. Kubernetes manifests (`feature/kubernetes`)
3. Monitoring & alerting (`feature/monitoring`)
4. Enterprise documentation (`docs/enterprise`)

---

## Claude Code Best Practices

### 1. Use /compact Frequently

```text
After every major feature, run /compact to free context
```text

### 2. Reference CLAUDE.md

```text
Always start sessions with: "Read CLAUDE.md, then..."
```text

### 3. Work in Vertical Slices

```text
Complete one feature end-to-end before starting next:
Code → Tests → Manual Test → PR → Merge
```text

### 4. Small, Focused Commands

```text
Bad:  "Build the entire knowledge system"
Good: "Create VectorStore class with add and search methods"
```text

### 5. Test-First When Possible

```text
"Create tests for SQLAgent that verify:
1. Valid SQL generation
2. Business rule application
Then implement SQLAgent to pass these tests"
```text

### 6. Review Before Commit

```text
After Claude generates code:
1. Read the diff
2. Run tests
3. Manual smoke test
4. Then commit
```text

### 7. Keep Context Updated

```text
Update CLAUDE.md "Current Development Focus" section
after each PR merge
```text

---

## Weekly Rhythm

### Monday

- Review week's goals
- Plan Claude Code sessions
- Update CLAUDE.md focus section

### Tuesday-Thursday

- Implement features
- Write tests
- Create PRs

### Friday

- Code review
- Merge PRs
- Run full test suite
- Update documentation
- /compact Claude context

---

## Troubleshooting

### Claude Code Lost Context

```text
"Read CLAUDE.md. I'm working on [feature]. 
Here's what's already done: [summary].
Continue with: [next step]"
```text

### Tests Failing

```text
"The test test_xyz is failing with error: [error].
The relevant code is in [file].
Fix the issue while maintaining existing behavior."
```text

### Stuck on Design

```text
"I need to implement [feature]. 
Requirements: [list]
Constraints: [list]
Suggest 2-3 approaches with tradeoffs."
```text

---

*This playbook is your roadmap. Follow it step-by-step, and you'll have DataChat v1.0 in 8 weeks, v1.1 in 16 weeks, and a full enterprise product by month 8.*
