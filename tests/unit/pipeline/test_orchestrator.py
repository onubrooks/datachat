"""
Unit tests for DataChatPipeline orchestrator.

Tests pipeline execution including:
- Correct agent execution order
- Retry loop on validation failure
- Max retries enforcement
- Streaming status updates
- Error handling and recovery
- State management
"""

from unittest.mock import AsyncMock

import pytest

from backend.models import (
    AgentMetadata,
    ClassifierAgentOutput,
    ContextAgentOutput,
    ContextAnswer,
    ContextAnswerAgentOutput,
    ExecutedQuery,
    ExecutorAgentOutput,
    GeneratedSQL,
    InvestigationMemory,
    QueryClassification,
    QueryResult,
    RetrievedDataPoint,
    SQLAgentOutput,
    SQLValidationError,
    ToolPlan,
    ToolPlannerAgentOutput,
    ValidatedSQL,
    ValidatorAgentOutput,
)
from backend.pipeline.orchestrator import DataChatPipeline


class TestPipelineExecution:
    """Test basic pipeline execution flow."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever."""
        retriever = AsyncMock()
        return retriever

    @pytest.fixture
    def mock_connector(self):
        """Mock database connector."""
        connector = AsyncMock()
        connector.connect = AsyncMock()
        connector.close = AsyncMock()
        return connector

    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider."""
        provider = AsyncMock()
        provider.generate = AsyncMock(return_value="mock response")
        provider.stream = AsyncMock()
        provider.count_tokens = AsyncMock(return_value=100)
        provider.get_model_info = AsyncMock(return_value={"name": "mock-model"})
        provider.provider = "mock"
        provider.model = "mock-model"
        return provider

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_connector, mock_llm_provider, mock_openai_api_key):
        """Create pipeline with mocked dependencies."""
        pipeline = DataChatPipeline(
            retriever=mock_retriever,
            connector=mock_connector,
            max_retries=3,
        )
        # Inject mock LLM providers into agents to avoid real API calls
        pipeline.classifier.llm = mock_llm_provider
        pipeline.sql.llm = mock_llm_provider
        pipeline.executor.llm = mock_llm_provider
        pipeline.tool_planner.execute = AsyncMock(
            return_value=ToolPlannerAgentOutput(
                success=True,
                plan=ToolPlan(tool_calls=[], rationale="No tools needed.", fallback="pipeline"),
                metadata=AgentMetadata(agent_name="ToolPlannerAgent", llm_calls=0),
            )
        )
        pipeline.response_synthesis.execute = AsyncMock(return_value="Found 1 result.")
        return pipeline

    @pytest.fixture
    def mock_agents(self, pipeline):
        """Mock all agents in pipeline."""
        # Mock ClassifierAgent
        pipeline.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="data_query",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )

        # Mock ContextAgent
        pipeline.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="test query",
                    datapoints=[
                        RetrievedDataPoint(
                            datapoint_id="table_001",
                            datapoint_type="Schema",
                            name="Test Table",
                            score=0.9,
                            source="vector",
                            metadata={"type": "Schema"},
                        )
                    ],
                    retrieval_mode="hybrid",
                    total_retrieved=1,
                    sources_used=["vector"],
                ),
                context_confidence=0.2,
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )

        pipeline.context_answer.execute = AsyncMock(
            return_value=ContextAnswerAgentOutput(
                success=True,
                context_answer=ContextAnswer(
                    answer="Here is a context-only answer.",
                    confidence=0.8,
                    evidence=[],
                    needs_sql=False,
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="ContextAnswerAgent", llm_calls=1),
            )
        )

        # Mock SQLAgent
        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql="SELECT * FROM test_table",
                    explanation="Simple select query",
                    confidence=0.95,
                    used_datapoints=["table_001"],
                    assumptions=[],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=1),
            )
        )

        # Mock ValidatorAgent (passing)
        pipeline.validator.execute = AsyncMock(
            return_value=ValidatorAgentOutput(
                success=True,
                validated_sql=ValidatedSQL(
                    sql="SELECT * FROM test_table",
                    is_valid=True,
                    is_safe=True,
                    errors=[],
                    warnings=[],
                    performance_score=0.8,
                ),
                metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
            )
        )

        # Mock ExecutorAgent
        pipeline.executor.execute = AsyncMock(
            return_value=ExecutorAgentOutput(
                success=True,
                executed_query=ExecutedQuery(
                    query_result=QueryResult(
                        rows=[{"id": 1, "name": "test"}],
                        row_count=1,
                        columns=["id", "name"],
                        execution_time_ms=50.0,
                        was_truncated=False,
                    ),
                    natural_language_answer="Found 1 result.",
                    visualization_hint="table",
                    key_insights=[],
                    source_citations=["table_001"],
                ),
                metadata=AgentMetadata(agent_name="ExecutorAgent", llm_calls=1),
            )
        )

        return pipeline

    @pytest.mark.asyncio
    async def test_pipeline_executes_in_correct_order(self, mock_agents):
        """Test that agents execute in correct order."""
        result = await mock_agents.run("test query")

        # Verify all agents were called
        assert mock_agents.classifier.execute.called
        assert mock_agents.context.execute.called
        assert mock_agents.sql.execute.called
        assert mock_agents.validator.execute.called
        assert mock_agents.executor.execute.called
        assert not mock_agents.context_answer.execute.called

        # Verify final state
        assert result["natural_language_answer"] == "Found 1 result."
        assert result["query_result"]["row_count"] == 1
        assert result["validation_passed"] is True
        assert result.get("error") is None

    @pytest.mark.asyncio
    async def test_pipeline_tracks_metadata(self, mock_agents):
        """Test that pipeline tracks cost and latency."""
        result = await mock_agents.run("test query")

        # Verify metadata tracking
        assert "agent_timings" in result
        assert "classifier" in result["agent_timings"]
        assert "context" in result["agent_timings"]
        assert "sql" in result["agent_timings"]
        assert "validator" in result["agent_timings"]
        assert "executor" in result["agent_timings"]

        assert result["llm_calls"] == 4  # classifier + sql + executor + response_synthesis
        assert result["total_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_simple_sql_can_skip_response_synthesis(self, mock_agents):
        """Request-level override should skip synthesis for simple SQL responses."""
        result = await mock_agents.run("test query", synthesize_simple_sql=False)

        assert result["natural_language_answer"] == "Found 1 result."
        assert not mock_agents.response_synthesis.execute.called
        assert result["llm_calls"] == 3  # classifier + sql + executor

    @pytest.mark.asyncio
    async def test_selective_tool_planner_skips_standard_data_query(self, mock_agents):
        """Tool planner should not run for plain SQL data requests."""
        await mock_agents.run("Show first 5 rows from orders")
        assert not mock_agents.tool_planner.execute.called

    @pytest.mark.asyncio
    async def test_selective_tool_planner_runs_for_tool_like_intent(self, mock_agents):
        """Tool planner should run for tool/action-style requests."""
        await mock_agents.run("Profile database and generate datapoints")
        assert mock_agents.tool_planner.execute.called

    @pytest.mark.asyncio
    async def test_pipeline_includes_all_outputs(self, mock_agents):
        """Test that final state includes all agent outputs."""
        result = await mock_agents.run("test query")

        # ClassifierAgent outputs
        assert result["intent"] == "data_query"
        assert result["complexity"] == "simple"

        # ContextAgent outputs
        assert len(result["retrieved_datapoints"]) == 1
        assert result["retrieved_datapoints"][0]["datapoint_id"] == "table_001"

        # SQLAgent outputs
        assert result["generated_sql"] == "SELECT * FROM test_table"
        assert result["sql_confidence"] == 0.95

        # ValidatorAgent outputs
        assert result["validated_sql"] == "SELECT * FROM test_table"
        assert result["performance_score"] == 0.8

        # ExecutorAgent outputs
        assert result["natural_language_answer"] == "Found 1 result."
        assert result["visualization_hint"] == "table"

    @pytest.mark.asyncio
    async def test_sql_success_clears_stale_classifier_clarification(self, mock_agents):
        """Ensure classifier clarification flags do not block SQL success path."""
        mock_agents.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="data_query",
                    entities=[],
                    complexity="simple",
                    clarification_needed=True,
                    clarifying_questions=["Which table should I use?"],
                    confidence=0.7,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )

        result = await mock_agents.run("show me the first 5 rows from sales")

        assert result["validation_passed"] is True
        assert result["answer_source"] == "sql"
        assert result["clarification_needed"] is False
        assert result["clarifying_questions"] == []

    @pytest.mark.asyncio
    async def test_pipeline_passes_database_context_to_sql_agent(self, mock_agents):
        """Ensure per-request database type/url reach SQLAgentInput."""
        await mock_agents.run(
            "test query",
            database_type="clickhouse",
            database_url="clickhouse://user:pass@click.example.com:8123/analytics",
        )
        sql_input = mock_agents.sql.execute.call_args.args[0]
        assert sql_input.database_type == "clickhouse"
        assert sql_input.database_url == "clickhouse://user:pass@click.example.com:8123/analytics"

    @pytest.mark.asyncio
    async def test_live_table_catalog_uses_database_url_connector(self, pipeline):
        mock_catalog = AsyncMock()
        mock_catalog.is_connected = False
        mock_catalog.connect = AsyncMock()
        mock_catalog.get_schema = AsyncMock(return_value=[])
        mock_catalog.close = AsyncMock()
        pipeline._build_catalog_connector = lambda *_args, **_kwargs: mock_catalog

        result = await pipeline._get_live_table_catalog(
            database_type="postgresql",
            database_url="postgresql://user:pass@localhost:5432/warehouse",
        )

        assert result is None
        mock_catalog.connect.assert_awaited()
        mock_catalog.get_schema.assert_awaited()
        mock_catalog.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_routes_to_context_answer_for_exploration(self, mock_agents):
        mock_agents.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="exploration",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )
        mock_agents.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="test query",
                    datapoints=[
                        RetrievedDataPoint(
                            datapoint_id="table_001",
                            datapoint_type="Schema",
                            name="Test Table",
                            score=0.9,
                            source="vector",
                            metadata={"type": "Schema"},
                        )
                    ],
                    retrieval_mode="hybrid",
                    total_retrieved=1,
                    sources_used=["vector"],
                ),
                context_confidence=0.8,
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )
        mock_agents.context_answer.execute = AsyncMock(
            return_value=ContextAnswerAgentOutput(
                success=True,
                context_answer=ContextAnswer(
                    answer="Context-only response.",
                    confidence=0.7,
                    evidence=[],
                    needs_sql=False,
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="ContextAnswerAgent", llm_calls=1),
            )
        )

        result = await mock_agents.run("Explain what this dataset is about.")

        assert mock_agents.context_answer.execute.called
        assert not mock_agents.sql.execute.called
        assert result["answer_source"] == "context"

    @pytest.mark.asyncio
    async def test_context_answer_can_fall_through_to_sql(self, mock_agents):
        mock_agents.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="exploration",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )
        mock_agents.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="test query",
                    datapoints=[
                        RetrievedDataPoint(
                            datapoint_id="table_001",
                            datapoint_type="Schema",
                            name="Test Table",
                            score=0.9,
                            source="vector",
                            metadata={"type": "Schema"},
                        )
                    ],
                    retrieval_mode="hybrid",
                    total_retrieved=1,
                    sources_used=["vector"],
                ),
                context_confidence=0.8,
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )
        mock_agents.context_answer.execute = AsyncMock(
            return_value=ContextAnswerAgentOutput(
                success=True,
                context_answer=ContextAnswer(
                    answer="Need numbers.",
                    confidence=0.5,
                    evidence=[],
                    needs_sql=True,
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="ContextAnswerAgent", llm_calls=1),
            )
        )

        result = await mock_agents.run("show me counts")

        assert mock_agents.context_answer.execute.called
        assert mock_agents.sql.execute.called
        assert result["answer_source"] == "sql"


class TestRetryLogic:
    """Test SQL validation retry logic."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever."""
        return AsyncMock()

    @pytest.fixture
    def mock_connector(self):
        """Mock connector."""
        connector = AsyncMock()
        connector.connect = AsyncMock()
        connector.close = AsyncMock()
        return connector

    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider."""
        provider = AsyncMock()
        provider.provider = "mock"
        provider.model = "mock-model"
        return provider

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_connector, mock_llm_provider, mock_openai_api_key):
        """Create pipeline."""
        pipeline = DataChatPipeline(
            retriever=mock_retriever,
            connector=mock_connector,
            max_retries=3,
        )
        # Inject mock LLM providers
        pipeline.classifier.llm = mock_llm_provider
        pipeline.sql.llm = mock_llm_provider
        pipeline.executor.llm = mock_llm_provider
        pipeline.tool_planner.execute = AsyncMock(
            return_value=ToolPlannerAgentOutput(
                success=True,
                plan=ToolPlan(tool_calls=[], rationale="No tools needed.", fallback="pipeline"),
                metadata=AgentMetadata(agent_name="ToolPlannerAgent", llm_calls=0),
            )
        )
        pipeline.response_synthesis.execute = AsyncMock(return_value="Found 1 result.")
        return pipeline

    @pytest.mark.asyncio
    async def test_retry_on_validation_failure(self, pipeline):
        """Test that pipeline retries SQL on validation failure."""
        # Mock agents
        pipeline.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="data_query",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )

        pipeline.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="test",
                    datapoints=[],
                    retrieval_mode="hybrid",
                    total_retrieved=0,
                    sources_used=[],
                ),
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )

        # SQLAgent succeeds
        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql="SELECT * FROM invalid_table",
                    explanation="Query",
                    confidence=0.8,
                    used_datapoint_ids=[],
                    assumptions=[],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=1),
            )
        )

        # ValidatorAgent fails first time, passes second time
        validation_call_count = 0

        async def validator_side_effect(*args, **kwargs):
            nonlocal validation_call_count
            validation_call_count += 1

            if validation_call_count == 1:
                # First call: validation fails
                return ValidatorAgentOutput(
                    success=False,
                    validated_sql=ValidatedSQL(
                        sql="SELECT * FROM invalid_table",
                        is_valid=False,
                        is_safe=False,
                        errors=[
                            SQLValidationError(
                                error_type="schema",
                                message="Table 'invalid_table' does not exist",
                            )
                        ],
                        warnings=[],
                        performance_score=0.5,
                    ),
                    metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
                )
            else:
                # Second call: validation passes
                return ValidatorAgentOutput(
                    success=True,
                    validated_sql=ValidatedSQL(
                        sql="SELECT * FROM valid_table",
                        is_valid=True,
                        is_safe=True,
                        errors=[],
                        warnings=[],
                        performance_score=0.9,
                    ),
                    metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
                )

        pipeline.validator.execute = AsyncMock(side_effect=validator_side_effect)

        # ExecutorAgent succeeds
        pipeline.executor.execute = AsyncMock(
            return_value=ExecutorAgentOutput(
                success=True,
                executed_query=ExecutedQuery(
                    query_result=QueryResult(
                        rows=[],
                        row_count=0,
                        columns=[],
                        execution_time_ms=10.0,
                    ),
                    natural_language_answer="No results",
                    visualization_hint="none",
                    key_insights=[],
                    source_citations=[],
                ),
                metadata=AgentMetadata(agent_name="ExecutorAgent", llm_calls=1),
            )
        )

        result = await pipeline.run("test query")

        # Verify retry happened
        assert pipeline.sql.execute.call_count == 2  # Initial + 1 retry
        assert pipeline.validator.execute.call_count == 2
        assert result["retry_count"] == 1
        assert result["validation_passed"] is True

    @pytest.mark.asyncio
    async def test_max_retries_enforced(self, pipeline):
        """Test that max retries is enforced."""
        # Mock agents
        pipeline.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="data_query",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )

        pipeline.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="test",
                    datapoints=[],
                    retrieval_mode="hybrid",
                    total_retrieved=0,
                    sources_used=[],
                ),
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )

        # SQLAgent always succeeds
        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql="SELECT * FROM bad_table",
                    explanation="Query",
                    confidence=0.8,
                    used_datapoints=[],
                    assumptions=[],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=1),
            )
        )

        # ValidatorAgent always fails
        pipeline.validator.execute = AsyncMock(
            return_value=ValidatorAgentOutput(
                success=False,
                validated_sql=ValidatedSQL(
                    sql="SELECT * FROM bad_table",
                    is_valid=False,
                    is_safe=False,
                    errors=[
                        SQLValidationError(
                            error_type="schema",
                            message="Table does not exist",
                        )
                    ],
                    warnings=[],
                    performance_score=0.0,
                ),
                metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
            )
        )

        result = await pipeline.run("test query")

        # Verify max retries hit
        assert pipeline.sql.execute.call_count == 4  # Initial + 3 retries
        assert pipeline.validator.execute.call_count == 4
        assert result["retry_count"] == 3
        assert result.get("error") is not None
        assert "after 3 attempts" in result["error"]


class TestIntentGate:
    """Test intent gate behavior."""

    @pytest.fixture
    def mock_retriever(self):
        return AsyncMock()

    @pytest.fixture
    def mock_connector(self):
        connector = AsyncMock()
        connector.connect = AsyncMock()
        connector.close = AsyncMock()
        connector.is_connected = True
        connector.get_schema = AsyncMock(return_value=[])
        return connector

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_connector, mock_openai_api_key):
        pipeline = DataChatPipeline(
            retriever=mock_retriever,
            connector=mock_connector,
            max_retries=3,
        )
        pipeline.tool_planner.execute = AsyncMock(
            return_value=ToolPlannerAgentOutput(
                success=True,
                plan=ToolPlan(tool_calls=[], rationale="No tools needed.", fallback="pipeline"),
                metadata=AgentMetadata(agent_name="ToolPlannerAgent", llm_calls=0),
            )
        )
        pipeline.response_synthesis.execute = AsyncMock(return_value="Result is 1")
        pipeline.classifier.execute = AsyncMock()
        return pipeline

    @pytest.mark.asyncio
    async def test_intent_gate_exit_short_circuits(self, pipeline):
        result = await pipeline.run("Ok I'm done for now")
        assert result.get("intent_gate") == "exit"
        assert result.get("answer_source") == "system"
        assert "Ending the session" in result.get("natural_language_answer", "")
        assert not pipeline.classifier.execute.called

    @pytest.mark.asyncio
    async def test_intent_gate_exit_detects_talk_later(self, pipeline):
        result = await pipeline.run("let's talk later")
        assert result.get("intent_gate") == "exit"
        assert result.get("answer_source") == "system"
        assert "Ending the session" in result.get("natural_language_answer", "")
        assert not pipeline.classifier.execute.called

    @pytest.mark.asyncio
    async def test_intent_gate_exit_detects_never_mind(self, pipeline):
        result = await pipeline.run("never mind, i'll ask later")
        assert result.get("intent_gate") == "exit"
        assert result.get("answer_source") == "system"
        assert "Ending the session" in result.get("natural_language_answer", "")
        assert not pipeline.classifier.execute.called

    @pytest.mark.asyncio
    async def test_intent_gate_ambiguous_prompts_clarification(self, pipeline):
        pipeline.intent_llm = None
        result = await pipeline.run("ok")
        assert result.get("intent_gate") == "clarify"
        assert result.get("answer_source") == "clarification"
        assert result.get("clarifying_questions")
        assert "data" in result.get("natural_language_answer", "").lower()
        assert not pipeline.classifier.execute.called

    @pytest.mark.asyncio
    async def test_intent_gate_out_of_scope_short_circuits(self, pipeline):
        result = await pipeline.run("Tell me a joke")
        assert result.get("intent_gate") == "out_of_scope"
        assert result.get("answer_source") == "system"
        assert "connected data" in result.get("natural_language_answer", "").lower()
        assert not pipeline.classifier.execute.called

    @pytest.mark.asyncio
    async def test_intent_gate_fast_path_list_tables_handles_empty_investigation_memory(
        self, pipeline
    ):
        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql=(
                        "SELECT table_schema, table_name FROM information_schema.tables "
                        "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                    ),
                    explanation="Catalog table listing query.",
                    confidence=0.95,
                    used_datapoints=[],
                    assumptions=[],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=0),
            )
        )
        pipeline.validator.execute = AsyncMock(
            return_value=ValidatorAgentOutput(
                success=True,
                validated_sql=ValidatedSQL(
                    sql="SELECT table_schema, table_name FROM information_schema.tables",
                    is_valid=True,
                    is_safe=True,
                    errors=[],
                    warnings=[],
                    performance_score=0.9,
                ),
                metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
            )
        )
        pipeline.executor.execute = AsyncMock(
            return_value=ExecutorAgentOutput(
                success=True,
                executed_query=ExecutedQuery(
                    query_result=QueryResult(
                        rows=[{"table_schema": "public", "table_name": "sales"}],
                        row_count=1,
                        columns=["table_schema", "table_name"],
                        execution_time_ms=5.0,
                        was_truncated=False,
                    ),
                    natural_language_answer="Found 1 table.",
                    visualization_hint="table",
                    key_insights=[],
                    source_citations=[],
                ),
                metadata=AgentMetadata(agent_name="ExecutorAgent", llm_calls=0),
            )
        )

        result = await pipeline.run("list tables")

        assert result.get("error") is None
        assert result.get("intent_gate") == "data_query"
        assert result.get("fast_path") is True
        assert pipeline.sql.execute.called

    @pytest.mark.asyncio
    async def test_reply_after_clarification_continues_previous_goal(self, pipeline):
        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql="SELECT * FROM public.petra_campuses LIMIT 2",
                    explanation="Sample rows from selected replacement table.",
                    confidence=0.95,
                    used_datapoints=[],
                    assumptions=[],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=0),
            )
        )
        pipeline.validator.execute = AsyncMock(
            return_value=ValidatorAgentOutput(
                success=True,
                validated_sql=ValidatedSQL(
                    sql="SELECT * FROM public.petra_campuses LIMIT 2",
                    is_valid=True,
                    is_safe=True,
                    errors=[],
                    warnings=[],
                    performance_score=0.9,
                ),
                metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
            )
        )
        pipeline.executor.execute = AsyncMock(
            return_value=ExecutorAgentOutput(
                success=True,
                executed_query=ExecutedQuery(
                    query_result=QueryResult(
                        rows=[{"id": 1, "location": "Abuja"}],
                        row_count=1,
                        columns=["id", "location"],
                        execution_time_ms=3.0,
                        was_truncated=False,
                    ),
                    natural_language_answer="Returned 1 sample row.",
                    visualization_hint="table",
                    key_insights=[],
                    source_citations=[],
                ),
                metadata=AgentMetadata(agent_name="ExecutorAgent", llm_calls=0),
            )
        )

        history = [
            {"role": "user", "content": "show 2 rows in public.sales"},
            {
                "role": "assistant",
                "content": (
                    "I couldn't find public.sales in the connected database schema.\n"
                    "Clarifying questions:\n"
                    "- Which existing table should I use instead?"
                ),
            },
        ]

        result = await pipeline.run("petra_campuses", conversation_history=history)

        assert result.get("intent_gate") == "data_query"
        assert result.get("fast_path") is True
        assert result.get("original_query") == "petra_campuses"
        assert result.get("query") == "Show 2 rows from petra_campuses"
        assert pipeline.sql.execute.called
        sql_input = pipeline.sql.execute.call_args.args[0]
        assert sql_input.query == "Show 2 rows from petra_campuses"
        assert not pipeline.classifier.execute.called

    @pytest.mark.asyncio
    async def test_reply_after_clarification_can_switch_to_new_exit_intent(self, pipeline):
        pipeline.sql.execute = AsyncMock()
        history = [
            {"role": "user", "content": "show 2 rows in public.sales"},
            {
                "role": "assistant",
                "content": (
                    "I couldn't find public.sales in the connected database schema.\n"
                    "Clarifying questions:\n"
                    "- Which existing table should I use instead?"
                ),
            },
        ]

        result = await pipeline.run(
            "never mind, i'll ask later",
            conversation_history=history,
        )

        assert result.get("intent_gate") == "exit"
        assert result.get("answer_source") == "system"
        assert "Ending the session" in result.get("natural_language_answer", "")
        assert not pipeline.sql.execute.called
        assert not pipeline.classifier.execute.called

    def test_short_command_like_message_not_treated_as_followup_hint(self, pipeline):
        assert pipeline._is_short_followup("show columns") is False

    def test_clean_hint_handles_regarding_prefix(self, pipeline):
        hint = pipeline._clean_hint(
            'Regarding "Which table should I list columns for?": vbs_registrations'
        )
        assert hint == "vbs_registrations"

    def test_merge_query_with_table_hint_rewrites_explicit_table(self, pipeline):
        merged = pipeline._merge_query_with_table_hint(
            "show 2 rows in public.sales",
            "petra_campuses",
        )
        assert merged == "Show 2 rows from petra_campuses"

    @pytest.mark.asyncio
    async def test_low_confidence_semantic_sql_triggers_clarification(self, pipeline):
        pipeline.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="data_query",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=0),
            )
        )
        pipeline.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="Show revenue by campus this month",
                    datapoints=[
                        RetrievedDataPoint(
                            datapoint_id="table_orders_001",
                            datapoint_type="Schema",
                            name="Orders",
                            score=0.91,
                            source="vector",
                            metadata={"table_name": "public.orders"},
                        )
                    ],
                    retrieval_mode="hybrid",
                    total_retrieved=1,
                    sources_used=["table_orders_001"],
                ),
                context_confidence=0.1,
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )
        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql="SELECT SUM(amount) FROM public.orders",
                    explanation="Guessing revenue from orders amount.",
                    confidence=0.32,
                    used_datapoints=["table_orders_001"],
                    assumptions=["revenue maps to orders.amount"],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=1),
            )
        )
        pipeline.validator.execute = AsyncMock()
        pipeline.executor.execute = AsyncMock()

        result = await pipeline.run("Show revenue by campus this month")

        assert result.get("answer_source") == "clarification"
        assert result.get("clarifying_questions")
        assert "revenue" in result.get("clarifying_questions", [""])[0].lower()
        assert pipeline.validator.execute.call_count == 0
        assert pipeline.executor.execute.call_count == 0

    def test_normalize_answer_metadata_assigns_defaults(self, pipeline):
        state = {
            "natural_language_answer": "Found rows.",
            "validated_sql": "SELECT * FROM public.orders LIMIT 2",
            "answer_source": None,
            "answer_confidence": None,
        }

        pipeline._normalize_answer_metadata(state)

        assert state["answer_source"] == "sql"
        assert state["answer_confidence"] == 0.7

    def test_normalize_answer_metadata_treats_generated_sql_as_sql(self, pipeline):
        state = {
            "generated_sql": "SELECT * FROM public.orders LIMIT 2",
            "answer_source": None,
            "answer_confidence": None,
            "error": None,
        }

        pipeline._normalize_answer_metadata(state)

        assert state["answer_source"] == "sql"
        assert state["answer_confidence"] == 0.7


class TestStreaming:
    """Test streaming functionality."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever."""
        return AsyncMock()

    @pytest.fixture
    def mock_connector(self):
        """Mock connector."""
        connector = AsyncMock()
        connector.connect = AsyncMock()
        connector.close = AsyncMock()
        return connector

    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider."""
        provider = AsyncMock()
        provider.provider = "mock"
        provider.model = "mock-model"
        return provider

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_connector, mock_llm_provider, mock_openai_api_key):
        """Create pipeline."""
        pipeline = DataChatPipeline(
            retriever=mock_retriever,
            connector=mock_connector,
            max_retries=3,
        )
        # Inject mock LLM providers
        pipeline.classifier.llm = mock_llm_provider
        pipeline.sql.llm = mock_llm_provider
        pipeline.executor.llm = mock_llm_provider
        pipeline.tool_planner.execute = AsyncMock(
            return_value=ToolPlannerAgentOutput(
                success=True,
                plan=ToolPlan(tool_calls=[], rationale="No tools needed.", fallback="pipeline"),
                metadata=AgentMetadata(agent_name="ToolPlannerAgent", llm_calls=0),
            )
        )
        pipeline.response_synthesis.execute = AsyncMock(return_value="Result is 1")
        return pipeline

    @pytest.mark.asyncio
    async def test_streaming_emits_updates(self, pipeline):
        """Test that streaming emits status updates for each agent."""
        # Mock all agents
        pipeline.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="data_query",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )

        pipeline.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="test",
                    datapoints=[],
                    retrieval_mode="hybrid",
                    total_retrieved=0,
                    sources_used=[],
                ),
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )

        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql="SELECT 1",
                    explanation="Test",
                    confidence=0.9,
                    used_datapoints=[],
                    assumptions=[],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=1),
            )
        )

        pipeline.validator.execute = AsyncMock(
            return_value=ValidatorAgentOutput(
                success=True,
                validated_sql=ValidatedSQL(
                    sql="SELECT 1",
                    is_valid=True,
                    is_safe=True,
                    errors=[],
                    warnings=[],
                    performance_score=1.0,
                ),
                metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
            )
        )

        pipeline.executor.execute = AsyncMock(
            return_value=ExecutorAgentOutput(
                success=True,
                executed_query=ExecutedQuery(
                    query_result=QueryResult(
                        rows=[{"result": 1}],
                        row_count=1,
                        columns=["result"],
                        execution_time_ms=5.0,
                    ),
                    natural_language_answer="Result is 1",
                    visualization_hint="none",
                    key_insights=[],
                    source_citations=[],
                ),
                metadata=AgentMetadata(agent_name="ExecutorAgent", llm_calls=1),
            )
        )

        # Collect streaming updates
        updates = []
        async for update in pipeline.stream("test query"):
            updates.append(update)

        # Verify we got updates for each agent
        nodes = [u["node"] for u in updates]
        assert "classifier" in nodes
        assert "context" in nodes
        assert "sql" in nodes
        assert "validator" in nodes
        assert "executor" in nodes

        # Verify each update has required fields
        for update in updates:
            assert "node" in update
            assert "current_agent" in update
            assert "status" in update
            assert "state" in update


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever."""
        return AsyncMock()

    @pytest.fixture
    def mock_connector(self):
        """Mock connector."""
        connector = AsyncMock()
        connector.connect = AsyncMock()
        connector.close = AsyncMock()
        return connector

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_connector):
        """Create pipeline."""
        pipeline = DataChatPipeline(
            retriever=mock_retriever,
            connector=mock_connector,
            max_retries=2,
        )
        pipeline.tool_planner.execute = AsyncMock(
            return_value=ToolPlannerAgentOutput(
                success=True,
                plan=ToolPlan(tool_calls=[], rationale="No tools needed.", fallback="pipeline"),
                metadata=AgentMetadata(agent_name="ToolPlannerAgent", llm_calls=0),
            )
        )
        pipeline.response_synthesis.execute = AsyncMock(return_value="Found 1 result.")
        return pipeline

    @pytest.mark.asyncio
    async def test_classifier_error_is_captured(self, pipeline):
        """Test that errors in classifier are captured."""
        # Mock classifier to raise error
        pipeline.classifier.execute = AsyncMock(side_effect=Exception("Classification failed"))

        result = await pipeline.run("test query")

        # Verify error is captured
        assert result.get("error") is not None
        assert "Classification failed" in result["error"]

    @pytest.mark.asyncio
    async def test_error_handler_provides_graceful_message(self, pipeline):
        """Test that error handler provides user-friendly message."""
        # Mock all agents to succeed except executor
        pipeline.classifier.execute = AsyncMock(
            return_value=ClassifierAgentOutput(
                success=True,
                classification=QueryClassification(
                    intent="data_query",
                    entities=[],
                    complexity="simple",
                    clarification_needed=False,
                    clarifying_questions=[],
                    confidence=0.9,
                ),
                metadata=AgentMetadata(agent_name="ClassifierAgent", llm_calls=1),
            )
        )

        pipeline.context.execute = AsyncMock(
            return_value=ContextAgentOutput(
                success=True,
                data={},
                investigation_memory=InvestigationMemory(
                    query="test",
                    datapoints=[],
                    retrieval_mode="hybrid",
                    total_retrieved=0,
                    sources_used=[],
                ),
                metadata=AgentMetadata(agent_name="ContextAgent", llm_calls=0),
            )
        )

        pipeline.sql.execute = AsyncMock(
            return_value=SQLAgentOutput(
                success=True,
                generated_sql=GeneratedSQL(
                    sql="SELECT 1",
                    explanation="Test",
                    confidence=0.9,
                    used_datapoints=[],
                    assumptions=[],
                    clarifying_questions=[],
                ),
                metadata=AgentMetadata(agent_name="SQLAgent", llm_calls=1),
            )
        )

        # ValidatorAgent always fails to trigger max retries
        pipeline.validator.execute = AsyncMock(
            return_value=ValidatorAgentOutput(
                success=False,
                validated_sql=ValidatedSQL(
                    sql="SELECT 1",
                    is_valid=False,
                    is_safe=False,
                    errors=[
                        SQLValidationError(
                            error_type="other",
                            message="Error",
                        )
                    ],
                    warnings=[],
                    performance_score=0.0,
                ),
                metadata=AgentMetadata(agent_name="ValidatorAgent", llm_calls=0),
            )
        )

        result = await pipeline.run("test query")

        # Verify error message exists and is user-friendly
        assert result.get("natural_language_answer") is not None
        assert "error" in result["natural_language_answer"].lower()
