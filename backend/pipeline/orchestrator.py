"""
DataChat Pipeline Orchestrator

LangGraph-based pipeline that orchestrates all agents:
- ClassifierAgent → ContextAgent → SQLAgent → ValidatorAgent → ExecutorAgent
- Self-correction loop: ValidatorAgent can send back to SQLAgent (max 3 retries)
- Streaming support for real-time status updates
- Cost and latency tracking
- Error recovery and graceful degradation
"""

import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.agents.classifier import ClassifierAgent
from backend.agents.context import ContextAgent
from backend.agents.context_answer import ContextAnswerAgent
from backend.agents.executor import ExecutorAgent
from backend.agents.sql import SQLAgent
from backend.agents.validator import ValidatorAgent
from backend.config import get_settings
from backend.connectors.base import BaseConnector
from backend.connectors.clickhouse import ClickHouseConnector
from backend.connectors.postgres import PostgresConnector
from backend.knowledge.retriever import Retriever
from backend.models import (
    ClassifierAgentInput,
    ContextAnswerAgentInput,
    ContextAgentInput,
    ExecutorAgentInput,
    EvidenceItem,
    GeneratedSQL,
    Message,
    SQLAgentInput,
    ValidatorAgentInput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline State Schema
# ============================================================================


class PipelineState(TypedDict, total=False):
    """
    State schema for the DataChat pipeline.

    Tracks the complete flow through all agents with all intermediate outputs.
    """

    # Input
    query: str
    conversation_history: list[Message]
    database_type: str
    database_url: str | None

    # Classifier output
    intent: str | None
    entities: list[dict[str, Any]]
    complexity: str | None
    clarification_needed: bool
    clarifying_questions: list[str]

    # Context output
    investigation_memory: dict[str, Any] | None
    retrieved_datapoints: list[dict[str, Any]]
    context_confidence: float | None

    # SQL output
    generated_sql: str | None
    sql_explanation: str | None
    sql_confidence: float | None
    used_datapoints: list[str]
    assumptions: list[str]

    # Validator output
    validated_sql: str | None
    validation_passed: bool
    validation_errors: list[dict[str, Any]]
    validation_warnings: list[dict[str, Any]]
    performance_score: float | None
    retry_count: int
    retries_exhausted: bool
    validated_sql_object: Any
    is_safe: bool | None

    # Executor output
    query_result: dict[str, Any] | None
    natural_language_answer: str | None
    visualization_hint: str | None
    key_insights: list[str]
    answer_source: str | None
    answer_confidence: float | None
    evidence: list[dict[str, Any]]

    # Pipeline metadata
    current_agent: str | None
    error: str | None
    total_cost: float
    total_latency_ms: float
    agent_timings: dict[str, float]
    llm_calls: int


# ============================================================================
# DataChat Pipeline
# ============================================================================


class DataChatPipeline:
    """
    LangGraph-based pipeline orchestrating all DataChat agents.

    Flow:
        1. ClassifierAgent: Understand intent and extract entities
        2. ContextAgent: Retrieve relevant DataPoints
        3. SQLAgent: Generate SQL query
        4. ValidatorAgent: Validate SQL (retry loop if fails)
        5. ExecutorAgent: Execute query and format results

    Features:
        - Self-correction loop (max 3 retries)
        - Streaming status updates
        - Cost and latency tracking
        - Error recovery

    Usage:
        pipeline = DataChatPipeline(retriever, connector)
        result = await pipeline.run("What were sales last month?")

        # Or with streaming:
        async for update in pipeline.stream("Show revenue by region"):
            print(f"Agent: {update['current_agent']}, Status: {update['status']}")
    """

    def __init__(
        self,
        retriever: Retriever,
        connector: BaseConnector,
        max_retries: int = 3,
    ):
        """
        Initialize pipeline with dependencies.

        Args:
            retriever: Knowledge retriever for ContextAgent
            connector: Database connector for ExecutorAgent
            max_retries: Maximum SQL validation retry attempts
        """
        self.retriever = retriever
        self.connector = connector
        self.max_retries = max_retries
        self.config = get_settings()

        # Initialize agents
        self.classifier = ClassifierAgent()
        self.context = ContextAgent(retriever=retriever)
        self.context_answer = ContextAnswerAgent()
        self.sql = SQLAgent()
        self.validator = ValidatorAgent()
        self.executor = ExecutorAgent()

        # Build LangGraph
        self.graph = self._build_graph()

        logger.info("DataChatPipeline initialized")

    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph state machine.

        Returns:
            Compiled LangGraph
        """
        workflow = StateGraph(PipelineState)

        # Add nodes
        workflow.add_node("classifier", self._run_classifier)
        workflow.add_node("context", self._run_context)
        workflow.add_node("context_answer", self._run_context_answer)
        workflow.add_node("sql", self._run_sql)
        workflow.add_node("validator", self._run_validator)
        workflow.add_node("executor", self._run_executor)
        workflow.add_node("error_handler", self._handle_error)

        # Set entry point
        workflow.set_entry_point("classifier")

        # Add edges
        workflow.add_edge("classifier", "context")
        workflow.add_conditional_edges(
            "context",
            self._should_use_context_answer,
            {
                "context": "context_answer",
                "sql": "sql",
            },
        )

        # Conditional edge from validator
        workflow.add_conditional_edges(
            "validator",
            self._should_retry_sql,
            {
                "retry": "sql",  # Retry SQL generation
                "execute": "executor",  # Proceed to execution
                "error": "error_handler",  # Max retries exceeded
            },
        )

        workflow.add_edge("sql", "validator")
        workflow.add_edge("executor", END)
        workflow.add_edge("context_answer", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile()

    # ========================================================================
    # Agent Execution Methods
    # ========================================================================

    async def _run_classifier(self, state: PipelineState) -> PipelineState:
        """Run ClassifierAgent."""
        start_time = time.time()
        state["current_agent"] = "ClassifierAgent"

        try:
            input_data = ClassifierAgentInput(
                query=state["query"],
                conversation_history=state.get("conversation_history", []),
            )

            output = await self.classifier.execute(input_data)

            # Update state
            state["intent"] = output.classification.intent
            state["entities"] = [
                {
                    "entity_type": e.entity_type,
                    "value": e.value,
                    "confidence": e.confidence,
                    "normalized_value": e.normalized_value,
                }
                for e in output.classification.entities
            ]
            state["complexity"] = output.classification.complexity
            state["clarification_needed"] = output.classification.clarification_needed
            state["clarifying_questions"] = output.classification.clarifying_questions

            # Update metadata
            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["classifier"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
            state["llm_calls"] = state.get("llm_calls", 0) + output.metadata.llm_calls

            logger.info(
                f"ClassifierAgent complete: intent={state['intent']}, complexity={state['complexity']}"
            )

        except Exception as e:
            logger.error(f"ClassifierAgent failed: {e}")
            state["error"] = f"Classification failed: {e}"

        return state

    async def _run_context(self, state: PipelineState) -> PipelineState:
        """Run ContextAgent."""
        start_time = time.time()
        state["current_agent"] = "ContextAgent"

        try:
            # Convert entities dict to ExtractedEntity objects
            from backend.models import ExtractedEntity

            entities = [ExtractedEntity(**e) for e in state.get("entities", [])]

            input_data = ContextAgentInput(
                query=state["query"],
                conversation_history=state.get("conversation_history", []),
                entities=entities,
                retrieval_mode="hybrid",
                max_datapoints=10,
            )

            output = await self.context.execute(input_data)

            # Update state
            state["investigation_memory"] = {
                "query": output.investigation_memory.query,
                "datapoints": [
                    {
                        "datapoint_id": dp.datapoint_id,
                        "datapoint_type": dp.datapoint_type,
                        "name": dp.name,
                        "score": dp.score,
                        "source": dp.source,
                        "metadata": dp.metadata,
                    }
                    for dp in output.investigation_memory.datapoints
                ],
                "total_retrieved": output.investigation_memory.total_retrieved,
                "retrieval_mode": output.investigation_memory.retrieval_mode,
                "sources_used": output.investigation_memory.sources_used,
            }
            state["retrieved_datapoints"] = state["investigation_memory"]["datapoints"]
            state["context_confidence"] = output.context_confidence

            # Update metadata
            elapsed = (time.time() - start_time) * 1000
            state["agent_timings"]["context"] = elapsed
            state["total_latency_ms"] += elapsed

            logger.info(
                f"ContextAgent complete: retrieved {len(state['retrieved_datapoints'])} datapoints"
            )

        except Exception as e:
            logger.error(f"ContextAgent failed: {e}")
            state["error"] = f"Context retrieval failed: {e}"

        return state

    async def _run_context_answer(self, state: PipelineState) -> PipelineState:
        """Run ContextAnswerAgent."""
        start_time = time.time()
        state["current_agent"] = "ContextAnswerAgent"

        if state.get("error"):
            return state

        try:
            from backend.models import InvestigationMemory, RetrievedDataPoint

            datapoints = [
                RetrievedDataPoint(
                    datapoint_id=dp["datapoint_id"],
                    datapoint_type=dp["datapoint_type"],
                    name=dp["name"],
                    score=dp["score"],
                    source=dp["source"],
                    metadata=dp["metadata"],
                )
                for dp in state.get("retrieved_datapoints", [])
            ]

            investigation_memory = InvestigationMemory(
                query=state["query"],
                datapoints=datapoints,
                total_retrieved=state.get("investigation_memory", {}).get(
                    "total_retrieved", len(datapoints)
                ),
                retrieval_mode=state.get("investigation_memory", {}).get(
                    "retrieval_mode", "hybrid"
                ),
                sources_used=state.get("investigation_memory", {}).get("sources_used", []),
            )

            input_data = ContextAnswerAgentInput(
                query=state["query"],
                conversation_history=state.get("conversation_history", []),
                investigation_memory=investigation_memory,
                intent=state.get("intent"),
                context_confidence=state.get("context_confidence"),
            )

            output = await self.context_answer.execute(input_data)

            state["natural_language_answer"] = output.context_answer.answer
            state["answer_source"] = "context"
            state["answer_confidence"] = output.context_answer.confidence
            state["evidence"] = [
                evidence.model_dump()
                for evidence in output.context_answer.evidence
            ]
            state["generated_sql"] = None
            state["validated_sql"] = None
            state["query_result"] = None
            state["visualization_hint"] = None
            state["clarification_needed"] = bool(output.context_answer.clarifying_questions)
            state["clarifying_questions"] = output.context_answer.clarifying_questions

            elapsed = (time.time() - start_time) * 1000
            state["agent_timings"]["context_answer"] = elapsed
            state["total_latency_ms"] += elapsed
            state["llm_calls"] += output.metadata.llm_calls

            logger.info("ContextAnswerAgent complete")

        except Exception as e:
            logger.error(f"ContextAnswerAgent failed: {e}")
            state["error"] = f"Context answer failed: {e}"

        return state

    async def _run_sql(self, state: PipelineState) -> PipelineState:
        """Run SQLAgent."""
        start_time = time.time()
        state["current_agent"] = "SQLAgent"

        if state.get("error"):
            return state

        try:
            # Reconstruct InvestigationMemory from state
            from backend.models import InvestigationMemory, RetrievedDataPoint

            datapoints = [
                RetrievedDataPoint(
                    datapoint_id=dp["datapoint_id"],
                    datapoint_type=dp["datapoint_type"],
                    name=dp["name"],
                    score=dp["score"],
                    source=dp["source"],
                    metadata=dp["metadata"],
                )
                for dp in state.get("retrieved_datapoints", [])
            ]

            investigation_memory = InvestigationMemory(
                query=state["query"],
                datapoints=datapoints,
                total_retrieved=state.get("investigation_memory", {}).get(
                    "total_retrieved", len(datapoints)
                ),
                retrieval_mode=state.get("investigation_memory", {}).get(
                    "retrieval_mode", "hybrid"
                ),
                sources_used=state.get("investigation_memory", {}).get(
                    "sources_used",
                    list({dp["source"] for dp in state.get("retrieved_datapoints", [])}),
                ),
            )

            input_data = SQLAgentInput(
                query=state["query"],
                conversation_history=state.get("conversation_history", []),
                investigation_memory=investigation_memory,
                database_type=state.get("database_type", "postgresql"),
            )

            output = await self.sql.execute(input_data)

            # Update state
            state["generated_sql"] = output.generated_sql.sql
            state["sql_explanation"] = output.generated_sql.explanation
            state["sql_confidence"] = output.generated_sql.confidence
            state["used_datapoints"] = getattr(
                output.generated_sql,
                "used_datapoints",
                getattr(output.generated_sql, "used_datapoint_ids", []),
            )
            state["assumptions"] = output.generated_sql.assumptions

            # Update metadata
            elapsed = (time.time() - start_time) * 1000
            state["agent_timings"]["sql"] = elapsed
            state["total_latency_ms"] += elapsed
            state["llm_calls"] += output.metadata.llm_calls

            retry_info = (
                f" (retry {state.get('retry_count', 0)})" if state.get("retry_count", 0) > 0 else ""
            )
            logger.info(f"SQLAgent complete{retry_info}: confidence={state['sql_confidence']:.2f}")

        except Exception as e:
            logger.error(f"SQLAgent failed: {e}")
            state["error"] = f"SQL generation failed: {e}"

        return state

    async def _run_validator(self, state: PipelineState) -> PipelineState:
        """Run ValidatorAgent."""
        start_time = time.time()
        state["current_agent"] = "ValidatorAgent"

        if state.get("error"):
            return state

        try:
            if not state.get("generated_sql"):
                raise ValueError("Missing generated SQL for validation")

            generated_sql = GeneratedSQL(
                sql=state["generated_sql"],
                explanation=state.get("sql_explanation", ""),
                used_datapoints=state.get("used_datapoints", []),
                confidence=state.get("sql_confidence", 0.0),
                assumptions=state.get("assumptions", []),
                clarifying_questions=state.get("clarifying_questions", []),
            )

            input_data = ValidatorAgentInput(
                query=state["query"],
                conversation_history=state.get("conversation_history", []),
                generated_sql=generated_sql,
                target_database=state.get("database_type", "postgresql"),
                strict_mode=False,  # Allow warnings
            )

            output = await self.validator.execute(input_data)

            # Update state
            state["validated_sql_object"] = output.validated_sql
            state["validated_sql"] = output.validated_sql.sql if output.validated_sql else None
            state["validation_passed"] = (
                output.validated_sql.is_valid if output.validated_sql else output.success
            )
            state["is_safe"] = output.validated_sql.is_safe if output.validated_sql else None
            state["validation_errors"] = (
                [
                    {
                        "message": e.message,
                        "error_type": getattr(e, "error_type", None),
                        "severity": getattr(e, "severity", None),
                        "location": getattr(e, "location", None),
                    }
                    for e in output.validated_sql.errors
                ]
                if hasattr(output.validated_sql, "errors")
                else []
            )
            state["validation_warnings"] = (
                [
                    {
                        "message": w.message,
                        "warning_type": getattr(w, "warning_type", None),
                        "suggestion": getattr(w, "suggestion", None),
                    }
                    for w in output.validated_sql.warnings
                ]
                if hasattr(output.validated_sql, "warnings")
                else []
            )
            state["performance_score"] = (
                output.validated_sql.performance_score
                if hasattr(output.validated_sql, "performance_score")
                else None
            )

            # Update metadata
            elapsed = (time.time() - start_time) * 1000
            state["agent_timings"]["validator"] = elapsed
            state["total_latency_ms"] += elapsed

            if state["validation_passed"]:
                logger.info(
                    f"ValidatorAgent complete: PASSED (warnings: {len(state['validation_warnings'])})"
                )
            else:
                next_retry = state.get("retry_count", 0) + 1
                if next_retry > self.max_retries:
                    state["retry_count"] = self.max_retries
                    state["retries_exhausted"] = True
                    state["error"] = (
                        f"Failed to generate valid SQL after {self.max_retries} attempts"
                    )
                else:
                    state["retry_count"] = next_retry
                    state["retries_exhausted"] = False
                logger.warning(
                    f"ValidatorAgent complete: FAILED (errors: {len(state['validation_errors'])})"
                )

        except Exception as e:
            logger.error(f"ValidatorAgent failed: {e}")
            state["error"] = f"Validation failed: {e}"
            state["validation_passed"] = False

        return state

    async def _run_executor(self, state: PipelineState) -> PipelineState:
        """Run ExecutorAgent."""
        start_time = time.time()
        state["current_agent"] = "ExecutorAgent"

        if state.get("error"):
            return state

        try:
            from backend.models import ValidatedSQL

            validated_sql = state.get("validated_sql_object")
            if validated_sql is None:
                validated_sql = ValidatedSQL(
                    sql=state["validated_sql"],
                    is_valid=True,
                    errors=[],
                    warnings=[],
                    suggestions=[],
                    is_safe=state.get("is_safe", True),
                    performance_score=state.get("performance_score", 1.0),
                )

            input_data = ExecutorAgentInput(
                query=state["query"],
                conversation_history=state.get("conversation_history", []),
                validated_sql=validated_sql,
                database_type=state.get("database_type", "postgresql"),
                database_url=state.get("database_url"),
                max_rows=1000,
                timeout_seconds=30,
                source_datapoints=state.get("used_datapoints", []),
            )

            output = await self.executor.execute(input_data)

            # Update state
            state["query_result"] = {
                "rows": output.executed_query.query_result.rows,
                "row_count": output.executed_query.query_result.row_count,
                "columns": output.executed_query.query_result.columns,
                "execution_time_ms": output.executed_query.query_result.execution_time_ms,
                "was_truncated": output.executed_query.query_result.was_truncated,
            }
            state["natural_language_answer"] = output.executed_query.natural_language_answer
            state["visualization_hint"] = output.executed_query.visualization_hint
            state["key_insights"] = output.executed_query.key_insights
            state["answer_source"] = "sql"
            state["answer_confidence"] = state.get("sql_confidence", 0.7)
            state["evidence"] = [
                evidence.model_dump()
                for evidence in self._build_evidence_items(state)
            ]

            # Update metadata
            elapsed = (time.time() - start_time) * 1000
            state["agent_timings"]["executor"] = elapsed
            state["total_latency_ms"] += elapsed
            state["llm_calls"] += output.metadata.llm_calls

            logger.info(
                f"ExecutorAgent complete: {state['query_result']['row_count']} rows, "
                f"viz={state['visualization_hint']}"
            )

        except Exception as e:
            logger.error(f"ExecutorAgent failed: {e}")
            state["error"] = f"Execution failed: {e}"
            if not state.get("natural_language_answer"):
                state["natural_language_answer"] = (
                    f"I encountered an error while processing your query: {state.get('error')}. "
                    "Please try rephrasing your question or contact support if the issue persists."
                )

        return state

    async def _handle_error(self, state: PipelineState) -> PipelineState:
        """Handle pipeline errors."""
        state["current_agent"] = "ErrorHandler"
        state["answer_source"] = "error"
        logger.error(f"Pipeline error: {state.get('error', 'Unknown error')}")

        # Provide graceful error message
        if not state.get("natural_language_answer"):
            state["natural_language_answer"] = (
                f"I encountered an error while processing your query: {state.get('error')}. "
                "Please try rephrasing your question or contact support if the issue persists."
            )

        return state

    # ========================================================================
    # Conditional Edge Logic
    # ========================================================================

    def _should_use_context_answer(self, state: PipelineState) -> str:
        if state.get("error"):
            return "sql"

        intent = state.get("intent") or "data_query"
        confidence = state.get("context_confidence") or 0.0
        query = (state.get("query") or "").lower()

        if intent in ("exploration", "explanation", "meta"):
            return "context"

        if self._query_requires_sql(query):
            return "sql"

        if confidence >= 0.7:
            return "context"

        return "sql"

    def _query_requires_sql(self, query: str) -> bool:
        keywords = (
            "total",
            "sum",
            "count",
            "average",
            "avg",
            "min",
            "max",
            "trend",
            "by",
            "per",
            "over time",
            "last",
            "this month",
            "this year",
            "yesterday",
        )
        return any(keyword in query for keyword in keywords)

    def _build_evidence_items(self, state: PipelineState) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        datapoints = {dp.get("datapoint_id"): dp for dp in state.get("retrieved_datapoints", [])}
        for datapoint_id in state.get("used_datapoints", []):
            dp = datapoints.get(datapoint_id, {})
            evidence.append(
                EvidenceItem(
                    datapoint_id=datapoint_id,
                    name=dp.get("name"),
                    type=dp.get("datapoint_type", dp.get("type")),
                    reason="Used for SQL generation",
                )
            )
        return evidence

    def _should_retry_sql(self, state: PipelineState) -> str:
        """
        Determine if SQL should be retried or execution should proceed.

        Returns:
            "retry": Retry SQL generation
            "execute": Proceed to execution
            "error": Max retries exceeded
        """
        if state.get("error"):
            return "error"

        if state.get("validation_passed"):
            return "execute"

        if state.get("retries_exhausted"):
            logger.error(f"Max retries ({self.max_retries}) exceeded")
            return "error"

        retry_count = state.get("retry_count", 0)
        if retry_count <= self.max_retries:
            logger.info(f"Retrying SQL generation (attempt {retry_count}/{self.max_retries})")
            return "retry"

        return "error"

    # ========================================================================
    # Public API
    # ========================================================================

    async def run(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        database_type: str = "postgresql",
        database_url: str | None = None,
    ) -> PipelineState:
        """
        Run pipeline synchronously (wait for completion).

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages
            database_type: Database type (postgresql, clickhouse, mysql)
            database_url: Database URL override for execution

        Returns:
            Final pipeline state with all outputs
        """
        initial_state: PipelineState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "database_type": database_type,
            "database_url": database_url,
            "current_agent": None,
            "error": None,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "agent_timings": {},
            "llm_calls": 0,
            "retry_count": 0,
            "retries_exhausted": False,
            "clarification_needed": False,
            "clarifying_questions": [],
            "entities": [],
            "validation_passed": False,
            "validation_errors": [],
            "validation_warnings": [],
            "key_insights": [],
            "used_datapoints": [],
            "assumptions": [],
            "investigation_memory": None,
            "retrieved_datapoints": [],
            "context_confidence": None,
            "answer_source": None,
            "answer_confidence": None,
            "evidence": [],
        }

        logger.info(f"Starting pipeline for query: {query[:100]}...")
        start_time = time.time()

        # Run graph
        result = await self.graph.ainvoke(initial_state)

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"Pipeline complete in {total_time:.1f}ms ({result.get('llm_calls', 0)} LLM calls)"
        )

        return result

    async def stream(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        database_type: str = "postgresql",
        database_url: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run pipeline with streaming updates.

        Yields status updates as each agent completes.

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages
            database_type: Database type
            database_url: Database URL override for execution

        Yields:
            Status updates with current agent and progress
        """
        initial_state: PipelineState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "database_type": database_type,
            "database_url": database_url,
            "current_agent": None,
            "error": None,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "agent_timings": {},
            "llm_calls": 0,
            "retry_count": 0,
            "retries_exhausted": False,
            "clarification_needed": False,
            "clarifying_questions": [],
            "entities": [],
            "validation_passed": False,
            "validation_errors": [],
            "validation_warnings": [],
            "key_insights": [],
            "used_datapoints": [],
            "assumptions": [],
            "investigation_memory": None,
            "retrieved_datapoints": [],
            "context_confidence": None,
            "answer_source": None,
            "answer_confidence": None,
            "evidence": [],
        }

        logger.info(f"Starting streaming pipeline for query: {query[:100]}...")

        # Stream graph execution
        async for update in self.graph.astream(initial_state):
            # Extract current state from update
            for node_name, state_update in update.items():
                yield {
                    "node": node_name,
                    "current_agent": state_update.get("current_agent"),
                    "status": "running",
                    "state": state_update,
                }

        logger.info("Pipeline streaming complete")

    async def run_with_streaming(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        database_type: str = "postgresql",
        database_url: str | None = None,
        event_callback: Any = None,
    ) -> PipelineState:
        """
        Run pipeline with callback-based streaming for WebSocket support.

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages
            database_type: Database type
            database_url: Database URL override for execution
            event_callback: Async callback function for streaming events
                           Signature: async def callback(event_type: str, event_data: dict)

        Returns:
            Final pipeline state with all outputs

        Event Types:
            - agent_start: Agent begins execution
            - agent_complete: Agent finishes execution
            - data_chunk: Intermediate data from agent (optional)
        """
        from datetime import datetime

        initial_state: PipelineState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "database_type": database_type,
            "database_url": database_url,
            "current_agent": None,
            "error": None,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "agent_timings": {},
            "llm_calls": 0,
            "retry_count": 0,
            "retries_exhausted": False,
            "clarification_needed": False,
            "clarifying_questions": [],
            "entities": [],
            "validation_passed": False,
            "validation_errors": [],
            "validation_warnings": [],
            "key_insights": [],
            "used_datapoints": [],
            "assumptions": [],
            "investigation_memory": None,
            "retrieved_datapoints": [],
            "context_confidence": None,
            "answer_source": None,
            "answer_confidence": None,
            "evidence": [],
        }

        logger.info(f"Starting streaming pipeline for query: {query[:100]}...")
        pipeline_start = time.time()

        # Stream graph execution and emit events
        agent_start_times: dict[str, float] = {}
        final_state: PipelineState | None = None

        async for update in self.graph.astream(initial_state):
            for _node_name, state_update in update.items():
                current_agent = state_update.get("current_agent")

                # Agent start event
                if current_agent and current_agent not in agent_start_times:
                    agent_start_times[current_agent] = time.time()
                    if event_callback:
                        await event_callback(
                            "agent_start",
                            {
                                "agent": current_agent,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )

                # Agent complete event
                if current_agent and current_agent in agent_start_times:
                    duration_ms = (time.time() - agent_start_times[current_agent]) * 1000
                    if event_callback:
                        # Extract relevant data for this agent
                        agent_data = {}
                        if current_agent == "ClassifierAgent":
                            agent_data = {
                                "intent": state_update.get("intent"),
                                "entities": state_update.get("entities", []),
                                "complexity": state_update.get("complexity"),
                            }
                        elif current_agent == "ContextAgent":
                            agent_data = {
                                "datapoints_found": len(
                                    state_update.get("retrieved_datapoints", [])
                                ),
                            }
                        elif current_agent == "SQLAgent":
                            agent_data = {
                                "sql_generated": bool(state_update.get("generated_sql")),
                                "confidence": state_update.get("sql_confidence"),
                            }
                        elif current_agent == "ValidatorAgent":
                            agent_data = {
                                "validation_passed": state_update.get("validation_passed", False),
                                "issues_found": len(state_update.get("validation_errors", [])),
                            }
                        elif current_agent == "ExecutorAgent":
                            query_result = state_update.get("query_result")
                            agent_data = {
                                "rows_returned": (
                                    query_result.get("row_count", 0) if query_result else 0
                                ),
                                "visualization_hint": state_update.get("visualization_hint"),
                            }
                        elif current_agent == "ContextAnswerAgent":
                            agent_data = {
                                "answer_source": state_update.get("answer_source"),
                                "confidence": state_update.get("answer_confidence"),
                                "evidence_count": len(state_update.get("evidence", [])),
                            }

                        await event_callback(
                            "agent_complete",
                            {
                                "agent": current_agent,
                                "data": agent_data,
                                "duration_ms": duration_ms,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )

                final_state = state_update

        # Calculate total latency
        total_latency_ms = (time.time() - pipeline_start) * 1000
        if final_state:
            final_state["total_latency_ms"] = total_latency_ms

        logger.info(
            f"Pipeline streaming complete in {total_latency_ms:.1f}ms "
            f"({final_state.get('llm_calls', 0) if final_state else 0} LLM calls)"
        )

        return final_state or initial_state


# ============================================================================
# Helper Functions
# ============================================================================


async def create_pipeline(
    database_type: str = "postgresql",
    database_url: str | None = None,
) -> DataChatPipeline:
    """
    Create a DataChatPipeline with all dependencies initialized.

    Args:
        database_type: Database type (postgresql, clickhouse)
        database_url: Database connection URL (uses config if not provided)

    Returns:
        Initialized pipeline
    """
    config = get_settings()

    # Initialize retriever
    from backend.knowledge.graph import KnowledgeGraph
    from backend.knowledge.vectors import VectorStore

    vector_store = VectorStore()
    await vector_store.initialize()

    knowledge_graph = KnowledgeGraph()

    retriever = Retriever(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    # Initialize connector
    db_url = database_url or config.database.url
    if not db_url:
        raise ValueError("DATABASE_URL must be set or provided to create a pipeline.")

    if database_type == "postgresql":
        # Parse PostgreSQL URL: postgresql://user:password@host:port/database
        from urllib.parse import urlparse

        # Convert Pydantic URL to string if needed
        db_url_str = str(db_url) if not isinstance(db_url, str) else db_url
        parsed = urlparse(db_url_str)
        connector = PostgresConnector(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "datachat",
            user=parsed.username or "postgres",
            password=parsed.password or "",
        )
    elif database_type == "clickhouse":
        connector = ClickHouseConnector(db_url)
    else:
        raise ValueError(f"Unsupported database type: {database_type}")

    await connector.connect()

    # Create pipeline
    pipeline = DataChatPipeline(
        retriever=retriever,
        connector=connector,
        max_retries=3,
    )

    return pipeline
