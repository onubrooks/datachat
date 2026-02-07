"""
DataChat Pipeline Orchestrator

LangGraph-based pipeline that orchestrates all agents:
- ClassifierAgent → ContextAgent → SQLAgent → ValidatorAgent → ExecutorAgent
- Self-correction loop: ValidatorAgent can send back to SQLAgent (max 3 retries)
- Streaming support for real-time status updates
- Cost and latency tracking
- Error recovery and graceful degradation
"""

import json
import logging
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, StateGraph

from backend.agents.classifier import ClassifierAgent
from backend.agents.context import ContextAgent
from backend.agents.context_answer import ContextAnswerAgent
from backend.agents.executor import ExecutorAgent
from backend.agents.response_synthesis import ResponseSynthesisAgent
from backend.agents.sql import SQLAgent
from backend.agents.tool_planner import ToolPlannerAgent
from backend.agents.validator import ValidatorAgent
from backend.config import get_settings
from backend.connectors.base import BaseConnector
from backend.connectors.clickhouse import ClickHouseConnector
from backend.connectors.postgres import PostgresConnector
from backend.knowledge.retriever import Retriever
from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMMessage, LLMRequest
from backend.models import (
    ClassifierAgentInput,
    ContextAgentInput,
    ContextAnswerAgentInput,
    EvidenceItem,
    ExecutorAgentInput,
    GeneratedSQL,
    Message,
    SQLAgentInput,
    ToolPlannerAgentInput,
    ValidatorAgentInput,
)
from backend.tools import ToolExecutor, initialize_tools
from backend.tools.base import ToolContext
from backend.tools.policy import ToolPolicyError
from backend.tools.registry import ToolRegistry

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
    original_query: str | None
    conversation_history: list[Message]
    database_type: str
    database_url: str | None
    user_id: str | None
    correlation_id: str | None
    tool_approved: bool
    intent_gate: str | None
    intent_summary: dict[str, Any] | None
    clarification_turn_count: int
    clarification_limit: int
    fast_path: bool
    skip_response_synthesis: bool

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
    context_needs_sql: bool | None
    context_preface: str | None
    context_evidence: list[dict[str, Any]]

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

    # Tool planner/executor output
    tool_plan: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_error: str | None
    tool_used: bool
    tool_approval_required: bool
    tool_approval_message: str | None
    tool_approval_calls: list[dict[str, Any]]

    # Pipeline metadata
    current_agent: str | None
    error: str | None
    total_cost: float
    total_latency_ms: float
    agent_timings: dict[str, float]
    llm_calls: int
    schema_refresh_attempted: bool


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
        self.max_clarifications = 3
        self.config = get_settings()

        # Initialize agents
        self.classifier = ClassifierAgent()
        self.context = ContextAgent(retriever=retriever)
        self.context_answer = ContextAnswerAgent()
        self.sql = SQLAgent()
        self.validator = ValidatorAgent()
        self.executor = ExecutorAgent()
        self.response_synthesis = ResponseSynthesisAgent()
        self.tool_planner = ToolPlannerAgent()
        self.tool_executor = ToolExecutor()
        try:
            self.intent_llm = LLMProviderFactory.create_default_provider(
                self.config.llm, model_type="mini"
            )
        except Exception as exc:
            logger.warning(f"Intent LLM disabled: {exc}")
            self.intent_llm = None
        self.tooling_enabled = self.config.tools.enabled
        self.tool_planner_enabled = self.config.tools.planner_enabled
        initialize_tools(self.config.tools.policy_path)

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
        workflow.add_node("intent_gate", self._run_intent_gate)
        workflow.add_node("tool_planner", self._run_tool_planner)
        workflow.add_node("tool_executor", self._run_tool_executor)
        workflow.add_node("classifier", self._run_classifier)
        workflow.add_node("context", self._run_context)
        workflow.add_node("context_answer", self._run_context_answer)
        workflow.add_node("sql", self._run_sql)
        workflow.add_node("validator", self._run_validator)
        workflow.add_node("executor", self._run_executor)
        workflow.add_node("response_synthesis", self._run_response_synthesis)
        workflow.add_node("error_handler", self._handle_error)

        # Set entry point
        workflow.set_entry_point("intent_gate")

        workflow.add_conditional_edges(
            "intent_gate",
            self._should_continue_after_intent_gate,
            {
                "end": END,
                "sql": "sql",
                "tool_planner": "tool_planner",
                "classifier": "classifier",
            },
        )

        workflow.add_conditional_edges(
            "tool_planner",
            self._should_use_tools,
            {
                "tools": "tool_executor",
                "pipeline": "classifier",
            },
        )
        workflow.add_conditional_edges(
            "tool_executor",
            self._should_continue_after_tool_execution,
            {
                "end": END,
                "pipeline": "classifier",
            },
        )

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
        workflow.add_conditional_edges(
            "context_answer",
            self._should_execute_after_context_answer,
            {
                "sql": "sql",
                "end": END,
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

        workflow.add_conditional_edges(
            "sql",
            self._should_validate_sql,
            {
                "validate": "validator",
                "clarify": END,
            },
        )
        workflow.add_conditional_edges(
            "executor",
            self._should_synthesize_response,
            {
                "synthesize": "response_synthesis",
                "end": END,
            },
        )
        workflow.add_edge("response_synthesis", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile()

    # ========================================================================
    # Agent Execution Methods
    # ========================================================================

    async def _run_intent_gate(self, state: PipelineState) -> PipelineState:
        """Run a lightweight intent gate before the main pipeline."""
        start_time = time.time()
        state["current_agent"] = "IntentGate"
        state["clarification_limit"] = self.max_clarifications
        state["clarification_turn_count"] = max(
            state.get("clarification_turn_count", 0),
            self._current_clarification_count(state),
        )
        state.setdefault("fast_path", False)
        state.setdefault("skip_response_synthesis", False)

        query = state.get("query") or ""
        summary = self._build_intent_summary(
            query, state.get("conversation_history", [])
        )
        state["intent_summary"] = summary

        resolved_query = summary.get("resolved_query")
        if resolved_query and resolved_query != query:
            state["original_query"] = query
            state["query"] = resolved_query

        fast_path = self._is_deterministic_sql_query(state.get("query") or "")
        if fast_path:
            state["intent"] = "data_query"
            state["intent_gate"] = "data_query"
            state["fast_path"] = True
            state["skip_response_synthesis"] = True
            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["intent_gate"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
            return state

        intent_gate = self._classify_intent_gate(state.get("query") or "")
        if intent_gate == "clarify":
            self._apply_clarification_response(
                state,
                ["What would you like to do with your data?"],
                default_intro=(
                    "I can help with questions about your connected data, but I need a bit more detail."
                ),
            )
            state["intent_gate"] = "clarify"
            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["intent_gate"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
            return state

        if intent_gate == "data_query" and self._should_call_intent_llm(state, summary):
            llm_result, llm_calls = await self._llm_intent_gate(
                state.get("query") or "", summary
            )
            if llm_calls:
                state["llm_calls"] = state.get("llm_calls", 0) + llm_calls
            if llm_result:
                intent_gate = llm_result.get("intent", intent_gate)
                confidence = llm_result.get("confidence", 0.0)
                question = llm_result.get("clarifying_question")
                if intent_gate == "clarify" or (
                    intent_gate == "data_query"
                    and confidence < 0.45
                    and self._is_ambiguous_intent(state, summary)
                ):
                    question = question or "What would you like to do with your data?"
                    self._apply_clarification_response(
                        state,
                        [question],
                        default_intro=(
                            "I can help with questions about your connected data, but I need a bit more detail."
                        ),
                    )
                    state["intent_gate"] = "clarify"
                    elapsed = (time.time() - start_time) * 1000
                    state.setdefault("agent_timings", {})["intent_gate"] = elapsed
                    state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
                    return state
                if (
                    intent_gate == "data_query"
                    and self._is_non_actionable_utterance(state.get("query") or "")
                ):
                    self._apply_clarification_response(
                        state,
                        ["What would you like to do with your data?"],
                        default_intro=(
                            "I can help with questions about your connected data, but I need a bit more detail."
                        ),
                    )
                    state["intent_gate"] = "clarify"
                    elapsed = (time.time() - start_time) * 1000
                    state.setdefault("agent_timings", {})["intent_gate"] = elapsed
                    state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
                    return state

        if intent_gate == "data_query" and self._is_ambiguous_intent(state, summary):
            questions = summary.get("last_clarifying_questions") or []
            question = questions[0] if questions else "What would you like to do with your data?"
            self._apply_clarification_response(
                state,
                [question],
                default_intro=(
                    "I can help with questions about your connected data, but I need a bit more detail."
                ),
            )
            state["intent_gate"] = "clarify"
            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["intent_gate"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
            return state
        state["intent_gate"] = intent_gate

        if intent_gate in {"exit", "out_of_scope", "small_talk", "setup_help"}:
            state["intent"] = intent_gate
            state["answer_source"] = "system"
            state["answer_confidence"] = 0.8
            state["natural_language_answer"] = self._build_intent_gate_response(
                intent_gate
            )
            state["clarification_needed"] = False
            state["clarifying_questions"] = []

        elapsed = (time.time() - start_time) * 1000
        state.setdefault("agent_timings", {})["intent_gate"] = elapsed
        state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed

        return state

    async def _run_tool_planner(self, state: PipelineState) -> PipelineState:
        """Run ToolPlannerAgent."""
        start_time = time.time()
        state["current_agent"] = "ToolPlannerAgent"

        try:
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category.value,
                    "parameters_schema": tool.parameters_schema,
                }
                for tool in ToolRegistry.list_definitions()
            ]
            output = await self.tool_planner.execute(
                ToolPlannerAgentInput(
                    query=state["query"],
                    conversation_history=self._augment_history_with_summary(state),
                    available_tools=tools,
                )
            )
            plan = output.plan
            state["tool_plan"] = plan.model_dump()
            state["tool_calls"] = [call.model_dump() for call in plan.tool_calls]
            state["tool_used"] = bool(plan.tool_calls)

            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["tool_planner"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
            state["llm_calls"] = state.get("llm_calls", 0) + output.metadata.llm_calls

        except Exception as exc:
            logger.error(f"ToolPlannerAgent failed: {exc}")
            state["tool_error"] = str(exc)
            state["tool_calls"] = []
            state["tool_used"] = False

        return state

    async def _run_tool_executor(self, state: PipelineState) -> PipelineState:
        """Execute planned tools."""
        start_time = time.time()
        state["current_agent"] = "ToolExecutor"

        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return state

        if not state.get("tool_approved"):
            approval_calls = []
            for call in tool_calls:
                definition = ToolRegistry.get_definition(call.get("name", ""))
                if definition and definition.policy.requires_approval:
                    approval_calls.append(call)
            if approval_calls:
                state["tool_approval_required"] = True
                state["tool_approval_calls"] = approval_calls
                state["tool_approval_message"] = (
                    "Approval required to run this tool."
                )
                state["answer_source"] = "approval"
                state["natural_language_answer"] = (
                    "This action needs approval before I can proceed."
                )
                return state

        ctx = ToolContext(
            user_id=state.get("user_id", "unknown"),
            correlation_id=state.get("correlation_id", "unknown"),
            approved=state.get("tool_approved", False),
            metadata={
                "retriever": self.retriever,
                "connector": self.connector,
                "database_type": state.get("database_type") or self.config.database.db_type,
                "database_url": state.get("database_url"),
            },
            state=state,
        )

        try:
            results = await self.tool_executor.execute_plan(tool_calls, ctx)
            state["tool_results"] = results

            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["tool_executor"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed

            self._apply_tool_results(state, results)
        except ToolPolicyError as exc:
            state["tool_error"] = str(exc)
            state["tool_results"] = []
        except Exception as exc:
            state["tool_error"] = str(exc)
            state["tool_results"] = []

        return state

    async def _run_classifier(self, state: PipelineState) -> PipelineState:
        """Run ClassifierAgent."""
        start_time = time.time()
        state["current_agent"] = "ClassifierAgent"

        try:
            input_data = ClassifierAgentInput(
                query=state["query"],
                conversation_history=self._augment_history_with_summary(state),
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
                conversation_history=self._augment_history_with_summary(state),
                entities=entities,
                retrieval_mode="hybrid",
                max_datapoints=10,
            )

            output = await self.context.execute(input_data)

            # Update state
            filtered_datapoints = await self._filter_datapoints_by_live_schema(
                [
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
                database_type=state.get("database_type"),
                database_url=state.get("database_url"),
            )
            sources_used = list({dp["datapoint_id"] for dp in filtered_datapoints})
            state["investigation_memory"] = {
                "query": output.investigation_memory.query,
                "datapoints": filtered_datapoints,
                "total_retrieved": len(filtered_datapoints),
                "retrieval_mode": output.investigation_memory.retrieval_mode,
                "sources_used": sources_used,
            }
            state["retrieved_datapoints"] = state["investigation_memory"]["datapoints"]
            state["context_confidence"] = output.context_confidence
            self._maybe_set_schema_preface(state)

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

            investigation_memory_state = state.get("investigation_memory") or {}
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
                total_retrieved=investigation_memory_state.get(
                    "total_retrieved", len(datapoints)
                ),
                retrieval_mode=investigation_memory_state.get(
                    "retrieval_mode", "hybrid"
                ),
                sources_used=investigation_memory_state.get("sources_used", []),
            )

            input_data = ContextAnswerAgentInput(
                query=state["query"],
                conversation_history=self._augment_history_with_summary(state),
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
            state["context_needs_sql"] = output.context_answer.needs_sql
            state["generated_sql"] = None
            state["validated_sql"] = None
            state["query_result"] = None
            state["visualization_hint"] = None
            if output.context_answer.needs_sql:
                state["context_preface"] = output.context_answer.answer
                state["context_evidence"] = [
                    evidence.model_dump()
                    for evidence in output.context_answer.evidence
                ]
            if output.context_answer.clarifying_questions:
                self._apply_clarification_response(
                    state,
                    output.context_answer.clarifying_questions,
                    default_intro="I need a bit more detail before I can continue.",
                )
            else:
                state["clarification_needed"] = False
                state["clarifying_questions"] = []

            elapsed = (time.time() - start_time) * 1000
            state["agent_timings"]["context_answer"] = elapsed
            state["total_latency_ms"] += elapsed
            state["llm_calls"] += output.metadata.llm_calls

            logger.info("ContextAnswerAgent complete")

        except Exception as e:
            logger.error(f"ContextAnswerAgent failed: {e}")
            state["error"] = f"Context answer failed: {e}"

        return state

    async def _filter_datapoints_by_live_schema(
        self,
        datapoints: list[dict[str, Any]],
        database_type: str | None = None,
        database_url: str | None = None,
    ) -> list[dict[str, Any]]:
        live_tables = await self._get_live_table_catalog(
            database_type=database_type,
            database_url=database_url,
        )
        if not live_tables:
            return datapoints

        filtered: list[dict[str, Any]] = []
        for dp in datapoints:
            table_key = self._extract_datapoint_table_key(dp)
            if table_key and table_key not in live_tables:
                continue
            filtered.append(dp)

        if len(filtered) != len(datapoints):
            logger.info(
                "Filtered %s datapoints not present in live schema",
                len(datapoints) - len(filtered),
            )

        return filtered

    async def _get_live_table_catalog(
        self,
        database_type: str | None = None,
        database_url: str | None = None,
    ) -> set[str] | None:
        connector = self.connector
        close_connector = False
        if database_url:
            connector = self._build_catalog_connector(database_type, database_url)
            close_connector = connector is not None
        if connector is None:
            return None

        try:
            if not connector.is_connected:
                await connector.connect()
            tables = await connector.get_schema()
        except Exception as exc:
            logger.warning(f"Failed to fetch live schema catalog: {exc}")
            return None
        finally:
            if close_connector:
                await connector.close()

        catalog: set[str] = set()
        for table in tables:
            schema_name = getattr(table, "schema_name", None) or getattr(table, "schema", None)
            table_name = getattr(table, "table_name", None)
            if schema_name and table_name:
                catalog.add(f"{schema_name}.{table_name}".lower())
            elif table_name:
                catalog.add(str(table_name).lower())
        return catalog or None

    def _build_catalog_connector(
        self, database_type: str | None, database_url: str
    ) -> BaseConnector | None:
        parsed = urlparse(database_url)
        scheme = parsed.scheme.split("+")[0].lower() if parsed.scheme else ""
        if database_type:
            target_type = database_type
        elif scheme in {"postgres", "postgresql"}:
            target_type = "postgresql"
        elif scheme == "clickhouse":
            target_type = "clickhouse"
        else:
            target_type = getattr(self.config.database, "db_type", "postgresql")

        if target_type == "postgresql":
            if scheme not in {"postgres", "postgresql"} or not parsed.hostname:
                return None
            return PostgresConnector(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") if parsed.path else "datachat",
                user=parsed.username or "postgres",
                password=parsed.password or "",
            )
        if target_type == "clickhouse":
            if scheme != "clickhouse" or not parsed.hostname:
                return None
            return ClickHouseConnector(
                host=parsed.hostname,
                port=parsed.port or 8123,
                database=parsed.path.lstrip("/") if parsed.path else "default",
                user=parsed.username or "default",
                password=parsed.password or "",
            )
        return None

    def _extract_datapoint_table_key(self, datapoint: dict[str, Any]) -> str | None:
        metadata = datapoint.get("metadata") or {}
        table_name = (
            metadata.get("table_name")
            or metadata.get("table")
            or metadata.get("table_key")
            or datapoint.get("table_name")
        )
        if not table_name:
            return None
        schema = metadata.get("schema")
        table_key = str(table_name)
        if "." not in table_key and schema:
            table_key = f"{schema}.{table_key}"
        return table_key.lower()

    def _maybe_set_schema_preface(self, state: PipelineState) -> None:
        if state.get("context_preface"):
            return
        datapoints = state.get("retrieved_datapoints") or []
        for datapoint in datapoints:
            if datapoint.get("datapoint_type") != "Schema":
                continue
            datapoint_id = datapoint.get("datapoint_id")
            if not datapoint_id:
                continue
            summary = self._load_schema_preface(datapoint_id)
            if summary:
                state["context_preface"] = summary
                return

    def _load_schema_preface(self, datapoint_id: str) -> str | None:
        data_dir = Path("datapoints") / "managed"
        path = data_dir / f"{datapoint_id}.json"
        if not path.exists():
            return None
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

        name = payload.get("name")
        table_name = payload.get("table_name")
        business_purpose = payload.get("business_purpose")
        key_columns = payload.get("key_columns") or []
        column_names = []
        for col in key_columns:
            if isinstance(col, dict) and col.get("name"):
                column_names.append(col["name"])
            if len(column_names) >= 5:
                break

        parts = []
        if name and table_name:
            parts.append(f"{name} ({table_name})")
        elif table_name:
            parts.append(table_name)
        elif name:
            parts.append(name)

        if business_purpose:
            parts.append(business_purpose)

        if column_names:
            parts.append(f"Key columns include {', '.join(column_names)}.")

        if not parts:
            return None

        return " ".join(parts)

    async def _run_sql(self, state: PipelineState) -> PipelineState:
        """Run SQLAgent."""
        start_time = time.time()
        state["current_agent"] = "SQLAgent"

        if state.get("error"):
            return state

        try:
            # Reconstruct InvestigationMemory from state
            from backend.models import InvestigationMemory, RetrievedDataPoint

            investigation_memory_state = state.get("investigation_memory") or {}
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
                total_retrieved=investigation_memory_state.get(
                    "total_retrieved", len(datapoints)
                ),
                retrieval_mode=investigation_memory_state.get(
                    "retrieval_mode", "hybrid"
                ),
                sources_used=investigation_memory_state.get(
                    "sources_used",
                    list({dp["source"] for dp in state.get("retrieved_datapoints", [])}),
                ),
            )

            resolved_query = await self._maybe_apply_any_table_hint(state)
            if resolved_query != state.get("query"):
                state["original_query"] = state.get("original_query") or state.get("query")
                state["query"] = resolved_query

            input_data = SQLAgentInput(
                query=state["query"],
                conversation_history=self._augment_history_with_summary(state),
                investigation_memory=investigation_memory,
                database_type=state.get("database_type", "postgresql"),
                database_url=state.get("database_url"),
            )

            output = await self.sql.execute(input_data)

            if output.needs_clarification:
                questions = output.generated_sql.clarifying_questions or []
                if not questions:
                    questions = ["Which table should I use to answer this?"]
                self._apply_clarification_response(state, questions)
                return state

            # Clear any stale clarification flags from earlier agents.
            state["clarification_needed"] = False
            state["clarifying_questions"] = []

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
                conversation_history=self._augment_history_with_summary(state),
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
                conversation_history=self._augment_history_with_summary(state),
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
            sql_evidence = [
                evidence.model_dump()
                for evidence in self._build_evidence_items(state)
            ]
            context_evidence = state.get("context_evidence") or []
            seen_ids = set()
            merged_evidence = []
            for item in [*context_evidence, *sql_evidence]:
                datapoint_id = item.get("datapoint_id")
                if datapoint_id and datapoint_id not in seen_ids:
                    seen_ids.add(datapoint_id)
                    merged_evidence.append(item)
            state["evidence"] = merged_evidence
            context_preface = state.get("context_preface")
            if context_preface:
                state["natural_language_answer"] = (
                    f"{context_preface}\n\n{state['natural_language_answer']}"
                )

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
            error_message = str(e)
            state["error"] = f"Execution failed: {error_message}"
            if "Missing table in live schema" in error_message:
                if not state.get("schema_refresh_attempted"):
                    state["schema_refresh_attempted"] = True
                    await self._run_schema_refresh(state)
                    if state.get("tool_error"):
                        state["error"] = (
                            "Schema refresh failed. Please check database connectivity "
                            "or refresh manually."
                        )
                        return await self._handle_error(state)
                    state.pop("error", None)
                    return await self._rerun_after_schema_refresh(state)
                state["answer_source"] = "error"
                state["natural_language_answer"] = (
                    "I refreshed the database schema, but the referenced table is still missing. "
                    "Please verify the table name or update your DataPoints."
                )
            elif not state.get("natural_language_answer"):
                state["natural_language_answer"] = (
                    f"I encountered an error while processing your query: {state.get('error')}. "
                    "Please try rephrasing your question or contact support if the issue persists."
                )

        return state

    async def _run_schema_refresh(self, state: PipelineState) -> None:
        if not self.tooling_enabled:
            logger.warning("Tooling disabled; skipping schema refresh.")
            return
        ctx = ToolContext(
            user_id=state.get("user_id"),
            correlation_id=state.get("correlation_id"),
            approved=True,
        )
        try:
            result = await self.tool_executor.execute(
                "profile_and_generate_datapoints",
                {"depth": "schema_only", "batch_size": 10},
                ctx,
            )
            state["tool_results"] = [result]
        except Exception as exc:
            logger.error(f"Schema refresh failed: {exc}")
            state["tool_error"] = str(exc)

    async def _rerun_after_schema_refresh(self, state: PipelineState) -> PipelineState:
        state = await self._run_context(state)
        if self._should_use_context_answer(state) == "context":
            state = await self._run_context_answer(state)
            if self._should_execute_after_context_answer(state) == "end":
                return state

        while True:
            state = await self._run_sql(state)
            state = await self._run_validator(state)
            decision = self._should_retry_sql(state)
            if decision == "retry":
                continue
            if decision == "execute":
                return await self._run_executor(state)
            return await self._handle_error(state)

    async def _run_response_synthesis(self, state: PipelineState) -> PipelineState:
        """Run ResponseSynthesisAgent."""
        start_time = time.time()
        state["current_agent"] = "ResponseSynthesisAgent"
        try:
            query_result = state.get("query_result") or {}
            rows = query_result.get("rows") or []
            columns = query_result.get("columns") or []
            preview_rows = rows[:3]
            result_summary = {
                "row_count": query_result.get("row_count"),
                "columns": columns,
                "preview": preview_rows,
            }
            synthesized = await self.response_synthesis.execute(
                query=state.get("query", ""),
                sql=state.get("validated_sql") or state.get("generated_sql") or "",
                result_summary=json.dumps(result_summary, default=str),
                context_preface=state.get("context_preface"),
            )
            state["natural_language_answer"] = synthesized
            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["response_synthesis"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
            state["llm_calls"] = state.get("llm_calls", 0) + 1
        except Exception as exc:
            logger.error(f"ResponseSynthesisAgent failed: {exc}")
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

    def _should_continue_after_intent_gate(self, state: PipelineState) -> str:
        intent_gate = state.get("intent_gate")
        if intent_gate in {"exit", "out_of_scope", "small_talk", "setup_help", "clarify"}:
            return "end"
        if state.get("fast_path"):
            return "sql"
        if self.tooling_enabled and self.tool_planner_enabled:
            return "tool_planner"
        return "classifier"

    def _should_validate_sql(self, state: PipelineState) -> str:
        if state.get("clarification_needed"):
            return "clarify"
        return "validate"

    def _should_use_context_answer(self, state: PipelineState) -> str:
        if state.get("error"):
            return "sql"

        intent = state.get("intent") or "data_query"
        confidence = state.get("context_confidence") or 0.0
        query = (state.get("query") or "").lower()
        retrieved = state.get("retrieved_datapoints") or []

        if not retrieved:
            return "sql"

        if self._query_is_table_list(query):
            return "sql"

        if "datapoint" in query or "data point" in query:
            return "context"

        if intent in ("exploration", "explanation", "meta"):
            return "context"

        if self._query_requires_sql(query):
            return "sql"

        if confidence >= 0.7:
            return "context"

        return "sql"

    def _query_is_table_list(self, query: str) -> bool:
        patterns = [
            r"\bwhat tables\b",
            r"\blist tables\b",
            r"\bshow tables\b",
            r"\bavailable tables\b",
            r"\bwhich tables\b",
            r"\bwhat tables exist\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _should_use_tools(self, state: PipelineState) -> str:
        if state.get("tool_error"):
            return "pipeline"
        tool_calls = state.get("tool_calls", [])
        tool_plan = state.get("tool_plan") or {}
        if tool_plan.get("fallback") == "pipeline":
            return "pipeline"
        if tool_calls:
            return "tools"
        return "pipeline"

    def _should_continue_after_tool_execution(self, state: PipelineState) -> str:
        if state.get("tool_error"):
            return "pipeline"
        if state.get("tool_approval_required"):
            return "end"
        if state.get("tool_used") and state.get("natural_language_answer"):
            return "end"
        return "pipeline"

    def _query_requires_sql(self, query: str) -> bool:
        if "datapoint" in query or "data point" in query:
            return False
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

    def _should_execute_after_context_answer(self, state: PipelineState) -> str:
        if state.get("error"):
            return "end"
        if state.get("context_needs_sql"):
            return "sql"
        return "end"

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

    def _apply_tool_results(
        self, state: PipelineState, results: list[dict[str, Any]]
    ) -> None:
        for result in results:
            payload = result.get("result") or {}
            answer = payload.get("answer")
            if answer:
                state["natural_language_answer"] = answer
            if payload.get("answer_source"):
                state["answer_source"] = payload.get("answer_source")
            if payload.get("confidence") is not None:
                state["answer_confidence"] = payload.get("confidence")
            if payload.get("evidence"):
                state["evidence"] = payload.get("evidence")
            if payload.get("sql"):
                state["validated_sql"] = payload.get("sql")
            if payload.get("data"):
                state["query_result"] = payload.get("data")
            if payload.get("visualization_hint"):
                state["visualization_hint"] = payload.get("visualization_hint")
            if payload.get("retrieved_datapoints"):
                state["retrieved_datapoints"] = payload.get("retrieved_datapoints")
            if payload.get("used_datapoints"):
                state["used_datapoints"] = payload.get("used_datapoints")
            if payload.get("validation_warnings"):
                state["validation_warnings"] = payload.get("validation_warnings")
            if payload.get("validation_errors"):
                state["validation_errors"] = payload.get("validation_errors")

        if state.get("used_datapoints") and not state.get("evidence"):
            state["evidence"] = [
                item.model_dump() for item in self._build_evidence_items(state)
            ]

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

    def _should_synthesize_response(self, state: PipelineState) -> str:
        if state.get("error"):
            return "end"
        if state.get("skip_response_synthesis"):
            return "end"
        if state.get("validated_sql") and state.get("query_result"):
            return "synthesize"
        return "end"

    def _format_clarifying_response(
        self, query: str, questions: list[str]
    ) -> str:
        if not questions:
            return "I need a bit more detail to generate SQL. Which table should I use?"
        prompt = "I need a bit more detail to generate SQL:"
        formatted = "\n".join(f"- {question}" for question in questions)
        return f"{prompt}\n{formatted}"

    def _current_clarification_count(self, state: PipelineState) -> int:
        summary = state.get("intent_summary") or {}
        from_summary = int(summary.get("clarification_count", 0) or 0)
        from_state = int(state.get("clarification_turn_count", 0) or 0)
        return max(from_summary, from_state)

    def _apply_clarification_response(
        self,
        state: PipelineState,
        questions: list[str],
        default_intro: str | None = None,
    ) -> None:
        limit = int(state.get("clarification_limit", self.max_clarifications) or 0)
        current_count = self._current_clarification_count(state)
        state["clarification_turn_count"] = current_count

        if current_count >= max(limit, 0):
            fallback = self._format_clarification_limit_message(state)
            state["clarification_needed"] = False
            state["clarifying_questions"] = []
            state["answer_source"] = "system"
            state["answer_confidence"] = 0.5
            state["natural_language_answer"] = fallback
            state["intent"] = "meta"
            return

        if not questions:
            questions = ["Which table should I use to answer this?"]

        state["clarification_turn_count"] = current_count + 1
        state["clarification_needed"] = True
        state["clarifying_questions"] = questions
        state["answer_source"] = "clarification"
        state["answer_confidence"] = 0.2
        intro = default_intro or "I need a bit more detail to generate SQL:"
        formatted = "\n".join(f"- {question}" for question in questions)
        state["natural_language_answer"] = f"{intro}\n{formatted}"
        state["intent"] = "clarify"

    def _format_clarification_limit_message(self, state: PipelineState) -> str:
        candidates = self._collect_table_candidates(state)
        options: list[str] = []
        if candidates:
            options.append(
                f"1. Pick one table to continue: {', '.join(candidates[:5])}."
            )
        options.append("2. Ask me to list available tables.")
        options.append("3. Ask a fully specified question with table + metric.")
        options.append("4. Type `exit` to end the session.")
        return (
            "I still cannot answer confidently after several clarifications.\n"
            + "\n".join(options)
        )

    def _build_intent_summary(
        self, query: str, history: list[Message]
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "last_goal": query.strip() if query else None,
            "last_clarifying_question": None,
            "last_clarifying_questions": [],
            "table_hints": [],
            "column_hints": [],
            "clarification_count": 0,
            "resolved_query": None,
            "any_table": False,
            "slots": {
                "table": None,
                "metric": None,
                "time_range": None,
            },
        }

        if not history:
            return summary

        last_clarifying_questions: list[str] = []
        last_clarifying_index = None
        clarification_count = 0

        for msg in history:
            role, content = self._message_role_content(msg)
            if role == "assistant" and self._is_clarification_prompt(content):
                clarification_count += 1

        for idx in range(len(history) - 1, -1, -1):
            role, content = self._message_role_content(history[idx])
            if role == "assistant" and self._is_clarification_prompt(content):
                last_clarifying_questions = self._extract_clarifying_questions(content)
                last_clarifying_index = idx
                break

        summary["clarification_count"] = clarification_count
        summary["last_clarifying_questions"] = last_clarifying_questions
        summary["last_clarifying_question"] = (
            last_clarifying_questions[0] if last_clarifying_questions else None
        )

        previous_user_text = None
        if last_clarifying_index is not None:
            for idx in range(last_clarifying_index - 1, -1, -1):
                role, content = self._message_role_content(history[idx])
                if role == "user" and content:
                    previous_user_text = content
                    break

        if previous_user_text:
            summary["last_goal"] = previous_user_text

        if last_clarifying_questions and self._is_short_followup(query):
            cleaned_hint = self._clean_hint(query)
            if self._is_any_table_request(query):
                summary["any_table"] = True
                if previous_user_text:
                    summary["resolved_query"] = f"{previous_user_text} Use any table."
                return summary

            if cleaned_hint:
                combined_questions = " ".join(last_clarifying_questions).lower()
                if "table" in combined_questions:
                    summary["table_hints"] = [cleaned_hint]
                    summary["slots"]["table"] = cleaned_hint
                    if previous_user_text:
                        summary["resolved_query"] = (
                            f"{previous_user_text} Use table {cleaned_hint}."
                        )
                elif "column" in combined_questions or "field" in combined_questions:
                    summary["column_hints"] = [cleaned_hint]
                    summary["slots"]["metric"] = cleaned_hint
                    if previous_user_text:
                        summary["resolved_query"] = (
                            f"{previous_user_text} Use column {cleaned_hint}."
                        )
                elif "date" in combined_questions or "time" in combined_questions:
                    summary["slots"]["time_range"] = cleaned_hint

        return summary

    def _augment_history_with_summary(self, state: PipelineState) -> list[Message]:
        history = state.get("conversation_history") or []
        summary = state.get("intent_summary") or {}
        summary_text = self._format_intent_summary(summary)
        if not summary_text:
            return history
        return [*history, {"role": "system", "content": summary_text}]

    def _format_intent_summary(self, summary: dict[str, Any]) -> str | None:
        if not summary:
            return None
        parts = []
        if summary.get("last_goal"):
            parts.append(f"last_goal={summary['last_goal']}")
        if summary.get("table_hints"):
            parts.append(f"table_hints={', '.join(summary['table_hints'])}")
        if summary.get("column_hints"):
            parts.append(f"column_hints={', '.join(summary['column_hints'])}")
        slots = summary.get("slots") or {}
        slot_parts = [f"{k}:{v}" for k, v in slots.items() if v]
        if slot_parts:
            parts.append(f"slots={', '.join(slot_parts)}")
        if summary.get("clarification_count"):
            parts.append(f"clarifications={summary['clarification_count']}")
        questions = summary.get("last_clarifying_questions") or []
        if questions:
            parts.append(f"last_questions={'; '.join(questions[:2])}")
        if not parts:
            return None
        return "Intent summary: " + " | ".join(parts)

    async def _maybe_apply_any_table_hint(self, state: PipelineState) -> str:
        query = state.get("query") or ""
        if not self._is_any_table_request(query):
            return query

        candidates = self._collect_table_candidates(state)
        if not candidates:
            live_tables = await self._get_live_table_catalog(
                database_type=state.get("database_type"),
                database_url=state.get("database_url"),
            )
            if live_tables:
                candidates = sorted(live_tables)

        if not candidates:
            return query

        ranked = self._rank_table_candidates(query, candidates, limit=1)
        if not ranked:
            return query

        selected = ranked[0]
        if "use table" in query.lower():
            return query

        summary = state.get("intent_summary") or {}
        if selected not in summary.get("table_hints", []):
            summary.setdefault("table_hints", []).append(selected)
            state["intent_summary"] = summary

        return f"{query.rstrip('. ')} Use table {selected}."

    async def _build_clarification_fallback(
        self, state: PipelineState
    ) -> dict[str, Any] | None:
        candidates = self._collect_table_candidates(state)
        if not candidates:
            live_tables = await self._get_live_table_catalog(
                database_type=state.get("database_type"),
                database_url=state.get("database_url"),
            )
            if live_tables:
                candidates = sorted(live_tables)

        if not candidates:
            return None

        ranked = self._rank_table_candidates(state.get("query") or "", candidates, limit=5)
        if not ranked:
            return None

        table_list = ", ".join(ranked)
        answer = (
            "I still need a bit more detail to generate SQL. "
            f"Here are a few tables that look relevant: {table_list}. "
            "Which table should I use?"
        )
        return {"answer": answer, "questions": ["Which table should I use?"]}

    def _collect_table_candidates(self, state: PipelineState) -> list[str]:
        candidates: list[str] = []
        for dp in state.get("retrieved_datapoints", []) or []:
            if not isinstance(dp, dict):
                continue
            if dp.get("datapoint_type") != "Schema":
                continue
            metadata = dp.get("metadata") or {}
            table_name = metadata.get("table_name") or metadata.get("table")
            if table_name:
                candidates.append(str(table_name))
        return list(dict.fromkeys(candidates))

    def _rank_table_candidates(
        self, query: str, candidates: list[str], limit: int = 5
    ) -> list[str]:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        stopwords = {
            "show",
            "me",
            "the",
            "a",
            "an",
            "first",
            "rows",
            "row",
            "count",
            "total",
            "sum",
            "average",
            "avg",
            "use",
            "table",
            "any",
            "from",
            "of",
            "in",
            "for",
            "with",
            "please",
            "pick",
            "select",
            "list",
        }
        tokens = {token for token in tokens if token not in stopwords}

        scored = []
        for name in candidates:
            name_tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
            score = len(tokens & name_tokens)
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))

        if scored and scored[0][0] == 0:
            return [name for _, name in scored[:limit]]

        return [name for score, name in scored if score > 0][:limit]

    def _classify_intent_gate(self, query: str) -> str:
        text = query.strip().lower()
        if not text:
            return "data_query"
        if self._is_exit_intent(text):
            return "exit"
        if self._is_setup_help_intent(text):
            return "setup_help"
        if self._is_small_talk(text):
            return "small_talk"
        if self._is_out_of_scope(text):
            return "out_of_scope"
        if self._is_non_actionable_utterance(text):
            return "clarify"
        return "data_query"

    def _build_intent_gate_response(self, intent: str) -> str:
        if intent == "exit":
            return "Got it. Ending the session. If you need more, just start a new chat."
        if intent == "setup_help":
            return (
                "To connect a database, open Settings -> Database Manager in the web app "
                "or run `datachat setup` / `datachat connect` in the CLI. "
                "Then ask questions like: list tables, show first 5 rows of a table, "
                "or total sales last month."
            )
        if intent == "small_talk":
            return (
                "Hi! I can help you explore your connected data. Try: "
                "list tables, show first 5 rows of a table, or total sales last month."
            )
        return (
            "I can help with questions about your connected data. Try: "
            "list tables, show first 5 rows of a table, or total sales last month."
        )

    def _format_intent_clarification(self, question: str) -> str:
        return (
            "I can help with questions about your connected data, but I need a bit more detail.\n"
            f"- {question}"
        )

    def _is_exit_intent(self, text: str) -> bool:
        if text in {"exit", "quit", "bye", "goodbye", "stop", "end"}:
            return True
        if re.search(r"\b(i'?m|im|we'?re|were)\s+done\b", text):
            return True
        if re.search(r"\b(done for now|done here|that'?s all|all set)\b", text):
            return True
        if re.search(r"\b(let'?s\s+)?talk\s+later\b", text):
            return True
        if re.search(r"\b(talk|see)\s+you\s+later\b", text):
            return True
        if re.search(r"\b(no\s+more|no\s+further)\s+questions\b", text):
            return True
        if re.search(r"\b(end|stop|quit|exit)\b.*\b(chat|conversation|session)\b", text):
            return True
        return False

    def _is_setup_help_intent(self, text: str) -> bool:
        patterns = [
            r"\bsetup\b",
            r"\bconnect\b",
            r"\bconfigure\b",
            r"\bconfiguration\b",
            r"\binstall\b",
            r"\bapi key\b",
            r"\bcredentials?\b",
            r"\bdatabase url\b",
            r"\bhow do i\b.*\bconnect\b",
            r"\bwhat can you do\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_small_talk(self, text: str) -> bool:
        greetings = [
            r"\bhi\b",
            r"\bhello\b",
            r"\bhey\b",
            r"\bhow are you\b",
            r"\bwhat'?s up\b",
            r"\bgood morning\b",
            r"\bgood afternoon\b",
            r"\bgood evening\b",
        ]
        return any(re.search(pattern, text) for pattern in greetings)

    def _is_out_of_scope(self, text: str) -> bool:
        if self._contains_data_keywords(text):
            return False
        out_of_scope = [
            r"\bjoke\b",
            r"\bweather\b",
            r"\bnews\b",
            r"\bsports\b",
            r"\bmovie\b",
            r"\bmusic\b",
            r"\bstock\b",
            r"\brecipe\b",
            r"\btranslate\b",
            r"\bwrite\b.*\bemail\b",
            r"\bcompose\b.*\bmessage\b",
            r"\bpoem\b",
            r"\bstory\b",
        ]
        return any(re.search(pattern, text) for pattern in out_of_scope)

    def _contains_data_keywords(self, text: str) -> bool:
        keywords = {
            "table",
            "tables",
            "column",
            "columns",
            "row",
            "rows",
            "schema",
            "database",
            "sql",
            "query",
            "count",
            "sum",
            "average",
            "avg",
            "min",
            "max",
            "join",
            "group",
            "order",
            "select",
            "from",
            "data",
            "dataset",
            "warehouse",
        }
        return any(word in text for word in keywords)

    def _is_non_actionable_utterance(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return True
        canned = {
            "ok",
            "okay",
            "k",
            "kk",
            "sure",
            "yes",
            "no",
            "cool",
            "fine",
            "great",
            "thanks",
            "thank you",
            "alright",
            "continue",
            "next",
            "go on",
        }
        if normalized in canned:
            return True
        if re.fullmatch(r"(ok|okay|sure|yes|no|thanks|thank you)[.!]*", normalized):
            return True
        return False

    def _is_deterministic_sql_query(self, query: str) -> bool:
        text = query.strip().lower()
        if not text:
            return False
        if self._query_is_table_list(text):
            return True

        table_ref = self._extract_table_reference(text)
        if not table_ref:
            return False

        sample_patterns = [
            r"\bshow\b.*\brows\b",
            r"\bfirst\s+\d+\b",
            r"\btop\s+\d+\b",
            r"\blimit\s+\d+\b",
            r"\bpreview\b",
            r"\bsample\b",
        ]
        if any(re.search(pattern, text) for pattern in sample_patterns):
            return True

        count_patterns = [
            r"\bhow\s+many\s+rows?\b",
            r"\brow\s+count\b",
            r"\bcount\s+of\s+rows?\b",
            r"\bhow\s+many\s+records?\b",
            r"\brecord\s+count\b",
        ]
        if any(re.search(pattern, text) for pattern in count_patterns):
            return True

        return False

    def _extract_table_reference(self, query: str) -> str | None:
        patterns = [
            r"\b(?:from|in|of)\s+([a-zA-Z0-9_.]+)",
            r"\btable\s+([a-zA-Z0-9_.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if not match:
                continue
            table = match.group(1).strip().rstrip(".,;:?)")
            if table and table not in {"table", "tables", "rows", "row"}:
                return table
        return None

    def _is_any_table_request(self, text: str) -> bool:
        patterns = [
            r"\bany\s+table\b",
            r"\bpick\s+any\s+table\b",
            r"\bany\s+table\s+from\b",
            r"\bwhatever\s+table\b",
        ]
        return any(re.search(pattern, text.lower()) for pattern in patterns)

    def _should_call_intent_llm(
        self, state: PipelineState, summary: dict[str, Any]
    ) -> bool:
        query = (state.get("query") or "").strip().lower()
        if not query:
            return False
        if not self.intent_llm:
            return False
        if self._is_ambiguous_intent(state, summary):
            return True
        generic_responses = {
            "ok",
            "okay",
            "sure",
            "yes",
            "no",
            "maybe",
            "help",
            "next",
            "continue",
            "go on",
            "not sure",
        }
        if query in generic_responses:
            return True
        if re.fullmatch(r"[a-z]+", query) and len(query.split()) <= 2:
            return True
        return False

    def _is_ambiguous_intent(self, state: PipelineState, summary: dict[str, Any]) -> bool:
        query = (state.get("query") or "").strip().lower()
        if not query:
            return False
        if self._contains_data_keywords(query):
            return False
        if self._is_any_table_request(query):
            return False
        if summary.get("last_clarifying_questions") and self._is_short_followup(query):
            return True
        return len(query.split()) <= 3

    async def _llm_intent_gate(
        self, query: str, summary: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, int]:
        summary_text = self._format_intent_summary(summary) or "None"
        system_prompt = (
            "You are a fast intent router for a data assistant. "
            "Classify the user's message into one of: "
            "data_query, exit, out_of_scope, small_talk, setup_help, clarify. "
            "Return JSON with keys: intent, confidence (0-1), "
            "clarifying_question (optional, only if intent=clarify)."
        )
        user_prompt = (
            f"User message: {query}\n"
            f"Intent summary: {summary_text}\n"
            "Return JSON only."
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            temperature=0.0,
            max_tokens=200,
        )
        try:
            response = await self.intent_llm.generate(request)
            return self._parse_intent_llm_response(response.content), 1
        except Exception as exc:
            logger.warning(f"Intent LLM fallback failed: {exc}")
            return None, 0

    def _parse_intent_llm_response(self, content: str) -> dict[str, Any] | None:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return None
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None
        intent = str(data.get("intent", "")).strip()
        allowed = {"data_query", "exit", "out_of_scope", "small_talk", "setup_help", "clarify"}
        if intent not in allowed:
            return None
        return {
            "intent": intent,
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "clarifying_question": data.get("clarifying_question"),
        }

    def _is_short_followup(self, text: str) -> bool:
        tokens = text.strip().split()
        return 0 < len(tokens) <= 5

    def _extract_clarifying_questions(self, text: str) -> list[str]:
        questions: list[str] = []
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            candidate = re.sub(r"^[\-\*\u2022]\s*", "", candidate).strip()
            if not candidate:
                continue
            if "?" in candidate:
                questions.append(candidate.rstrip())
        if not questions and "?" in text:
            chunks = [chunk.strip() for chunk in text.split("?") if chunk.strip()]
            for chunk in chunks[:3]:
                questions.append(f"{chunk}?")
        return questions[:3]

    def _is_clarification_prompt(self, text: str) -> bool:
        lower = text.lower()
        triggers = [
            "clarifying question",
            "clarifying questions",
            "need a bit more detail",
            "which table",
            "which column",
            "what information are you trying",
            "is there a specific",
            "do you want to see",
            "are you looking for",
        ]
        return any(trigger in lower for trigger in triggers)

    def _clean_hint(self, text: str) -> str | None:
        cleaned = re.sub(r"[^\w.]+", " ", text.lower()).strip()
        if not cleaned:
            return None
        tokens = [
            token
            for token in cleaned.split()
            if token
            and token
            not in {"table", "column", "field", "use", "the", "a", "an", "any"}
        ]
        if not tokens:
            return None
        return tokens[0] if len(tokens) == 1 else "_".join(tokens)

    def _message_role_content(self, msg: Message) -> tuple[str, str]:
        if isinstance(msg, dict):
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
        else:
            role = str(getattr(msg, "role", "user"))
            content = str(getattr(msg, "content", ""))
        return role, content.strip()

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
            "original_query": None,
            "conversation_history": conversation_history or [],
            "database_type": database_type,
            "database_url": database_url,
            "user_id": "anonymous",
            "correlation_id": f"local-{int(time.time() * 1000)}",
            "tool_approved": False,
            "intent_gate": None,
            "intent_summary": None,
            "clarification_turn_count": 0,
            "clarification_limit": self.max_clarifications,
            "fast_path": False,
            "skip_response_synthesis": False,
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
            "context_needs_sql": None,
            "context_preface": None,
            "context_evidence": [],
            "answer_source": None,
            "answer_confidence": None,
            "evidence": [],
            "tool_plan": None,
            "tool_calls": [],
            "tool_results": [],
            "tool_error": None,
            "tool_used": False,
            "tool_approval_required": False,
            "tool_approval_message": None,
            "tool_approval_calls": [],
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
            "original_query": None,
            "conversation_history": conversation_history or [],
            "database_type": database_type,
            "database_url": database_url,
            "user_id": "anonymous",
            "correlation_id": f"stream-{int(time.time() * 1000)}",
            "tool_approved": False,
            "intent_gate": None,
            "intent_summary": None,
            "clarification_turn_count": 0,
            "clarification_limit": self.max_clarifications,
            "fast_path": False,
            "skip_response_synthesis": False,
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
            "context_needs_sql": None,
            "context_preface": None,
            "context_evidence": [],
            "answer_source": None,
            "answer_confidence": None,
            "evidence": [],
            "tool_plan": None,
            "tool_calls": [],
            "tool_results": [],
            "tool_error": None,
            "tool_used": False,
            "tool_approval_required": False,
            "tool_approval_message": None,
            "tool_approval_calls": [],
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
            "original_query": None,
            "conversation_history": conversation_history or [],
            "database_type": database_type,
            "database_url": database_url,
            "user_id": "anonymous",
            "correlation_id": f"stream-{int(time.time() * 1000)}",
            "tool_approved": False,
            "intent_gate": None,
            "intent_summary": None,
            "clarification_turn_count": 0,
            "clarification_limit": self.max_clarifications,
            "fast_path": False,
            "skip_response_synthesis": False,
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
            "context_needs_sql": None,
            "context_preface": None,
            "context_evidence": [],
            "answer_source": None,
            "answer_confidence": None,
            "evidence": [],
            "tool_plan": None,
            "tool_calls": [],
            "tool_results": [],
            "tool_error": None,
            "tool_used": False,
            "tool_approval_required": False,
            "tool_approval_message": None,
            "tool_approval_calls": [],
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
                        elif current_agent == "ToolPlannerAgent":
                            agent_data = {
                                "tool_calls": len(state_update.get("tool_calls", [])),
                            }
                        elif current_agent == "ToolExecutor":
                            agent_data = {
                                "tool_results": len(state_update.get("tool_results", [])),
                                "tool_error": state_update.get("tool_error"),
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
