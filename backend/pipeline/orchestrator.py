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
from backend.connectors.factory import create_connector
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
    session_summary: str | None
    session_state: dict[str, Any] | None
    database_type: str
    database_url: str | None
    target_connection_id: str | None
    user_id: str | None
    correlation_id: str | None
    tool_approved: bool
    intent_gate: str | None
    intent_summary: dict[str, Any] | None
    clarification_turn_count: int
    clarification_limit: int
    fast_path: bool
    skip_response_synthesis: bool
    synthesize_simple_sql: bool | None

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
    sql_formatter_fallback_calls: int
    sql_formatter_fallback_successes: int
    query_compiler_llm_calls: int
    query_compiler_llm_refinements: int
    query_compiler_latency_ms: float
    query_compiler: dict[str, Any] | None
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
    visualization_metadata: dict[str, Any] | None
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
    sub_answers: list[dict[str, Any]]
    decision_trace: list[dict[str, Any]]


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
        self.routing_policy = self._build_routing_policy()

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
        self.max_subqueries = 3

        # Build LangGraph
        self.graph = self._build_graph()

        logger.info("DataChatPipeline initialized")

    def _build_routing_policy(self) -> dict[str, float | int]:
        pipeline_cfg = getattr(self.config, "pipeline", None)
        return {
            "intent_llm_confidence_threshold": float(
                getattr(pipeline_cfg, "intent_llm_confidence_threshold", 0.45)
            ),
            "context_answer_confidence_threshold": float(
                getattr(pipeline_cfg, "context_answer_confidence_threshold", 0.7)
            ),
            "semantic_sql_clarification_confidence_threshold": float(
                getattr(
                    pipeline_cfg,
                    "semantic_sql_clarification_confidence_threshold",
                    0.55,
                )
            ),
            "ambiguous_query_max_tokens": int(
                getattr(pipeline_cfg, "ambiguous_query_max_tokens", 3)
            ),
        }

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
        summary = self._build_intent_summary(query, state.get("conversation_history", []))
        summary = self._merge_session_state_into_summary(
            summary,
            state.get("session_state"),
        )

        contextual_rewrite = self._rewrite_contextual_followup(query, summary)
        if contextual_rewrite and contextual_rewrite != query:
            summary["resolved_query"] = contextual_rewrite
        state["intent_summary"] = summary

        resolved_query = summary.get("resolved_query")
        if resolved_query and resolved_query != query:
            state["original_query"] = query
            state["query"] = resolved_query
            self._record_decision(
                state,
                stage="intent_gate.rewrite",
                decision="rewrite_query",
                reason="resolved_followup",
                details={"from": query, "to": resolved_query},
            )

        fast_path = self._is_deterministic_sql_query(state.get("query") or "")
        if fast_path:
            state["intent"] = "data_query"
            state["intent_gate"] = "data_query"
            state["fast_path"] = True
            state["skip_response_synthesis"] = True
            self._record_decision(
                state,
                stage="intent_gate",
                decision="data_query_fast_path",
                reason="deterministic_sql_query",
            )
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
            self._record_decision(
                state,
                stage="intent_gate",
                decision="clarify",
                reason="rule_based_non_actionable",
            )
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
                    and confidence < float(self.routing_policy["intent_llm_confidence_threshold"])
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
                    self._record_decision(
                        state,
                        stage="intent_gate",
                        decision="clarify",
                        reason="intent_llm_low_confidence",
                        details={
                            "intent": intent_gate,
                            "confidence": confidence,
                            "threshold": self.routing_policy["intent_llm_confidence_threshold"],
                        },
                    )
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
                    self._record_decision(
                        state,
                        stage="intent_gate",
                        decision="clarify",
                        reason="intent_llm_non_actionable_followup",
                    )
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
            self._record_decision(
                state,
                stage="intent_gate",
                decision="clarify",
                reason="ambiguous_query_requires_clarification",
            )
            elapsed = (time.time() - start_time) * 1000
            state.setdefault("agent_timings", {})["intent_gate"] = elapsed
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + elapsed
            return state
        state["intent_gate"] = intent_gate
        self._record_decision(
            state,
            stage="intent_gate",
            decision=str(intent_gate),
            reason="rule_or_llm_classification",
        )

        if intent_gate in {
            "exit",
            "out_of_scope",
            "small_talk",
            "setup_help",
            "datapoint_help",
        }:
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
            connection_scoped_datapoints = self._filter_datapoints_by_target_connection(
                [
                    {
                        "datapoint_id": dp.datapoint_id,
                        "datapoint_type": dp.datapoint_type,
                        "name": dp.name,
                        "score": dp.score,
                        "source": dp.source,
                        "metadata": dp.metadata,
                        "content": dp.content,
                    }
                    for dp in output.investigation_memory.datapoints
                ],
                target_connection_id=state.get("target_connection_id"),
            )
            filtered_datapoints = await self._filter_datapoints_by_live_schema(
                connection_scoped_datapoints,
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

    def _filter_datapoints_by_target_connection(
        self,
        datapoints: list[dict[str, Any]],
        target_connection_id: str | None,
    ) -> list[dict[str, Any]]:
        """
        Scope retrieved DataPoints to the selected connection.

        Rules:
        - Keep datapoints with matching metadata.connection_id
        - Keep explicitly global datapoints (metadata.scope in {global, shared}
          or metadata.shared=True)
        - If no scoped/global datapoints were retrieved at all, fallback to
          legacy unscoped datapoints for backwards compatibility.
        """
        if not target_connection_id:
            return datapoints

        scoped: list[dict[str, Any]] = []
        global_items: list[dict[str, Any]] = []
        unscoped: list[dict[str, Any]] = []
        removed_foreign = 0
        for dp in datapoints:
            metadata = dp.get("metadata") or {}
            scope = str(metadata.get("scope", "")).strip().lower()
            shared_raw = metadata.get("shared")
            shared_flag = (
                shared_raw is True
                or str(shared_raw).strip().lower() in {"1", "true", "yes", "y"}
            )
            if scope in {"global", "shared"} or shared_flag:
                global_items.append(dp)
                continue

            connection_id = metadata.get("connection_id")
            if connection_id is None:
                unscoped.append(dp)
                continue

            if str(connection_id) != str(target_connection_id):
                removed_foreign += 1
                continue
            scoped.append(dp)

        if removed_foreign:
            logger.info(
                "Filtered %s datapoints scoped to a different connection",
                removed_foreign,
            )

        # Preferred mode: only scoped + global datapoints.
        if scoped or global_items:
            return [*scoped, *global_items]

        # Backwards compatibility: old datasets may be unscoped.
        if unscoped:
            logger.info(
                "No scoped datapoints matched target connection; using %s unscoped datapoints",
                len(unscoped),
            )
            return unscoped

        return []

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
                    content=dp.get("content"),
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
                clarification_intro = (
                    output.context_answer.answer
                    or "I need a bit more detail before I can continue."
                )
                self._apply_clarification_response(
                    state,
                    output.context_answer.clarifying_questions,
                    default_intro=clarification_intro,
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
            table_keys = self._extract_datapoint_table_keys(dp)
            if table_keys and table_keys.isdisjoint(live_tables):
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
        try:
            return create_connector(
                database_url=database_url,
                database_type=database_type or getattr(self.config.database, "db_type", None),
                pool_size=self.config.database.pool_size,
            )
        except Exception:
            return None

    def _extract_datapoint_table_keys(self, datapoint: dict[str, Any]) -> set[str]:
        metadata = datapoint.get("metadata") or {}
        schema = metadata.get("schema") or datapoint.get("schema")

        def _normalize_table(value: Any) -> str | None:
            if value is None:
                return None
            table_key = str(value).strip()
            if not table_key:
                return None
            if "." not in table_key and schema:
                table_key = f"{schema}.{table_key}"
            return table_key.lower()

        keys: set[str] = set()

        for value in (
            metadata.get("table_name"),
            metadata.get("table"),
            metadata.get("table_key"),
            datapoint.get("table_name"),
        ):
            normalized = _normalize_table(value)
            if normalized:
                keys.add(normalized)

        related_tables = datapoint.get("related_tables")
        if isinstance(related_tables, list):
            for value in related_tables:
                normalized = _normalize_table(value)
                if normalized:
                    keys.add(normalized)

        metadata_related = metadata.get("related_tables")
        if isinstance(metadata_related, str):
            for value in metadata_related.split(","):
                normalized = _normalize_table(value)
                if normalized:
                    keys.add(normalized)
        elif isinstance(metadata_related, list):
            for value in metadata_related:
                normalized = _normalize_table(value)
                if normalized:
                    keys.add(normalized)

        return keys

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
                    content=dp.get("content"),
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

            if await self._should_gate_low_confidence_sql(state, output):
                fallback = await self._build_clarification_fallback(state)
                if fallback:
                    self._apply_clarification_response(
                        state,
                        fallback["questions"],
                        default_intro=fallback["answer"],
                    )
                else:
                    self._apply_clarification_response(
                        state,
                        ["Which table should I use to answer this?"],
                        default_intro=(
                            "I am not confident enough to run this query yet. "
                            "Please confirm the table first."
                        ),
                    )
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
            state["sql_formatter_fallback_calls"] = int(
                (output.data or {}).get("formatter_fallback_calls", 0)
            )
            state["sql_formatter_fallback_successes"] = int(
                (output.data or {}).get("formatter_fallback_successes", 0)
            )
            state["query_compiler_llm_calls"] = int(
                (output.data or {}).get("query_compiler_llm_calls", 0)
            )
            state["query_compiler_llm_refinements"] = int(
                (output.data or {}).get("query_compiler_llm_refinements", 0)
            )
            state["query_compiler_latency_ms"] = float(
                (output.data or {}).get("query_compiler_latency_ms", 0.0)
            )
            query_compiler_summary = (output.data or {}).get("query_compiler")
            if isinstance(query_compiler_summary, dict):
                state["query_compiler"] = query_compiler_summary
                self._record_decision(
                    state,
                    stage="query_compiler",
                    decision=str(query_compiler_summary.get("path") or "unknown"),
                    reason=str(query_compiler_summary.get("reason") or "n/a"),
                    details={
                        "confidence": query_compiler_summary.get("confidence"),
                        "selected_tables": query_compiler_summary.get("selected_tables", []),
                        "candidate_tables": query_compiler_summary.get("candidate_tables", []),
                        "operators": query_compiler_summary.get("operators", []),
                    },
                )

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
                max_rows=10,
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
            state["visualization_metadata"] = output.executed_query.visualization_metadata
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
                missing = error_message.split("Missing table in live schema:", 1)[-1].strip()
                missing = re.sub(r"\.\s*Schema refresh required\.?", "", missing).strip()
                state["answer_source"] = "clarification"
                state["natural_language_answer"] = (
                    f"I couldn't find `{missing}` in the connected database schema. "
                    "Please verify the table name or run `list tables` to choose a valid table."
                )
                state["answer_confidence"] = 0.3
                state["clarification_needed"] = True
                state["clarifying_questions"] = [
                    "Which existing table should I use instead?",
                ]
                state.pop("error", None)
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

    def _record_decision(
        self,
        state: PipelineState,
        *,
        stage: str,
        decision: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        trace = state.setdefault("decision_trace", [])
        entry: dict[str, Any] = {
            "stage": stage,
            "decision": decision,
            "reason": reason,
        }
        if details:
            entry["details"] = details
        trace.append(entry)

    def _should_continue_after_intent_gate(self, state: PipelineState) -> str:
        intent_gate = state.get("intent_gate")
        if intent_gate in {
            "exit",
            "out_of_scope",
            "small_talk",
            "setup_help",
            "datapoint_help",
            "clarify",
        }:
            self._record_decision(
                state,
                stage="continue_after_intent_gate",
                decision="end",
                reason=f"intent_gate={intent_gate}",
            )
            return "end"
        if state.get("fast_path"):
            self._record_decision(
                state,
                stage="continue_after_intent_gate",
                decision="sql",
                reason="fast_path",
            )
            return "sql"
        if self._should_run_tool_planner(state):
            self._record_decision(
                state,
                stage="continue_after_intent_gate",
                decision="tool_planner",
                reason="tool_planner_enabled_for_query",
            )
            return "tool_planner"
        self._record_decision(
            state,
            stage="continue_after_intent_gate",
            decision="classifier",
            reason="default_classifier_path",
        )
        return "classifier"

    def _should_run_tool_planner(self, state: PipelineState) -> bool:
        if not (self.tooling_enabled and self.tool_planner_enabled):
            return False
        pipeline_cfg = getattr(self.config, "pipeline", None)
        selective = (
            True
            if pipeline_cfg is None
            else bool(getattr(pipeline_cfg, "selective_tool_planner_enabled", True))
        )
        if not selective:
            return True
        return self._query_likely_requires_tools(state.get("query", ""))

    def _query_likely_requires_tools(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False
        tool_patterns = [
            r"\bprofile\b",
            r"\bdatapoint quality\b",
            r"\bquality report\b",
            r"\bsync datapoints?\b",
            r"\bgenerate datapoints?\b",
            r"\brun tool\b",
            r"\bexecute tool\b",
            r"\bapprove\b",
            r"\brefresh profile\b",
        ]
        return any(re.search(pattern, text) for pattern in tool_patterns)

    def _should_validate_sql(self, state: PipelineState) -> str:
        if state.get("clarification_needed"):
            return "clarify"
        return "validate"

    def _should_use_context_answer(self, state: PipelineState) -> str:
        if state.get("error"):
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="sql",
                reason="state_error_present",
            )
            return "sql"

        intent = state.get("intent") or "data_query"
        confidence = state.get("context_confidence") or 0.0
        query = (state.get("query") or "").lower()
        retrieved = state.get("retrieved_datapoints") or []

        if not retrieved:
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="sql",
                reason="no_retrieved_datapoints",
            )
            return "sql"

        if self._query_is_table_list(query):
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="sql",
                reason="deterministic_table_list_query",
            )
            return "sql"

        if "datapoint" in query or "data point" in query:
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="context",
                reason="datapoint_definition_request",
            )
            return "context"

        if self._query_is_definition_intent(query):
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="context",
                reason="definition_intent",
            )
            return "context"

        if self._query_requires_sql(query):
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="sql",
                reason="query_requires_sql_keywords",
            )
            return "sql"

        if intent in ("exploration", "explanation", "meta"):
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="context",
                reason=f"intent={intent}",
            )
            return "context"

        threshold = float(self.routing_policy["context_answer_confidence_threshold"])
        if confidence >= threshold:
            self._record_decision(
                state,
                stage="context_vs_sql",
                decision="context",
                reason="context_confidence_threshold_met",
                details={"confidence": confidence, "threshold": threshold},
            )
            return "context"

        self._record_decision(
            state,
            stage="context_vs_sql",
            decision="sql",
            reason="context_confidence_below_threshold",
            details={"confidence": confidence, "threshold": threshold},
        )
        return "sql"

    def _query_is_definition_intent(self, query: str) -> bool:
        patterns = [
            r"^\s*define\b",
            r"\bdefinition of\b",
            r"\bwhat does\b.*\b(mean|stand for)\b",
            r"\bmeaning of\b",
            r"\bhow is\b.*\b(calculated|computed|defined)\b",
            r"\bhow do (?:i|we|you)\b.*\bcalculate\b",
            r"\bbusiness rules?\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

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

    def _query_is_column_list(self, query: str) -> bool:
        patterns = [
            r"\bshow columns\b",
            r"\blist columns\b",
            r"\bwhat columns\b",
            r"\bwhich columns\b",
            r"\bdescribe table\b",
            r"\btable schema\b",
            r"\bcolumn list\b",
            r"\bfields in\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _should_use_tools(self, state: PipelineState) -> str:
        if state.get("tool_error"):
            self._record_decision(
                state,
                stage="tool_plan_resolution",
                decision="pipeline",
                reason="tool_error_present",
            )
            return "pipeline"
        tool_calls = state.get("tool_calls", [])
        tool_plan = state.get("tool_plan") or {}
        if tool_plan.get("fallback") == "pipeline":
            self._record_decision(
                state,
                stage="tool_plan_resolution",
                decision="pipeline",
                reason="tool_plan_fallback_pipeline",
            )
            return "pipeline"
        if tool_calls:
            self._record_decision(
                state,
                stage="tool_plan_resolution",
                decision="tools",
                reason="tool_calls_planned",
                details={"tool_calls": len(tool_calls)},
            )
            return "tools"
        self._record_decision(
            state,
            stage="tool_plan_resolution",
            decision="pipeline",
            reason="no_tool_calls",
        )
        return "pipeline"

    def _should_continue_after_tool_execution(self, state: PipelineState) -> str:
        if state.get("tool_error"):
            self._record_decision(
                state,
                stage="tool_execution_resolution",
                decision="pipeline",
                reason="tool_execution_error",
            )
            return "pipeline"
        if state.get("tool_approval_required"):
            self._record_decision(
                state,
                stage="tool_execution_resolution",
                decision="end",
                reason="tool_approval_required",
            )
            return "end"
        if state.get("tool_used") and state.get("natural_language_answer"):
            self._record_decision(
                state,
                stage="tool_execution_resolution",
                decision="end",
                reason="tool_answer_ready",
            )
            return "end"
        self._record_decision(
            state,
            stage="tool_execution_resolution",
            decision="pipeline",
            reason="continue_pipeline_after_tools",
        )
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
            "rate",
            "ratio",
            "percent",
            "percentage",
            "pct",
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
            synthesize_simple = state.get("synthesize_simple_sql")
            if synthesize_simple is None:
                pipeline_cfg = getattr(self.config, "pipeline", None)
                synthesize_simple = (
                    True
                    if pipeline_cfg is None
                    else bool(getattr(pipeline_cfg, "synthesize_simple_sql_answers", True))
                )
            if not synthesize_simple and self._is_simple_sql_response(state):
                return "end"
            return "synthesize"
        return "end"

    def _is_simple_sql_response(self, state: PipelineState) -> bool:
        sql = (state.get("validated_sql") or "").strip()
        if not sql:
            return False
        if re.search(r"\b(JOIN|GROUP\s+BY|WITH|UNION|OVER|HAVING)\b", sql, flags=re.IGNORECASE):
            return False

        query_result = state.get("query_result") or {}
        row_count = query_result.get("row_count")
        if row_count is None and isinstance(query_result.get("rows"), list):
            row_count = len(query_result.get("rows", []))
        try:
            row_count_num = int(row_count) if row_count is not None else 0
        except (TypeError, ValueError):
            row_count_num = 0
        if row_count_num > 10:
            return False

        columns = query_result.get("columns")
        if isinstance(columns, list) and len(columns) > 8:
            return False
        return True

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
            "last_goal": None,
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
            "target_subquery_index": None,
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

        target_subquery_index = None
        for question in last_clarifying_questions:
            target_subquery_index = self._extract_subquery_index(question)
            if target_subquery_index:
                break
        if target_subquery_index and previous_user_text:
            split_prior = self._split_multi_query(previous_user_text)
            if 1 <= target_subquery_index <= len(split_prior):
                previous_user_text = split_prior[target_subquery_index - 1]
                summary["last_goal"] = previous_user_text
                summary["target_subquery_index"] = target_subquery_index

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
                        summary["resolved_query"] = self._merge_query_with_table_hint(
                            previous_user_text,
                            cleaned_hint,
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
        history = (state.get("conversation_history") or [])[-12:]
        summary = state.get("intent_summary") or {}
        summary_text = self._format_intent_summary(summary)
        session_summary = (state.get("session_summary") or "").strip()

        system_messages: list[Message] = []
        if session_summary:
            system_messages.append({"role": "system", "content": f"Session memory: {session_summary}"})
        if summary_text:
            system_messages.append({"role": "system", "content": summary_text})
        if not system_messages:
            return history
        return [*system_messages, *history]

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
        if summary.get("target_subquery_index"):
            parts.append(f"target_subquery=Q{summary['target_subquery_index']}")
        if not parts:
            return None
        return "Intent summary: " + " | ".join(parts)

    def _merge_session_state_into_summary(
        self,
        summary: dict[str, Any],
        session_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not session_state:
            return summary

        merged = dict(summary)
        slots = dict(merged.get("slots") or {})
        prior_slots = session_state.get("slots") if isinstance(session_state, dict) else {}
        if isinstance(prior_slots, dict):
            for key, value in prior_slots.items():
                if value and not slots.get(key):
                    slots[key] = value
        merged["slots"] = slots

        for key in ("table_hints", "column_hints", "last_clarifying_questions"):
            current = list(merged.get(key) or [])
            prior = list(session_state.get(key) or [])
            combined: list[str] = []
            for value in [*prior, *current]:
                if value and value not in combined:
                    combined.append(value)
            merged[key] = combined

        merged["clarification_count"] = max(
            int(merged.get("clarification_count", 0) or 0),
            int(session_state.get("clarification_count", 0) or 0),
        )

        if not merged.get("last_goal") and session_state.get("last_goal"):
            merged["last_goal"] = str(session_state.get("last_goal"))
        if not merged.get("target_subquery_index") and session_state.get("target_subquery_index"):
            merged["target_subquery_index"] = session_state.get("target_subquery_index")
        if session_state.get("any_table"):
            merged["any_table"] = True
        if not merged.get("resolved_query") and session_state.get("resolved_query"):
            merged["resolved_query"] = str(session_state.get("resolved_query"))
        if not merged.get("last_clarifying_question"):
            prior_questions = merged.get("last_clarifying_questions") or []
            if prior_questions:
                merged["last_clarifying_question"] = prior_questions[0]
        return merged

    def _rewrite_contextual_followup(self, query: str, summary: dict[str, Any]) -> str | None:
        text = (query or "").strip()
        if not text:
            return None
        if not self._is_contextual_followup_query(text):
            return None
        if self._contains_data_keywords(text):
            return None

        last_goal = str(summary.get("last_goal") or "").strip()
        if not last_goal:
            return None
        focus = self._extract_followup_focus(text)
        if not focus:
            return None

        last_goal_lower = last_goal.lower()
        if "how many " in last_goal_lower:
            return f"How many {focus} do we have?"
        if last_goal_lower.startswith("list "):
            return f"List {focus}"
        if last_goal_lower.startswith("show "):
            return f"Show {focus}"
        if "total " in last_goal_lower:
            return f"What is total {focus}?"
        return None

    def _is_contextual_followup_query(self, text: str) -> bool:
        lowered = text.strip().lower()
        patterns = [
            r"^what\s+about\b",
            r"^how\s+about\b",
            r"^what\s+of\b",
            r"^and\b",
            r"^about\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _extract_followup_focus(self, text: str) -> str | None:
        cleaned = text.strip().strip("\"'").strip()
        cleaned = re.sub(r"^(what\s+about|how\s+about|what\s+of|and|about)\s+", "", cleaned, flags=re.I)
        cleaned = cleaned.strip(" .,!?:;\"'")
        cleaned = re.sub(r"^(the|our|their)\s+", "", cleaned, flags=re.I)
        if not cleaned:
            return None
        return cleaned.lower()

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
        focus_hint = self._extract_focus_hint(state.get("query") or "")
        focus_text = f" for {focus_hint}" if focus_hint else ""
        table_list = ", ".join(ranked)
        answer = (
            "I still need a bit more detail to generate SQL. "
            f"Here are a few tables that look relevant: {table_list}. "
            "Which table should I use?"
        )
        return {
            "answer": answer,
            "questions": [f"Which table should I use{focus_text}?"],
        }

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

    async def _should_gate_low_confidence_sql(
        self,
        state: PipelineState,
        output: Any,
    ) -> bool:
        """Block low-confidence semantic SQL and request clarification first."""
        query = (state.get("query") or "").strip()
        if not query:
            return False
        if self._is_deterministic_sql_query(query):
            return False
        if self._query_is_table_list(query.lower()) or self._query_is_column_list(query.lower()):
            return False
        if self._extract_table_reference(query.lower()):
            return False

        confidence = float(output.generated_sql.confidence or 0.0)
        threshold = float(self.routing_policy["semantic_sql_clarification_confidence_threshold"])
        if confidence >= threshold:
            return False

        if self._extract_focus_hint(query):
            return True

        if self._contains_data_keywords(query.lower()):
            return True

        summary = state.get("intent_summary") or {}
        return bool(self._is_ambiguous_intent(state, summary))

    def _classify_intent_gate(self, query: str) -> str:
        text = query.strip().lower()
        if not text:
            return "data_query"
        if self._is_exit_intent(text):
            return "exit"
        if self._is_datapoint_help_intent(text):
            return "datapoint_help"
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
        if intent == "datapoint_help":
            return (
                "You can manage and inspect DataPoints without writing SQL. "
                "In the UI, open Database Manager and review Pending/Approved DataPoints. "
                "In the CLI, use `datachat dp list` for indexed DataPoints and "
                "`datachat pending list` for approval queue."
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
        if re.search(r"\bnever\s*mind\b", text):
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

    def _merge_query_with_table_hint(self, previous_query: str, table_hint: str) -> str:
        base = previous_query.strip()
        hint = table_hint.strip().strip("`").strip('"')
        if not base or not hint:
            return previous_query

        lower = base.lower()
        limit_match = re.search(r"\b(first|top|limit|show)\s+(\d+)\s+rows?\b", lower)
        if limit_match:
            limit = max(1, min(int(limit_match.group(2)), 10))
            return f"Show {limit} rows from {hint}"
        if re.search(r"\b(show|sample|preview)\b.*\brows?\b", lower):
            return f"Show 3 rows from {hint}"
        if "column" in lower or "columns" in lower or "fields" in lower:
            return f"Show columns in {hint}"
        if re.search(r"\b(row count|how many rows|records?)\b", lower):
            return f"How many rows are in {hint}?"
        return f"{base.rstrip('. ')} Use table {hint}."

    def _extract_focus_hint(self, query: str) -> str | None:
        lowered = query.lower()
        for token in (
            "revenue",
            "sales",
            "growth",
            "churn",
            "retention",
            "conversion",
            "registrations",
            "orders",
            "users",
            "customers",
        ):
            if token in lowered:
                return token
        return None

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

    def _is_datapoint_help_intent(self, text: str) -> bool:
        if re.search(r"\b(explain|definition|meaning|calculate|metric|sql|query)\b", text):
            return False
        patterns = [
            r"^\s*(show|list|view)\s+(all\s+)?data\s*points?\b",
            r"^\s*(show|list|view)\s+(approved|pending|managed)\s+data\s*points?\b",
            r"^\s*available\s+data\s*points?\b",
            r"^\s*what\s+data\s*points?\s+(are\s+available|do\s+i\s+have)\b",
            r"^\s*data\s*points?\s+(list|overview)\b",
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
        if self._query_is_table_list(text) or self._query_is_column_list(text):
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
        max_tokens = int(self.routing_policy["ambiguous_query_max_tokens"])
        return len(query.split()) <= max_tokens

    async def _llm_intent_gate(
        self, query: str, summary: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, int]:
        summary_text = self._format_intent_summary(summary) or "None"
        system_prompt = (
            "You are a fast intent router for a data assistant. "
            "Classify the user's message into one of: "
            "data_query, exit, out_of_scope, small_talk, setup_help, datapoint_help, clarify. "
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
        allowed = {
            "data_query",
            "exit",
            "out_of_scope",
            "small_talk",
            "setup_help",
            "datapoint_help",
            "clarify",
        }
        if intent not in allowed:
            return None
        return {
            "intent": intent,
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "clarifying_question": data.get("clarifying_question"),
        }

    def _is_short_followup(self, text: str) -> bool:
        candidate = text.strip().lower()
        if ":" in candidate:
            candidate = candidate.rsplit(":", 1)[-1].strip()
        tokens = [token for token in candidate.split() if token]
        if not (0 < len(tokens) <= 5):
            return False
        disallowed = {
            "show",
            "list",
            "count",
            "select",
            "describe",
            "rows",
            "columns",
            "help",
        }
        return not any(token in disallowed for token in tokens)

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
        candidate = text.strip()
        if ":" in candidate:
            candidate = candidate.rsplit(":", 1)[-1].strip()
        cleaned = re.sub(r"[^\w.]+", " ", candidate.lower()).strip()
        if not cleaned:
            return None
        tokens = [
            token
            for token in cleaned.split()
            if token
            and token
            not in {
                "table",
                "column",
                "field",
                "use",
                "the",
                "a",
                "an",
                "any",
                "which",
                "what",
                "how",
                "should",
                "show",
                "list",
                "rows",
                "columns",
                "for",
                "i",
                "to",
                "do",
                "we",
            }
        ]
        if not tokens:
            return None
        if len(tokens) > 3:
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

    def _build_initial_state(
        self,
        *,
        query: str,
        conversation_history: list[Message] | None,
        session_summary: str | None,
        session_state: dict[str, Any] | None,
        database_type: str,
        database_url: str | None,
        target_connection_id: str | None,
        synthesize_simple_sql: bool | None,
        correlation_prefix: str,
    ) -> PipelineState:
        return {
            "query": query,
            "original_query": None,
            "conversation_history": conversation_history or [],
            "session_summary": session_summary,
            "session_state": session_state or {},
            "database_type": database_type,
            "database_url": database_url,
            "target_connection_id": target_connection_id,
            "user_id": "anonymous",
            "correlation_id": f"{correlation_prefix}-{int(time.time() * 1000)}",
            "tool_approved": False,
            "intent_gate": None,
            "intent_summary": None,
            "clarification_turn_count": 0,
            "clarification_limit": self.max_clarifications,
            "fast_path": False,
            "skip_response_synthesis": False,
            "synthesize_simple_sql": synthesize_simple_sql,
            "current_agent": None,
            "error": None,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "agent_timings": {},
            "decision_trace": [],
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
            "visualization_metadata": None,
            "used_datapoints": [],
            "assumptions": [],
            "sql_formatter_fallback_calls": 0,
            "sql_formatter_fallback_successes": 0,
            "query_compiler_llm_calls": 0,
            "query_compiler_llm_refinements": 0,
            "query_compiler_latency_ms": 0.0,
            "query_compiler": None,
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
            "sub_answers": [],
        }

    def _split_multi_query(self, query: str) -> list[str]:
        text = (query or "").strip()
        if not text:
            return []
        if len(text) < 20:
            return [text]

        parts: list[str] = []
        if text.count("?") >= 1:
            parts = [segment.strip(" ?\n\t") for segment in re.split(r"\?\s*", text)]
            parts = [segment for segment in parts if segment]
        if len(parts) <= 1:
            connector_split = re.split(
                r"\s+(?:and then|then|also|plus|and)\s+"
                r"(?=(?:what|how|show|list|give|define|explain|which|who|where|when|is|are|do|does|count|sum)\b)",
                text,
                flags=re.IGNORECASE,
            )
            connector_split = [segment.strip(" .") for segment in connector_split if segment.strip()]
            if len(connector_split) > 1:
                parts = connector_split

        if len(parts) <= 1:
            return [text]

        normalized: list[str] = []
        for part in parts[: self.max_subqueries]:
            candidate = part.strip(" .")
            if not candidate or len(candidate) < 4:
                continue
            normalized.append(candidate)
        if len(normalized) <= 1:
            return [text]
        return normalized

    def _extract_subquery_index(self, text: str) -> int | None:
        match = re.search(r"\[q(\d+)\]", text.strip(), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            value = int(match.group(1))
        except ValueError:
            return None
        return value if value > 0 else None

    async def _run_single_query(
        self,
        *,
        query: str,
        conversation_history: list[Message] | None = None,
        session_summary: str | None = None,
        session_state: dict[str, Any] | None = None,
        database_type: str = "postgresql",
        database_url: str | None = None,
        target_connection_id: str | None = None,
        synthesize_simple_sql: bool | None = None,
    ) -> PipelineState:
        initial_state = self._build_initial_state(
            query=query,
            conversation_history=conversation_history,
            session_summary=session_summary,
            session_state=session_state,
            database_type=database_type,
            database_url=database_url,
            target_connection_id=target_connection_id,
            synthesize_simple_sql=synthesize_simple_sql,
            correlation_prefix="local",
        )

        logger.info(f"Starting pipeline for query: {query[:100]}...")
        start_time = time.time()
        result = await self.graph.ainvoke(initial_state)
        self._normalize_answer_metadata(result)
        self._finalize_session_memory(result)
        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"Pipeline complete in {total_time:.1f}ms ({result.get('llm_calls', 0)} LLM calls)"
        )
        return result

    def _build_sub_answer(self, index: int, query: str, result: PipelineState) -> dict[str, Any]:
        answer = result.get("natural_language_answer")
        if not answer:
            if result.get("error"):
                answer = f"I encountered an error: {result.get('error')}"
            else:
                answer = "No answer generated"
        sql_text = result.get("validated_sql") or result.get("generated_sql")
        query_result = result.get("query_result")
        data: dict[str, list[Any]] | None = None
        if isinstance(query_result, dict):
            candidate = query_result.get("data")
            if isinstance(candidate, dict):
                data = candidate
            else:
                rows = query_result.get("rows")
                columns = query_result.get("columns")
                if isinstance(rows, list) and isinstance(columns, list):
                    data = {str(col): [row.get(col) for row in rows] for col in columns}
        return {
            "index": index,
            "query": query,
            "answer": answer,
            "answer_source": result.get("answer_source"),
            "answer_confidence": result.get("answer_confidence"),
            "sql": sql_text,
            "data": data,
            "visualization_hint": result.get("visualization_hint"),
            "visualization_metadata": result.get("visualization_metadata"),
            "clarifying_questions": result.get("clarifying_questions", []),
            "error": result.get("error"),
        }

    def _aggregate_multi_results(
        self,
        *,
        original_query: str,
        sub_results: list[PipelineState],
        sub_answers: list[dict[str, Any]],
        conversation_history: list[Message] | None,
        session_summary: str | None,
        session_state: dict[str, Any] | None,
        database_type: str,
        database_url: str | None,
        target_connection_id: str | None,
        synthesize_simple_sql: bool | None,
    ) -> PipelineState:
        merged = self._build_initial_state(
            query=original_query,
            conversation_history=conversation_history,
            session_summary=session_summary,
            session_state=session_state,
            database_type=database_type,
            database_url=database_url,
            target_connection_id=target_connection_id,
            synthesize_simple_sql=synthesize_simple_sql,
            correlation_prefix="local",
        )
        merged["sub_answers"] = sub_answers
        primary_sub_answer = None
        for item in sub_answers:
            if item.get("sql") or item.get("data"):
                primary_sub_answer = item
                break
        if primary_sub_answer is None and sub_answers:
            primary_sub_answer = sub_answers[0]

        section_lines = ["I handled your request as multiple questions:"]
        for item in sub_answers:
            section_lines.append(f"\n{item['index']}. {item['query']}")
            section_lines.append(str(item.get("answer") or "No answer generated"))
        merged["natural_language_answer"] = "\n".join(section_lines).strip()

        clarifications: list[str] = []
        for item in sub_answers:
            for question in item.get("clarifying_questions", []):
                clarifications.append(f"[Q{item['index']}] {question}")
        merged["clarifying_questions"] = clarifications
        merged["clarification_needed"] = bool(clarifications)

        merged["answer_source"] = "multi"
        if primary_sub_answer:
            merged["generated_sql"] = primary_sub_answer.get("sql")
            merged["visualization_hint"] = primary_sub_answer.get("visualization_hint")
            merged["visualization_metadata"] = primary_sub_answer.get("visualization_metadata")
            sub_data = primary_sub_answer.get("data")
            if isinstance(sub_data, dict):
                merged["query_result"] = {
                    "data": sub_data,
                    "rows": [],
                    "columns": list(sub_data.keys()),
                    "row_count": (
                        max((len(values) for values in sub_data.values()), default=0)
                        if sub_data
                        else 0
                    ),
                }

        confidence_values = [
            float(item.get("answer_confidence"))
            for item in sub_answers
            if item.get("answer_confidence") is not None
        ]
        if confidence_values:
            merged["answer_confidence"] = max(0.0, min(1.0, sum(confidence_values) / len(confidence_values)))

        merged["llm_calls"] = sum(int(result.get("llm_calls", 0) or 0) for result in sub_results)
        merged["retry_count"] = sum(int(result.get("retry_count", 0) or 0) for result in sub_results)
        merged["total_latency_ms"] = sum(
            float(result.get("total_latency_ms", 0.0) or 0.0) for result in sub_results
        )
        merged["sql_formatter_fallback_calls"] = sum(
            int(result.get("sql_formatter_fallback_calls", 0) or 0) for result in sub_results
        )
        merged["sql_formatter_fallback_successes"] = sum(
            int(result.get("sql_formatter_fallback_successes", 0) or 0) for result in sub_results
        )
        merged["query_compiler_llm_calls"] = sum(
            int(result.get("query_compiler_llm_calls", 0) or 0) for result in sub_results
        )
        merged["query_compiler_llm_refinements"] = sum(
            int(result.get("query_compiler_llm_refinements", 0) or 0) for result in sub_results
        )
        merged["query_compiler_latency_ms"] = sum(
            float(result.get("query_compiler_latency_ms", 0.0) or 0.0) for result in sub_results
        )
        for result in sub_results:
            summary = result.get("query_compiler")
            if isinstance(summary, dict):
                merged["query_compiler"] = summary
                break
        merged["decision_trace"] = [
            {
                "stage": "subquery",
                "decision": f"Q{idx + 1}",
                "reason": result.get("query"),
                "details": {
                    "answer_source": result.get("answer_source"),
                    "decision_trace": result.get("decision_trace", []),
                },
            }
            for idx, result in enumerate(sub_results)
        ]

        merged_agent_timings: dict[str, float] = {}
        for result in sub_results:
            for agent, duration in (result.get("agent_timings") or {}).items():
                merged_agent_timings[agent] = merged_agent_timings.get(agent, 0.0) + float(duration or 0.0)
        merged["agent_timings"] = merged_agent_timings

        all_sources: list[dict[str, Any]] = []
        seen_source_ids: set[str] = set()
        all_evidence: list[dict[str, Any]] = []
        seen_evidence_ids: set[tuple[str, str]] = set()
        errors: list[str] = []
        for result in sub_results:
            for dp in result.get("retrieved_datapoints", []):
                datapoint_id = str(dp.get("datapoint_id", ""))
                if datapoint_id and datapoint_id not in seen_source_ids:
                    seen_source_ids.add(datapoint_id)
                    all_sources.append(dp)
            for item in result.get("evidence", []):
                key = (str(item.get("datapoint_id", "")), str(item.get("reason", "")))
                if key not in seen_evidence_ids:
                    seen_evidence_ids.add(key)
                    all_evidence.append(item)
            if result.get("error"):
                errors.append(str(result["error"]))

        merged["retrieved_datapoints"] = all_sources
        merged["evidence"] = all_evidence
        merged["validation_errors"] = [
            item for result in sub_results for item in result.get("validation_errors", [])
        ]
        merged["validation_warnings"] = [
            item for result in sub_results for item in result.get("validation_warnings", [])
        ]
        if errors and not merged["natural_language_answer"]:
            merged["error"] = errors[0]

        self._normalize_answer_metadata(merged)
        if sub_results:
            merged["session_summary"] = sub_results[-1].get("session_summary")
            merged["session_state"] = sub_results[-1].get("session_state")
        self._finalize_session_memory(merged)
        return merged

    # ========================================================================
    # Public API
    # ========================================================================

    async def run(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        session_summary: str | None = None,
        session_state: dict[str, Any] | None = None,
        database_type: str = "postgresql",
        database_url: str | None = None,
        target_connection_id: str | None = None,
        synthesize_simple_sql: bool | None = None,
    ) -> PipelineState:
        """
        Run pipeline synchronously (wait for completion).

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages
            session_summary: Compact summary carried across turns
            session_state: Structured session memory carried across turns
            database_type: Database type (postgresql, clickhouse, mysql)
            database_url: Database URL override for execution

        Returns:
            Final pipeline state with all outputs
        """
        parts = self._split_multi_query(query)
        if len(parts) <= 1:
            return await self._run_single_query(
                query=query,
                conversation_history=conversation_history,
                session_summary=session_summary,
                session_state=session_state,
                database_type=database_type,
                database_url=database_url,
                target_connection_id=target_connection_id,
                synthesize_simple_sql=synthesize_simple_sql,
            )

        sub_results: list[PipelineState] = []
        sub_answers: list[dict[str, Any]] = []
        for index, part in enumerate(parts, start=1):
            result = await self._run_single_query(
                query=part,
                conversation_history=conversation_history,
                session_summary=session_summary,
                session_state=session_state,
                database_type=database_type,
                database_url=database_url,
                target_connection_id=target_connection_id,
                synthesize_simple_sql=synthesize_simple_sql,
            )
            sub_results.append(result)
            sub_answers.append(self._build_sub_answer(index, part, result))

        return self._aggregate_multi_results(
            original_query=query,
            sub_results=sub_results,
            sub_answers=sub_answers,
            conversation_history=conversation_history,
            session_summary=session_summary,
            session_state=session_state,
            database_type=database_type,
            database_url=database_url,
            target_connection_id=target_connection_id,
            synthesize_simple_sql=synthesize_simple_sql,
        )

    async def stream(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        session_summary: str | None = None,
        session_state: dict[str, Any] | None = None,
        database_type: str = "postgresql",
        database_url: str | None = None,
        target_connection_id: str | None = None,
        synthesize_simple_sql: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run pipeline with streaming updates.

        Yields status updates as each agent completes.

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages
            session_summary: Compact summary carried across turns
            session_state: Structured session memory carried across turns
            database_type: Database type
            database_url: Database URL override for execution

        Yields:
            Status updates with current agent and progress
        """
        parts = self._split_multi_query(query)
        if len(parts) > 1:
            result = await self.run(
                query=query,
                conversation_history=conversation_history,
                session_summary=session_summary,
                session_state=session_state,
                database_type=database_type,
                database_url=database_url,
                target_connection_id=target_connection_id,
                synthesize_simple_sql=synthesize_simple_sql,
            )
            yield {
                "node": "MultiQueryAggregator",
                "current_agent": "MultiQueryAggregator",
                "status": "completed",
                "state": result,
            }
            return

        initial_state = self._build_initial_state(
            query=query,
            conversation_history=conversation_history,
            session_summary=session_summary,
            session_state=session_state,
            database_type=database_type,
            database_url=database_url,
            target_connection_id=target_connection_id,
            synthesize_simple_sql=synthesize_simple_sql,
            correlation_prefix="stream",
        )

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
        session_summary: str | None = None,
        session_state: dict[str, Any] | None = None,
        database_type: str = "postgresql",
        database_url: str | None = None,
        target_connection_id: str | None = None,
        synthesize_simple_sql: bool | None = None,
        event_callback: Any = None,
    ) -> PipelineState:
        """
        Run pipeline with callback-based streaming for WebSocket support.

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages
            session_summary: Compact summary carried across turns
            session_state: Structured session memory carried across turns
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

        parts = self._split_multi_query(query)
        if len(parts) > 1:
            if event_callback:
                await event_callback(
                    "decompose_complete",
                    {
                        "parts": parts,
                        "part_count": len(parts),
                    },
                )
            return await self.run(
                query=query,
                conversation_history=conversation_history,
                session_summary=session_summary,
                session_state=session_state,
                database_type=database_type,
                database_url=database_url,
                target_connection_id=target_connection_id,
                synthesize_simple_sql=synthesize_simple_sql,
            )

        initial_state = self._build_initial_state(
            query=query,
            conversation_history=conversation_history,
            session_summary=session_summary,
            session_state=session_state,
            database_type=database_type,
            database_url=database_url,
            target_connection_id=target_connection_id,
            synthesize_simple_sql=synthesize_simple_sql,
            correlation_prefix="stream",
        )

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
            self._normalize_answer_metadata(final_state)
            self._finalize_session_memory(final_state)

        logger.info(
            f"Pipeline streaming complete in {total_latency_ms:.1f}ms "
            f"({final_state.get('llm_calls', 0) if final_state else 0} LLM calls)"
        )

        return final_state or initial_state

    def _finalize_session_memory(self, state: PipelineState) -> None:
        """Persist compact memory fields for the next turn."""
        intent_summary = dict(state.get("intent_summary") or {})
        session_state = dict(state.get("session_state") or {})

        merged = self._merge_session_state_into_summary(intent_summary, session_state)
        merged["clarification_count"] = self._current_clarification_count(state)

        latest_goal = (state.get("original_query") or state.get("query") or "").strip()
        if latest_goal:
            merged["last_goal"] = latest_goal

        questions = state.get("clarifying_questions") or []
        if questions:
            merged["last_clarifying_questions"] = questions[:3]
            merged["last_clarifying_question"] = questions[0]

        sql_text = (state.get("validated_sql") or state.get("generated_sql") or "").strip()
        if sql_text:
            table_hint = self._extract_table_reference(sql_text)
            if table_hint:
                table_hints = list(merged.get("table_hints") or [])
                if table_hint not in table_hints:
                    table_hints.append(table_hint)
                merged["table_hints"] = table_hints
                slots = dict(merged.get("slots") or {})
                slots["table"] = slots.get("table") or table_hint
                merged["slots"] = slots

        merged["updated_at"] = int(time.time())
        state["session_state"] = merged
        state["session_summary"] = self._format_intent_summary(merged)

    def _normalize_answer_metadata(self, state: PipelineState) -> None:
        """Ensure answer source/confidence are consistently populated."""
        source = state.get("answer_source")
        if not source:
            if state.get("tool_approval_required"):
                source = "approval"
            elif state.get("clarification_needed") or state.get("clarifying_questions"):
                source = "clarification"
            elif state.get("error"):
                source = "error"
            elif (
                state.get("validated_sql")
                or state.get("generated_sql")
                or state.get("query_result")
            ):
                source = "sql"
            elif state.get("intent_gate") in {"exit", "out_of_scope", "small_talk", "setup_help"}:
                source = "system"
            elif state.get("natural_language_answer"):
                source = "context"
            else:
                source = "error"
            state["answer_source"] = source

        if state.get("answer_confidence") is None:
            defaults = {
                "sql": 0.7,
                "context": 0.6,
                "clarification": 0.2,
                "system": 0.8,
                "approval": 0.5,
                "multi": 0.65,
                "error": 0.0,
            }
            state["answer_confidence"] = defaults.get(source, 0.5)
        else:
            confidence = float(state.get("answer_confidence", 0.5))
            state["answer_confidence"] = max(0.0, min(1.0, confidence))


# ============================================================================
# Helper Functions
# ============================================================================


async def create_pipeline(
    database_type: str | None = None,
    database_url: str | None = None,
) -> DataChatPipeline:
    """
    Create a DataChatPipeline with all dependencies initialized.

    Args:
        database_type: Database type (postgresql, clickhouse, mysql). If omitted,
            inferred from database URL.
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

    db_url_str = str(db_url) if not isinstance(db_url, str) else db_url
    connector = create_connector(
        database_url=db_url_str,
        database_type=database_type,
        pool_size=config.database.pool_size,
    )

    await connector.connect()

    # Create pipeline
    pipeline = DataChatPipeline(
        retriever=retriever,
        connector=connector,
        max_retries=3,
    )

    return pipeline
