"""
SQLAgent

Generates SQL queries from natural language using LLM and retrieved context.
Includes self-correction capability to fix syntax errors, missing columns, etc.

The SQLAgent:
1. Takes user query + InvestigationMemory from ContextAgent
2. Builds a prompt with schema context, business rules, and examples
3. Generates SQL using configured LLM (GPT-4o, Claude, etc.)
4. Self-validates the generated SQL
5. Self-corrects if issues are found (max 3 attempts)
6. Returns GeneratedSQL with explanation and metadata
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from backend.agents.base import BaseAgent
from backend.config import get_settings
from backend.connectors.base import BaseConnector
from backend.connectors.factory import create_connector
from backend.database.catalog import CatalogIntelligence
from backend.database.catalog_templates import (
    get_catalog_aliases,
    get_catalog_schemas,
    get_list_tables_query,
)
from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMMessage, LLMRequest
from backend.models.agent import (
    AgentMetadata,
    CorrectionAttempt,
    GeneratedSQL,
    LLMError,
    SQLAgentInput,
    SQLAgentOutput,
    SQLGenerationError,
    ValidationIssue,
)
from backend.profiling.cache import load_profile_cache
from backend.prompts.loader import PromptLoader

logger = logging.getLogger(__name__)
_INTERNAL_SERVICE_TABLES = {
    "database_connections",
    "profiling_jobs",
    "profiling_profiles",
    "pending_datapoints",
    "datapoint_generation_jobs",
}


@dataclass
class TableResolution:
    """Lightweight table-resolution plan produced before SQL generation."""

    candidate_tables: list[str]
    column_hints: list[str]
    confidence: float
    needs_clarification: bool = False
    clarifying_question: str | None = None


class SQLClarificationNeeded(Exception):
    """Raised when SQL generation needs user clarification."""

    def __init__(self, questions: list[str]) -> None:
        super().__init__("SQL generation needs clarification")
        self.questions = questions


class SQLAgent(BaseAgent):
    """
    SQL generation agent with self-correction.

    Generates SQL queries from natural language using LLM and context from
    ContextAgent. Includes self-correction to fix syntax errors, missing
    columns, table name issues, etc.

    Usage:
        agent = SQLAgent()

        input = SQLAgentInput(
            query="What were total sales last quarter?",
            investigation_memory=context_output.investigation_memory
        )

        output = await agent(input)
        sql = output.generated_sql.sql
    """

    def __init__(self, llm_provider=None):
        """
        Initialize SQLAgent with LLM provider.

        Args:
            llm_provider: Optional LLM provider. If None, creates default provider.
        """
        super().__init__(name="SQLAgent")

        # Get configuration
        self.config = get_settings()

        # Create LLM provider using factory (respects sql_provider override)
        if llm_provider is None:
            self.llm = LLMProviderFactory.create_agent_provider(
                agent_name="sql",
                config=self.config.llm,
                model_type="main",  # Use main model (GPT-4o) for SQL generation
            )
            self.fast_llm = LLMProviderFactory.create_agent_provider(
                agent_name="sql",
                config=self.config.llm,
                model_type="mini",
            )
            self.formatter_llm = self.fast_llm
        else:
            self.llm = llm_provider
            self.fast_llm = llm_provider
            self.formatter_llm = llm_provider

        provider_name = getattr(self.llm, "provider", "unknown")
        model_name = getattr(self.llm, "model", "unknown")
        logger.info(
            f"SQLAgent initialized with {provider_name} provider",
            extra={"provider": provider_name, "model": model_name},
        )
        self.prompts = PromptLoader()
        self.catalog = CatalogIntelligence()
        self._live_schema_cache: dict[str, str] = {}
        self._live_schema_snapshot_cache: dict[str, dict[str, Any]] = {}
        self._live_schema_tables_cache: dict[str, set[str]] = {}
        self._live_profile_cache: dict[str, dict[str, dict[str, object]]] = {}
        self._max_safe_row_limit = 10
        self._default_row_limit = 5

    def _pipeline_flag(self, name: str, default: bool) -> bool:
        pipeline_cfg = getattr(self.config, "pipeline", None)
        if pipeline_cfg is None:
            return default
        value = getattr(pipeline_cfg, name, default)
        if self._is_mock_value(value):
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            return default
        if isinstance(value, int):
            return bool(value)
        return default

    def _pipeline_int(self, name: str, default: int) -> int:
        pipeline_cfg = getattr(self.config, "pipeline", None)
        if pipeline_cfg is None:
            return default
        value = getattr(pipeline_cfg, name, default)
        if self._is_mock_value(value):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _pipeline_float(self, name: str, default: float) -> float:
        pipeline_cfg = getattr(self.config, "pipeline", None)
        if pipeline_cfg is None:
            return default
        value = getattr(pipeline_cfg, name, default)
        if self._is_mock_value(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_mock_value(value: Any) -> bool:
        return value.__class__.__module__.startswith("unittest.mock")

    def _providers_are_equivalent(self, primary: Any, secondary: Any) -> bool:
        """Return True when two providers resolve to the same effective model endpoint."""
        def _stored_attr(obj: Any, name: str) -> Any:
            data = getattr(obj, "__dict__", {})
            if isinstance(data, dict) and name in data:
                return data.get(name)
            return None

        if primary is secondary:
            return True
        if primary is None or secondary is None:
            return False

        primary_provider = (
            _stored_attr(primary, "provider_name")
            or _stored_attr(primary, "provider")
            or primary.__class__.__name__
        )
        secondary_provider = (
            _stored_attr(secondary, "provider_name")
            or _stored_attr(secondary, "provider")
            or secondary.__class__.__name__
        )
        if str(primary_provider).lower() != str(secondary_provider).lower():
            return False

        primary_model = _stored_attr(primary, "model")
        secondary_model = _stored_attr(secondary, "model")
        if (primary_model is None) != (secondary_model is None):
            return False
        if primary_model is not None and str(primary_model).lower() != str(secondary_model).lower():
            return False

        primary_base_url = _stored_attr(primary, "base_url")
        secondary_base_url = _stored_attr(secondary, "base_url")
        if (primary_base_url is None) != (secondary_base_url is None):
            return False
        if primary_base_url is not None and str(primary_base_url).rstrip("/") != str(
            secondary_base_url
        ).rstrip("/"):
            return False

        return True

    async def execute(self, input: SQLAgentInput) -> SQLAgentOutput:
        """
        Execute SQL generation with self-correction.

        Args:
            input: SQLAgentInput with query and investigation memory

        Returns:
            SQLAgentOutput with generated SQL and correction history

        Raises:
            SQLGenerationError: If SQL generation fails after all attempts
        """
        # Validate input type
        self._validate_input(input)

        logger.info(
            f"Generating SQL for query: {input.query[:100]}...",
            extra={
                "query_length": len(input.query),
                "num_datapoints": len(input.investigation_memory.datapoints),
            },
        )

        metadata = AgentMetadata(agent_name=self.name)
        correction_attempts: list[CorrectionAttempt] = []
        runtime_stats = {
            "formatter_fallback_calls": 0,
            "formatter_fallback_successes": 0,
        }

        try:
            # Initial SQL generation
            try:
                generated_sql = await self._generate_sql(input, metadata, runtime_stats)
            except SQLClarificationNeeded as exc:
                generated_sql = GeneratedSQL(
                    sql="SELECT 1",
                    explanation="Clarification needed before generating SQL.",
                    used_datapoints=[],
                    confidence=0.0,
                    assumptions=[],
                    clarifying_questions=exc.questions,
                )
                return SQLAgentOutput(
                    success=True,
                    data=runtime_stats,
                    metadata=metadata,
                    next_agent="ValidatorAgent",
                    generated_sql=generated_sql,
                    correction_attempts=[],
                    needs_clarification=True,
                )

            # Self-validation
            issues = self._validate_sql(generated_sql, input)

            # Self-correction loop if issues found
            attempt_number = 1
            while issues and attempt_number <= input.max_correction_attempts:
                logger.warning(
                    f"SQL validation found {len(issues)} issues, attempting correction #{attempt_number}",
                    extra={"issues": [issue.issue_type for issue in issues]},
                )

                # Record correction attempt
                original_sql = generated_sql.sql

                # Attempt correction
                corrected_sql = await self._correct_sql(
                    generated_sql=generated_sql,
                    issues=issues,
                    input=input,
                    metadata=metadata,
                    runtime_stats=runtime_stats,
                )

                # Validate corrected SQL
                new_issues = self._validate_sql(corrected_sql, input)
                success = len(new_issues) == 0

                # Record attempt
                correction_attempts.append(
                    CorrectionAttempt(
                        attempt_number=attempt_number,
                        original_sql=original_sql,
                        issues_found=issues,
                        corrected_sql=corrected_sql.sql,
                        success=success,
                    )
                )

                # Update for next iteration
                generated_sql = corrected_sql
                issues = new_issues
                attempt_number += 1

            # Check if we have unresolved issues
            if issues:
                logger.error(
                    f"Failed to resolve {len(issues)} validation issues after {input.max_correction_attempts} attempts",
                    extra={"issues": [issue.message for issue in issues]},
                )
                # Still return the best attempt we have, but mark needs_clarification
                needs_clarification = True
            else:
                needs_clarification = bool(generated_sql.clarifying_questions)

            logger.info(
                "SQL generation complete",
                extra={
                    "correction_attempts": len(correction_attempts),
                    "needs_clarification": needs_clarification,
                    "confidence": generated_sql.confidence,
                    "prompt_version": self.prompts.get_metadata(
                        "agents/sql_generator.md"
                    ).get("version"),
                },
            )

            return SQLAgentOutput(
                success=True,
                data=runtime_stats,
                metadata=metadata,
                next_agent="ValidatorAgent",
                generated_sql=generated_sql,
                correction_attempts=correction_attempts,
                needs_clarification=needs_clarification,
            )

        except Exception as e:
            metadata.error = str(e)
            logger.error(f"SQL generation failed: {e}", exc_info=True)
            raise SQLGenerationError(
                agent=self.name,
                message=f"Failed to generate SQL: {e}",
                recoverable=False,
                context={"query": input.query},
            ) from e

    async def _generate_sql(
        self, input: SQLAgentInput, metadata: AgentMetadata, runtime_stats: dict[str, int]
    ) -> GeneratedSQL:
        """
        Generate SQL from user query and context.

        Args:
            input: SQLAgentInput
            metadata: AgentMetadata to track LLM calls

        Returns:
            GeneratedSQL with query and metadata

        Raises:
            LLMError: If LLM call fails
        """
        resolved_query, _ = self._resolve_followup_query(input)
        if resolved_query != input.query:
            input = input.model_copy(update={"query": resolved_query})

        catalog_plan = self.catalog.plan_query(
            query=input.query,
            database_type=input.database_type,
            investigation_memory=input.investigation_memory,
        )
        if catalog_plan and self._should_bypass_catalog_plan(input.query, catalog_plan):
            catalog_plan = None
        if catalog_plan and catalog_plan.clarifying_questions:
            raise SQLClarificationNeeded(catalog_plan.clarifying_questions)

        if catalog_plan and catalog_plan.sql:
            generated = GeneratedSQL(
                sql=catalog_plan.sql,
                explanation=catalog_plan.explanation,
                used_datapoints=[],
                confidence=catalog_plan.confidence,
                assumptions=[],
                clarifying_questions=[],
            )
            return self._apply_row_limit_policy(generated, input.query)

        table_resolution = await self._resolve_tables_with_llm(input)
        resolver_threshold = self._pipeline_float("sql_table_resolver_confidence_threshold", 0.55)
        if (
            table_resolution
            and table_resolution.needs_clarification
            and (
                not table_resolution.candidate_tables
                or table_resolution.confidence < resolver_threshold
            )
        ):
            question = table_resolution.clarifying_question or "Which table should I use to answer this?"
            raise SQLClarificationNeeded([question])

        # Build prompt with context
        prompt = await self._build_generation_prompt(input, table_resolution=table_resolution)

        # Create LLM request
        llm_request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=self._get_system_prompt()),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,  # Deterministic for SQL generation
            max_tokens=2000,
        )

        try:
            use_two_stage = (
                self._pipeline_flag("sql_two_stage_enabled", True)
                and not self._providers_are_equivalent(self.fast_llm, self.llm)
            )

            if use_two_stage:
                fast_generated = await self._request_sql_from_llm(
                    provider=self.fast_llm,
                    llm_request=llm_request,
                    input=input,
                    runtime_stats=runtime_stats,
                )
                if self._should_accept_fast_sql(fast_generated, input):
                    generated_sql = fast_generated
                else:
                    generated_sql = await self._request_sql_from_llm(
                        provider=self.llm,
                        llm_request=llm_request,
                        input=input,
                        runtime_stats=runtime_stats,
                    )
            else:
                generated_sql = await self._request_sql_from_llm(
                    provider=self.llm,
                    llm_request=llm_request,
                    input=input,
                    runtime_stats=runtime_stats,
                )

            logger.debug(
                f"Generated SQL: {generated_sql.sql[:200]}...",
                extra={"confidence": generated_sql.confidence},
            )

            return generated_sql

        except SQLClarificationNeeded:
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise LLMError(
                agent=self.name,
                message=f"LLM generation failed: {e}",
                context={"query": input.query},
            ) from e

    async def _resolve_tables_with_llm(self, input: SQLAgentInput) -> TableResolution | None:
        """Use a lightweight LLM step to infer likely source tables before SQL generation."""
        if not self._pipeline_flag("sql_table_resolver_enabled", False):
            return None
        if not self._should_run_table_resolver(input):
            return None

        table_columns = self._schema_table_columns_from_memory(input)
        if len(table_columns) < 2:
            live_candidates = await self._load_live_table_candidates_for_fallback(input)
            for table_name, columns in live_candidates:
                existing = table_columns.get(table_name, set())
                table_columns[table_name] = existing | {column.lower() for column in columns}
        if len(table_columns) < 2:
            return None

        scores = {
            table_name: self._score_table_candidate(input.query, table_name, columns)
            for table_name, columns in table_columns.items()
        }
        if scores and max(scores.values()) <= 0:
            return None

        ranked = self._rank_table_candidates_for_query(input.query, table_columns)
        max_tables = self._pipeline_int("sql_table_resolver_max_tables", 20)
        limited_tables = ranked[:max_tables]
        if len(limited_tables) < 2:
            return None

        prompt = self._build_table_resolver_prompt(
            query=input.query,
            table_columns=table_columns,
            ranked_tables=limited_tables,
        )
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are a SQL table resolver. Return strict JSON only. "
                        "Do not include markdown fences."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            max_tokens=400,
        )

        try:
            response = await self.fast_llm.generate(request)
            self._track_llm_call(tokens=response.usage.total_tokens)
            parsed = self._parse_table_resolver_response(
                content=response.content,
                ranked_tables=limited_tables,
                table_columns=table_columns,
            )
            if parsed is None:
                return None
            threshold = self._pipeline_float("sql_table_resolver_confidence_threshold", 0.55)
            if parsed.needs_clarification and parsed.confidence < threshold:
                return parsed
            if parsed.needs_clarification and parsed.candidate_tables and parsed.confidence >= threshold:
                return TableResolution(
                    candidate_tables=parsed.candidate_tables,
                    column_hints=parsed.column_hints,
                    confidence=parsed.confidence,
                    needs_clarification=False,
                    clarifying_question=None,
                )
            if not parsed.candidate_tables and parsed.confidence < threshold:
                clarifying = parsed.clarifying_question or self._build_table_resolver_question(
                    limited_tables
                )
                return TableResolution(
                    candidate_tables=[],
                    column_hints=[],
                    confidence=parsed.confidence,
                    needs_clarification=True,
                    clarifying_question=clarifying,
                )
            return parsed
        except Exception as exc:
            logger.debug(f"Table resolver fallback skipped due to error: {exc}")
            return None

    def _should_run_table_resolver(self, input: SQLAgentInput) -> bool:
        query = (input.query or "").strip()
        if not query:
            return False
        if self._extract_explicit_table_name(query):
            return False
        if len(query.split()) < 4:
            return False

        lowered = query.lower()
        if any(term in lowered for term in ("exit", "quit", "goodbye", "thanks", "thank you")):
            return False
        if self.catalog.is_list_tables_query(lowered):
            return False
        return True

    def _should_bypass_catalog_plan(self, query: str, catalog_plan: Any) -> bool:
        """Avoid catalog sample-row shortcuts for semantic analytic questions."""
        operation = str(getattr(catalog_plan, "operation", "")).lower()
        if operation != "sample_rows":
            return False
        text = query.lower()
        semantic_markers = (
            " risk",
            " rate",
            " trend",
            " by ",
            " total ",
            " revenue",
            " margin",
            " waste",
            " stockout",
            " average ",
            " sum ",
        )
        return any(marker in f" {text} " for marker in semantic_markers)

    def _rank_table_candidates_for_query(
        self, query: str, table_columns: dict[str, set[str]]
    ) -> list[str]:
        scored: list[tuple[int, str]] = []
        for table_name, columns in table_columns.items():
            score = self._score_table_candidate(query, table_name, columns)
            scored.append((score, table_name))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [table for _, table in scored]

    def _score_table_candidate(self, query: str, table_name: str, columns: set[str]) -> int:
        tokens = self._tokenize_query(query)
        if not tokens:
            return 0

        score = 0
        table_tokens = re.findall(r"[a-z0-9_]+", table_name.lower().replace(".", "_"))
        column_tokens = {col.lower() for col in columns}
        for token in tokens:
            if token in table_tokens:
                score += 4
            if any(token in piece for piece in table_tokens):
                score += 1
            if token in column_tokens:
                score += 3
            if any(token in col for col in column_tokens):
                score += 1
        return score

    def _build_table_resolver_prompt(
        self,
        *,
        query: str,
        table_columns: dict[str, set[str]],
        ranked_tables: list[str],
    ) -> str:
        lines = []
        for idx, table_name in enumerate(ranked_tables, start=1):
            columns = sorted(table_columns.get(table_name, set()))
            preview = ", ".join(columns[:10]) if columns else "no column metadata"
            lines.append(f"{idx}. {table_name} | columns: {preview}")

        return (
            "Choose the table(s) most likely needed to answer the question.\n"
            "Return JSON with keys:\n"
            '- candidate_tables: array of table names (use names exactly from CANDIDATE_TABLES)\n'
            '- column_hints: array of relevant columns\n'
            "- confidence: number 0..1\n"
            "- needs_clarification: boolean\n"
            "- clarifying_question: string|null\n\n"
            "Use needs_clarification=true only when multiple plausible tables remain and "
            "you cannot choose confidently.\n\n"
            f"QUESTION:\n{query}\n\n"
            f"CANDIDATE_TABLES:\n" + "\n".join(lines)
        )

    def _parse_table_resolver_response(
        self,
        *,
        content: str,
        ranked_tables: list[str],
        table_columns: dict[str, set[str]],
    ) -> TableResolution | None:
        json_text = None
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced:
            json_text = fenced.group(1)
        else:
            obj = re.search(r"\{.*\}", content, re.DOTALL)
            if obj:
                json_text = obj.group(0)
        if not json_text:
            return None

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        allowed = {table.lower(): table for table in ranked_tables}
        short_name_map: dict[str, list[str]] = {}
        for table in ranked_tables:
            short_name = table.split(".")[-1].lower()
            short_name_map.setdefault(short_name, []).append(table)

        resolved_tables: list[str] = []
        raw_tables = data.get("candidate_tables", [])
        if isinstance(raw_tables, str):
            raw_tables = [raw_tables]
        if isinstance(raw_tables, list):
            for raw in raw_tables:
                if not isinstance(raw, str):
                    continue
                candidate = raw.strip().strip('"`')
                if not candidate:
                    continue
                lowered = candidate.lower()
                canonical = allowed.get(lowered)
                if canonical is None and "." not in lowered:
                    options = short_name_map.get(lowered, [])
                    if len(options) == 1:
                        canonical = options[0]
                if canonical and canonical not in resolved_tables:
                    resolved_tables.append(canonical)

        column_hints: list[str] = []
        raw_columns = data.get("column_hints", [])
        if isinstance(raw_columns, str):
            raw_columns = [raw_columns]
        if isinstance(raw_columns, list):
            for value in raw_columns:
                if isinstance(value, str) and value.strip():
                    column_hints.append(value.strip())
        if not column_hints and resolved_tables:
            first = resolved_tables[0]
            column_hints = sorted(table_columns.get(first, set()))[:6]

        confidence = data.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence_value = max(0.0, min(confidence_value, 1.0))

        needs_clarification = bool(data.get("needs_clarification", False))
        clarifying = data.get("clarifying_question")
        if isinstance(clarifying, str):
            clarifying_question = clarifying.strip() or None
        else:
            clarifying_question = None
        if needs_clarification and not clarifying_question:
            clarifying_question = self._build_table_resolver_question(ranked_tables)

        return TableResolution(
            candidate_tables=resolved_tables,
            column_hints=column_hints,
            confidence=confidence_value,
            needs_clarification=needs_clarification,
            clarifying_question=clarifying_question,
        )

    def _build_table_resolver_question(self, ranked_tables: list[str]) -> str:
        options = [table.split(".")[-1] for table in ranked_tables[:3]]
        if not options:
            return "Which table should I use to answer this?"
        joined = ", ".join(options)
        return f"Which table should I use? Possible matches: {joined}."

    async def _build_store_metric_bundle_fallback(
        self,
        input: SQLAgentInput,
        *,
        table_resolution: TableResolution | None = None,
    ) -> GeneratedSQL | None:
        """Deterministic fallback for store-level revenue/margin/waste bundle queries."""
        text = (input.query or "").strip().lower()
        if not text:
            return None
        if "by store" not in text:
            return None

        has_revenue = ("revenue" in text) or ("total sales" in text)
        has_margin = "gross margin" in text
        has_waste = "waste cost" in text or ("waste" in text and "cost" in text)
        if not (has_revenue and has_margin and has_waste):
            return None

        schema_tables = list(table_resolution.candidate_tables) if table_resolution else []
        schema_tables.extend(self._schema_tables_from_memory(input))
        schema_tables = list(dict.fromkeys(schema_tables))
        if len(schema_tables) < 4:
            schema_tables.extend(await self._load_live_tables_for_fallback(input))
            schema_tables = list(dict.fromkeys(schema_tables))
        sales_table = self._pick_table_name(
            schema_tables, includes=("grocery", "sales"), fallback_contains=("transactions",)
        )
        products_table = self._pick_table_name(
            schema_tables, includes=("grocery", "products"), fallback_contains=("products",)
        )
        waste_table = self._pick_table_name(
            schema_tables, includes=("grocery", "waste"), fallback_contains=("waste",)
        )
        stores_table = self._pick_table_name(
            schema_tables, includes=("grocery", "stores"), fallback_contains=("stores",)
        )
        if not (sales_table and products_table and waste_table and stores_table):
            return None

        db_type = (input.database_type or "").strip().lower()
        window_days = self._extract_last_days_window(text, default_days=30)
        if db_type == "mysql":
            sales_window = f"st.business_date >= DATE_SUB(CURDATE(), INTERVAL {window_days} DAY)"
            waste_window = f"we.event_date >= DATE_SUB(CURDATE(), INTERVAL {window_days} DAY)"
        else:
            sales_window = f"st.business_date >= CURRENT_DATE - INTERVAL '{window_days} days'"
            waste_window = f"we.event_date >= CURRENT_DATE - INTERVAL '{window_days} days'"

        sql = (
            "WITH sales_window AS ("
            f" SELECT st.store_id, "
            "SUM(st.total_amount) AS total_revenue, "
            "SUM(st.total_amount - (st.quantity * p.unit_cost)) "
            "/ NULLIF(SUM(st.total_amount), 0) AS gross_margin"
            f" FROM {sales_table} st"
            f" JOIN {products_table} p ON p.product_id = st.product_id"
            f" WHERE {sales_window}"
            " GROUP BY st.store_id"
            "), waste_window AS ("
            " SELECT we.store_id, SUM(we.estimated_cost) AS waste_cost"
            f" FROM {waste_table} we"
            f" WHERE {waste_window}"
            " GROUP BY we.store_id"
            ") "
            "SELECT s.store_id, s.store_name, "
            "COALESCE(sw.total_revenue, 0) AS total_revenue, "
            "COALESCE(sw.gross_margin, 0) AS gross_margin, "
            "COALESCE(ww.waste_cost, 0) AS waste_cost "
            f"FROM {stores_table} s "
            "LEFT JOIN sales_window sw ON sw.store_id = s.store_id "
            "LEFT JOIN waste_window ww ON ww.store_id = s.store_id "
            "ORDER BY total_revenue DESC"
        )

        used_datapoints = [
            dp.datapoint_id
            for dp in input.investigation_memory.datapoints
            if dp.datapoint_type in {"Schema", "Business"}
        ][:8]
        return GeneratedSQL(
            sql=sql,
            explanation=(
                "Deterministic fallback for store-level revenue, gross margin, and waste cost "
                f"over the last {window_days} days."
            ),
            used_datapoints=used_datapoints,
            confidence=0.92,
            assumptions=[],
            clarifying_questions=[],
        )

    async def _build_weekday_weekend_sales_lift_fallback(
        self,
        input: SQLAgentInput,
        *,
        table_resolution: TableResolution | None = None,
    ) -> GeneratedSQL | None:
        """Deterministic fallback for weekend-vs-weekday sales lift by store and category."""
        text = (input.query or "").strip().lower()
        if not text:
            return None

        has_weekend_weekday = "weekend" in text and "weekday" in text
        has_sales_signal = any(token in text for token in ("sales", "revenue"))
        has_dimensions = "category" in text and "store" in text
        if not (has_weekend_weekday and has_sales_signal and has_dimensions):
            return None

        table_columns: dict[str, set[str]] = {}
        if table_resolution:
            for table_name in table_resolution.candidate_tables:
                table_columns.setdefault(table_name, set())
        table_columns.update(self._schema_table_columns_from_memory(input))
        if len(table_columns) < 3:
            live_candidates = await self._load_live_table_candidates_for_fallback(input)
            for table_name, columns in live_candidates:
                existing = table_columns.get(table_name, set())
                table_columns[table_name] = existing | {column.lower() for column in columns}
        schema_tables = list(table_columns.keys()) or self._schema_tables_from_memory(input)

        sales_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "sales"),
            fallback_contains=("sales_transactions", "transactions", "sales"),
        )
        if sales_table is None:
            sales_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"business_date", "store_id", "product_id", "total_amount"},
            )

        products_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "products"),
            fallback_contains=("products", "product"),
        )
        if products_table is None:
            products_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"product_id", "category"},
                excluded_tables={sales_table} if sales_table else None,
            )

        stores_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "stores"),
            fallback_contains=("stores", "store"),
        )
        if stores_table is None:
            stores_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"store_id", "store_name"},
                excluded_tables={sales_table, products_table}
                if sales_table and products_table
                else None,
            )

        if not (sales_table and products_table and stores_table):
            return None

        db_type = (input.database_type or "").strip().lower()
        window_days = self._extract_last_days_window(text, default_days=90)

        if db_type == "mysql":
            date_filter = f"st.business_date >= DATE_SUB(CURDATE(), INTERVAL {window_days} DAY)"
            day_type_expr = (
                "CASE WHEN DAYOFWEEK(st.business_date) IN (1, 7) "
                "THEN 'weekend' ELSE 'weekday' END"
            )
            order_clause = (
                "ORDER BY weekend_sales_lift_pct IS NULL ASC, "
                "weekend_sales_lift_pct DESC, a.category ASC, s.store_name ASC"
            )
        else:
            date_filter = f"st.business_date >= CURRENT_DATE - INTERVAL '{window_days} days'"
            day_type_expr = (
                "CASE WHEN DATE_PART('dow', st.business_date) IN (0, 6) "
                "THEN 'weekend' ELSE 'weekday' END"
            )
            order_clause = "ORDER BY weekend_sales_lift_pct DESC NULLS LAST, a.category ASC, s.store_name ASC"

        sql = (
            "WITH daily_sales AS ("
            " SELECT "
            "st.store_id, "
            "p.category, "
            "st.business_date, "
            f"{day_type_expr} AS day_type, "
            "SUM(st.total_amount) AS daily_sales "
            f"FROM {sales_table} st "
            f"JOIN {products_table} p ON p.product_id = st.product_id "
            f"WHERE {date_filter} "
            "GROUP BY st.store_id, p.category, st.business_date, day_type"
            "), aggregated AS ("
            " SELECT "
            "store_id, "
            "category, "
            "AVG(CASE WHEN day_type = 'weekend' THEN daily_sales END) AS avg_weekend_sales, "
            "AVG(CASE WHEN day_type = 'weekday' THEN daily_sales END) AS avg_weekday_sales "
            "FROM daily_sales "
            "GROUP BY store_id, category"
            ") "
            "SELECT "
            "s.store_id, "
            "s.store_name, "
            "a.category, "
            "ROUND(COALESCE(a.avg_weekend_sales, 0), 2) AS avg_weekend_sales, "
            "ROUND(COALESCE(a.avg_weekday_sales, 0), 2) AS avg_weekday_sales, "
            "ROUND( "
            "CASE "
            "WHEN COALESCE(a.avg_weekday_sales, 0) = 0 THEN NULL "
            "ELSE ((a.avg_weekend_sales / a.avg_weekday_sales) - 1) * 100 "
            "END, "
            "2"
            ") AS weekend_sales_lift_pct "
            "FROM aggregated a "
            f"JOIN {stores_table} s ON s.store_id = a.store_id "
            f"{order_clause}"
        )

        used_datapoints = [
            dp.datapoint_id
            for dp in input.investigation_memory.datapoints
            if dp.datapoint_type in {"Schema", "Business"}
        ][:8]
        return GeneratedSQL(
            sql=sql,
            explanation=(
                "Deterministic fallback comparing weekend vs weekday average daily sales by "
                f"store and category over the last {window_days} days."
            ),
            used_datapoints=used_datapoints,
            confidence=0.91,
            assumptions=[],
            clarifying_questions=[],
        )

    async def _build_inventory_movement_sales_gap_fallback(
        self,
        input: SQLAgentInput,
        *,
        table_resolution: TableResolution | None = None,
    ) -> GeneratedSQL | None:
        """
        Deterministic fallback for inventory-movement vs recorded-sales gap by store.
        """
        text = (input.query or "").strip().lower()
        if not text:
            return None

        has_store = "store" in text
        has_inventory = "inventory" in text
        has_movement = "movement" in text or "moved" in text
        has_sales = "sales" in text
        has_gap = "gap" in text or "difference" in text
        if not (has_store and has_inventory and has_movement and has_sales and has_gap):
            return None

        table_columns: dict[str, set[str]] = {}
        if table_resolution:
            for table_name in table_resolution.candidate_tables:
                table_columns.setdefault(table_name, set())
        table_columns.update(self._schema_table_columns_from_memory(input))
        if len(table_columns) < 3:
            live_candidates = await self._load_live_table_candidates_for_fallback(input)
            for table_name, columns in live_candidates:
                existing = table_columns.get(table_name, set())
                table_columns[table_name] = existing | {column.lower() for column in columns}
        schema_tables = list(table_columns.keys()) or self._schema_tables_from_memory(input)

        inventory_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "inventory"),
            fallback_contains=("inventory_snapshots", "inventory", "stock"),
        )
        if inventory_table is None:
            inventory_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"snapshot_date", "store_id", "on_hand_qty"},
            )

        sales_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "sales"),
            fallback_contains=("sales_transactions", "sales", "transactions"),
        )
        if sales_table is None:
            sales_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"store_id", "quantity"},
                excluded_tables={inventory_table} if inventory_table else None,
            )

        stores_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "stores"),
            fallback_contains=("stores", "store"),
        )
        if stores_table is None:
            stores_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"store_id", "store_name"},
                excluded_tables={inventory_table, sales_table}
                if inventory_table and sales_table
                else None,
            )

        if not (inventory_table and sales_table and stores_table):
            return None

        inventory_cols = table_columns.get(inventory_table, set())
        sales_cols = table_columns.get(sales_table, set())
        stores_cols = table_columns.get(stores_table, set())

        if inventory_cols and ("snapshot_date" not in inventory_cols or "on_hand_qty" not in inventory_cols):
            return None

        window_days = self._extract_last_days_window(text, default_days=30)
        db_type = (input.database_type or "").strip().lower()
        top_n = self._extract_top_n(text, default=10, max_value=25)

        inventory_filter = (
            f"snapshot_date >= DATE_SUB(CURDATE(), INTERVAL {window_days} DAY)"
            if db_type == "mysql"
            else f"snapshot_date >= CURRENT_DATE - INTERVAL '{window_days} days'"
        )

        if "business_date" in sales_cols:
            sales_date_col = "business_date"
        elif "sold_at" in sales_cols:
            sales_date_col = "DATE(sold_at)" if db_type == "mysql" else "sold_at::date"
        elif "event_date" in sales_cols:
            sales_date_col = "event_date"
        else:
            sales_date_col = None

        if sales_date_col:
            sales_filter = (
                f"{sales_date_col} >= DATE_SUB(CURDATE(), INTERVAL {window_days} DAY)"
                if db_type == "mysql"
                else f"{sales_date_col} >= CURRENT_DATE - INTERVAL '{window_days} days'"
            )
        else:
            sales_filter = "1=1"

        sales_units_expr = "SUM(quantity)" if "quantity" in sales_cols else "COUNT(*)"
        store_name_expr = (
            "s.store_name"
            if "store_name" in stores_cols
            else ("s.store_code" if "store_code" in stores_cols else "s.store_id")
        )

        gap_order = (
            "ORDER BY ABS(COALESCE(i.movement_units, 0) - COALESCE(sa.recorded_sales_units, 0)) DESC"
        )
        sql = (
            "WITH inventory_daily AS ("
            " SELECT snapshot_date, store_id, SUM(on_hand_qty) AS on_hand_units"
            f" FROM {inventory_table}"
            f" WHERE {inventory_filter}"
            " GROUP BY snapshot_date, store_id"
            "), inventory_delta AS ("
            " SELECT store_id, snapshot_date, "
            "ABS(on_hand_units - LAG(on_hand_units) OVER (PARTITION BY store_id ORDER BY snapshot_date)) "
            "AS movement_delta "
            "FROM inventory_daily"
            "), inventory_movement AS ("
            " SELECT store_id, SUM(COALESCE(movement_delta, 0)) AS movement_units"
            " FROM inventory_delta"
            " GROUP BY store_id"
            "), sales_agg AS ("
            f" SELECT store_id, {sales_units_expr} AS recorded_sales_units"
            f" FROM {sales_table}"
            f" WHERE {sales_filter}"
            " GROUP BY store_id"
            ") "
            "SELECT "
            "s.store_id, "
            f"{store_name_expr} AS store_name, "
            "COALESCE(i.movement_units, 0) AS inventory_movement_units, "
            "COALESCE(sa.recorded_sales_units, 0) AS recorded_sales_units, "
            "(COALESCE(i.movement_units, 0) - COALESCE(sa.recorded_sales_units, 0)) AS movement_sales_gap "
            f"FROM {stores_table} s "
            "LEFT JOIN inventory_movement i ON i.store_id = s.store_id "
            "LEFT JOIN sales_agg sa ON sa.store_id = s.store_id "
            f"{gap_order} "
            f"LIMIT {top_n}"
        )

        used_datapoints = [
            dp.datapoint_id
            for dp in input.investigation_memory.datapoints
            if dp.datapoint_type in {"Schema", "Business"}
        ][:8]
        return GeneratedSQL(
            sql=sql,
            explanation=(
                "Deterministic fallback comparing inventory movement against recorded sales "
                f"by store over the last {window_days} days."
            ),
            used_datapoints=used_datapoints,
            confidence=0.9,
            assumptions=[],
            clarifying_questions=[],
        )

    async def _build_stockout_risk_ranking_fallback(
        self,
        input: SQLAgentInput,
        *,
        table_resolution: TableResolution | None = None,
    ) -> GeneratedSQL | None:
        """Deterministic fallback for SKU-level stockout risk ranking queries."""
        text = (input.query or "").strip().lower()
        if not text:
            return None

        has_stockout = any(
            phrase in text
            for phrase in (
                "stockout",
                "stock-out",
                "out of stock",
            )
        )
        has_risk = "risk" in text
        has_sku_or_product = any(token in text for token in ("sku", "skus", "product", "products"))
        has_inventory_signals = any(
            token in text
            for token in (
                "on-hand",
                "on hand",
                "on_hand",
                "reserved",
                "reorder",
                "reorder level",
            )
        )
        if not (has_stockout and has_risk and has_sku_or_product and has_inventory_signals):
            return None

        table_columns: dict[str, set[str]] = {}
        if table_resolution:
            for table_name in table_resolution.candidate_tables:
                table_columns.setdefault(table_name, set())
        table_columns.update(self._schema_table_columns_from_memory(input))
        if len(table_columns) < 2:
            live_candidates = await self._load_live_table_candidates_for_fallback(input)
            for table_name, columns in live_candidates:
                existing = table_columns.get(table_name, set())
                table_columns[table_name] = existing | {column.lower() for column in columns}
        schema_tables = list(table_columns.keys())

        inventory_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "inventory"),
            fallback_contains=("inventory_snapshots", "inventory", "snapshots"),
        )
        if inventory_table is None:
            inventory_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"product_id", "store_id", "on_hand_qty", "reserved_qty"},
            )
        products_table = self._pick_table_name(
            schema_tables,
            includes=("grocery", "products"),
            fallback_contains=("products",),
        )
        if products_table is None:
            products_table = self._pick_table_by_columns(
                table_columns,
                required_columns={"product_id", "sku", "reorder_level"},
                excluded_tables={inventory_table} if inventory_table else None,
            )
        if not (inventory_table and products_table):
            return None

        db_type = (input.database_type or "").strip().lower()
        snapshot_filter, window_desc = self._build_snapshot_window_filter(
            db_type,
            text,
            inventory_table,
        )
        top_n = self._extract_top_n(text, default=5, max_value=25)

        if db_type == "mysql":
            risk_expr = (
                "CASE "
                "WHEN p.reorder_level <= 0 THEN 0.0 "
                "WHEN (inv.on_hand_qty - inv.reserved_qty) <= 0 THEN 1.0 "
                "ELSE GREATEST(0.0, LEAST(1.0, "
                "CAST((p.reorder_level - (inv.on_hand_qty - inv.reserved_qty)) AS DECIMAL(12,4)) "
                "/ NULLIF(p.reorder_level, 0) "
                ")) "
                "END"
            )
        else:
            risk_expr = (
                "CASE "
                "WHEN p.reorder_level <= 0 THEN 0.0 "
                "WHEN (inv.on_hand_qty - inv.reserved_qty) <= 0 THEN 1.0 "
                "ELSE GREATEST(0.0, LEAST(1.0, "
                "((p.reorder_level - (inv.on_hand_qty - inv.reserved_qty))::numeric "
                "/ NULLIF(p.reorder_level, 0)) "
                ")) "
                "END"
            )

        sql = (
            "WITH latest_snapshot AS ("
            " SELECT store_id, product_id, MAX(snapshot_date) AS snapshot_date"
            f" FROM {inventory_table}"
            f" WHERE {snapshot_filter}"
            " GROUP BY store_id, product_id"
            "), stock_base AS ("
            " SELECT "
            "p.sku, p.product_name, p.reorder_level, "
            "inv.store_id, inv.on_hand_qty, inv.reserved_qty, "
            "(inv.on_hand_qty - inv.reserved_qty) AS available_qty, "
            f"{risk_expr} AS stockout_risk_score "
            f"FROM {inventory_table} inv "
            "JOIN latest_snapshot ls "
            "ON ls.store_id = inv.store_id "
            "AND ls.product_id = inv.product_id "
            "AND ls.snapshot_date = inv.snapshot_date "
            f"JOIN {products_table} p ON p.product_id = inv.product_id"
            ") "
            "SELECT "
            "sku, "
            "product_name, "
            "MAX(reorder_level) AS reorder_level, "
            "ROUND(AVG(on_hand_qty), 2) AS avg_on_hand_qty, "
            "ROUND(AVG(reserved_qty), 2) AS avg_reserved_qty, "
            "ROUND(AVG(available_qty), 2) AS avg_available_qty, "
            "ROUND(AVG(stockout_risk_score), 4) AS stockout_risk_score, "
            "SUM(CASE WHEN available_qty <= 0 THEN 1 ELSE 0 END) AS store_stockout_count, "
            "COUNT(*) AS store_coverage "
            "FROM stock_base "
            "GROUP BY sku, product_name "
            "ORDER BY stockout_risk_score DESC, avg_available_qty ASC "
            f"LIMIT {top_n}"
        )

        used_datapoints = [
            dp.datapoint_id
            for dp in input.investigation_memory.datapoints
            if dp.datapoint_type in {"Schema", "Business"}
        ][:8]
        return GeneratedSQL(
            sql=sql,
            explanation=(
                "Deterministic fallback for ranking SKU stockout risk using on-hand, "
                f"reserved, and reorder-level signals ({window_desc})."
            ),
            used_datapoints=used_datapoints,
            confidence=0.93,
            assumptions=[],
            clarifying_questions=[],
        )

    async def _load_live_tables_for_fallback(self, input: SQLAgentInput) -> list[str]:
        return [table_name for table_name, _ in await self._load_live_table_candidates_for_fallback(input)]

    async def _load_live_table_candidates_for_fallback(
        self, input: SQLAgentInput
    ) -> list[tuple[str, set[str]]]:
        db_type = input.database_type or getattr(self.config.database, "db_type", "postgresql")
        db_url = input.database_url or (
            str(self.config.database.url) if self.config.database.url else None
        )
        if not db_url:
            return []
        try:
            connector = create_connector(
                database_url=db_url,
                database_type=db_type,
                pool_size=self.config.database.pool_size,
                timeout=8,
            )
        except Exception:
            return []

        try:
            await connector.connect()
            tables_info = await connector.get_schema()
            live_tables: list[tuple[str, set[str]]] = []
            for table in tables_info:
                schema = getattr(table, "schema_name", None) or getattr(table, "schema", None)
                name = getattr(table, "table_name", None)
                if not name:
                    continue
                if str(name).lower() in _INTERNAL_SERVICE_TABLES:
                    continue
                qualified = f"{schema}.{name}" if schema else str(name)
                column_names: set[str] = set()
                columns = getattr(table, "columns", None) or []
                for column in columns:
                    column_name = getattr(column, "name", None)
                    if isinstance(column_name, str) and column_name.strip():
                        column_names.add(column_name.strip().lower())
                live_tables.append((qualified, column_names))
            return live_tables
        except Exception:
            return []
        finally:
            await connector.close()

    def _schema_tables_from_memory(self, input: SQLAgentInput) -> list[str]:
        tables: list[str] = []
        for dp in input.investigation_memory.datapoints:
            metadata = dp.metadata if isinstance(dp.metadata, dict) else {}
            if dp.datapoint_type == "Schema":
                table_name = metadata.get("table_name") or metadata.get("table")
                if isinstance(table_name, str) and table_name.strip():
                    tables.append(table_name.strip())

            related_tables = metadata.get("related_tables")
            if isinstance(related_tables, str):
                values = [item.strip() for item in related_tables.split(",") if item.strip()]
                tables.extend(values)
            elif isinstance(related_tables, list):
                for value in related_tables:
                    if isinstance(value, str) and value.strip():
                        tables.append(value.strip())

            dp_related_tables = getattr(dp, "related_tables", None)
            if isinstance(dp_related_tables, list):
                for value in dp_related_tables:
                    if isinstance(value, str) and value.strip():
                        tables.append(value.strip())
        return list(dict.fromkeys(tables))

    def _schema_table_columns_from_memory(self, input: SQLAgentInput) -> dict[str, set[str]]:
        """Extract table->columns mapping from retrieved schema datapoints."""
        table_columns: dict[str, set[str]] = {}
        for dp in input.investigation_memory.datapoints:
            if dp.datapoint_type != "Schema":
                continue
            metadata = dp.metadata if isinstance(dp.metadata, dict) else {}
            table_name = metadata.get("table_name") or metadata.get("table")
            if not isinstance(table_name, str) or not table_name.strip():
                continue
            normalized_table = table_name.strip()
            columns = table_columns.setdefault(normalized_table, set())
            key_columns = metadata.get("key_columns")
            if isinstance(key_columns, list):
                for column in key_columns:
                    if not isinstance(column, dict):
                        continue
                    column_name = column.get("name")
                    if isinstance(column_name, str) and column_name.strip():
                        columns.add(column_name.strip().lower())
        return table_columns

    def _pick_table_name(
        self,
        tables: list[str],
        *,
        includes: tuple[str, ...],
        fallback_contains: tuple[str, ...],
    ) -> str | None:
        for table in tables:
            lowered = table.lower()
            if all(token in lowered for token in includes):
                return table
        for table in tables:
            lowered = table.lower()
            if any(token in lowered for token in fallback_contains):
                return table
        return None

    def _pick_table_by_columns(
        self,
        table_columns: dict[str, set[str]],
        *,
        required_columns: set[str],
        excluded_tables: set[str] | None = None,
    ) -> str | None:
        excluded = {value.lower() for value in (excluded_tables or set())}
        best_table = None
        best_score = -1
        required_lower = {column.lower() for column in required_columns}
        for table_name, columns in table_columns.items():
            if table_name.lower() in excluded:
                continue
            score = len(required_lower.intersection({column.lower() for column in columns}))
            if score > best_score and score >= max(2, len(required_lower) - 1):
                best_score = score
                best_table = table_name
        return best_table

    def _extract_last_days_window(self, query: str, *, default_days: int = 30) -> int:
        match = re.search(r"\blast\s+(\d+)\s+days?\b", query)
        if not match:
            return default_days
        try:
            value = int(match.group(1))
        except ValueError:
            return default_days
        return max(1, min(value, 365))

    def _build_snapshot_window_filter(
        self,
        db_type: str,
        query: str,
        table_name: str,
    ) -> tuple[str, str]:
        """Build a deterministic time window filter for inventory snapshot queries."""
        text = query.lower()
        anchor = self._snapshot_anchor_expr(db_type, table_name)
        if "this week" in text:
            if db_type == "mysql":
                return (
                    f"snapshot_date >= DATE_SUB(DATE({anchor}), INTERVAL WEEKDAY(DATE({anchor})) DAY)",
                    "this week",
                )
            return (f"snapshot_date >= DATE_TRUNC('week', {anchor})::date", "this week")

        if "last week" in text:
            if db_type == "mysql":
                return (
                    f"snapshot_date >= DATE_SUB(DATE_SUB(DATE({anchor}), INTERVAL WEEKDAY(DATE({anchor})) DAY), INTERVAL 7 DAY) "
                    f"AND snapshot_date < DATE_SUB(DATE({anchor}), INTERVAL WEEKDAY(DATE({anchor})) DAY)",
                    "last week",
                )
            return (
                f"snapshot_date >= (DATE_TRUNC('week', {anchor})::date - INTERVAL '7 days') "
                f"AND snapshot_date < DATE_TRUNC('week', {anchor})::date",
                "last week",
            )

        if "week" in text:
            if db_type == "mysql":
                return (
                    f"snapshot_date >= DATE_SUB(DATE({anchor}), INTERVAL 7 DAY)",
                    "last 7 days",
                )
            return (
                f"snapshot_date >= ({anchor})::date - INTERVAL '7 days'",
                "last 7 days",
            )

        days = self._extract_last_days_window(text, default_days=30)
        if db_type == "mysql":
            return (
                f"snapshot_date >= DATE_SUB(DATE({anchor}), INTERVAL {days} DAY)",
                f"last {days} days",
            )
        return (
            f"snapshot_date >= ({anchor})::date - INTERVAL '{days} days'",
            f"last {days} days",
        )

    def _snapshot_anchor_expr(self, db_type: str, table_name: str) -> str:
        if db_type == "mysql":
            return f"(SELECT COALESCE(MAX(snapshot_date), CURDATE()) FROM {table_name})"
        return f"(SELECT COALESCE(MAX(snapshot_date), CURRENT_DATE) FROM {table_name})"

    def _extract_top_n(self, query: str, *, default: int = 5, max_value: int = 25) -> int:
        text = query.lower()
        patterns = [
            r"\btop\s+(\d+)\b",
            r"\bfirst\s+(\d+)\b",
            r"\blimit\s+(\d+)\b",
            r"\bwhich\s+(\d+)\s+(?:skus?|products?)\b",
            r"\b(\d+)\s+(?:skus?|products?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            return max(1, min(value, max_value))
        return default

    async def _request_sql_from_llm(
        self,
        *,
        provider: Any,
        llm_request: LLMRequest,
        input: SQLAgentInput,
        runtime_stats: dict[str, int],
    ) -> GeneratedSQL:
        response = await provider.generate(llm_request)
        self._track_llm_call(tokens=response.usage.total_tokens)

        try:
            return self._parse_llm_response(response.content, input)
        except ValueError:
            retry_request = llm_request.model_copy()
            retry_request.messages.append(
                LLMMessage(
                    role="system",
                    content="Return ONLY JSON with a top-level 'sql' field.",
                )
            )
            retry_response = await provider.generate(retry_request)
            self._track_llm_call(tokens=retry_response.usage.total_tokens)
            try:
                return self._parse_llm_response(retry_response.content, input)
            except ValueError as exc:
                recovered = await self._recover_sql_with_formatter(
                    raw_content=retry_response.content or response.content,
                    input=input,
                    runtime_stats=runtime_stats,
                )
                if recovered is not None:
                    return recovered
                raise SQLClarificationNeeded(
                    self._build_clarifying_questions(input.query)
                ) from exc

    def _should_accept_fast_sql(self, generated_sql: GeneratedSQL, input: SQLAgentInput) -> bool:
        if generated_sql.clarifying_questions:
            return False
        threshold = self._pipeline_float("sql_two_stage_confidence_threshold", 0.78)
        if generated_sql.confidence < threshold:
            return False
        issues = self._validate_sql(generated_sql, input)
        return len(issues) == 0

    async def _correct_sql(
        self,
        generated_sql: GeneratedSQL,
        issues: list[ValidationIssue],
        input: SQLAgentInput,
        metadata: AgentMetadata,
        runtime_stats: dict[str, int],
    ) -> GeneratedSQL:
        """
        Self-correct SQL based on validation issues.

        Args:
            generated_sql: Original generated SQL
            issues: Validation issues found
            input: Original input
            metadata: Metadata to track LLM calls

        Returns:
            Corrected GeneratedSQL

        Raises:
            LLMError: If correction fails
        """
        # Build correction prompt
        prompt = self._build_correction_prompt(generated_sql, issues, input)

        # Create LLM request
        llm_request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=self._get_system_prompt()),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            max_tokens=2000,
        )

        try:
            # Call LLM
            response = await self.llm.generate(llm_request)

            # Track LLM call and tokens
            self._track_llm_call(tokens=response.usage.total_tokens)

            # Parse corrected response
            try:
                corrected_sql = self._parse_llm_response(response.content, input)
            except ValueError as parse_error:
                recovered = await self._recover_sql_with_formatter(
                    raw_content=response.content,
                    input=input,
                    runtime_stats=runtime_stats,
                )
                if recovered is None:
                    raise parse_error
                corrected_sql = recovered

            logger.debug(
                f"Corrected SQL: {corrected_sql.sql[:200]}...",
                extra={"issues_addressed": len(issues)},
            )

            return corrected_sql

        except Exception as e:
            logger.error(f"SQL correction failed: {e}", exc_info=True)
            raise LLMError(
                agent=self.name,
                message=f"SQL correction failed: {e}",
                context={"original_sql": generated_sql.sql},
            ) from e

    async def _recover_sql_with_formatter(
        self, *, raw_content: str, input: SQLAgentInput, runtime_stats: dict[str, int]
    ) -> GeneratedSQL | None:
        """Attempt to recover malformed SQL-agent output with a formatter model."""
        if not self._pipeline_flag("sql_formatter_fallback_enabled", True):
            return None
        if not raw_content or not raw_content.strip():
            return None

        provider = getattr(self, "formatter_llm", None) or self.fast_llm or self.llm
        if provider is None:
            return None

        formatter_model = getattr(self.config.llm, "sql_formatter_model", None)
        if not isinstance(formatter_model, str):
            formatter_model = None
        elif not formatter_model.strip():
            formatter_model = None

        formatter_prompt = self._build_sql_formatter_prompt(raw_content, input.query)
        formatter_request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are a strict JSON formatter for SQL generation outputs. "
                        "Return only valid JSON."
                    ),
                ),
                LLMMessage(role="user", content=formatter_prompt),
            ],
            temperature=0.0,
            max_tokens=600,
            model=formatter_model,
        )

        try:
            runtime_stats["formatter_fallback_calls"] = (
                int(runtime_stats.get("formatter_fallback_calls", 0)) + 1
            )
            formatter_response = await provider.generate(formatter_request)
            self._track_llm_call(tokens=formatter_response.usage.total_tokens)
            parsed = self._parse_llm_response(formatter_response.content, input)
            runtime_stats["formatter_fallback_successes"] = (
                int(runtime_stats.get("formatter_fallback_successes", 0)) + 1
            )
            return parsed
        except Exception as exc:
            logger.debug(
                "Formatter fallback failed to recover SQL response",
                extra={"error": str(exc)},
            )
            return None

    def _build_sql_formatter_prompt(self, raw_content: str, query: str) -> str:
        max_chars = self._pipeline_int("sql_prompt_max_context_chars", 12000)
        clipped_content = raw_content.strip()[:max_chars]
        return (
            "Reformat the MODEL_OUTPUT into strict JSON.\n"
            "Output requirements:\n"
            '- Return ONLY one JSON object with keys: "sql", "explanation", '
            '"used_datapoints", "confidence", "assumptions", "clarifying_questions".\n'
            "- sql must contain executable SQL when present.\n"
            "- If SQL is not recoverable from MODEL_OUTPUT, set sql to an empty string and "
            "add one concise clarifying question.\n"
            "- used_datapoints and assumptions must be JSON arrays.\n"
            "- confidence must be a number between 0 and 1.\n\n"
            f"USER_QUERY:\n{query}\n\n"
            f"MODEL_OUTPUT:\n{clipped_content}\n"
        )

    def _validate_sql(
        self, generated_sql: GeneratedSQL, input: SQLAgentInput
    ) -> list[ValidationIssue]:
        """
        Validate generated SQL for common issues.

        Performs basic validation checks:
        - Basic syntax check (SELECT, FROM keywords)
        - Table names match available DataPoints (excluding CTEs and subqueries)
        - Column names referenced in DataPoints
        - No obvious SQL injection patterns

        Args:
            generated_sql: Generated SQL to validate
            input: Original input with context

        Returns:
            List of validation issues (empty if valid)
        """
        issues: list[ValidationIssue] = []
        sql = generated_sql.sql.strip().upper()
        db_type = input.database_type or getattr(self.config.database, "db_type", "postgresql")
        is_show_statement = sql.startswith("SHOW") or sql.startswith("DESCRIBE") or sql.startswith(
            "DESC"
        )

        # Basic syntax checks
        if is_show_statement:
            return issues
        if not sql.startswith("SELECT") and not sql.startswith("WITH"):
            issues.append(
                ValidationIssue(
                    issue_type="syntax",
                    message="SQL must start with SELECT or WITH",
                    suggested_fix="Ensure query begins with SELECT or WITH (for CTEs)",
                )
            )

        if "FROM" not in sql:
            issues.append(
                ValidationIssue(
                    issue_type="syntax",
                    message="SQL missing FROM clause",
                    suggested_fix="Add FROM clause to specify table(s)",
                )
            )

        # Extract CTE names (Common Table Expressions) from WITH clause
        # Pattern: WITH cte_name AS (...), another_cte AS (...)
        cte_names = set()
        cte_pattern = r"WITH\s+([a-zA-Z0-9_]+)\s+AS\s*\("
        cte_matches = re.findall(cte_pattern, sql, re.IGNORECASE)
        for cte_name in cte_matches:
            cte_names.add(cte_name.upper())

        # Also match comma-separated CTEs: , cte_name AS (
        additional_cte_pattern = r",\s*([a-zA-Z0-9_]+)\s+AS\s*\("
        additional_ctes = re.findall(additional_cte_pattern, sql, re.IGNORECASE)
        for cte_name in additional_ctes:
            cte_names.add(cte_name.upper())

        # Extract table names from SQL (FROM and JOIN clauses)
        table_pattern = r"FROM\s+([a-zA-Z0-9_.]+)|JOIN\s+([a-zA-Z0-9_.]+)"
        table_matches = re.findall(table_pattern, sql, re.IGNORECASE)
        referenced_tables = {match[0] or match[1] for match in table_matches}
        referenced_table_lowers = {table.lower() for table in referenced_tables}
        catalog_schemas = self._catalog_schemas_for_db(db_type)
        catalog_aliases = self._catalog_aliases_for_db(db_type)
        catalog_tables = {
            table
            for table in referenced_table_lowers
            if self._is_catalog_table(table, catalog_schemas, catalog_aliases)
        }
        is_catalog_only = referenced_tables and len(catalog_tables) == len(
            referenced_table_lowers
        )

        # Get available tables from DataPoints
        available_tables = set()
        for dp in input.investigation_memory.datapoints:
            if not isinstance(dp.metadata, dict):
                continue
            table_values: list[str] = []
            table_name = dp.metadata.get("table_name")
            if isinstance(table_name, str) and table_name.strip():
                table_values.append(table_name.strip())

            related_tables = dp.metadata.get("related_tables")
            if isinstance(related_tables, str):
                table_values.extend(
                    value.strip() for value in related_tables.split(",") if value.strip()
                )
            elif isinstance(related_tables, list):
                for value in related_tables:
                    if isinstance(value, str) and value.strip():
                        table_values.append(value.strip())

            for table_value in table_values:
                available_tables.add(table_value.upper())
                if "." in table_value:
                    available_tables.add(table_value.split(".")[-1].upper())
        available_table_lowers = {table.lower() for table in available_tables}

        live_schema_tables: set[str] = set()
        db_url = input.database_url or (
            str(self.config.database.url) if self.config.database.url else None
        )
        if db_url:
            schema_key = f"{db_type}::{db_url}"
            live_schema_tables = self._live_schema_tables_cache.get(schema_key, set())

        table_validation_candidates = (
            available_table_lowers if available_table_lowers else live_schema_tables
        )
        has_table_validation = bool(table_validation_candidates)

        # Check for missing tables (excluding CTEs and special tables)
        for table in referenced_tables:
            table_upper = table.upper()

            # Skip if this is a CTE name
            if table_upper in cte_names:
                continue

            # Skip special tables
            if table_upper in ("DUAL", "LATERAL"):
                continue
            if is_catalog_only:
                continue
            if table.lower().startswith(("information_schema.", "pg_catalog.")):
                continue
            if table.lower() in catalog_aliases:
                continue
            if not has_table_validation:
                continue

            # Check if table exists in DataPoints
            if table.lower() not in table_validation_candidates and "." in table:
                # Check without schema
                table_no_schema = table.split(".")[-1].lower()
                if (
                    table_no_schema not in table_validation_candidates
                    and table_no_schema.upper() not in cte_names
                ):
                    issues.append(
                        ValidationIssue(
                            issue_type="missing_table",
                            message=f"Table '{table}' not found in available DataPoints",
                            suggested_fix=(
                                f"Use one of: {', '.join(sorted(available_tables))}"
                                if available_tables
                                else "Use a table from the live schema snapshot."
                            ),
                        )
                    )
            elif table.lower() not in table_validation_candidates:
                # Simple table name not found
                issues.append(
                    ValidationIssue(
                        issue_type="missing_table",
                        message=f"Table '{table}' not found in available DataPoints",
                        suggested_fix=(
                            f"Use one of: {', '.join(sorted(available_tables))}"
                            if available_tables
                            else "Use a table from the live schema snapshot."
                        ),
                    )
                )

        return issues

    def _build_introspection_query(
        self,
        query: str,
        database_type: str | None = None,
    ) -> str | None:
        text = query.lower().strip()
        if not self.catalog.is_list_tables_query(text):
            return None
        target_db_type = database_type or getattr(self.config.database, "db_type", "postgresql")
        return get_list_tables_query(target_db_type)

    def _build_list_columns_fallback(self, input: SQLAgentInput) -> str | None:
        plan = self.catalog.plan_query(
            query=input.query,
            database_type=input.database_type,
            investigation_memory=input.investigation_memory,
        )
        if plan and plan.operation == "list_columns":
            return plan.sql
        return None

    def _build_row_count_fallback(self, input: SQLAgentInput) -> str | None:
        plan = self.catalog.plan_query(
            query=input.query,
            database_type=input.database_type,
            investigation_memory=input.investigation_memory,
        )
        if plan and plan.operation == "row_count":
            return plan.sql
        return None

    def _requires_row_count(self, query: str) -> bool:
        text = query.lower()
        patterns = [
            r"\brow count\b",
            r"\bhow many rows\b",
            r"\bnumber of rows\b",
            r"\bcount of rows\b",
            r"\btotal rows\b",
            r"\brow total\b",
            r"\bhow many records\b",
            r"\brecord count\b",
            r"\brecords in\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _build_sample_rows_fallback(self, input: SQLAgentInput) -> str | None:
        plan = self.catalog.plan_query(
            query=input.query,
            database_type=input.database_type,
            investigation_memory=input.investigation_memory,
        )
        if plan and plan.operation == "sample_rows":
            return plan.sql
        return None

    def _requires_sample_rows(self, query: str) -> bool:
        text = query.lower()
        patterns = [
            r"\bshow\b.*\brows\b",
            r"\bfirst\s+\d+\b",
            r"\btop\s+\d+\b",
            r"\bpreview\b",
            r"\bsample\b",
            r"\bexample\b",
            r"\bshow me\b.*\brows\b",
            r"\bdisplay\b.*\brows\b",
            r"\blimit\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _extract_sample_limit(self, query: str) -> int:
        text = query.lower()
        match = re.search(r"\b(first|top|limit)\s+(\d+)\b", text)
        if match:
            try:
                value = int(match.group(2))
                return max(1, min(value, 25))
            except ValueError:
                return 3
        return 3

    def _build_clarifying_questions(self, query: str) -> list[str]:
        questions = ["Which table should I use to answer this?"]
        text = query.lower()
        if any(term in text for term in ("total", "sum", "average", "avg", "count")):
            questions.append(
                "Which column should I aggregate (for example: amount, total_amount, revenue)?"
            )
        if any(term in text for term in ("date", "time", "month", "year", "quarter")):
            questions.append("Is there a specific date or time range I should use?")
        return questions

    def _resolve_followup_query(
        self, input: SQLAgentInput
    ) -> tuple[str, str | None]:
        history = input.conversation_history or []
        if not history:
            return input.query, None

        last_assistant = None
        last_assistant_index = None
        for idx in range(len(history) - 1, -1, -1):
            msg = history[idx]
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                last_assistant = msg
                last_assistant_index = idx
                break

        if last_assistant is None:
            return input.query, None

        assistant_text = (
            str(last_assistant.get("content", ""))
            if isinstance(last_assistant, dict)
            else str(getattr(last_assistant, "content", ""))
        ).lower()
        if "which table" not in assistant_text and "clarifying" not in assistant_text:
            return input.query, None

        query_text = input.query.strip()
        if (
            not query_text
            or len(query_text.split()) > 4
            or not self._looks_like_followup_hint(query_text)
        ):
            return input.query, None

        previous_user = None
        if last_assistant_index is not None:
            for idx in range(last_assistant_index - 1, -1, -1):
                msg = history[idx]
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                if role == "user":
                    previous_user = msg
                    break
        if previous_user is None:
            return input.query, None

        previous_text = (
            str(previous_user.get("content", ""))
            if isinstance(previous_user, dict)
            else str(getattr(previous_user, "content", ""))
        ).strip()
        if not previous_text:
            return input.query, None

        table_hint = re.sub(r"[^\w.]+", "", query_text)
        if not table_hint:
            return input.query, None

        resolved_query = self._merge_query_with_table_hint(previous_text, table_hint)
        return resolved_query, table_hint

    def _looks_like_followup_hint(self, text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False
        if ":" in lowered:
            lowered = lowered.rsplit(":", 1)[-1].strip()
        disallowed = {
            "show",
            "list",
            "count",
            "select",
            "describe",
            "help",
            "what",
            "which",
            "how",
            "rows",
            "columns",
            "table",
            "tables",
        }
        words = [word for word in re.split(r"\s+", lowered) if word]
        if any(word in disallowed for word in words):
            return False
        return bool(re.fullmatch(r"[a-zA-Z0-9_.`\"'-]+", lowered))

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

    def _extract_explicit_table_name(self, query: str) -> str | None:
        patterns = [
            r"\bhow\s+many\s+rows?\s+(?:are\s+)?in\s+([a-zA-Z0-9_.]+)",
            r"\brows?\s+in\s+([a-zA-Z0-9_.]+)",
            r"\bcount\s+of\s+rows?\s+in\s+([a-zA-Z0-9_.]+)",
            r"\brecords?\s+in\s+([a-zA-Z0-9_.]+)",
            r"\b(?:first|top|last)\s+\d+\s+rows?\s+(?:from|in|of)\s+([a-zA-Z0-9_.]+)",
            r"\bshow\s+me\s+(?:the\s+)?(?:first|top|last)?\s*\d*\s*rows?\s+(?:from|in|of)\s+([a-zA-Z0-9_.]+)",
            r"\b(?:preview|sample)\s+(?:rows?\s+(?:from|in|of)\s+)?([a-zA-Z0-9_.]+)",
            r"\btable\s+([a-zA-Z0-9_.]+)",
        ]
        lowered = query.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                table_name = match.group(1).rstrip(".,;:?)")
                if table_name and table_name not in {
                    "table",
                    "tables",
                    "row",
                    "rows",
                }:
                    return table_name
        return None

    def _select_schema_table(self, input: SQLAgentInput) -> str | None:
        for dp in input.investigation_memory.datapoints:
            if dp.datapoint_type != "Schema":
                continue
            metadata = dp.metadata if isinstance(dp.metadata, dict) else {}
            table_name = metadata.get("table_name") or metadata.get("table")
            if table_name:
                return table_name
        return None

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for SQL generation.

        Returns:
            System prompt string
        """
        return self.prompts.load("system/main.md")

    async def _build_generation_prompt(
        self,
        input: SQLAgentInput,
        *,
        table_resolution: TableResolution | None = None,
    ) -> str:
        """
        Build prompt for SQL generation.

        Args:
            input: SQLAgentInput with query and context

        Returns:
            Formatted prompt string
        """
        resolved_query, _ = self._resolve_followup_query(input)
        # Extract schema and business context
        schema_context = self._format_schema_context(input.investigation_memory)
        ranked_catalog_context = self.catalog.build_ranked_schema_context(
            query=resolved_query,
            investigation_memory=input.investigation_memory,
        )
        if ranked_catalog_context:
            if schema_context == "No schema context available":
                schema_context = ranked_catalog_context
            else:
                schema_context = f"{schema_context}\n\n{ranked_catalog_context}"
        include_profile = not input.investigation_memory.datapoints
        focus_tables = table_resolution.candidate_tables if table_resolution else None
        live_context = await self._get_live_schema_context(
            query=resolved_query,
            database_type=input.database_type,
            database_url=input.database_url,
            include_profile=include_profile,
            focus_tables=focus_tables,
        )
        if live_context:
            if schema_context == "No schema context available":
                schema_context = live_context
            else:
                schema_context = (
                    f"{schema_context}\n\n**Live schema snapshot (authoritative):**\n"
                    f"{live_context}"
                )
        if table_resolution and table_resolution.candidate_tables:
            table_lines = ", ".join(table_resolution.candidate_tables[:8])
            column_lines = ", ".join(table_resolution.column_hints[:10])
            resolver_context = f"**Likely source tables (resolver):** {table_lines}"
            if column_lines:
                resolver_context = (
                    f"{resolver_context}\n**Likely relevant columns:** {column_lines}"
                )
            if schema_context == "No schema context available":
                schema_context = resolver_context
            else:
                schema_context = f"{resolver_context}\n\n{schema_context}"
        if self._pipeline_flag("sql_prompt_budget_enabled", True):
            max_chars = self._pipeline_int("sql_prompt_max_context_chars", 12000)
            schema_context = self._truncate_context(schema_context, max_chars)
        business_context = self._format_business_context(input.investigation_memory)
        conversation_context = self._format_conversation_context(
            input.conversation_history
        )
        return self.prompts.render(
            "agents/sql_generator.md",
            user_query=resolved_query,
            schema_context=schema_context,
            business_context=business_context,
            conversation_context=conversation_context,
            backend=input.database_type or getattr(self.config.database, "db_type", "postgresql"),
            user_preferences={"default_limit": 10},
        )

    def _truncate_context(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars].rstrip()
        return (
            f"{truncated}\n\n"
            "[Context truncated for latency budget. Ask a narrower query for more schema detail.]"
        )

    async def _get_live_schema_context(
        self,
        query: str,
        database_type: str | None = None,
        database_url: str | None = None,
        include_profile: bool = False,
        focus_tables: list[str] | None = None,
    ) -> str | None:
        db_type = database_type or getattr(self.config.database, "db_type", "postgresql")

        db_url = database_url or (
            str(self.config.database.url) if self.config.database.url else None
        )
        if not db_url:
            return None

        focus_key = ",".join(sorted(focus_tables or []))
        cache_key = (
            f"{db_type}::{db_url}::{query.lower().strip()}::{include_profile}::{focus_key}"
        )
        schema_key = f"{db_type}::{db_url}"
        cached = self._live_schema_cache.get(cache_key)
        if cached:
            return cached

        try:
            connector = create_connector(
                database_url=db_url,
                database_type=db_type,
                pool_size=self.config.database.pool_size,
                timeout=10,
            )
        except Exception:
            return None

        try:
            await connector.connect()
            context, qualified_tables = await self._fetch_live_schema_context(
                connector,
                query,
                schema_key,
                include_profile,
                focus_tables=focus_tables,
                db_type=db_type,
                db_url=db_url,
            )
            if context:
                self._live_schema_cache[cache_key] = context
            if qualified_tables:
                expanded_tables = set()
                for table in qualified_tables:
                    expanded_tables.add(table.lower())
                    if "." in table:
                        expanded_tables.add(table.split(".")[-1].lower())
                if expanded_tables:
                    self._live_schema_tables_cache[schema_key] = expanded_tables
            return context
        except Exception as exc:
            logger.warning(f"Live schema lookup failed: {exc}")
            return None
        finally:
            await connector.close()

    async def _fetch_live_schema_context(
        self,
        connector: BaseConnector,
        query: str,
        schema_key: str,
        include_profile: bool,
        focus_tables: list[str] | None = None,
        *,
        db_type: str,
        db_url: str,
    ) -> tuple[str | None, list[str]]:
        qualified_tables, columns_by_table = await self._load_schema_snapshot(
            connector=connector,
            schema_key=schema_key,
            db_type=db_type,
        )
        if not qualified_tables:
            return None, []

        max_tables = 200
        if self._pipeline_flag("sql_prompt_budget_enabled", True):
            max_tables = self._pipeline_int("sql_prompt_max_tables", 80)

        entries = []
        for qualified in qualified_tables[:max_tables]:
            entries.append(qualified)

        if not entries:
            return None, []

        tables = ", ".join(entries)

        columns_context, resolved_focus_tables = self._build_columns_context_from_map(
            query=query,
            qualified_tables=qualified_tables,
            columns_by_table=columns_by_table,
            preferred_tables=focus_tables,
        )

        join_context = ""
        profile_context = ""
        cached_profile_context = ""
        if include_profile and columns_by_table:
            join_context = self._build_join_hints_context(columns_by_table, resolved_focus_tables)
        if include_profile and columns_by_table and db_type == "postgresql":
            profile_context = await self._build_lightweight_profile_context(
                connector, schema_key, query, columns_by_table, resolved_focus_tables
            )
            cached_profile_context = self._build_cached_profile_context(
                db_type=db_type,
                db_url=db_url,
                focus_tables=resolved_focus_tables,
            )

        return (
            f"**Tables in database (compact list):** {tables}"
            f"{columns_context}"
            f"{join_context}"
            f"{profile_context}"
            f"{cached_profile_context}"
        ), qualified_tables

    async def _load_schema_snapshot(
        self,
        *,
        connector: BaseConnector,
        schema_key: str,
        db_type: str,
    ) -> tuple[list[str], dict[str, list[tuple[str, str | None]]]]:
        use_snapshot_cache = self._pipeline_flag("schema_snapshot_cache_enabled", True)
        snapshot_ttl_seconds = self._pipeline_int("schema_snapshot_cache_ttl_seconds", 21600)
        if use_snapshot_cache:
            cached = self._live_schema_snapshot_cache.get(schema_key)
            if cached:
                tables = cached.get("tables")
                columns = cached.get("columns")
                cached_at = cached.get("cached_at")
                is_expired = False
                if snapshot_ttl_seconds > 0 and isinstance(cached_at, int | float):
                    is_expired = (time.time() - float(cached_at)) > snapshot_ttl_seconds
                if not is_expired and isinstance(tables, list) and isinstance(columns, dict):
                    return list(tables), columns
                if is_expired:
                    self._live_schema_snapshot_cache.pop(schema_key, None)

        if db_type == "postgresql":
            tables_query = (
                "SELECT table_schema, table_name "
                "FROM information_schema.tables "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                "AND table_name NOT IN ("
                "'database_connections', "
                "'profiling_jobs', "
                "'profiling_profiles', "
                "'pending_datapoints', "
                "'datapoint_generation_jobs'"
                ") "
                "ORDER BY table_schema, table_name"
            )
            result = await connector.execute(tables_query)
            if not result.rows:
                return [], {}

            qualified_tables: list[str] = []
            for row in result.rows:
                schema = row.get("table_schema")
                table = row.get("table_name")
                if schema and table:
                    qualified_tables.append(f"{schema}.{table}")
                elif table:
                    qualified_tables.append(str(table))

            if not qualified_tables:
                return [], {}

            qualified_set = set(qualified_tables)
            columns_query = (
                "SELECT table_schema, table_name, column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                "ORDER BY table_schema, table_name, ordinal_position"
            )
            columns_result = await connector.execute(columns_query)
            columns_by_table: dict[str, list[tuple[str, str | None]]] = {}
            for row in columns_result.rows:
                schema = row.get("table_schema")
                table = row.get("table_name")
                column = row.get("column_name")
                dtype = row.get("data_type")
                if not (schema and table and column):
                    continue
                key = f"{schema}.{table}"
                if key not in qualified_set:
                    continue
                columns_by_table.setdefault(key, []).append((str(column), dtype))
        else:
            tables_info = await connector.get_schema()
            qualified_tables = []
            columns_by_table = {}
            for table in tables_info:
                schema = getattr(table, "schema_name", None) or getattr(table, "schema", None)
                name = getattr(table, "table_name", None)
                if not name:
                    continue
                key = f"{schema}.{name}" if schema else str(name)
                qualified_tables.append(key)
                cols = []
                for column in getattr(table, "columns", []):
                    col_name = getattr(column, "name", None)
                    if not col_name:
                        continue
                    cols.append((str(col_name), getattr(column, "data_type", None)))
                if cols:
                    columns_by_table[key] = cols

        if use_snapshot_cache:
            self._live_schema_snapshot_cache[schema_key] = {
                "tables": list(qualified_tables),
                "columns": columns_by_table,
                "cached_at": time.time(),
            }

        return qualified_tables, columns_by_table

    def _build_columns_context_from_map(
        self,
        *,
        query: str,
        qualified_tables: list[str],
        columns_by_table: dict[str, list[tuple[str, str | None]]],
        preferred_tables: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        if not qualified_tables or not columns_by_table:
            return "", []

        query_lower = query.lower()
        is_list_tables = bool(
            re.search(r"\b(list|show|what|which)\s+tables\b", query_lower)
            or "tables exist" in query_lower
            or "available tables" in query_lower
        )

        focus_limit = 10
        if self._pipeline_flag("sql_prompt_budget_enabled", True):
            focus_limit = self._pipeline_int("sql_prompt_focus_tables", 8)

        if preferred_tables:
            allowed = {value.lower() for value in preferred_tables}
            matched = []
            for table in columns_by_table.keys():
                lower_table = table.lower()
                short = lower_table.split(".")[-1]
                if lower_table in allowed or short in allowed:
                    matched.append(table)
            if matched:
                focus_tables = matched[:focus_limit]
            else:
                focus_tables = self._rank_tables_by_query(query, columns_by_table)[:focus_limit]
        elif is_list_tables:
            focus_tables = sorted(columns_by_table.keys())[:focus_limit]
        else:
            focus_tables = self._rank_tables_by_query(query, columns_by_table)[:focus_limit]

        if not focus_tables:
            return "", []

        max_columns = 30
        if self._pipeline_flag("sql_prompt_budget_enabled", True):
            max_columns = self._pipeline_int("sql_prompt_max_columns_per_table", 18)

        lines = []
        for table in focus_tables:
            columns = columns_by_table.get(table, [])
            if columns:
                formatted_columns = []
                for name, dtype in columns[:max_columns]:
                    formatted_columns.append(f"{name} ({dtype})" if dtype else name)
                lines.append(f"- {table}: {', '.join(formatted_columns)}")

        if not lines:
            return "", focus_tables

        header = (
            "**Columns (all tables):**"
            if is_list_tables
            else "**Columns (top matched tables):**"
        )
        return f"\n{header}\n" + "\n".join(lines), focus_tables

    def _rank_tables_by_query(
        self, query: str, columns_by_table: dict[str, list[tuple[str, str | None]]]
    ) -> list[str]:
        tokens = self._tokenize_query(query)
        if not tokens:
            return sorted(columns_by_table.keys())

        scores: dict[str, int] = {}
        for table, columns in columns_by_table.items():
            score = 0
            table_tokens = table.lower().replace(".", "_").split("_")
            column_names = [name.lower() for name, _ in columns]
            for token in tokens:
                if token in table_tokens:
                    score += 3
                if any(token in col for col in column_names):
                    score += 2
            if len(tokens) >= 2:
                bigrams = {" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)}
                for bigram in bigrams:
                    if bigram.replace(" ", "_") in table.lower():
                        score += 4
            scores[table] = score

        return sorted(
            scores.keys(),
            key=lambda item: (scores[item], item),
            reverse=True,
        )

    def _tokenize_query(self, query: str) -> list[str]:
        text = re.sub(r"[^a-z0-9_\\s]", " ", query.lower())
        raw_tokens = [token for token in text.split() if len(token) > 1]
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
            "show",
            "list",
            "tables",
            "table",
            "in",
            "of",
            "for",
            "to",
            "and",
            "or",
            "by",
            "with",
            "on",
            "from",
            "does",
            "exist",
            "exists",
        }
        return [token for token in raw_tokens if token not in stopwords]

    def _build_join_hints_context(
        self,
        columns_by_table: dict[str, list[tuple[str, str | None]]],
        focus_tables: list[str],
    ) -> str:
        if not focus_tables:
            return ""

        table_aliases: dict[str, set[str]] = {}
        table_columns: dict[str, list[str]] = {}
        for table in focus_tables[:8]:
            base = table.split(".")[-1].lower()
            aliases = {base}
            if base.endswith("s") and len(base) > 1:
                aliases.add(base[:-1])
            else:
                aliases.add(f"{base}s")
            table_aliases[table] = aliases
            table_columns[table] = [name.lower() for name, _ in columns_by_table.get(table, [])]

        hints: list[str] = []
        seen = set()
        for source_table, columns in table_columns.items():
            for column in columns:
                if not column.endswith("_id"):
                    continue
                key = column[:-3]
                for target_table, aliases in table_aliases.items():
                    if target_table == source_table:
                        continue
                    if key not in aliases:
                        continue
                    target_columns = table_columns.get(target_table, [])
                    target_column = None
                    if "id" in target_columns:
                        target_column = "id"
                    elif f"{key}_id" in target_columns:
                        target_column = f"{key}_id"
                    hint = (
                        f"- {source_table}.{column} -> "
                        f"{target_table}.{target_column or 'id'}"
                    )
                    if hint not in seen:
                        seen.add(hint)
                        hints.append(hint)
                if len(hints) >= 8:
                    break
            if len(hints) >= 8:
                break

        if not hints:
            return ""
        return "\n**Join hints (heuristic):**\n" + "\n".join(hints)

    async def _build_lightweight_profile_context(
        self,
        connector: BaseConnector,
        schema_key: str,
        query: str,
        columns_by_table: dict[str, list[tuple[str, str | None]]],
        focus_tables: list[str],
    ) -> str:
        if not focus_tables:
            return ""

        profile_cache = self._live_profile_cache.setdefault(schema_key, {})
        lines: list[str] = []
        for table in focus_tables[:3]:
            if table not in profile_cache:
                profile_cache[table] = await self._fetch_table_profile(
                    connector, table, query, columns_by_table.get(table, [])
                )
            profile = profile_cache.get(table, {})
            if not profile:
                continue
            row_count = profile.get("row_count")
            columns = profile.get("columns", {})
            if not columns:
                continue
            line = f"- {table}"
            if row_count is not None:
                line += f" (~{row_count} rows)"
            lines.append(line)
            for column_name, stats in list(columns.items())[:5]:
                parts = []
                null_frac = stats.get("null_frac")
                if null_frac is not None:
                    parts.append(f"null_frac={null_frac:.2f}")
                n_distinct = stats.get("n_distinct")
                if n_distinct is not None:
                    parts.append(f"n_distinct={n_distinct}")
                common_vals = stats.get("common_vals")
                if common_vals:
                    preview = ", ".join(common_vals[:3])
                    parts.append(f"examples=[{preview}]")
                if parts:
                    lines.append(f"  - {column_name}: " + ", ".join(parts))

        if not lines:
            return ""
        return "\n**Lightweight stats (cached):**\n" + "\n".join(lines)

    async def _fetch_table_profile(
        self,
        connector: BaseConnector,
        table: str,
        query: str,
        columns: list[tuple[str, str | None]],
    ) -> dict[str, object]:
        schema = "public"
        table_name = table
        if "." in table:
            schema, table_name = table.split(".", 1)

        row_count = await self._fetch_table_row_estimate(connector, schema, table_name)
        selected_columns = self._select_profile_columns(query, columns)
        column_stats = await self._fetch_column_stats(
            connector, schema, table_name, selected_columns
        )
        return {
            "row_count": row_count,
            "columns": column_stats,
        }

    def _select_profile_columns(
        self, query: str, columns: list[tuple[str, str | None]]
    ) -> list[str]:
        if not columns:
            return []
        tokens = set(self._tokenize_query(query))
        tokens.update({"amount", "total", "revenue", "price", "cost", "sales"})
        numeric_types = {
            "smallint",
            "integer",
            "bigint",
            "numeric",
            "decimal",
            "real",
            "double precision",
            "float",
        }
        preferred: list[str] = []
        fallback: list[str] = []
        for name, dtype in columns:
            dtype_norm = (dtype or "").lower()
            if dtype_norm and dtype_norm in numeric_types:
                fallback.append(name)
                if any(token in name.lower() for token in tokens):
                    preferred.append(name)

        selected = preferred or fallback
        return selected[:5]

    async def _fetch_table_row_estimate(
        self, connector: BaseConnector, schema: str, table: str
    ) -> int | None:
        query = (
            "SELECT reltuples::BIGINT AS estimate "
            "FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = $1 AND c.relname = $2"
        )
        try:
            result = await connector.execute(query, params=[schema, table])
        except Exception:
            return None
        if not result.rows:
            return None
        value = result.rows[0].get("estimate")
        return int(value) if value is not None else None

    async def _fetch_column_stats(
        self,
        connector: BaseConnector,
        schema: str,
        table: str,
        columns: list[str],
    ) -> dict[str, dict[str, object]]:
        if not columns:
            return {}
        query = (
            "SELECT attname, null_frac, n_distinct, most_common_vals "
            "FROM pg_stats "
            "WHERE schemaname = $1 AND tablename = $2 AND attname = ANY($3::text[])"
        )
        try:
            result = await connector.execute(query, params=[schema, table, columns])
        except Exception:
            return {}

        stats: dict[str, dict[str, object]] = {}
        for row in result.rows:
            attname = row.get("attname")
            if not attname:
                continue
            common_vals = row.get("most_common_vals") or []
            common_vals = [str(value) for value in common_vals if value is not None]
            stats[str(attname)] = {
                "null_frac": row.get("null_frac"),
                "n_distinct": row.get("n_distinct"),
                "common_vals": common_vals,
            }
        return stats

    def _build_cached_profile_context(
        self,
        *,
        db_type: str,
        db_url: str,
        focus_tables: list[str],
    ) -> str:
        snapshot = load_profile_cache(database_type=db_type, database_url=db_url)
        if not snapshot:
            return ""
        table_entries = snapshot.get("tables")
        if not isinstance(table_entries, list) or not table_entries:
            return ""

        selected = []
        focus_set = {table.lower() for table in focus_tables}
        for entry in table_entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            if focus_set and name.lower() not in focus_set:
                continue
            selected.append(entry)
            if len(selected) >= 3:
                break

        if not selected:
            for entry in table_entries[:3]:
                if isinstance(entry, dict):
                    selected.append(entry)

        if not selected:
            return ""

        lines = []
        for entry in selected:
            table_name = str(entry.get("name") or "unknown")
            status = str(entry.get("status") or "unknown")
            row_count = entry.get("row_count")
            line = f"- {table_name}"
            if row_count is not None:
                line += f" (~{row_count} rows)"
            if status != "completed":
                line += f" [{status}]"
            lines.append(line)
            columns = entry.get("columns")
            if isinstance(columns, list):
                for column in columns[:4]:
                    if not isinstance(column, dict):
                        continue
                    col_name = str(column.get("name") or "")
                    data_type = str(column.get("data_type") or "")
                    if not col_name:
                        continue
                    if data_type:
                        lines.append(f"  - {col_name} ({data_type})")
                    else:
                        lines.append(f"  - {col_name}")
        if not lines:
            return ""
        return "\n**Auto-profile cache snapshot:**\n" + "\n".join(lines)

    def _catalog_schemas_for_db(self, db_type: str) -> set[str]:
        return get_catalog_schemas(db_type)

    def _catalog_aliases_for_db(self, db_type: str) -> set[str]:
        return get_catalog_aliases(db_type)

    def _is_catalog_table(
        self, table: str, catalog_schemas: set[str], catalog_aliases: set[str]
    ) -> bool:
        if table in catalog_aliases:
            return True
        if table in catalog_schemas:
            return True
        for schema in catalog_schemas:
            if table.startswith(f"{schema}."):
                return True
            if f".{schema}." in table:
                return True
        return False

    def _build_correction_prompt(
        self, generated_sql: GeneratedSQL, issues: list[ValidationIssue], input: SQLAgentInput
    ) -> str:
        """
        Build prompt for SQL self-correction.

        Args:
            generated_sql: Original generated SQL
            issues: Validation issues found
            input: Original input

        Returns:
            Correction prompt string
        """
        # Extract schema context
        schema_context = self._format_schema_context(input.investigation_memory)

        # Format issues
        issues_text = "\n".join(
            [
                f"- {issue.issue_type.upper()}: {issue.message}"
                + (f" (Suggested fix: {issue.suggested_fix})" if issue.suggested_fix else "")
                for issue in issues
            ]
        )

        return self.prompts.render(
            "agents/sql_correction.md",
            original_sql=generated_sql.sql,
            issues=issues_text,
            schema_context=schema_context,
        )

    def _format_schema_context(self, memory) -> str:
        """
        Format schema DataPoints into readable context.

        Args:
            memory: InvestigationMemory with DataPoints

        Returns:
            Formatted schema context string
        """
        schema_parts = []

        for dp in memory.datapoints:
            if dp.datapoint_type == "Schema":
                # Access metadata as dict
                metadata = dp.metadata if isinstance(dp.metadata, dict) else {}

                table_name = metadata.get("table_name", "unknown")
                table_schema = metadata.get("schema", "")
                business_purpose = metadata.get("business_purpose", "")

                # Only prefix schema if table_name doesn't already include it
                # Avoids double-qualification like "analytics.analytics.fact_sales"
                if table_schema and "." not in table_name:
                    full_table_name = f"{table_schema}.{table_name}"
                else:
                    full_table_name = table_name
                schema_parts.append(f"\n**Table: {full_table_name}**")
                if business_purpose:
                    schema_parts.append(f"Purpose: {business_purpose}")

                # Add columns
                columns = self._coerce_metadata_list(metadata.get("key_columns", []))
                if columns:
                    schema_parts.append("Columns:")
                    for col in columns:
                        if isinstance(col, dict):
                            col_name = col.get("name", "unknown")
                            col_type = col.get("type", "unknown")
                            col_meaning = col.get("business_meaning", "")
                            schema_parts.append(f"  - {col_name} ({col_type}): {col_meaning}")
                        elif isinstance(col, str):
                            schema_parts.append(f"  - {col}")

                # Add relationships
                relationships = self._coerce_metadata_list(metadata.get("relationships", []))
                if relationships:
                    schema_parts.append("Relationships:")
                    for rel in relationships:
                        if isinstance(rel, dict):
                            target = rel.get("target_table", "unknown")
                            join_col = rel.get("join_column", "unknown")
                            cardinality = rel.get("cardinality", "")
                            schema_parts.append(f"  - JOIN {target} ON {join_col} ({cardinality})")
                        elif isinstance(rel, str):
                            schema_parts.append(f"  - {rel}")

                # Add gotchas/common queries
                gotchas = self._coerce_string_list(metadata.get("gotchas", []))
                if gotchas:
                    schema_parts.append(f"Important Notes: {'; '.join(gotchas)}")

        return "\n".join(schema_parts) if schema_parts else "No schema context available"

    def _format_business_context(self, memory) -> str:
        """
        Format business DataPoints into readable context.

        Args:
            memory: InvestigationMemory with DataPoints

        Returns:
            Formatted business context string
        """
        business_parts = []

        for dp in memory.datapoints:
            if dp.datapoint_type == "Business":
                # Access metadata as dict
                metadata = dp.metadata if isinstance(dp.metadata, dict) else {}

                name = dp.name
                calculation = metadata.get("calculation", "")
                synonyms = self._coerce_string_list(metadata.get("synonyms", []))
                business_rules = self._coerce_string_list(metadata.get("business_rules", []))

                business_parts.append(f"\n**Metric: {name}**")
                if calculation:
                    business_parts.append(f"Calculation: {calculation}")
                if synonyms:
                    business_parts.append(f"Also known as: {', '.join(synonyms)}")
                if business_rules:
                    business_parts.append("Business Rules:")
                    for rule in business_rules:
                        business_parts.append(f"  - {rule}")

        return "\n".join(business_parts) if business_parts else "No business rules available"

    def _coerce_metadata_list(self, value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    return []
        return []

    def _coerce_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    pass
            if "," in stripped:
                return [part.strip() for part in stripped.split(",") if part.strip()]
            return [stripped]
        return []

    def _format_conversation_context(self, history: list[dict] | list) -> str:
        if not history:
            return "No conversation context available."

        recent = history[-6:]
        lines = []
        for msg in recent:
            role = "user"
            content = ""
            if isinstance(msg, dict):
                role = str(msg.get("role", role))
                content = str(msg.get("content", ""))
            else:
                role = str(getattr(msg, "role", role))
                content = str(getattr(msg, "content", ""))
            if content:
                lines.append(f"{role}: {content}")

        return "\n".join(lines) if lines else "No conversation context available."

    def _parse_llm_response(self, content: str, input: SQLAgentInput) -> GeneratedSQL:
        """
        Parse LLM response into GeneratedSQL.

        Args:
            content: Raw LLM response content
            input: Original input

        Returns:
            GeneratedSQL object

        Raises:
            ValueError: If response cannot be parsed
        """
        # Try JSON first.
        json_str = None
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

        if json_str:
            try:
                data = json.loads(json_str)
                sql_value = data.get("sql") or data.get("query")
                if sql_value:
                    explanation = (
                        data.get("explanation")
                        or data.get("reasoning")
                        or data.get("rationale")
                        or ""
                    )
                    generated = GeneratedSQL(
                        sql=sql_value,
                        explanation=explanation,
                        used_datapoints=data.get("used_datapoints", []),
                        confidence=data.get("confidence", 0.8),
                        assumptions=data.get("assumptions", []),
                        clarifying_questions=data.get("clarifying_questions", []),
                    )
                    return self._apply_row_limit_policy(generated, input.query)
            except json.JSONDecodeError as e:
                logger.debug(f"Invalid JSON in LLM response: {e}")

        # Fallback: extract SQL from markdown/code/text output.
        sql_text = self._extract_sql_from_response(content)
        if sql_text:
            generated = GeneratedSQL(
                sql=sql_text,
                explanation="Generated SQL",
                used_datapoints=[],
                confidence=0.6,
                assumptions=[],
                clarifying_questions=[],
            )
            return self._apply_row_limit_policy(generated, input.query)

        logger.warning("Failed to parse LLM response: Response missing 'sql' field")
        logger.debug(f"LLM response content: {content}")
        raise ValueError("Failed to parse LLM response: Response missing 'sql' field")

    def _apply_row_limit_policy(self, generated: GeneratedSQL, query: str) -> GeneratedSQL:
        """Normalize SQL limits for safe interactive responses."""
        sql = generated.sql.strip()
        if not sql:
            return generated

        sql_upper = sql.upper()
        if sql_upper.startswith("SHOW") or sql_upper.startswith("DESCRIBE") or sql_upper.startswith("DESC"):
            return generated
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return generated
        if self._is_single_value_query(sql_upper):
            return generated

        requested_limit = self._extract_requested_limit(query)
        if requested_limit is not None:
            target_limit = max(1, min(requested_limit, self._max_safe_row_limit))
            force_limit = True
        else:
            target_limit = self._default_row_limit
            if self.catalog.is_list_tables_query(query.lower()) or self._is_catalog_metadata_query(sql_upper):
                target_limit = self._max_safe_row_limit
            force_limit = False

        rewritten = self._rewrite_sql_limit(sql, target_limit, force_limit=force_limit)
        if rewritten == sql:
            return generated
        return generated.model_copy(update={"sql": rewritten})

    def _extract_requested_limit(self, query: str) -> int | None:
        """Extract explicit row limit requested by the user."""
        text = (query or "").lower()
        if not text:
            return None

        patterns = (
            r"\b(first|top|limit)\s+(\d+)\b",
            r"\bshow\s+(\d+)\s+rows?\b",
            r"\b(\d+)\s+rows?\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            numbers = [group for group in match.groups() if group and group.isdigit()]
            if not numbers:
                continue
            try:
                return int(numbers[0])
            except ValueError:
                continue
        return None

    def _rewrite_sql_limit(self, sql: str, target_limit: int, *, force_limit: bool) -> str:
        """Rewrite SQL LIMIT to enforce bounded row returns."""
        top_level_limits = self._scan_top_level_limit_clauses(sql)
        numeric_limits = [entry for entry in top_level_limits if entry[2].isdigit()]

        if numeric_limits:
            value_start, value_end, value_token = numeric_limits[-1]
            current_limit = int(value_token)
            new_limit = target_limit if force_limit else min(current_limit, target_limit)
            if new_limit == current_limit:
                return sql
            return f"{sql[:value_start]}{new_limit}{sql[value_end:]}"

        # Avoid appending a second LIMIT when a top-level non-numeric variant already exists
        # (e.g. LIMIT $1). Subquery limits do not count.
        if top_level_limits:
            return sql

        stripped = sql.rstrip()
        if stripped.endswith(";"):
            stripped = stripped[:-1].rstrip()
            return f"{stripped} LIMIT {target_limit};"
        return f"{stripped} LIMIT {target_limit}"

    def _scan_top_level_limit_clauses(self, sql: str) -> list[tuple[int, int, str]]:
        """
        Return top-level LIMIT clause values.

        Returns tuples of: (value_start_index, value_end_index, value_token).
        """
        clauses: list[tuple[int, int, str]] = []
        i = 0
        depth = 0
        length = len(sql)

        while i < length:
            ch = sql[i]
            nxt = sql[i + 1] if i + 1 < length else ""

            # Skip line comments.
            if ch == "-" and nxt == "-":
                i += 2
                while i < length and sql[i] != "\n":
                    i += 1
                continue

            # Skip block comments.
            if ch == "/" and nxt == "*":
                i += 2
                while i + 1 < length and not (sql[i] == "*" and sql[i + 1] == "/"):
                    i += 1
                i = min(i + 2, length)
                continue

            # Skip quoted strings/identifiers.
            if ch in ("'", '"', "`"):
                quote = ch
                i += 1
                while i < length:
                    if sql[i] == quote:
                        # Handle doubled quotes: '' or "" or ``
                        if i + 1 < length and sql[i + 1] == quote:
                            i += 2
                            continue
                        i += 1
                        break
                    i += 1
                continue

            if ch == "(":
                depth += 1
                i += 1
                continue

            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue

            if depth == 0 and self._matches_keyword(sql, i, "LIMIT"):
                value_start = i + len("LIMIT")
                while value_start < length and sql[value_start].isspace():
                    value_start += 1
                value_end = value_start
                while value_end < length and re.match(r"[A-Za-z0-9_$:]", sql[value_end]):
                    value_end += 1
                value_token = sql[value_start:value_end]
                clauses.append((value_start, value_end, value_token))
                i = value_end
                continue

            i += 1

        return clauses

    @staticmethod
    def _matches_keyword(sql: str, index: int, keyword: str) -> bool:
        end = index + len(keyword)
        if sql[index:end].upper() != keyword:
            return False
        before = sql[index - 1] if index > 0 else " "
        after = sql[end] if end < len(sql) else " "
        return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")

    def _is_single_value_query(self, sql_upper: str) -> bool:
        """Heuristic: single aggregate queries usually return one row and don't need LIMIT."""
        if "GROUP BY" in sql_upper:
            return False
        aggregate_tokens = ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "BOOL_OR(", "BOOL_AND(")
        return any(token in sql_upper for token in aggregate_tokens)

    def _is_catalog_metadata_query(self, sql_upper: str) -> bool:
        """Detect catalog/system metadata SQL where a larger default preview is acceptable."""
        catalog_markers = (
            "INFORMATION_SCHEMA.TABLES",
            "INFORMATION_SCHEMA.COLUMNS",
            "PG_CATALOG.",
            "SYSTEM.TABLES",
            "SVV_TABLES",
            "PG_TABLE_DEF",
        )
        return any(marker in sql_upper for marker in catalog_markers)

    def _extract_sql_from_response(self, content: str) -> str | None:
        """Extract SQL from non-JSON LLM output."""
        code_blocks = re.findall(r"```(?:sql)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            candidate = block.strip()
            extracted = self._extract_sql_statement(candidate)
            if extracted:
                return extracted

        # Common key-value style fallback: sql: SELECT ...
        inline_sql_field = re.search(
            r"\bsql\s*[:=]\s*(.+)$",
            content,
            re.IGNORECASE | re.MULTILINE,
        )
        if inline_sql_field:
            candidate = inline_sql_field.group(1).strip()
            extracted = self._extract_sql_statement(candidate)
            if extracted:
                return extracted

        return self._extract_sql_statement(content)

    def _extract_sql_statement(self, text: str) -> str | None:
        """Extract a SQL statement from free text."""
        # Prefer SQL-looking statements at line boundaries to avoid
        # accidentally parsing natural language like "show you ...".
        select_like = re.compile(
            r"(?:^|[\r\n])\s*(SELECT|WITH|EXPLAIN|DESCRIBE|DESC)\b[\s\S]*?(?:;|$)",
            re.IGNORECASE,
        )
        show_like = re.compile(
            r"(?:^|[\r\n])\s*SHOW\s+"
            r"(?:TABLES|FULL\s+TABLES|COLUMNS|DATABASES|SCHEMAS|CREATE\s+TABLE)\b"
            r"[\s\S]*?(?:;|$)",
            re.IGNORECASE,
        )

        for pattern in (select_like, show_like):
            match = pattern.search(text)
            if not match:
                continue
            statement = match.group(0).strip()
            statement = re.sub(r"\s+", " ", statement).strip().rstrip(";")
            if statement and "{" not in statement and "}" not in statement:
                return statement

        return None

    def _validate_input(self, input: SQLAgentInput) -> None:
        """
        Validate input type.

        Args:
            input: Input to validate

        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(input, SQLAgentInput):
            from backend.models.agent import ValidationError

            raise ValidationError(
                agent=self.name,
                message=f"Expected SQLAgentInput, got {type(input).__name__}",
                context={"input_type": type(input).__name__},
            )
