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
from typing import Any
from urllib.parse import urlparse

from backend.agents.base import BaseAgent
from backend.config import get_settings
from backend.connectors.postgres import PostgresConnector
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
        else:
            self.llm = llm_provider
            self.fast_llm = llm_provider

        provider_name = getattr(self.llm, "provider", "unknown")
        model_name = getattr(self.llm, "model", "unknown")
        logger.info(
            f"SQLAgent initialized with {provider_name} provider",
            extra={"provider": provider_name, "model": model_name},
        )
        self.prompts = PromptLoader()
        self.catalog = CatalogIntelligence()
        self._live_schema_cache: dict[str, str] = {}
        self._live_schema_snapshot_cache: dict[
            str, dict[str, list[str] | dict[str, list[tuple[str, str | None]]]]
        ] = {}
        self._live_schema_tables_cache: dict[str, set[str]] = {}
        self._live_profile_cache: dict[str, dict[str, dict[str, object]]] = {}
        self._max_safe_row_limit = 10
        self._default_row_limit = 5

    def _pipeline_flag(self, name: str, default: bool) -> bool:
        pipeline_cfg = getattr(self.config, "pipeline", None)
        if pipeline_cfg is None:
            return default
        return bool(getattr(pipeline_cfg, name, default))

    def _pipeline_int(self, name: str, default: int) -> int:
        pipeline_cfg = getattr(self.config, "pipeline", None)
        if pipeline_cfg is None:
            return default
        value = getattr(pipeline_cfg, name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _pipeline_float(self, name: str, default: float) -> float:
        pipeline_cfg = getattr(self.config, "pipeline", None)
        if pipeline_cfg is None:
            return default
        value = getattr(pipeline_cfg, name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

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

        try:
            # Initial SQL generation
            try:
                generated_sql = await self._generate_sql(input, metadata)
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
                    data={},
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
                    generated_sql=generated_sql, issues=issues, input=input, metadata=metadata
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
                data={},
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

    async def _generate_sql(self, input: SQLAgentInput, metadata: AgentMetadata) -> GeneratedSQL:
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

        # Build prompt with context
        prompt = await self._build_generation_prompt(input)

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
                and self.fast_llm is not self.llm
            )

            if use_two_stage:
                fast_generated = await self._request_sql_from_llm(
                    provider=self.fast_llm,
                    llm_request=llm_request,
                    input=input,
                )
                if self._should_accept_fast_sql(fast_generated, input):
                    generated_sql = fast_generated
                else:
                    generated_sql = await self._request_sql_from_llm(
                        provider=self.llm,
                        llm_request=llm_request,
                        input=input,
                    )
            else:
                generated_sql = await self._request_sql_from_llm(
                    provider=self.llm,
                    llm_request=llm_request,
                    input=input,
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

    async def _request_sql_from_llm(
        self,
        *,
        provider: Any,
        llm_request: LLMRequest,
        input: SQLAgentInput,
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
            corrected_sql = self._parse_llm_response(response.content, input)

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
            if isinstance(dp.metadata, dict) and "table_name" in dp.metadata:
                table_name = dp.metadata["table_name"]
                available_tables.add(table_name.upper())
                # Also add without schema prefix
                if "." in table_name:
                    available_tables.add(table_name.split(".")[-1].upper())
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

    async def _build_generation_prompt(self, input: SQLAgentInput) -> str:
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
        live_context = await self._get_live_schema_context(
            query=resolved_query,
            database_type=input.database_type,
            database_url=input.database_url,
            include_profile=include_profile,
        )
        if live_context:
            if schema_context == "No schema context available":
                schema_context = live_context
            else:
                schema_context = (
                    f"{schema_context}\n\n**Live schema snapshot (authoritative):**\n"
                    f"{live_context}"
                )
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
    ) -> str | None:
        db_type = database_type or getattr(self.config.database, "db_type", "postgresql")
        if db_type != "postgresql":
            return None

        db_url = database_url or (
            str(self.config.database.url) if self.config.database.url else None
        )
        if not db_url:
            return None

        cache_key = f"{db_type}::{db_url}::{query.lower().strip()}::{include_profile}"
        schema_key = f"{db_type}::{db_url}"
        cached = self._live_schema_cache.get(cache_key)
        if cached:
            return cached

        parsed = urlparse(db_url.replace("postgresql+asyncpg://", "postgresql://"))
        if not parsed.hostname:
            return None

        connector = PostgresConnector(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "postgres",
            user=parsed.username or "postgres",
            password=parsed.password or "",
        )

        try:
            await connector.connect()
            context, qualified_tables = await self._fetch_live_schema_context(
                connector,
                query,
                schema_key,
                include_profile,
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
        connector: PostgresConnector,
        query: str,
        schema_key: str,
        include_profile: bool,
        *,
        db_type: str,
        db_url: str,
    ) -> tuple[str | None, list[str]]:
        qualified_tables, columns_by_table = await self._load_schema_snapshot(
            connector=connector,
            schema_key=schema_key,
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

        columns_context, focus_tables = self._build_columns_context_from_map(
            query=query,
            qualified_tables=qualified_tables,
            columns_by_table=columns_by_table,
        )

        join_context = ""
        profile_context = ""
        cached_profile_context = ""
        if include_profile and columns_by_table:
            join_context = self._build_join_hints_context(columns_by_table, focus_tables)
            profile_context = await self._build_lightweight_profile_context(
                connector, schema_key, query, columns_by_table, focus_tables
            )
            cached_profile_context = self._build_cached_profile_context(
                db_type=db_type,
                db_url=db_url,
                focus_tables=focus_tables,
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
        connector: PostgresConnector,
        schema_key: str,
    ) -> tuple[list[str], dict[str, list[tuple[str, str | None]]]]:
        use_snapshot_cache = self._pipeline_flag("schema_snapshot_cache_enabled", True)
        if use_snapshot_cache:
            cached = self._live_schema_snapshot_cache.get(schema_key)
            if cached:
                tables = cached.get("tables")
                columns = cached.get("columns")
                if isinstance(tables, list) and isinstance(columns, dict):
                    return list(tables), columns

        tables_query = (
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
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

        if use_snapshot_cache:
            self._live_schema_snapshot_cache[schema_key] = {
                "tables": list(qualified_tables),
                "columns": columns_by_table,
            }

        return qualified_tables, columns_by_table

    def _build_columns_context_from_map(
        self,
        *,
        query: str,
        qualified_tables: list[str],
        columns_by_table: dict[str, list[tuple[str, str | None]]],
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

        if is_list_tables:
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
        connector: PostgresConnector,
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
        connector: PostgresConnector,
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
        self, connector: PostgresConnector, schema: str, table: str
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
        connector: PostgresConnector,
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
