"""
ExecutorAgent: Query execution and result formatting.

This agent executes validated SQL queries and formats results:
- Executes queries with timeout protection
- Generates natural language summaries using GPT-4o-mini
- Suggests visualization types based on data shape
- Handles pagination/truncation for large results
- Includes source citations from pipeline

Uses database connectors for execution and LLM for summarization.
"""

import asyncio
import json
import logging
import re
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.agents.base import BaseAgent
from backend.config import get_settings
from backend.connectors.base import BaseConnector, QueryError
from backend.connectors.clickhouse import ClickHouseConnector
from backend.connectors.postgres import PostgresConnector
from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMMessage, LLMRequest
from backend.models import (
    ExecutedQuery,
    ExecutorAgentInput,
    ExecutorAgentOutput,
    QueryResult,
)
from backend.prompts.loader import PromptLoader

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    """
    Query execution and result formatting agent.

    Executes SQL queries, generates natural language summaries,
    and suggests appropriate visualizations.
    """

    def __init__(self, llm_provider=None):
        """
        Initialize ExecutorAgent with LLM provider.

        Args:
            llm_provider: Optional LLM provider. If None, creates default provider.
        """
        super().__init__(name="ExecutorAgent")

        # Get configuration
        self.config = get_settings()

        # Get LLM provider (use mini model for summarization/cost)
        if llm_provider is None:
            self.llm = LLMProviderFactory.create_default_provider(
                self.config.llm, model_type="mini"
            )
        else:
            self.llm = llm_provider
        self.prompts = PromptLoader()

    async def execute(self, input: ExecutorAgentInput) -> ExecutorAgentOutput:
        """
        Execute SQL query and format results.

        Args:
            input: ExecutorAgentInput with validated SQL and database config

        Returns:
            ExecutorAgentOutput with query results and summary

        Raises:
            QueryError: If query execution fails
            LLMError: If summary generation fails
        """
        logger.info(f"[{self.name}] Executing query on {input.database_type}")

        # Get database connector
        connector = await self._get_connector(input.database_type, input.database_url)
        llm_calls = 0
        sql_to_execute = input.validated_sql.sql

        try:
            # Execute query with timeout
            try:
                query_result = await self._execute_query(
                    connector,
                    sql_to_execute,
                    input.max_rows,
                    input.timeout_seconds,
                )
            except QueryError as exc:
                missing_relation = self._extract_missing_relation(exc)
                if missing_relation:
                    catalog = await self._fetch_schema_catalog(
                        connector, input.database_type
                    )
                    if catalog and not self._relation_in_catalog(
                        missing_relation, catalog
                    ):
                        raise QueryError(
                            f"Missing table in live schema: {missing_relation}. "
                            "Schema refresh required."
                        ) from exc

                schema_context = self._load_schema_context(input.source_datapoints)
                if self._should_probe_schema(exc, input.database_type):
                    db_context = await self._fetch_schema_context(connector, input.database_type)
                    if db_context:
                        if schema_context == "No schema datapoints available.":
                            schema_context = db_context
                        else:
                            schema_context = f"{schema_context}\n\n{db_context}"
                corrected_sql = await self._attempt_sql_correction(
                    input, sql_to_execute, exc, schema_context
                )
                if corrected_sql:
                    llm_calls += 1
                    sql_to_execute = corrected_sql
                    query_result = await self._execute_query(
                        connector,
                        sql_to_execute,
                        input.max_rows,
                        input.timeout_seconds,
                    )
                else:
                    raise

            # Generate natural language summary
            deterministic_summary = self._generate_deterministic_summary(
                input.query, sql_to_execute, query_result
            )
            if deterministic_summary:
                nl_answer, insights = deterministic_summary
            else:
                nl_answer, insights = await self._generate_summary(
                    input.query, sql_to_execute, query_result
                )
                llm_calls += 1

            # Suggest visualization
            viz_hint = self._suggest_visualization(query_result, input.query)

            # Build executed query
            executed_query = ExecutedQuery(
                query_result=query_result,
                natural_language_answer=nl_answer,
                visualization_hint=viz_hint,
                key_insights=insights,
                source_citations=input.source_datapoints,
            )

            logger.info(
                f"[{self.name}] Execution complete: {query_result.row_count} rows, "
                f"{query_result.execution_time_ms:.1f}ms"
            )

            # Create metadata with LLM call count
            metadata = self._create_metadata()
            metadata.llm_calls = llm_calls

            return ExecutorAgentOutput(
                success=True,
                executed_query=executed_query,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"[{self.name}] Execution failed: {e}")
            raise
        finally:
            await connector.close()

    async def _get_connector(
        self, database_type: str, database_url: str | None
    ) -> BaseConnector:
        """
        Get database connector for specified type.

        Args:
            database_type: Type of database (postgresql, clickhouse, mysql)

        Returns:
            Database connector instance
        """
        if database_type == "postgresql":
            if database_url:
                db_url = database_url
            elif self.config.database.url:
                db_url = str(self.config.database.url)
            else:
                raise ValueError("DATABASE_URL is not configured for query execution.")
            parsed = urlparse(db_url.replace("postgresql+asyncpg://", "postgresql://"))
            if not parsed.hostname:
                raise ValueError("Invalid PostgreSQL database URL.")
            connector = PostgresConnector(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") if parsed.path else "datachat",
                user=parsed.username or "postgres",
                password=parsed.password or "",
            )
        elif database_type == "clickhouse":
            if not database_url:
                raise ValueError("ClickHouse requires a database URL.")
            parsed = urlparse(database_url)
            if not parsed.hostname:
                raise ValueError("Invalid ClickHouse database URL.")
            connector = ClickHouseConnector(
                host=parsed.hostname,
                port=parsed.port or 8123,
                database=parsed.path.lstrip("/") if parsed.path else "default",
                user=parsed.username or "default",
                password=parsed.password or "",
            )
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

        await connector.connect()
        return connector

    async def _execute_query(
        self,
        connector: BaseConnector,
        sql: str,
        max_rows: int,
        timeout_seconds: int,
    ) -> QueryResult:
        """
        Execute SQL query with timeout and result truncation.

        Args:
            connector: Database connector
            sql: SQL query to execute
            max_rows: Maximum rows to return
            timeout_seconds: Query timeout in seconds

        Returns:
            QueryResult with data

        Raises:
            QueryError: If query execution fails
            TimeoutError: If query exceeds timeout
        """
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                connector.execute(sql),
                timeout=timeout_seconds,
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Truncate if needed
            rows = result.rows
            was_truncated = False
            if len(rows) > max_rows:
                rows = rows[:max_rows]
                was_truncated = True

            return QueryResult(
                rows=rows,
                row_count=len(rows),
                columns=result.columns,
                execution_time_ms=execution_time_ms,
                was_truncated=was_truncated,
                max_rows=max_rows if was_truncated else None,
            )

        except TimeoutError as e:
            raise TimeoutError(f"Query exceeded timeout of {timeout_seconds}s") from e
        except Exception as e:
            raise QueryError(f"Query execution failed: {e}") from e

    async def _attempt_sql_correction(
        self,
        input: ExecutorAgentInput,
        sql: str,
        error: Exception,
        schema_context: str,
    ) -> str | None:
        issues = "\n".join(
            [
                f"- EXECUTION_ERROR: {error}",
                f"- DATABASE_ENGINE: {input.database_type}",
            ]
        )
        prompt = self.prompts.render(
            "agents/sql_correction.md",
            original_sql=sql,
            issues=issues,
            schema_context=schema_context,
        )

        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=self.prompts.load("system/main.md")),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            max_tokens=2000,
        )

        try:
            response = await self.llm.generate(request)
            corrected = self._parse_correction_response(response.content)
            if not corrected or corrected.strip().lower() == sql.strip().lower():
                return None
            return corrected
        except Exception as exc:
            logger.error(f"SQL execution correction failed: {exc}")
            return None

    def _should_probe_schema(self, error: Exception, database_type: str) -> bool:
        if database_type != "postgresql":
            return False
        message = str(error).lower()
        return bool(
            re.search(r"relation .* does not exist", message)
            or "does not exist" in message
            or "undefined table" in message
        )

    async def _fetch_schema_context(
        self, connector: BaseConnector, database_type: str
    ) -> str | None:
        if database_type != "postgresql":
            return None

        tables_query = (
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY table_schema, table_name"
        )
        try:
            result = await connector.execute(tables_query)
        except Exception as exc:
            logger.warning(f"Failed to fetch schema context: {exc}")
            return None

        if not result.rows:
            return None

        entries = []
        qualified_tables = []
        for row in result.rows[:200]:
            schema = row.get("table_schema")
            table = row.get("table_name")
            if schema and table:
                qualified = f"{schema}.{table}"
                entries.append(qualified)
                qualified_tables.append(qualified)
            elif table:
                entries.append(str(table))

        if not entries:
            return None

        tables = ", ".join(entries)

        columns_context = ""
        if qualified_tables:
            columns_query = (
                "SELECT table_schema, table_name, column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                "ORDER BY table_schema, table_name, ordinal_position"
            )
            try:
                columns_result = await connector.execute(columns_query)
                if columns_result.rows:
                    columns_by_table: dict[str, list[str]] = {}
                    for row in columns_result.rows:
                        schema = row.get("table_schema")
                        table = row.get("table_name")
                        column = row.get("column_name")
                        dtype = row.get("data_type")
                        if not (schema and table and column):
                            continue
                        key = f"{schema}.{table}"
                        if key not in qualified_tables:
                            continue
                        columns_by_table.setdefault(key, []).append(
                            f"{column} ({dtype})" if dtype else str(column)
                        )
                    if columns_by_table:
                        lines = []
                        for table in sorted(columns_by_table):
                            columns = columns_by_table[table]
                            if columns:
                                lines.append(f"- {table}: {', '.join(columns[:30])}")
                        if lines:
                            columns_context = "\n**Columns:**\n" + "\n".join(lines)
            except Exception as exc:
                logger.warning(f"Failed to fetch column context: {exc}")

        return f"**Tables in database:** {tables}{columns_context}"

    async def _fetch_schema_catalog(
        self, connector: BaseConnector, database_type: str
    ) -> set[str] | None:
        if database_type != "postgresql":
            return None

        tables_query = (
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY table_schema, table_name"
        )
        try:
            result = await connector.execute(tables_query)
        except Exception as exc:
            logger.warning(f"Failed to fetch schema catalog: {exc}")
            return None

        catalog: set[str] = set()
        for row in result.rows:
            schema = row.get("table_schema")
            table = row.get("table_name")
            if schema and table:
                catalog.add(f"{schema}.{table}".lower())
            elif table:
                catalog.add(str(table).lower())
        return catalog or None

    def _extract_missing_relation(self, error: Exception) -> str | None:
        message = str(error)
        match = re.search(r'relation \"([^\"]+)\" does not exist', message, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _relation_in_catalog(self, relation: str, catalog: set[str]) -> bool:
        normalized = relation.lower()
        if normalized in catalog:
            return True
        if "." in normalized:
            _, table = normalized.split(".", 1)
            return table in catalog
        return False

    def _parse_correction_response(self, content: str) -> str | None:
        payload = self._extract_json_payload(content)
        if payload:
            sql = payload.get("sql")
            if isinstance(sql, str) and sql.strip():
                return sql.strip()

        fenced = self._extract_code_block(content)
        if fenced:
            return fenced
        return None

    def _extract_json_payload(self, content: str) -> dict | None:
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        fenced = self._extract_code_block(content, include_json=True)
        if fenced:
            stripped = fenced.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            snippet = content[brace_start : brace_end + 1]
            try:
                payload = json.loads(snippet)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

        return None

    def _extract_code_block(self, content: str, include_json: bool = False) -> str | None:
        if "```" not in content:
            return None
        chunks = content.split("```")
        for i in range(1, len(chunks), 2):
            block = chunks[i].strip()
            if block.startswith("sql"):
                block = block[3:].strip()
            elif include_json and block.startswith("json"):
                block = block[4:].strip()
            if block:
                return block
        return None

    def _load_schema_context(self, datapoint_ids: list[str]) -> str:
        if not datapoint_ids:
            return "No schema datapoints available."
        context_parts: list[str] = []
        data_dir = Path("datapoints") / "managed"
        for datapoint_id in datapoint_ids:
            path = data_dir / f"{datapoint_id}.json"
            if not path.exists():
                continue
            try:
                with path.open() as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            if payload.get("type") != "Schema":
                continue

            table_name = payload.get("table_name", "unknown")
            schema_name = payload.get("schema") or payload.get("schema_name")
            full_name = (
                f"{schema_name}.{table_name}"
                if schema_name and "." not in table_name
                else table_name
            )
            context_parts.append(f"\n**Table: {full_name}**")
            if payload.get("business_purpose"):
                context_parts.append(f"Purpose: {payload['business_purpose']}")
            columns = payload.get("key_columns") or []
            if columns:
                context_parts.append("Columns:")
                for column in columns:
                    column_name = column.get("name", "unknown")
                    column_type = column.get("type", "unknown")
                    context_parts.append(f"- {column_name} ({column_type})")

        return "\n".join(context_parts) if context_parts else "No schema datapoints available."

    async def _generate_summary(
        self, original_query: str, sql: str, query_result: QueryResult
    ) -> tuple[str, list[str]]:
        """
        Generate natural language summary of query results.

        Args:
            original_query: User's original question
            sql: SQL query that was executed
            query_result: Query results

        Returns:
            Tuple of (natural language answer, key insights)
        """
        try:
            # Build summary prompt
            prompt = self._build_summary_prompt(original_query, sql, query_result)

            # Call LLM
            request = LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=self.prompts.load("system/main.md"),
                    ),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0.3,
            )
            response = await self.llm.generate(request)

            # Parse response (expecting "Answer: ... Insights: ..." format)
            answer, insights = self._parse_summary_response(response.content, query_result)

            return answer, insights

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback to basic summary
            return self._generate_basic_summary(query_result), []

    def _generate_deterministic_summary(
        self, original_query: str, sql: str, query_result: QueryResult
    ) -> tuple[str, list[str]] | None:
        if query_result.row_count == 0:
            requested_table = self._extract_information_schema_target_table(sql)
            table_name = self._extract_table_name_from_sql(sql)
            if "information_schema.columns" in sql.lower() and table_name:
                return (
                    f"No columns were found for table `{requested_table or table_name}`.",
                    [],
                )
            if table_name:
                return (f"No results found for `{table_name}`.", [])
            return ("No results found.", [])

        lower_sql = sql.lower()
        if "information_schema.columns" in lower_sql:
            table_name = (
                self._extract_information_schema_target_table(sql)
                or self._extract_table_name_from_sql(sql)
                or "the selected table"
            )
            column_values = []
            for row in query_result.rows:
                value = row.get("column_name")
                if value is not None:
                    column_values.append(str(value))
            unique_columns = list(dict.fromkeys(column_values))
            if not unique_columns:
                return (f"No columns were found for table `{table_name}`.", [])
            preview = ", ".join(unique_columns[:10])
            suffix = (
                f" (and {len(unique_columns) - 10} more)"
                if len(unique_columns) > 10
                else ""
            )
            return (
                f"The `{table_name}` table has {len(unique_columns)} column(s): {preview}{suffix}.",
                [],
            )

        if "information_schema.tables" in lower_sql:
            table_names = []
            for row in query_result.rows:
                table = row.get("table_name")
                schema = row.get("table_schema")
                if table is None:
                    continue
                table_names.append(f"{schema}.{table}" if schema else str(table))
            unique_tables = list(dict.fromkeys(table_names))
            if not unique_tables:
                return ("No tables were found.", [])
            preview = ", ".join(unique_tables[:10])
            suffix = (
                f" (and {len(unique_tables) - 10} more)"
                if len(unique_tables) > 10
                else ""
            )
            return (
                f"Found {len(unique_tables)} table(s): {preview}{suffix}.",
                [],
            )

        if query_result.row_count == 1 and query_result.columns == ["row_count"]:
            value = query_result.rows[0].get("row_count")
            table_name = self._extract_table_name_from_sql(sql)
            if table_name:
                return (f"Table `{table_name}` has {value} row(s).", [])
            return (f"The row count is {value}.", [])

        if re.match(r"^\s*select\s+\*\s+from\s+", lower_sql):
            table_name = self._extract_table_name_from_sql(sql)
            if table_name:
                return (
                    f"Returned {query_result.row_count} row(s) from `{table_name}`.",
                    [],
                )

        return None

    def _extract_table_name_from_sql(self, sql: str) -> str | None:
        match = re.search(
            r'\bfrom\s+(`[^`]+`|(?:"[^"]+"(?:\."[^"]+")*)|[a-zA-Z0-9_.-]+)',
            sql,
            re.IGNORECASE,
        )
        if not match:
            return None

        identifier = match.group(1).strip()
        if identifier.startswith("`") and identifier.endswith("`"):
            return identifier[1:-1]
        if '"' in identifier:
            parts = [part.strip('"') for part in identifier.split(".")]
            return ".".join(parts)
        return identifier

    def _extract_information_schema_target_table(self, sql: str) -> str | None:
        match = re.search(
            r"\btable_name\s*=\s*'([^']+)'",
            sql,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip()

    def _build_summary_prompt(self, query: str, sql: str, query_result: QueryResult) -> str:
        """Build prompt for result summarization."""
        # Format results for prompt (limit to prevent token overflow)
        rows_sample = query_result.rows[:10]  # Show max 10 rows
        results_str = "\n".join([f"Row {i + 1}: {row}" for i, row in enumerate(rows_sample)])

        if query_result.was_truncated:
            results_str += f"\n... (showing {len(rows_sample)} of {query_result.row_count} rows)"

        return self.prompts.render(
            "agents/executor_summary.md",
            user_query=query,
            sql_query=sql,
            results=results_str,
        )

    def _parse_summary_response(
        self, response: str, query_result: QueryResult
    ) -> tuple[str, list[str]]:
        """Parse LLM summary response."""
        lines = response.strip().split("\n")

        answer = ""
        insights = []

        for line in lines:
            if line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
            elif line.startswith("Insights:"):
                insights_text = line.replace("Insights:", "").strip()
                insights = [i.strip("- ").strip() for i in insights_text.split("\n") if i.strip()]
            elif line.strip().startswith("-") or line.strip().startswith("•"):
                insights.append(line.strip("- •").strip())

        # Fallback if parsing fails
        if not answer:
            answer = self._generate_basic_summary(query_result)

        return answer, insights

    def _generate_basic_summary(self, query_result: QueryResult) -> str:
        """Generate basic summary without LLM."""
        if query_result.row_count == 0:
            return "No results found."

        if query_result.row_count == 1:
            # Single row - describe it
            row = query_result.rows[0]
            if len(row) == 1:
                key, value = list(row.items())[0]
                return f"The {key} is {value}."
            return f"Found 1 result with {len(row)} columns."

        return f"Found {query_result.row_count} results."

    def _suggest_visualization(
        self, query_result: QueryResult, original_query: str | None = None
    ) -> str:
        """
        Suggest visualization type based on query results.

        Args:
            query_result: Query results
            original_query: Original user query (used for chart preference hints)

        Returns:
            Visualization hint
        """
        if query_result.row_count == 0:
            return "none"

        num_cols = len(query_result.columns)
        num_rows = query_result.row_count
        sample_rows = query_result.rows[: min(len(query_result.rows), 50)]
        numeric_col_count = sum(
            1
            for col in query_result.columns
            if any(self._is_numeric_value(row.get(col)) for row in sample_rows)
        )
        has_time_dimension = any(
            self._is_temporal_column(col, sample_rows) for col in query_result.columns
        )

        # Single value
        if num_rows == 1 and num_cols == 1:
            return "none"

        requested_hint = self._requested_visualization_hint(original_query)
        if requested_hint == "none":
            return "none"
        if requested_hint == "table":
            return "table"
        if requested_hint == "line_chart" and num_cols >= 2 and num_rows >= 2 and has_time_dimension:
            return "line_chart"
        if requested_hint == "bar_chart" and num_cols >= 2 and num_rows >= 2 and numeric_col_count >= 1:
            return "bar_chart"
        if (
            requested_hint == "pie_chart"
            and num_cols >= 2
            and num_rows >= 2
            and num_rows <= 12
            and numeric_col_count >= 1
        ):
            return "pie_chart"
        if requested_hint == "scatter" and num_cols >= 2 and num_rows >= 2 and numeric_col_count >= 2:
            return "scatter"
        if requested_hint == "line_chart" and num_cols >= 2 and num_rows >= 2 and numeric_col_count >= 1:
            return "bar_chart"

        # Time series detection (prioritize over other 2-column logic)
        if has_time_dimension and num_cols >= 2 and numeric_col_count >= 1:
            return "line_chart"

        # Two columns - likely category + value
        if num_cols == 2:
            if num_rows <= 10:
                return "bar_chart"
            elif num_rows <= 20:
                return "line_chart"
            else:
                return "table"

        # Multiple columns - scatter or table
        if num_cols >= 3:
            if num_rows <= 100 and numeric_col_count >= 2:
                return "scatter"
            return "table"

        # Default
        return "table"

    def _requested_visualization_hint(self, query: str | None) -> str | None:
        """Infer user visualization preference from query text."""
        text = (query or "").lower()
        if not text:
            return None

        if re.search(r"\b(no chart|without chart|table only|just table)\b", text):
            return "table"
        if re.search(r"\b(no visualization|text only)\b", text):
            return "none"
        if re.search(r"\b(bar chart|bar graph|histogram)\b", text):
            return "bar_chart"
        if re.search(r"\bpie chart|donut\b", text):
            return "pie_chart"
        if re.search(r"\bscatter|correlation\b", text):
            return "scatter"
        if re.search(r"\b(line chart|line graph|trend|time series|over time)\b", text):
            return "line_chart"
        return None

    @staticmethod
    def _is_numeric_value(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float, Decimal))

    def _is_temporal_column(self, column_name: str, rows: list[dict[str, Any]]) -> bool:
        lowered = column_name.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
        direct_markers = {"date", "time", "timestamp", "datetime"}
        period_markers = {"day", "week", "month", "quarter", "year"}
        period_disqualifiers = {"type", "category", "name", "code"}

        if any(marker in tokens for marker in direct_markers):
            return True
        if any(marker in tokens for marker in period_markers) and not any(
            disqualifier in tokens for disqualifier in period_disqualifiers
        ):
            return True
        if len(tokens) >= 2 and tokens[-1] == "at" and tokens[-2] in {
            "created",
            "updated",
            "deleted",
            "opened",
            "closed",
            "posted",
            "processed",
            "occurred",
            "recorded",
        }:
            return True
        for row in rows[:20]:
            if self._is_temporal_value(row.get(column_name)):
                return True
        return False

    @staticmethod
    def _is_temporal_value(value: Any) -> bool:
        if isinstance(value, (datetime, date)):
            return True
        if not isinstance(value, str):
            return False
        candidate = value.strip()
        if len(candidate) < 8:
            return False
        try:
            normalized = candidate.replace("Z", "+00:00")
            datetime.fromisoformat(normalized)
            return True
        except ValueError:
            return False
