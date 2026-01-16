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
import logging
import time

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

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    """
    Query execution and result formatting agent.

    Executes SQL queries, generates natural language summaries,
    and suggests appropriate visualizations.
    """

    def __init__(self):
        """Initialize ExecutorAgent with LLM provider."""
        super().__init__(name="ExecutorAgent")

        # Get configuration
        self.config = get_settings()

        # Get LLM provider (use mini model for summarization/cost)
        self.llm = LLMProviderFactory.create_default_provider(
            self.config.llm,
            model_type="mini"
        )

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
        connector = await self._get_connector(input.database_type)

        try:
            # Execute query with timeout
            query_result = await self._execute_query(
                connector,
                input.validated_sql.sql,
                input.max_rows,
                input.timeout_seconds,
            )

            # Generate natural language summary
            nl_answer, insights = await self._generate_summary(
                input.query, input.validated_sql.sql, query_result
            )

            # Suggest visualization
            viz_hint = self._suggest_visualization(query_result)

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
            metadata.llm_calls = 1

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

    async def _get_connector(self, database_type: str) -> BaseConnector:
        """
        Get database connector for specified type.

        Args:
            database_type: Type of database (postgresql, clickhouse, mysql)

        Returns:
            Database connector instance
        """
        if database_type == "postgresql":
            connector = PostgresConnector(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database_name,
                user=self.config.database.username,
                password=self.config.database.password,
            )
        elif database_type == "clickhouse":
            connector = ClickHouseConnector(
                host=self.config.database.host,
                port=self.config.database.port or 8123,
                database=self.config.database.database_name,
                user=self.config.database.username,
                password=self.config.database.password,
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
                        content="You are a data assistant that summarizes query results.",
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

    def _build_summary_prompt(self, query: str, sql: str, query_result: QueryResult) -> str:
        """Build prompt for result summarization."""
        # Format results for prompt (limit to prevent token overflow)
        rows_sample = query_result.rows[:10]  # Show max 10 rows
        results_str = "\n".join([f"Row {i + 1}: {row}" for i, row in enumerate(rows_sample)])

        if query_result.was_truncated:
            results_str += f"\n... (showing {len(rows_sample)} of {query_result.row_count} rows)"

        return f"""You are a data assistant. Summarize these query results in natural language.

**User Question:** {query}

**SQL Query:** {sql}

**Results:**
{results_str}

**Response Format:**
Answer: [1-2 sentence natural language answer]
Insights: [Bullet points of key insights, if any]

Be concise and focus on answering the user's question."""

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

    def _suggest_visualization(self, query_result: QueryResult) -> str:
        """
        Suggest visualization type based on query results.

        Args:
            query_result: Query results

        Returns:
            Visualization hint
        """
        if query_result.row_count == 0:
            return "none"

        num_cols = len(query_result.columns)
        num_rows = query_result.row_count

        # Single value
        if num_rows == 1 and num_cols == 1:
            return "none"

        # Time series detection (prioritize over other 2-column logic)
        has_date = any(
            "date" in col.lower() or "time" in col.lower() for col in query_result.columns
        )
        if has_date and num_cols == 2:
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
            if num_rows <= 100:
                return "scatter"
            return "table"

        # Default
        return "table"
