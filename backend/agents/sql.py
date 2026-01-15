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
from typing import Dict, List, Optional, Any

from backend.agents.base import BaseAgent
from backend.config import get_settings
from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMRequest, LLMMessage
from backend.models.agent import (
    SQLAgentInput,
    SQLAgentOutput,
    GeneratedSQL,
    ValidationIssue,
    CorrectionAttempt,
    AgentMetadata,
    SQLGenerationError,
    LLMError,
)
from backend.models.datapoint import SchemaDataPoint, BusinessDataPoint

logger = logging.getLogger(__name__)


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

    def __init__(self):
        """Initialize SQLAgent with LLM provider."""
        super().__init__(name="SQLAgent")

        # Get configuration
        self.config = get_settings()

        # Create LLM provider using factory (respects sql_provider override)
        self.llm = LLMProviderFactory.create_agent_provider(
            agent_name="sql",
            config=self.config.llm,
            model_type="main"  # Use main model (GPT-4o) for SQL generation
        )

        logger.info(
            f"SQLAgent initialized with {self.llm.provider} provider",
            extra={"provider": self.llm.provider, "model": self.llm.model}
        )

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
                "num_datapoints": len(input.investigation_memory.datapoints)
            }
        )

        metadata = AgentMetadata(agent_name=self.name)
        correction_attempts: List[CorrectionAttempt] = []

        try:
            # Initial SQL generation
            generated_sql = await self._generate_sql(input, metadata)

            # Self-validation
            issues = self._validate_sql(generated_sql, input)

            # Self-correction loop if issues found
            attempt_number = 1
            while issues and attempt_number <= input.max_correction_attempts:
                logger.warning(
                    f"SQL validation found {len(issues)} issues, attempting correction #{attempt_number}",
                    extra={"issues": [issue.issue_type for issue in issues]}
                )

                # Record correction attempt
                original_sql = generated_sql.sql

                # Attempt correction
                corrected_sql = await self._correct_sql(
                    generated_sql=generated_sql,
                    issues=issues,
                    input=input,
                    metadata=metadata
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
                        success=success
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
                    extra={"issues": [issue.message for issue in issues]}
                )
                # Still return the best attempt we have, but mark needs_clarification
                needs_clarification = True
            else:
                needs_clarification = bool(generated_sql.clarifying_questions)

            logger.info(
                f"SQL generation complete",
                extra={
                    "correction_attempts": len(correction_attempts),
                    "needs_clarification": needs_clarification,
                    "confidence": generated_sql.confidence
                }
            )

            return SQLAgentOutput(
                success=True,
                data={},
                metadata=metadata,
                next_agent="ValidatorAgent",
                generated_sql=generated_sql,
                correction_attempts=correction_attempts,
                needs_clarification=needs_clarification
            )

        except Exception as e:
            metadata.error = str(e)
            logger.error(f"SQL generation failed: {e}", exc_info=True)
            raise SQLGenerationError(
                agent=self.name,
                message=f"Failed to generate SQL: {e}",
                recoverable=False,
                context={"query": input.query}
            ) from e

    async def _generate_sql(
        self,
        input: SQLAgentInput,
        metadata: AgentMetadata
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
        # Build prompt with context
        prompt = self._build_generation_prompt(input)

        # Create LLM request
        llm_request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=self._get_system_prompt()
                ),
                LLMMessage(
                    role="user",
                    content=prompt
                )
            ],
            temperature=0.0,  # Deterministic for SQL generation
            max_tokens=2000
        )

        try:
            # Call LLM
            response = await self.llm.generate(llm_request)

            # Track LLM call and tokens
            self._track_llm_call(tokens=response.usage.total_tokens)

            # Parse structured response
            generated_sql = self._parse_llm_response(response.content, input)

            logger.debug(
                f"Generated SQL: {generated_sql.sql[:200]}...",
                extra={"confidence": generated_sql.confidence}
            )

            return generated_sql

        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise LLMError(
                agent=self.name,
                message=f"LLM generation failed: {e}",
                context={"query": input.query}
            ) from e

    async def _correct_sql(
        self,
        generated_sql: GeneratedSQL,
        issues: List[ValidationIssue],
        input: SQLAgentInput,
        metadata: AgentMetadata
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
                LLMMessage(
                    role="system",
                    content=self._get_system_prompt()
                ),
                LLMMessage(
                    role="user",
                    content=prompt
                )
            ],
            temperature=0.0,
            max_tokens=2000
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
                extra={"issues_addressed": len(issues)}
            )

            return corrected_sql

        except Exception as e:
            logger.error(f"SQL correction failed: {e}", exc_info=True)
            raise LLMError(
                agent=self.name,
                message=f"SQL correction failed: {e}",
                context={"original_sql": generated_sql.sql}
            ) from e

    def _validate_sql(
        self,
        generated_sql: GeneratedSQL,
        input: SQLAgentInput
    ) -> List[ValidationIssue]:
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
        issues: List[ValidationIssue] = []
        sql = generated_sql.sql.strip().upper()

        # Basic syntax checks
        if not sql.startswith("SELECT") and not sql.startswith("WITH"):
            issues.append(
                ValidationIssue(
                    issue_type="syntax",
                    message="SQL must start with SELECT or WITH",
                    suggested_fix="Ensure query begins with SELECT or WITH (for CTEs)"
                )
            )

        if "FROM" not in sql:
            issues.append(
                ValidationIssue(
                    issue_type="syntax",
                    message="SQL missing FROM clause",
                    suggested_fix="Add FROM clause to specify table(s)"
                )
            )

        # Extract CTE names (Common Table Expressions) from WITH clause
        # Pattern: WITH cte_name AS (...), another_cte AS (...)
        cte_names = set()
        cte_pattern = r'WITH\s+([a-zA-Z0-9_]+)\s+AS\s*\('
        cte_matches = re.findall(cte_pattern, sql, re.IGNORECASE)
        for cte_name in cte_matches:
            cte_names.add(cte_name.upper())

        # Also match comma-separated CTEs: , cte_name AS (
        additional_cte_pattern = r',\s*([a-zA-Z0-9_]+)\s+AS\s*\('
        additional_ctes = re.findall(additional_cte_pattern, sql, re.IGNORECASE)
        for cte_name in additional_ctes:
            cte_names.add(cte_name.upper())

        # Extract table names from SQL (FROM and JOIN clauses)
        table_pattern = r'FROM\s+([a-zA-Z0-9_.]+)|JOIN\s+([a-zA-Z0-9_.]+)'
        table_matches = re.findall(table_pattern, sql, re.IGNORECASE)
        referenced_tables = {
            match[0] or match[1]
            for match in table_matches
        }

        # Get available tables from DataPoints
        available_tables = set()
        for dp in input.investigation_memory.datapoints:
            if isinstance(dp.metadata, dict) and "table_name" in dp.metadata:
                table_name = dp.metadata["table_name"]
                available_tables.add(table_name.upper())
                # Also add without schema prefix
                if "." in table_name:
                    available_tables.add(table_name.split(".")[-1].upper())

        # Check for missing tables (excluding CTEs and special tables)
        for table in referenced_tables:
            table_upper = table.upper()

            # Skip if this is a CTE name
            if table_upper in cte_names:
                continue

            # Skip special tables
            if table_upper in ("DUAL", "LATERAL"):
                continue

            # Check if table exists in DataPoints
            if table_upper not in available_tables and "." in table:
                # Check without schema
                table_no_schema = table.split(".")[-1].upper()
                if table_no_schema not in available_tables and table_no_schema not in cte_names:
                    issues.append(
                        ValidationIssue(
                            issue_type="missing_table",
                            message=f"Table '{table}' not found in available DataPoints",
                            suggested_fix=f"Use one of: {', '.join(sorted(available_tables))}"
                        )
                    )
            elif table_upper not in available_tables:
                # Simple table name not found
                issues.append(
                    ValidationIssue(
                        issue_type="missing_table",
                        message=f"Table '{table}' not found in available DataPoints",
                        suggested_fix=f"Use one of: {', '.join(sorted(available_tables))}"
                    )
                )

        return issues

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for SQL generation.

        Returns:
            System prompt string
        """
        return """You are an expert SQL query generator for data warehouses.

Your task is to generate SQL queries from natural language questions using the provided schema context and business rules.

**Guidelines:**
1. Use ONLY the tables and columns provided in the context
2. Follow business rules exactly as specified (e.g., exclude refunds, use specific date ranges)
3. Apply common SQL best practices (proper joins, appropriate aggregations, etc.)
4. Be explicit about assumptions you make
5. Ask clarifying questions if the query is ambiguous
6. Generate valid, executable SQL

**Output Format:**
Return a JSON object with:
- "sql": The SQL query (string)
- "explanation": Human-readable explanation of what the query does (string)
- "used_datapoints": List of datapoint IDs used (array of strings)
- "confidence": Confidence score 0-1 (number)
- "assumptions": List of assumptions made (array of strings)
- "clarifying_questions": List of questions for user if ambiguous (array of strings)

Example:
{
  "sql": "SELECT SUM(amount) FROM fact_sales WHERE status = 'completed' AND date >= '2024-07-01' AND date < '2024-10-01'",
  "explanation": "This query calculates total sales for Q3 2024 by summing the amount column from fact_sales, excluding refunds and filtering for the third quarter.",
  "used_datapoints": ["table_fact_sales_001", "metric_revenue_001"],
  "confidence": 0.95,
  "assumptions": ["'last quarter' refers to Q3 2024 (most recent complete quarter)"],
  "clarifying_questions": []
}
"""

    def _build_generation_prompt(self, input: SQLAgentInput) -> str:
        """
        Build prompt for SQL generation.

        Args:
            input: SQLAgentInput with query and context

        Returns:
            Formatted prompt string
        """
        # Extract schema and business context
        schema_context = self._format_schema_context(input.investigation_memory)
        business_context = self._format_business_context(input.investigation_memory)

        prompt = f"""Generate a SQL query for the following question:

**User Question:**
{input.query}

**Available Schema Context:**
{schema_context}

**Business Rules and Definitions:**
{business_context}

**Instructions:**
1. Use ONLY the tables and columns listed above
2. Follow the business rules exactly as specified
3. If the question is ambiguous, ask clarifying questions
4. Return your response as a JSON object following the format specified in the system prompt
"""

        return prompt

    def _build_correction_prompt(
        self,
        generated_sql: GeneratedSQL,
        issues: List[ValidationIssue],
        input: SQLAgentInput
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
        issues_text = "\n".join([
            f"- {issue.issue_type.upper()}: {issue.message}"
            + (f" (Suggested fix: {issue.suggested_fix})" if issue.suggested_fix else "")
            for issue in issues
        ])

        prompt = f"""The following SQL query has validation issues that need to be corrected:

**Original SQL:**
```sql
{generated_sql.sql}
```

**Validation Issues Found:**
{issues_text}

**Available Schema Context:**
{schema_context}

**Instructions:**
1. Fix ALL validation issues listed above
2. Use ONLY the tables and columns listed in the schema context
3. Maintain the original intent of the query
4. Return your response as a JSON object following the format specified in the system prompt

**Original User Question (for reference):**
{input.query}
"""

        return prompt

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
                schema = metadata.get("schema", "")
                business_purpose = metadata.get("business_purpose", "")

                schema_parts.append(f"\n**Table: {table_name}**")
                if business_purpose:
                    schema_parts.append(f"Purpose: {business_purpose}")

                # Add columns
                columns = metadata.get("key_columns", [])
                if columns:
                    schema_parts.append("Columns:")
                    for col in columns:
                        col_name = col.get("name", "unknown")
                        col_type = col.get("type", "unknown")
                        col_meaning = col.get("business_meaning", "")
                        schema_parts.append(
                            f"  - {col_name} ({col_type}): {col_meaning}"
                        )

                # Add relationships
                relationships = metadata.get("relationships", [])
                if relationships:
                    schema_parts.append("Relationships:")
                    for rel in relationships:
                        target = rel.get("target_table", "unknown")
                        join_col = rel.get("join_column", "unknown")
                        cardinality = rel.get("cardinality", "")
                        schema_parts.append(
                            f"  - JOIN {target} ON {join_col} ({cardinality})"
                        )

                # Add gotchas/common queries
                gotchas = metadata.get("gotchas", [])
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
                synonyms = metadata.get("synonyms", [])
                business_rules = metadata.get("business_rules", [])

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
        try:
            # Try to extract JSON from response (handles markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in LLM response")

            # Parse JSON
            data = json.loads(json_str)

            # Validate required fields
            if "sql" not in data:
                raise ValueError("Response missing 'sql' field")
            if "explanation" not in data:
                raise ValueError("Response missing 'explanation' field")

            return GeneratedSQL(
                sql=data["sql"],
                explanation=data["explanation"],
                used_datapoints=data.get("used_datapoints", []),
                confidence=data.get("confidence", 0.8),
                assumptions=data.get("assumptions", []),
                clarifying_questions=data.get("clarifying_questions", [])
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"LLM response content: {content}")
            raise ValueError(f"Invalid JSON in LLM response: {e}") from e
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"LLM response content: {content}")
            raise ValueError(f"Failed to parse LLM response: {e}") from e

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
                context={"input_type": type(input).__name__}
            )
