"""LLM-backed DataPoint generation from profiles."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from backend.llm.factory import LLMProviderFactory
from backend.llm.models import LLMMessage, LLMRequest
from backend.models.datapoint import BusinessDataPoint, ColumnMetadata, SchemaDataPoint
from backend.profiling.models import (
    DatabaseProfile,
    GeneratedDataPoint,
    GeneratedDataPoints,
    TableProfile,
)

_DEFAULT_OWNER = "auto-profiler@datachat.ai"


class DataPointGenerator:
    """Generate DataPoints from a schema profile using LLM assistance."""

    def __init__(self, llm_provider=None) -> None:
        if llm_provider is None:
            from backend.config import get_settings

            settings = get_settings()
            self._llm = LLMProviderFactory.create_default_provider(
                settings.llm, model_type="mini"
            )
        else:
            self._llm = llm_provider

    async def generate_from_profile(
        self, profile: DatabaseProfile
    ) -> GeneratedDataPoints:
        schema_points: list[GeneratedDataPoint] = []
        business_points: list[GeneratedDataPoint] = []

        for idx, table in enumerate(profile.tables, start=1):
            schema_points.append(await self._generate_schema_datapoint(table, idx))
            business_points.extend(await self._generate_business_datapoints(table, idx))

        return GeneratedDataPoints(
            profile_id=profile.profile_id,
            schema_datapoints=schema_points,
            business_datapoints=business_points,
        )

    async def _generate_schema_datapoint(
        self, table: TableProfile, index: int
    ) -> GeneratedDataPoint:
        system_prompt = (
            "You are a data analyst helping document a database table. "
            "Return ONLY valid JSON."
        )
        user_prompt = self._build_schema_prompt(table)
        response = await self._llm.generate(
            LLMRequest(messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ])
        )
        payload = self._parse_json_response(response.content)

        business_purpose = payload.get(
            "business_purpose",
            f"Auto-profiled table {table.schema_name}.{table.name} for analytics.",
        )
        column_meanings = payload.get("columns", {}) if isinstance(payload, dict) else {}

        key_columns: list[ColumnMetadata] = []
        for col in table.columns:
            meaning = column_meanings.get(col.name)
            if not meaning:
                meaning = self._fallback_column_meaning(col.name, col.sample_values)
            key_columns.append(
                ColumnMetadata(
                    name=col.name,
                    type=col.data_type,
                    business_meaning=meaning,
                    nullable=col.nullable,
                    default_value=col.default_value,
                )
            )

        common_queries = self._normalize_list(payload.get("common_queries"))
        gotchas = self._normalize_list(payload.get("gotchas"))
        freshness = payload.get("freshness") if isinstance(payload, dict) else None

        if self._has_time_series(table) and not any("DATE_TRUNC" in q for q in common_queries):
            common_queries.append(
                "SELECT DATE_TRUNC('day', <timestamp_column>), COUNT(*) FROM "
                f"{table.schema_name}.{table.name} GROUP BY 1 ORDER BY 1;"
            )

        row_count = table.row_count if table.row_count is not None and table.row_count >= 0 else None
        schema_datapoint = SchemaDataPoint(
            datapoint_id=self._make_datapoint_id("table", table.name, index),
            name=self._title_case(table.name),
            table_name=f"{table.schema_name}.{table.name}",
            schema=table.schema_name,
            business_purpose=self._ensure_min_length(business_purpose, 10),
            key_columns=key_columns,
            relationships=[
                self._relationship_to_model(rel) for rel in table.relationships
            ],
            common_queries=common_queries,
            gotchas=gotchas,
            freshness=freshness,
            row_count=row_count,
            owner=_DEFAULT_OWNER,
            tags=["auto-profiled"],
            metadata={"source": "auto-profiler"},
        )

        return GeneratedDataPoint(
            datapoint=schema_datapoint.model_dump(mode="json", by_alias=True),
            confidence=float(payload.get("confidence", 0.7))
            if isinstance(payload, dict)
            else 0.7,
            explanation=payload.get("explanation") if isinstance(payload, dict) else None,
        )

    async def _generate_business_datapoints(
        self, table: TableProfile, index: int
    ) -> list[GeneratedDataPoint]:
        numeric_columns = [
            col for col in table.columns if self._is_numeric_type(col.data_type)
        ]
        if not numeric_columns:
            return []

        system_prompt = (
            "You are a data analyst defining KPIs from numeric columns. "
            "Return ONLY valid JSON."
        )
        user_prompt = self._build_metric_prompt(table, numeric_columns)
        response = await self._llm.generate(
            LLMRequest(messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ])
        )
        payload = self._parse_json_response(response.content)
        metrics = payload.get("metrics", []) if isinstance(payload, dict) else []

        generated: list[GeneratedDataPoint] = []
        for metric_index, metric in enumerate(metrics, start=1):
            name = metric.get("name") or f"{table.name} metric"
            calculation = metric.get("calculation") or f"SUM({numeric_columns[0].name})"
            business_rules = self._normalize_list(metric.get("business_rules"))
            synonyms = self._normalize_list(metric.get("synonyms"))

            business_datapoint = BusinessDataPoint(
                datapoint_id=self._make_datapoint_id("metric", name, index + metric_index),
                name=name,
                calculation=calculation,
                synonyms=synonyms,
                business_rules=business_rules,
                related_tables=[f"{table.schema_name}.{table.name}"],
                unit=metric.get("unit"),
                aggregation=metric.get("aggregation"),
                owner=_DEFAULT_OWNER,
                tags=["auto-profiled"],
                metadata={"source": "auto-profiler"},
            )
            generated.append(
                GeneratedDataPoint(
                    datapoint=business_datapoint.model_dump(mode="json", by_alias=True),
                    confidence=self._normalize_confidence(metric.get("confidence", 0.6)),
                    explanation=metric.get("explanation"),
                )
            )

        return generated

    @staticmethod
    def _parse_json_response(content: str) -> dict:
        response_text = content.strip()
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            return {}
        json_str = response_text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _normalize_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _is_numeric_type(data_type: str) -> bool:
        return any(token in data_type.lower() for token in ["int", "numeric", "decimal", "float"])

    @staticmethod
    def _has_time_series(table: TableProfile) -> bool:
        for col in table.columns:
            if any(
                keyword in col.name.lower()
                for keyword in ["date", "time", "timestamp", "created", "updated"]
            ):
                return True
        return False

    @staticmethod
    def _normalize_confidence(value: object) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(float(value), 1.0))
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"high", "high confidence"}:
                return 0.8
            if lowered in {"medium", "medium confidence"}:
                return 0.6
            if lowered in {"low", "low confidence"}:
                return 0.4
            try:
                return max(0.0, min(float(lowered), 1.0))
            except ValueError:
                return 0.6
        return 0.6

    @staticmethod
    def _fallback_column_meaning(name: str, samples: Iterable[str]) -> str:
        sample_preview = ", ".join(list(samples)[:3])
        if sample_preview:
            return f"Values such as {sample_preview}."
        return f"Auto-profiled column {name}."

    @staticmethod
    def _ensure_min_length(value: str, length: int) -> str:
        if len(value) >= length:
            return value
        return value + " (auto-profiled)"

    @staticmethod
    def _make_datapoint_id(prefix: str, name: str, index: int) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return f"{prefix}_{slug}_{index:03d}"

    @staticmethod
    def _title_case(value: str) -> str:
        return value.replace("_", " ").title()

    @staticmethod
    def _relationship_to_model(relationship) -> dict:
        return {
            "target_table": f"{relationship.target_table}",
            "join_column": relationship.source_column,
            "cardinality": relationship.cardinality,
            "relationship_type": relationship.relationship_type,
        }

    def _build_schema_prompt(self, table: TableProfile) -> str:
        columns = [
            {
                "name": col.name,
                "type": col.data_type,
                "nullable": col.nullable,
                "samples": col.sample_values[:3],
            }
            for col in table.columns
        ]
        return (
            "Document this table for business users. Return JSON with keys: "
            "business_purpose (string), columns (object mapping column name to business meaning), "
            "common_queries (array), gotchas (array), freshness (string or null), "
            "confidence (0-1), explanation (string).\n\n"
            f"Table: {table.schema_name}.{table.name}\n"
            f"Row count (estimate): {table.row_count}\n"
            f"Columns: {json.dumps(columns)}"
        )

    def _build_metric_prompt(self, table: TableProfile, numeric_columns: list) -> str:
        numeric_cols = [
            {
                "name": col.name,
                "type": col.data_type,
                "samples": col.sample_values[:3],
            }
            for col in numeric_columns
        ]
        return (
            "Suggest KPIs from numeric columns. Return JSON with key 'metrics', "
            "an array of objects with fields: name, calculation, aggregation, unit, "
            "synonyms, business_rules, confidence, explanation.\n\n"
            f"Table: {table.schema_name}.{table.name}\n"
            f"Numeric columns: {json.dumps(numeric_cols)}"
        )
