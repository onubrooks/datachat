"""Workflow-style response packaging helpers (finance-focused v1)."""

from __future__ import annotations

import math
import re
from typing import Any

from backend.models.api import (
    DataSource,
    WorkflowArtifacts,
    WorkflowDriver,
    WorkflowMetric,
    WorkflowSource,
)

FINANCE_SIGNAL_KEYWORDS = (
    "revenue",
    "deposit",
    "withdrawal",
    "net flow",
    "liquidity",
    "loan",
    "default",
    "delinquency",
    "interest",
    "fee",
    "balance",
    "bank_",
    "fx",
    "treasury",
    "risk",
)


def build_workflow_artifacts(
    *,
    query: str,
    answer: str,
    answer_source: str,
    data: dict[str, list] | None,
    sources: list[DataSource],
    validation_warnings: list[Any],
    clarifying_questions: list[str],
    has_datapoints: bool,
    workflow_mode: str | None = "auto",
) -> WorkflowArtifacts | None:
    """Build optional finance workflow package for decision-ready responses."""
    if answer_source in {"error", "clarification", "approval", "system"}:
        return None
    force_finance = (workflow_mode or "auto") == "finance_variance_v1"
    if not force_finance and not _looks_finance_like(query=query, answer=answer, sources=sources):
        return None

    metrics = _extract_workflow_metrics(data)
    drivers = _extract_workflow_drivers(data)
    caveats = _build_workflow_caveats(
        answer_source=answer_source,
        sources=sources,
        validation_warnings=validation_warnings,
        clarifying_questions=clarifying_questions,
        has_datapoints=has_datapoints,
    )
    follow_ups = _build_workflow_follow_ups(query=query, data=data, drivers=drivers)
    workflow_sources = [
        WorkflowSource(
            datapoint_id=source.datapoint_id,
            name=source.name,
            source_type=source.type,
        )
        for source in sources[:5]
    ]

    return WorkflowArtifacts(
        package_version="1.0",
        domain="finance",
        summary=_summarize_answer(answer),
        metrics=metrics,
        drivers=drivers,
        caveats=caveats,
        sources=workflow_sources,
        follow_ups=follow_ups,
    )


def _looks_finance_like(*, query: str, answer: str, sources: list[DataSource]) -> bool:
    combined = f"{query} {answer}".lower()
    if any(keyword in combined for keyword in FINANCE_SIGNAL_KEYWORDS):
        return True
    for source in sources:
        source_text = f"{source.datapoint_id} {source.name} {source.type}".lower()
        if "bank" in source_text or "loan" in source_text or "deposit" in source_text:
            return True
    return False


def _summarize_answer(answer: str) -> str:
    cleaned = re.sub(r"\s+", " ", answer or "").strip()
    if not cleaned:
        return "No summary available."
    first_sentence = re.split(r"(?<=[.!?])\s", cleaned, maxsplit=1)[0]
    return first_sentence[:280]


def _extract_workflow_metrics(data: dict[str, list] | None) -> list[WorkflowMetric]:
    if not data:
        return []
    columns = list(data.keys())
    if not columns:
        return []
    row_count = max((len(data[column]) for column in columns), default=0)
    if row_count == 0:
        return []

    metrics: list[WorkflowMetric] = []
    if row_count == 1:
        for column in columns:
            value = data[column][0] if data[column] else None
            if value is None:
                continue
            metrics.append(WorkflowMetric(label=_pretty_label(column), value=_format_value(value)))
            if len(metrics) >= 5:
                break
        return metrics

    for column in columns:
        series = data.get(column, [])
        numeric_values = [
            float(value)
            for value in series
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not numeric_values:
            continue
        avg_value = sum(numeric_values) / len(numeric_values)
        metrics.append(
            WorkflowMetric(label=f"Average {_pretty_label(column)}", value=_format_value(avg_value))
        )
        if len(metrics) >= 4:
            break
    return metrics


def _extract_workflow_drivers(data: dict[str, list] | None) -> list[WorkflowDriver]:
    if not data:
        return []
    columns = list(data.keys())
    if len(columns) < 2:
        return []

    row_count = max((len(data[column]) for column in columns), default=0)
    if row_count <= 1:
        return []

    dimension_col = next(
        (
            column
            for column in columns
            if any(
                isinstance(value, str) and value.strip()
                for value in data.get(column, [])[: min(row_count, 30)]
            )
        ),
        None,
    )
    measure_col = next(
        (
            column
            for column in columns
            if any(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in data.get(column, [])[: min(row_count, 30)]
            )
        ),
        None,
    )
    if not dimension_col or not measure_col:
        return []

    ranked_rows: list[tuple[float, str]] = []
    for index in range(row_count):
        dimension_value = data.get(dimension_col, [None] * row_count)[index]
        measure_value = data.get(measure_col, [None] * row_count)[index]
        if not isinstance(dimension_value, str) or not dimension_value.strip():
            continue
        if not isinstance(measure_value, (int, float)) or isinstance(measure_value, bool):
            continue
        numeric_measure = float(measure_value)
        if math.isfinite(numeric_measure):
            ranked_rows.append((numeric_measure, dimension_value.strip()))
    ranked_rows.sort(reverse=True, key=lambda item: item[0])

    drivers: list[WorkflowDriver] = []
    for measure_value, dimension_value in ranked_rows[:3]:
        drivers.append(
            WorkflowDriver(
                dimension=_pretty_label(dimension_col),
                value=dimension_value,
                contribution=f"{_pretty_label(measure_col)}: {_format_value(measure_value)}",
            )
        )
    return drivers


def _build_workflow_caveats(
    *,
    answer_source: str,
    sources: list[DataSource],
    validation_warnings: list[Any],
    clarifying_questions: list[str],
    has_datapoints: bool,
) -> list[str]:
    caveats: list[str] = []
    if answer_source == "context":
        caveats.append("Answer derived from context and metadata; verify with SQL for final reporting.")
    if not sources:
        caveats.append("No explicit DataPoint sources were attached to this answer.")
    if not has_datapoints:
        caveats.append("Running in live schema mode without DataPoints may reduce business-definition precision.")
    if clarifying_questions:
        caveats.append("Further clarification could improve answer precision.")
    for warning in validation_warnings[:2]:
        if isinstance(warning, dict):
            message = str(warning.get("message", "")).strip()
            if message:
                caveats.append(message)
    if not caveats:
        caveats.append("Review source assumptions before sharing externally.")
    return caveats[:4]


def _build_workflow_follow_ups(
    *,
    query: str,
    data: dict[str, list] | None,
    drivers: list[WorkflowDriver],
) -> list[str]:
    follow_ups: list[str] = []
    lower_query = query.lower()
    if "last" in lower_query or "trend" in lower_query or "week" in lower_query:
        follow_ups.append("Compare this result against the previous equivalent period.")
    else:
        follow_ups.append("Show the same metric as a weekly trend for the last 8 weeks.")

    if drivers:
        primary_dimension = drivers[0].dimension
        follow_ups.append(f"Break down this result further by {primary_dimension}.")
    elif data and len(data.keys()) > 1:
        first_column = _pretty_label(list(data.keys())[0])
        follow_ups.append(f"Show the top contributors by {first_column}.")
    else:
        follow_ups.append("Show top drivers of change by segment and country.")

    follow_ups.append("List caveats and data quality checks for this answer.")
    return follow_ups[:3]


def _pretty_label(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1_000:
            return f"{value:,.2f}"
        return f"{value:.4g}"
    if value is None:
        return "-"
    return str(value)
