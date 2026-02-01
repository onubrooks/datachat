#!/usr/bin/env python3
"""
Minimal evaluation runner for DataChat RAG checks.

Usage:
  python scripts/eval_runner.py --mode retrieval --dataset eval/retrieval.json
  python scripts/eval_runner.py --mode qa --dataset eval/qa.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


def _post_chat(api_base: str, message: str) -> dict[str, Any]:
    response = httpx.post(
        f"{api_base}/api/v1/chat",
        json={"message": message, "conversation_id": "eval_run"},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def _infer_answer_type(data: Any) -> str | None:
    if not data or not isinstance(data, list):
        return None
    if len(data) == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
        return "single_value"
    if len(data) > 1 and isinstance(data[0], dict):
        cols = [str(col).lower() for col in data[0].keys()]
        if any(token in col for col in cols for token in ("date", "time", "day")):
            return "time_series"
        return "table"
    return None


def run_retrieval(api_base: str, dataset: list[dict[str, Any]]) -> int:
    total = len(dataset)
    hits = 0
    recall_sum = 0.0

    for item in dataset:
        query = item["query"]
        expected = set(item["expected_datapoint_ids"])
        response = _post_chat(api_base, query)
        sources = response.get("sources") or []
        retrieved = {src.get("datapoint_id") for src in sources if src.get("datapoint_id")}

        hit = 1 if expected & retrieved else 0
        hits += hit
        recall = len(expected & retrieved) / len(expected) if expected else 0.0
        recall_sum += recall

        print(f"- {query}")
        print(f"  expected: {sorted(expected)}")
        print(f"  retrieved: {sorted(retrieved)}")
        print(f"  hit: {hit}  recall: {recall:.2f}")

    hit_rate = hits / total if total else 0.0
    avg_recall = recall_sum / total if total else 0.0
    print(f"\nHit rate: {hit_rate:.2f}  Avg recall: {avg_recall:.2f}")
    return 0


def run_qa(api_base: str, dataset: list[dict[str, Any]]) -> int:
    total = len(dataset)
    sql_matches = 0
    answer_matches = 0
    validation_errors = 0

    for item in dataset:
        query = item["query"]
        expected_sql_contains = [token.lower() for token in item["expected_sql_contains"]]
        expected_answer_type = item["expected_answer_type"]

        response = _post_chat(api_base, query)
        sql = (response.get("sql") or "").lower()
        data = response.get("data")
        answer_type = _infer_answer_type(data)
        errors = response.get("validation_errors") or []

        sql_ok = all(token in sql for token in expected_sql_contains)
        answer_ok = answer_type == expected_answer_type
        if sql_ok:
            sql_matches += 1
        if answer_ok:
            answer_matches += 1
        if errors:
            validation_errors += 1

        print(f"- {query}")
        print(f"  sql_ok: {sql_ok}  answer_type: {answer_type}  expected: {expected_answer_type}")
        if errors:
            print(f"  validation_errors: {len(errors)}")

    print(f"\nSQL match rate: {sql_matches}/{total}")
    print(f"Answer type match: {answer_matches}/{total}")
    print(f"Validation errors: {validation_errors}/{total}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DataChat eval runner")
    parser.add_argument("--mode", choices=["retrieval", "qa"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        with open(args.dataset, encoding="utf-8") as handle:
            dataset = json.load(handle)
    except OSError as exc:
        print(f"Failed to load dataset: {exc}")
        return 1

    if args.mode == "retrieval":
        return run_retrieval(args.api_base, dataset)
    if args.mode == "qa":
        return run_qa(args.api_base, dataset)
    return 1


if __name__ == "__main__":
    sys.exit(main())
