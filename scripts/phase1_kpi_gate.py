#!/usr/bin/env python3
"""
Phase 1 KPI gate runner.

Usage:
  python scripts/phase1_kpi_gate.py --mode ci
  python scripts/phase1_kpi_gate.py --mode release --api-base http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "scripts" / "eval_runner.py"
DEFAULT_CONFIG = ROOT / "config" / "phase1_kpi.json"


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: str
    return_code: int
    stdout: str
    stderr: str


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_command(name: str, command: str) -> CommandResult:
    completed = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandResult(
        name=name,
        command=command,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _print_command_result(result: CommandResult) -> None:
    status = "PASS" if result.return_code == 0 else "FAIL"
    print(f"[{status}] {result.name}")
    print(f"  $ {result.command}")
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip())


def run_ci_gate(config: dict[str, Any]) -> int:
    checks: list[dict[str, Any]] = config.get("ci_checks", [])
    if not checks:
        print("No ci_checks configured.")
        return 1

    failures = 0
    for item in checks:
        result = _run_command(item["name"], item["command"])
        _print_command_result(result)
        if result.return_code != 0:
            failures += 1

    if failures:
        print(f"\nCI KPI gate failed: {failures}/{len(checks)} checks failed.")
        return 1

    print(f"\nCI KPI gate passed: {len(checks)} checks passed.")
    return 0


def _build_eval_command(
    *,
    api_base: str,
    mode: str,
    dataset: str,
    thresholds: dict[str, Any] | None = None,
) -> str:
    command_parts = [
        sys.executable,
        str(EVAL_RUNNER),
        "--mode",
        mode,
        "--dataset",
        dataset,
        "--api-base",
        api_base,
    ]
    threshold_mapping = {
        "min_sql_match_rate": "--min-sql-match-rate",
        "min_source_match_rate": "--min-source-match-rate",
        "min_clarification_match_rate": "--min-clarification-match-rate",
        "min_hit_rate": "--min-hit-rate",
        "min_recall": "--min-recall",
        "min_mrr": "--min-mrr",
        "min_answer_type_rate": "--min-answer-type-rate",
    }
    for key, value in (thresholds or {}).items():
        flag = threshold_mapping.get(key)
        if flag is None:
            continue
        command_parts.extend([flag, str(value)])
    return " ".join(shlex.quote(part) for part in command_parts)


def _parse_metric(output: str, pattern: str) -> float | None:
    match = re.search(pattern, output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def run_release_gate(config: dict[str, Any], api_base: str) -> int:
    release_config = config.get("release_checks", {})
    eval_runs: list[dict[str, Any]] = release_config.get("eval_runs", [])
    if not eval_runs:
        print("No release eval_runs configured.")
        return 1

    failures = 0
    intent_latency_values: list[float] = []
    intent_llm_values: list[float] = []

    for run in eval_runs:
        command = _build_eval_command(
            api_base=api_base,
            mode=run["mode"],
            dataset=run["dataset"],
            thresholds=run.get("thresholds"),
        )
        result = _run_command(run["name"], command)
        _print_command_result(result)
        if result.return_code != 0:
            failures += 1
            continue

        if run["mode"] == "intent":
            latency = _parse_metric(result.stdout, r"Avg latency:\s*([0-9.]+)ms")
            llm_calls = _parse_metric(result.stdout, r"Avg LLM calls:\s*([0-9.]+)")
            if latency is not None:
                intent_latency_values.append(latency)
            if llm_calls is not None:
                intent_llm_values.append(llm_calls)

    max_latency = release_config.get("intent_avg_latency_ms_max")
    if max_latency is not None and intent_latency_values:
        measured = max(intent_latency_values)
        if measured > float(max_latency):
            failures += 1
            print(
                "Intent latency threshold failed: "
                f"{measured:.1f}ms > {float(max_latency):.1f}ms"
            )

    max_llm_calls = release_config.get("intent_avg_llm_calls_max")
    if max_llm_calls is not None and intent_llm_values:
        measured = max(intent_llm_values)
        if measured > float(max_llm_calls):
            failures += 1
            print(
                "Intent LLM-call threshold failed: "
                f"{measured:.2f} > {float(max_llm_calls):.2f}"
            )

    if failures:
        print(f"\nRelease KPI gate failed: {failures} checks failed.")
        return 1

    print("\nRelease KPI gate passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 KPI gate runner")
    parser.add_argument("--mode", choices=["ci", "release"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--api-base", default="http://localhost:8000")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"KPI config not found: {config_path}")
        return 1

    config = _load_config(config_path)
    if args.mode == "ci":
        return run_ci_gate(config)
    return run_release_gate(config, api_base=args.api_base)


if __name__ == "__main__":
    sys.exit(main())
