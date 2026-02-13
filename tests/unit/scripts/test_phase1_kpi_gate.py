"""Unit tests for scripts/phase1_kpi_gate.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "phase1_kpi_gate.py"
SPEC = importlib.util.spec_from_file_location("phase1_kpi_gate_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
KPI_GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = KPI_GATE
SPEC.loader.exec_module(KPI_GATE)


def test_parse_metric_returns_none_on_missing():
    assert KPI_GATE._parse_metric("hello", r"Avg latency:\s*([0-9.]+)ms") is None


def test_parse_metric_extracts_value():
    value = KPI_GATE._parse_metric("Avg latency: 1234.5ms", r"Avg latency:\s*([0-9.]+)ms")
    assert value == 1234.5


def test_build_eval_command_includes_thresholds():
    command = KPI_GATE._build_eval_command(
        api_base="http://localhost:8000",
        mode="catalog",
        dataset="eval/catalog/mysql_credentials.json",
        thresholds={
            "min_sql_match_rate": 0.7,
            "min_source_match_rate": 0.8,
            "min_clarification_match_rate": 0.8,
        },
    )
    assert "--mode catalog" in command
    assert "--dataset eval/catalog/mysql_credentials.json" in command
    assert "--min-sql-match-rate 0.7" in command
    assert "--min-source-match-rate 0.8" in command
    assert "--min-clarification-match-rate 0.8" in command


def test_run_release_gate_enforces_latency_threshold(monkeypatch):
    config = {
        "release_checks": {
            "intent_avg_latency_ms_max": 100.0,
            "eval_runs": [
                {
                    "name": "intent",
                    "mode": "intent",
                    "dataset": "eval/intent_credentials.json",
                }
            ],
        }
    }

    def _fake_run(name: str, command: str):
        return KPI_GATE.CommandResult(
            name=name,
            command=command,
            return_code=0,
            stdout="Avg latency: 250.0ms\nAvg LLM calls: 1.0",
            stderr="",
        )

    monkeypatch.setattr(KPI_GATE, "_run_command", _fake_run)
    rc = KPI_GATE.run_release_gate(config, api_base="http://localhost:8000")
    assert rc == 1


def test_run_ci_gate_passes_when_all_commands_pass(monkeypatch):
    config = {
        "ci_checks": [
            {"name": "check-1", "command": "echo ok"},
            {"name": "check-2", "command": "echo ok"},
        ]
    }

    def _fake_run(name: str, command: str):
        return KPI_GATE.CommandResult(
            name=name,
            command=command,
            return_code=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(KPI_GATE, "_run_command", _fake_run)
    rc = KPI_GATE.run_ci_gate(config)
    assert rc == 0
