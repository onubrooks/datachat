"""Unit tests for scripts/phase1_kpi_gate.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = ROOT / "scripts" / "phase1_kpi_gate.py"
SPEC = importlib.util.spec_from_file_location("phase1_kpi_gate_module", GATE_PATH)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def _command_result(name: str, stdout: str, return_code: int = 0):
    return GATE.CommandResult(
        name=name,
        command="python scripts/eval_runner.py",
        return_code=return_code,
        stdout=stdout,
        stderr="",
    )


def test_release_gate_fails_when_intent_metrics_missing(monkeypatch):
    config = {
        "release_checks": {
            "eval_runs": [{"name": "intent-check", "mode": "intent", "dataset": "eval/intent.json"}],
            "intent_avg_latency_ms_max": 5000,
            "intent_avg_llm_calls_max": 3.0,
        }
    }

    monkeypatch.setattr(
        GATE,
        "_run_command",
        lambda name, command: _command_result(name, "Intent accuracy: 1.00\n"),
    )

    assert GATE.run_release_gate(config, api_base="http://localhost:8000") == 1


def test_release_gate_fails_when_threshold_set_but_no_intent_runs(monkeypatch):
    config = {
        "release_checks": {
            "eval_runs": [{"name": "catalog-check", "mode": "catalog", "dataset": "eval/catalog.json"}],
            "intent_avg_latency_ms_max": 5000,
            "intent_avg_llm_calls_max": 3.0,
        }
    }

    monkeypatch.setattr(
        GATE,
        "_run_command",
        lambda name, command: _command_result(name, "Catalog match: 1.00\n"),
    )

    assert GATE.run_release_gate(config, api_base="http://localhost:8000") == 1


def test_release_gate_passes_when_intent_metrics_present_and_within_threshold(monkeypatch):
    config = {
        "release_checks": {
            "eval_runs": [{"name": "intent-check", "mode": "intent", "dataset": "eval/intent.json"}],
            "intent_avg_latency_ms_max": 5000,
            "intent_avg_llm_calls_max": 3.0,
        }
    }

    stdout = "Avg latency: 1200.5ms\nAvg LLM calls: 1.8\n"
    monkeypatch.setattr(
        GATE,
        "_run_command",
        lambda name, command: _command_result(name, stdout),
    )

    assert GATE.run_release_gate(config, api_base="http://localhost:8000") == 0
