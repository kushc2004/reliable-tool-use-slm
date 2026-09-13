"""Regression tests for the generated CV summary.

The reliability headline must compare positive-only Tool-SFT against Reliable
Tool-SFT. Base can have a 0% false-call rate simply by never calling a tool, so
using Base as the before-value would invert the interpretation of the result.
"""
from __future__ import annotations

import json

import pytest

from src.cv_metrics import collect, render


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_false_call_headline_uses_tool_sft_as_baseline(tmp_path):
    tool_metrics = {
        "overall": {
            "n": 3,
            "exact_match": 0.0,
            "tool_selection_accuracy": 0.0,
            "argument_accuracy": 0.0,
            "json_validity": 1.0,
            "unseen_function_accuracy": 0.0,
        }
    }
    decisions = {
        "base": {
            "n": 3,
            "decision_accuracy": 0.23,
            "macro_f1": 0.16,
            "tool_call_precision": None,
            "tool_call_recall": 0.0,
            "false_tool_call_rate": 0.0,
            "missing_info_accuracy": 0.66,
            "cannot_answer_accuracy": 0.11,
        },
        "tool_sft": {
            "n": 3,
            "decision_accuracy": 0.373,
            "macro_f1": 0.179,
            "tool_call_precision": 0.372,
            "tool_call_recall": 0.972,
            "false_tool_call_rate": 0.899,
            "missing_info_accuracy": 0.10,
            "cannot_answer_accuracy": 0.0,
        },
        "reliable_tool_sft": {
            "n": 3,
            "decision_accuracy": 0.621,
            "macro_f1": 0.474,
            "tool_call_precision": 0.702,
            "tool_call_recall": 0.482,
            "false_tool_call_rate": 0.112,
            "missing_info_accuracy": 0.734,
            "cannot_answer_accuracy": 0.667,
        },
    }

    for run in ("base", "tool_sft", "reliable_tool_sft"):
        _write_json(tmp_path / f"{run}_metrics.json", tool_metrics)
        _write_json(tmp_path / f"{run}_when2call.json", decisions[run])

    data = collect(tmp_path)
    headline = data["headline"]

    assert headline["tool_to_reliable_false_call_drop_abs"] == pytest.approx(0.787)
    assert headline["tool_to_reliable_false_call_drop_rel"] == pytest.approx(0.787 / 0.899)
    assert headline["tool_to_reliable_decision_accuracy_gain_abs"] == pytest.approx(0.248)

    text = render(data)
    assert "Tool-SFT false tool-call rate" in text
    assert "False tool-call reduction (Tool-SFT → Reliable)" in text
    assert "Cannot-answer accuracy" in text
    assert "Base false tool-call rate" not in text
