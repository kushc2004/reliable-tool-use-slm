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

    _write_json(
        tmp_path / "official_when2call_mcq" / "summary.json",
        {
            "runs": {
                "base": {
                    "n": 3652,
                    "accuracy": 0.4721,
                    "accuracy_norm": 0.5279,
                    "macro_f1": 0.3063,
                    "hallucination_rate": 0.2442,
                    "confusion_matrix": {
                        "true:tool_call": {"pred:direct": 1, "pred:tool_call": 8, "pred:request_for_info": 1, "pred:cannot_answer": 0},
                        "true:request_for_info": {"pred:direct": 0, "pred:tool_call": 4, "pred:request_for_info": 5, "pred:cannot_answer": 1},
                        "true:cannot_answer": {"pred:direct": 0, "pred:tool_call": 3, "pred:request_for_info": 2, "pred:cannot_answer": 5},
                    },
                },
                "tool_sft": {
                    "n": 3652,
                    "accuracy": 0.4381,
                    "accuracy_norm": 0.4981,
                    "macro_f1": 0.2626,
                    "hallucination_rate": 0.4070,
                    "confusion_matrix": {
                        "true:tool_call": {"pred:direct": 0, "pred:tool_call": 9, "pred:request_for_info": 1, "pred:cannot_answer": 0},
                        "true:request_for_info": {"pred:direct": 0, "pred:tool_call": 6, "pred:request_for_info": 4, "pred:cannot_answer": 0},
                        "true:cannot_answer": {"pred:direct": 0, "pred:tool_call": 5, "pred:request_for_info": 4, "pred:cannot_answer": 1},
                    },
                },
                "reliable_tool_sft": {
                    "n": 3652,
                    "accuracy": 0.6917,
                    "accuracy_norm": 0.7103,
                    "macro_f1": 0.5201,
                    "hallucination_rate": 0.0814,
                    "confusion_matrix": {
                        "true:tool_call": {"pred:direct": 0, "pred:tool_call": 8, "pred:request_for_info": 1, "pred:cannot_answer": 1},
                        "true:request_for_info": {"pred:direct": 0, "pred:tool_call": 2, "pred:request_for_info": 7, "pred:cannot_answer": 1},
                        "true:cannot_answer": {"pred:direct": 0, "pred:tool_call": 1, "pred:request_for_info": 2, "pred:cannot_answer": 7},
                    },
                },
            }
        },
    )

    data = collect(tmp_path)
    headline = data["headline"]

    assert headline["tool_to_reliable_false_call_drop_abs"] == pytest.approx(0.787)
    assert headline["tool_to_reliable_false_call_drop_rel"] == pytest.approx(0.787 / 0.899)
    assert headline["tool_to_reliable_decision_accuracy_gain_abs"] == pytest.approx(0.248)
    assert headline["official_tool_to_reliable_accuracy_norm_gain_abs"] == pytest.approx(0.2122)
    assert headline["official_tool_to_reliable_macro_f1_gain_abs"] == pytest.approx(0.2575)
    assert headline["official_tool_to_reliable_hallucination_drop_abs"] == pytest.approx(0.3256)
    assert headline["official_tool_to_reliable_hallucination_drop_rel"] == pytest.approx(0.3256 / 0.4070)

    text = render(data)
    assert "Tool-SFT false tool-call rate" in text
    assert "False tool-call reduction (Tool-SFT → Reliable)" in text
    assert "Cannot-answer accuracy" in text
    assert "Official When2Call MCQ" in text
    assert "Official relative hallucination reduction" in text
    assert "Base false tool-call rate" not in text
