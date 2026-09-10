"""Regression tests for the four-way When2Call classifier and its metrics.

The classifier is a heuristic over prose, which means it can be wrong in ways
that silently corrupt every downstream number. These tests pin the behaviour
that matters: that each of the four gold categories, written the way a model
would actually write it, lands in the right bucket.

The ``test_request_for_info_phrasing`` case is a real regression. An earlier
cue list matched "could you provide..." but not "what is the invoice identifier
you would like me to look up?", so a perfect oracle scored 93.75% instead of
100% and the missing-info metric read 75%. The oracle check in the pipeline
caught it; this test stops it coming back.
"""
from __future__ import annotations

import pytest

from src.decision_metrics import (
    DECISIONS,
    aggregate_decisions,
    classify_prediction,
    confusion_matrix,
    score_decision,
)
from src.schema import CALL_CLOSE, CALL_OPEN


def _call_block(name: str, arguments: dict) -> str:
    import json

    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{CALL_OPEN}\n{payload}\n{CALL_CLOSE}"


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def test_tool_call_detected():
    decision, source = classify_prediction(_call_block("get_weather", {"city": "Hanoi"}))
    assert decision == "tool_call"
    assert source == "tool_call"


def test_tool_call_wins_over_prose():
    """A clarifying sentence alongside a call is still a call."""
    text = "Sure, let me check that.\n" + _call_block("get_time", {"city": "Oslo"})
    assert classify_prediction(text)[0] == "tool_call"


@pytest.mark.parametrize(
    "text",
    [
        "To help you with that, could you please provide the order ID?",
        "Sure - what is the invoice identifier you would like me to look up?",
        "Happy to check. Could you tell me which order ID you mean?",
        "I can do that. Could you specify the invoice ID?",
        "Which city did you mean?",
        "Of course - what should I remind you about, and when?",
    ],
)
def test_request_for_info_phrasing(text):
    assert classify_prediction(text)[0] == "request_for_info"


@pytest.mark.parametrize(
    "text",
    [
        "Apologies, I'm unable to help with that - I don't have access to flight booking.",
        "I can't provide that; none of the available functions cover air quality data.",
        "Sorry, that is outside my capabilities - I have no parcel tracking function.",
        "I don't have access to recipe data, so I can't help with that request.",
        # The adverb slot. These are the dominant real phrasing in When2Call and
        # were the single largest source of oracle mismatches (110 of 176).
        "I'm sorry for the inconvenience, but I'm currently unable to provide that.",
        "I am currently unable to perform web searches.",
        "I'm now not able to access that data.",
        "I couldn't find a matching record.",
        "I don't have the capability to check application versions.",
        "抱歉，我无法提供实时天气信息。",
    ],
)
def test_cannot_answer_phrasing(text):
    assert classify_prediction(text)[0] == "cannot_answer"


@pytest.mark.parametrize(
    "text",
    [
        "To proceed, I need to know the status of the order.",
        "To proceed, I'll need to know which items you'd like.",
        "How many bathrooms are you looking for in the property?",
        "Which action should I take here?",
        "What do you need help with today?",
    ],
)
def test_request_for_info_procedural_phrasing(text):
    """The second-largest error family: procedural clarification frames."""
    assert classify_prediction(text)[0] == "request_for_info"


@pytest.mark.parametrize(
    "text",
    [
        "I can answer that directly without calling a function.",
        "17 times 24 is 408.",
        "TCP is connection-oriented; UDP is not.",
    ],
)
def test_direct_phrasing(text):
    assert classify_prediction(text)[0] == "direct"


def test_cannot_answer_not_shadowed_by_what_is_the():
    """'What is the air quality in Hanoi?' is a data request, not a clarification.

    The ``what is the <noun>`` cue must not fire on a bare noun phrase that is
    not in the argument-noun list.
    """
    decision, _ = classify_prediction(
        "I can't help with that - none of the available tools provide air quality data."
    )
    assert decision == "cannot_answer"


def test_empty_completion_is_cannot_answer():
    assert classify_prediction("")[0] == "cannot_answer"
    assert classify_prediction("   ")[0] == "cannot_answer"


# --------------------------------------------------------------------------- #
# Scoring and aggregation
# --------------------------------------------------------------------------- #


def _record(decision: str, calls=None):
    return {
        "id": "t",
        "decision": decision,
        "gold_calls": calls or [],
        "tools": [],
        "gold_response": "",
    }


def test_score_decision_correct_and_incorrect():
    ok = score_decision(_record("tool_call"), _call_block("f", {}))
    assert ok["correct"] is True

    bad = score_decision(_record("direct"), _call_block("f", {}))
    assert bad["correct"] is False
    assert bad["emitted_call"] is True


def test_confusion_matrix_shape_and_totals():
    scored = [
        score_decision(_record("tool_call"), _call_block("f", {})),
        score_decision(_record("direct"), "I can answer directly."),
        score_decision(_record("direct"), _call_block("f", {})),
    ]
    matrix = confusion_matrix(scored)
    assert set(matrix) == set(DECISIONS)
    assert sum(sum(row.values()) for row in matrix.values()) == 3
    assert matrix["tool_call"]["tool_call"] == 1
    assert matrix["direct"]["direct"] == 1
    assert matrix["direct"]["tool_call"] == 1


def test_false_tool_call_rate_counts_only_non_call_gold():
    scored = [
        score_decision(_record("tool_call"), _call_block("f", {})),   # correct call
        score_decision(_record("direct"), _call_block("f", {})),      # false call
        score_decision(_record("direct"), "I can answer directly."),  # correct abstain
    ]
    metrics = aggregate_decisions(scored)
    # One of two non-call rows produced a call.
    assert metrics["false_tool_call_rate"] == pytest.approx(0.5)
    assert metrics["tool_call_recall"] == pytest.approx(1.0)
    assert metrics["tool_call_precision"] == pytest.approx(0.5)


def test_missing_info_and_cannot_answer_accuracy():
    scored = [
        score_decision(_record("request_for_info"), "Could you please provide the ID?"),
        score_decision(_record("request_for_info"), "I can answer directly."),
        score_decision(_record("cannot_answer"), "I'm unable to help with that."),
    ]
    metrics = aggregate_decisions(scored)
    assert metrics["missing_info_accuracy"] == pytest.approx(0.5)
    assert metrics["cannot_answer_accuracy"] == pytest.approx(1.0)


def test_perfect_scorer_scores_perfectly():
    """The oracle property: gold answers must score 100%."""
    pairs = [
        (_record("tool_call", [{"name": "f", "arguments": {"a": 1}}]),
         _call_block("f", {"a": 1})),
        (_record("direct"), "I can answer that directly."),
        (_record("request_for_info"), "Could you please provide the invoice identifier?"),
        (_record("cannot_answer"), "I'm unable to help with that request."),
    ]
    scored = [score_decision(record, pred) for record, pred in pairs]
    metrics = aggregate_decisions(scored)
    assert metrics["decision_accuracy"] == pytest.approx(1.0)
    assert metrics["false_tool_call_rate"] == pytest.approx(0.0)


def test_aggregate_handles_empty_input():
    metrics = aggregate_decisions([])
    assert metrics["n"] == 0
    assert "confusion_matrix" in metrics