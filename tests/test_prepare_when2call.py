"""Tests for the When2Call loader.

The bug these pin: When2Call writes gold calls in **two** different shapes, and
the loader originally handled only one of them.

    train_pref   <TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>
    mcq          {"name": ..., "arguments": {...}}          (no tag)

``mcq`` is the evaluation split, so requiring the tag meant every eval
``tool_call`` row parsed to zero calls. The scorer reported 0% tool-call recall
and the oracle self-check (which echoes the gold answer and must score 100%)
caught it at 59.7%.
"""
from __future__ import annotations

import json

import pytest

from src.data.prepare_when2call import (
    DECISIONS,
    _coerce_payload,
    _extract_embedded_json,
    _extract_gold_calls,
    classify_gold_response,
    convert_eval_row,
)


# --------------------------------------------------------------------------- #
# Gold-call extraction: both shapes
# --------------------------------------------------------------------------- #


def test_untagged_object_is_extracted():
    """The mcq shape. This is the regression."""
    text = '{"name": "uber.ride", "arguments": {"loc": "Berkeley", "type": "plus"}}'
    calls = _extract_gold_calls(text)
    assert calls == [{"name": "uber.ride", "arguments": {"loc": "Berkeley", "type": "plus"}}]


def test_tagged_array_is_extracted():
    """The train_pref shape."""
    text = '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Hanoi"}}]</TOOLCALL>'
    assert _extract_gold_calls(text) == [
        {"name": "get_weather", "arguments": {"city": "Hanoi"}}
    ]


def test_tagged_multiple_calls_all_extracted():
    text = (
        '<TOOLCALL>[{"name": "a", "arguments": {}}, '
        '{"name": "b", "arguments": {"x": 1}}]</TOOLCALL>'
    )
    assert [c["name"] for c in _extract_gold_calls(text)] == ["a", "b"]


def test_arguments_given_as_json_string_are_decoded():
    text = '{"name": "f", "arguments": "{\\"city\\": \\"Hanoi\\"}"}'
    assert _extract_gold_calls(text) == [{"name": "f", "arguments": {"city": "Hanoi"}}]


def test_nested_function_wrapper_is_tolerated():
    text = '{"function": {"name": "f", "arguments": {"a": 1}}}'
    assert _extract_gold_calls(text) == [{"name": "f", "arguments": {"a": 1}}]


def test_parameters_key_accepted_as_alias_for_arguments():
    text = '{"name": "f", "parameters": {"a": 1}}'
    assert _extract_gold_calls(text) == [{"name": "f", "arguments": {"a": 1}}]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I can answer that directly.",
        "Apologies, but I'm unable to help with that.",
        "Could you please provide the order ID?",
        "{}",
        '{"arguments": {"a": 1}}',          # no name
        '{"name": ""}',                      # empty name
        "not json at all",
    ],
)
def test_non_calls_extract_nothing(text):
    assert _extract_gold_calls(text) == []


def test_call_embedded_in_prose_is_found():
    text = 'Let me check. {"name": "f", "arguments": {"a": 1}} One moment.'
    assert _extract_gold_calls(text) == [{"name": "f", "arguments": {"a": 1}}]


def test_nested_object_not_truncated_by_brace_scanning():
    """A regex like ``\\{.*\\}`` would cut at the first inner closing brace."""
    text = '{"name": "f", "arguments": {"nested": {"deep": [1, 2]}, "b": 2}}'
    assert _extract_gold_calls(text) == [
        {"name": "f", "arguments": {"nested": {"deep": [1, 2]}, "b": 2}}
    ]


def test_embedded_json_returns_none_without_braces():
    assert _extract_embedded_json("no json here") is None


def test_coerce_payload_handles_non_list():
    assert _coerce_payload("a string") == []
    assert _coerce_payload(None) == []
    assert _coerce_payload([1, 2, "x"]) == []


# --------------------------------------------------------------------------- #
# Classification uses the same extractor
# --------------------------------------------------------------------------- #


def test_untagged_object_classifies_as_tool_call():
    """Previously fell through to ``direct`` because only the tag was checked."""
    text = '{"name": "api.weather", "arguments": {"loc": "Ha Noi"}}'
    assert classify_gold_response(text) == "tool_call"


def test_tagged_array_classifies_as_tool_call():
    text = '<TOOLCALL>[{"name": "f", "arguments": {}}]</TOOLCALL>'
    assert classify_gold_response(text) == "tool_call"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I can answer that directly.", "direct"),
        ("Could you please provide the invoice identifier?", "request_for_info"),
        ("Apologies, but I'm unable to help with that.", "cannot_answer"),
    ],
)
def test_non_call_responses_still_classify(text, expected):
    assert classify_gold_response(text) == expected


# --------------------------------------------------------------------------- #
# Eval-row conversion
# --------------------------------------------------------------------------- #


def test_eval_row_recovers_calls_from_untagged_answer():
    row = {
        "uuid": "abc",
        "question": "Book me a ride.",
        "correct_answer": "tool_call",
        "answers": {
            "tool_call": '{"name": "uber.ride", "arguments": {"loc": "Berkeley"}}',
            "direct": "Sure.",
            "request_for_info": "Where to?",
            "cannot_answer": "I can't.",
        },
        "tools": ['{"name": "uber.ride", "parameters": {}}'],
    }
    record = convert_eval_row(row, 0)
    assert record is not None
    assert record["expects_call"] is True
    assert record["gold_calls"] == [{"name": "uber.ride", "arguments": {"loc": "Berkeley"}}]


def test_eval_row_answers_may_be_a_json_string():
    row = {
        "uuid": "abc",
        "question": "Book me a ride.",
        "correct_answer": "tool_call",
        "answers": json.dumps({"tool_call": '{"name": "f", "arguments": {}}'}),
        "tools": [],
    }
    record = convert_eval_row(row, 0)
    assert record["gold_calls"] == [{"name": "f", "arguments": {}}]


def test_eval_row_with_unknown_label_is_skipped():
    row = {"uuid": "x", "question": "q", "correct_answer": "not_a_label", "answers": {}}
    assert convert_eval_row(row, 0) is None


def test_eval_row_without_question_is_skipped():
    row = {"uuid": "x", "correct_answer": "tool_call", "answers": {}}
    assert convert_eval_row(row, 0) is None


def test_non_call_label_yields_no_gold_calls():
    row = {
        "uuid": "x",
        "question": "What is 2+2?",
        "correct_answer": "direct",
        "answers": {"direct": "4"},
        "tools": ['{"name": "f", "parameters": {}}'],
    }
    record = convert_eval_row(row, 0)
    assert record["expects_call"] is False
    assert record["gold_calls"] == []
    assert record["decision"] == "direct"


def test_eval_decision_labels_are_the_canonical_four():
    row = {
        "uuid": "x",
        "question": "q",
        "correct_answer": "cannot_answer",
        "answers": {"cannot_answer": "I can't."},
        "tools": [],
    }
    assert convert_eval_row(row, 0)["decision"] in DECISIONS