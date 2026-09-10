"""Offline tests for the parser and the metric layer.

These run with no network, no model and no dataset, which is deliberate: the
parser and the metric definitions are the parts most likely to be silently
wrong, and a wrong metric is worse than no metric.
"""
from __future__ import annotations

import json

import pytest

from src.metrics import compute_metrics, score_one
from src.schema import (
    ToolCall,
    canon_calls,
    has_tool_call,
    normalize_value,
    param_triples,
    parse_tool_calls,
    render_conversation,
    values_equal,
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parses_single_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Hanoi"}}\n</tool_call>'
    result = parse_tool_calls(text)
    assert result.json_valid
    assert result.calls == [ToolCall("get_weather", {"city": "Hanoi"})]


def test_parses_parallel_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    result = parse_tool_calls(text)
    assert result.json_valid
    assert [call.name for call in result.calls] == ["a", "b"]


def test_accepts_legacy_functioncall_tag():
    text = '<functioncall> {"name": "search", "arguments": {"q": "x"}} </functioncall>'
    result = parse_tool_calls(text)
    assert result.json_valid
    assert result.calls[0].name == "search"


def test_unterminated_block_is_still_recovered():
    text = '<tool_call>{"name": "search", "arguments": {"q": "x"}}'
    result = parse_tool_calls(text)
    assert result.n_blocks == 1
    assert result.calls[0].name == "search"


def test_malformed_block_marks_invalid_but_does_not_raise():
    text = "<tool_call>{not json at all</tool_call>"
    result = parse_tool_calls(text)
    assert result.n_blocks == 1
    assert result.n_valid == 0
    assert not result.json_valid
    assert result.calls == []


def test_plain_text_is_valid_and_has_no_calls():
    result = parse_tool_calls("I can answer that without a tool: 2 + 2 is 4.")
    assert result.json_valid
    assert result.calls == []
    assert not has_tool_call("I can answer that without a tool: 2 + 2 is 4.")


def test_arguments_as_json_string_are_decoded():
    text = '<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>'
    result = parse_tool_calls(text)
    assert result.calls[0].arguments == {"a": 1}


def test_missing_name_is_malformed():
    result = parse_tool_calls('<tool_call>{"arguments": {"a": 1}}</tool_call>')
    assert not result.json_valid


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "left, right",
    [
        (1, 1.0),
        ("1", 1.0),
        ("true", True),
        ("Hanoi ", "Hanoi"),
        (None, "null"),
        ([1, 2], ["2", 1.0]),
        ({"a": {"b": "true"}}, {"a": {"b": True}}),
    ],
)
def test_values_equal_accepts_benign_variation(left, right):
    assert values_equal(left, right)


@pytest.mark.parametrize(
    "left, right",
    [
        (1, 2),
        ("Hanoi", "Hanoi City"),
        (True, False),
        ([1, 2], [1, 2, 3]),
        ({"a": 1}, {"a": 1, "b": 2}),
        ("true", 1),
    ],
)
def test_values_equal_rejects_real_differences(left, right):
    assert not values_equal(left, right)


def test_normalize_value_coerces_numeric_strings():
    assert normalize_value("1,234") == 1234.0
    assert normalize_value(" 3.5 ") == 3.5
    assert normalize_value("Hanoi") == "Hanoi"


def test_canon_calls_is_order_insensitive():
    a = [ToolCall("x", {"k": 1}), ToolCall("y", {})]
    b = [ToolCall("y", {}), ToolCall("x", {"k": "1"})]
    assert canon_calls(a) == canon_calls(b)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _record(split="heldout_tools", gold=None, expects_call=True):
    return {
        "id": f"{split}-1",
        "split": split,
        "tools": [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}],
        "messages": [{"role": "user", "content": "weather in Hanoi?"}],
        "gold_calls": gold if gold is not None else [{"name": "get_weather", "arguments": {"city": "Hanoi"}}],
        "expects_call": expects_call,
    }


GOLD_TEXT = '<tool_call>{"name": "get_weather", "arguments": {"city": "Hanoi"}}</tool_call>'


def test_perfect_prediction_scores_one():
    row = score_one(_record(), GOLD_TEXT)
    assert row["json_valid"]
    assert row["selection_correct"]
    assert row["exact_match"]
    assert row["argument_score"] == 1.0


def test_wrong_function_fails_selection_and_exact_match():
    text = '<tool_call>{"name": "get_time", "arguments": {"city": "Hanoi"}}</tool_call>'
    row = score_one(_record(), text)
    assert not row["selection_correct"]
    assert not row["exact_match"]
    assert row["argument_score"] == 0.0


def test_extra_hallucinated_argument_lowers_argument_score():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Hanoi", "units": "F"}}</tool_call>'
    row = score_one(_record(), text)
    assert row["selection_correct"]
    assert not row["exact_match"]
    assert 0.0 < row["argument_score"] < 1.0


def test_false_tool_call_is_flagged_on_no_tool_record():
    record = _record(split="no_tool", gold=[], expects_call=False)
    row = score_one(record, GOLD_TEXT)
    assert row["emitted_call_when_none_expected"]
    assert not row["correctly_abstained"]


def test_correct_abstention_is_flagged():
    record = _record(split="no_tool", gold=[], expects_call=False)
    row = score_one(record, "Two plus two is four.")
    assert row["correctly_abstained"]
    assert not row["emitted_call_when_none_expected"]


def test_malformed_output_breaks_json_validity():
    row = score_one(_record(), "<tool_call>{oops</tool_call>")
    assert not row["json_valid"]


def test_compute_metrics_separates_splits():
    pairs = [
        (_record(split="heldout_tools"), GOLD_TEXT),
        (
            _record(
                split="unseen_functions",
                gold=[{"name": "book_flight", "arguments": {"to": "Hanoi"}}],
            ),
            '<tool_call>{"name": "book_flight", "arguments": {"to": "Hanoi"}}</tool_call>',
        ),
        (_record(split="no_tool", gold=[], expects_call=False), "No tool needed."),
        (_record(split="no_tool", gold=[], expects_call=False), GOLD_TEXT),
    ]
    metrics = compute_metrics(pairs)

    assert metrics["overall"]["exact_match"] == 1.0
    assert metrics["overall"]["tool_selection_accuracy"] == 1.0
    assert metrics["overall"]["unseen_function_accuracy"] == 1.0
    assert metrics["overall"]["json_validity"] == 1.0
    assert metrics["overall"]["false_tool_call_rate"] == 0.5
    assert metrics["overall"]["no_tool_accuracy"] == 0.5
    assert set(metrics["by_split"]) == {"heldout_tools", "no_tool", "unseen_functions"}


def test_metrics_report_none_not_zero_for_empty_slices():
    metrics = compute_metrics([(_record(), GOLD_TEXT)])
    assert metrics["overall"]["false_tool_call_rate"] is None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_render_conversation_masks_only_assistant_turns():
    from src.schema import build_segments

    record = _record()
    record["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Hanoi"}}]},
        {"role": "tool", "name": "get_weather", "content": '{"temp": 30}'},
    ]
    segments = build_segments(record)
    trainable = [text for text, is_trainable in segments if is_trainable]
    assert len(trainable) == 1
    assert "<tool_call>" in trainable[0]
    # The tool result must never be trained on.
    assert all("returned:" not in text for text in trainable)
    assert "<|im_start|>system" in render_conversation(record)
    assert render_conversation(record, add_generation_prompt=True).endswith("<|im_start|>assistant\n")