"""Tests for the Glaive / Hermes loader.

This module parses a third-party format, so the tests pin the parsing decisions
that matter: role mapping, call extraction, and — most importantly — that a
trajectory with no tool call is *skipped* rather than mislabelled as a positive.

That last case is the dangerous one. If a no-call conversation were admitted as
a positive example, the model would be trained to emit a call where none
belongs, which is the exact failure this project exists to measure.
"""
from __future__ import annotations

import pytest

from src.data.prepare_glaive import convert_row, _normalize_tools, _split_glaive_chat

GLAIVE_ROW = {
    "system": "SYSTEM: You are a helpful assistant.",
    "chat": (
        "USER: What is the weather in Hanoi?\n"
        'ASSISTANT: Let me check. <functioncall> {"name": "get_weather", '
        '"arguments": {"city": "Hanoi"}} </functioncall>\n'
        'FUNCTION RESPONSE: {"temp_c": 28}\n'
        "ASSISTANT: It is 28C in Hanoi."
    ),
    "functions": (
        '[{"name": "get_weather", "description": "Get weather", '
        '"parameters": {"type": "object", "properties": '
        '{"city": {"type": "string"}}}}]'
    ),
}


# --------------------------------------------------------------------------- #
# Chat splitting
# --------------------------------------------------------------------------- #


def test_roles_are_mapped():
    messages = _split_glaive_chat(GLAIVE_ROW["chat"])
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]


def test_function_response_becomes_tool_turn():
    messages = _split_glaive_chat(GLAIVE_ROW["chat"])
    tool_turns = [m for m in messages if m["role"] == "tool"]
    assert len(tool_turns) == 1
    assert "temp_c" in tool_turns[0]["content"]


def test_unmarked_chat_is_treated_as_one_user_turn():
    messages = _split_glaive_chat("just some text with no role markers")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_empty_chat_yields_nothing():
    assert _split_glaive_chat("") == []
    assert _split_glaive_chat(None) == []


# --------------------------------------------------------------------------- #
# Tool specs
# --------------------------------------------------------------------------- #


def test_tools_normalized_to_openai_shape():
    tools = _normalize_tools(GLAIVE_ROW["functions"])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"


def test_tools_accept_already_wrapped_shape():
    wrapped = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    tools = _normalize_tools(wrapped)
    assert tools[0]["function"]["name"] == "f"


@pytest.mark.parametrize("bad", [None, "", "not json", 42, {}])
def test_malformed_tools_degrade_to_empty(bad):
    assert _normalize_tools(bad) == []


# --------------------------------------------------------------------------- #
# Row conversion
# --------------------------------------------------------------------------- #


def test_valid_row_converts():
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    assert record is not None
    assert record["expects_call"] is True
    assert record["gold_calls"][0]["name"] == "get_weather"
    assert record["gold_calls"][0]["arguments"]["city"] == "Hanoi"
    assert record["messages"][-1]["role"] == "assistant"
    assert len(record["tools"]) == 1


def test_call_extracted_into_tool_calls():
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    assistant_turns = [m for m in record["messages"] if m["role"] == "assistant"]
    with_call = [m for m in assistant_turns if m.get("tool_calls")]
    assert len(with_call) == 1
    assert with_call[0]["tool_calls"][0]["name"] == "get_weather"


def test_row_without_call_is_skipped():
    """The critical case: no call means this is not a positive example."""
    row = dict(GLAIVE_ROW)
    row["chat"] = "USER: hi\nASSISTANT: hello there"
    assert convert_row(row, 0, "glaive") is None


def test_inline_markup_stripped_from_content():
    """The call must not appear in both ``content`` and ``tool_calls``.

    The renderer writes the prose and the structured calls, so leaving the
    inline markup in place would emit the same call twice and train the model
    to duplicate it.
    """
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    for message in record["messages"]:
        content = message.get("content") or ""
        assert "<functioncall>" not in content
        assert "</functioncall>" not in content
        assert "<tool_call>" not in content


def test_surrounding_prose_survives_markup_stripping():
    """'Let me check' is useful supervision and should not be discarded."""
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    with_call = [m for m in record["messages"] if m.get("tool_calls")][0]
    assert "Let me check" in with_call["content"]


def test_call_payload_not_left_in_content():
    """Stripping only the tags would leave the JSON behind as a second call."""
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    with_call = [m for m in record["messages"] if m.get("tool_calls")][0]
    content = with_call["content"]
    assert "get_weather" not in content
    assert '"arguments"' not in content
    assert "Hanoi" not in content


def test_call_count_not_doubled():
    """Exactly one call, not one-from-content plus one-from-tool_calls."""
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    assert len(record["gold_calls"]) == 1

    # And the rendered form should contain the call block exactly once.
    from src.schema import render_assistant

    with_call = [m for m in record["messages"] if m.get("tool_calls")][0]
    rendered = render_assistant(with_call)
    assert rendered.count("<tool_call>") == 1


def test_leading_system_turn_is_dropped():
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    assert record["messages"][0]["role"] != "system"


def test_row_ending_on_user_is_skipped():
    """A conversation must end on an assistant turn to be a training target."""
    row = dict(GLAIVE_ROW)
    row["chat"] = (
        'USER: weather?\nASSISTANT: <functioncall> {"name": "get_weather", '
        '"arguments": {}} </functioncall>\nUSER: thanks'
    )
    assert convert_row(row, 0, "glaive") is None


def test_hermes_dialect_reads_turn_list():
    row = {
        "tools": '[{"name": "get_time", "parameters": {}}]',
        "conversations": [
            {"from": "human", "value": "What time is it in Oslo?"},
            {"from": "gpt", "value": "Let me check.",
             "tool_calls": [{"name": "get_time", "arguments": {"city": "Oslo"}}]},
        ],
    }
    record = convert_row(row, 0, "hermes")
    assert record is not None
    assert record["gold_calls"][0]["name"] == "get_time"
    assert record["messages"][0]["role"] == "user"


def test_unknown_dialect_raises():
    with pytest.raises(ValueError):
        convert_row(GLAIVE_ROW, 0, "nope")