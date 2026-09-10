"""Tests for the Glaive / Hermes loader.

This module parses a third-party format against which three assumptions were
originally wrong, each of which is now pinned here:

1. **There is no ``functions`` column.** The tool schema lives inside the
   ``system`` string. Reading ``row["functions"]`` yields ``None`` for every
   row, so every record was dropped as "no tools advertised" and the corpus
   came out empty.
2. **A call is terminated by ``<|endoftext|>``, not by ``</functioncall>``.**
   Requiring the closing tag matches nothing in the dataset.
3. **Glaive conversations continue past the call** (tool response, then a
   summary). Requiring the conversation to *end* on the call rejects most of
   the corpus, so the record is truncated at the first calling turn instead.

There is also a regression guard for a bug that made the loader silently
return zero rows: ``_split_glaive_chat`` must not strip ``<|endoftext|>``
before call extraction runs, because that sentinel is the terminator the call
regex matches on.
"""
from __future__ import annotations

import pytest

from src.data.prepare_glaive import (
    EOT,
    _normalize_tool_list,
    _split_glaive_chat,
    convert_row,
    extract_tools_from_system,
)

SYSTEM_WITH_TOOLS = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n"
    '{\n  "name": "get_weather",\n  "description": "Get the current weather",\n'
    '  "parameters": {"type": "object", "properties": '
    '{"city": {"type": "string"}}, "required": ["city"]}\n}'
)

# Real shape: unclosed call terminated by the sentinel, conversation continues.
GLAIVE_ROW = {
    "system": SYSTEM_WITH_TOOLS,
    "chat": (
        "USER: What is the weather in Hanoi?\n"
        "ASSISTANT: Let me check. <functioncall> "
        '{"name": "get_weather", "arguments": {"city": "Hanoi"}} ' + EOT + "\n"
        'FUNCTION RESPONSE: {"temp_c": 28}\n'
        "ASSISTANT: It is 28C in Hanoi."
    ),
}


# --------------------------------------------------------------------------- #
# Tool extraction from the system string
# --------------------------------------------------------------------------- #


def test_tools_extracted_from_system():
    tools = extract_tools_from_system(SYSTEM_WITH_TOOLS)
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"
    assert tools[0]["function"]["parameters"]["required"] == ["city"]


def test_tools_absent_returns_empty():
    assert extract_tools_from_system("SYSTEM: You are a helpful assistant.") == []


@pytest.mark.parametrize("bad", [None, "", 42, {}, "no marker here at all"])
def test_tools_malformed_or_absent_degrade_to_empty(bad):
    assert extract_tools_from_system(bad) == []


def test_tools_marker_with_unparseable_json_returns_empty():
    broken = "SYSTEM: ... Use them if required -\n{not valid json at all"
    assert extract_tools_from_system(broken) == []


def test_tools_accepts_a_list_of_schemas():
    system = (
        "SYSTEM: ... Use them if required -\n"
        '[{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}]'
    )
    names = [t["function"]["name"] for t in extract_tools_from_system(system)]
    assert names == ["a", "b"]


# --------------------------------------------------------------------------- #
# Normalization of an already-decoded tool list
# --------------------------------------------------------------------------- #


def test_tool_list_normalized_to_openai_shape():
    tools = _normalize_tool_list('[{"name": "f", "parameters": {}}]')
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "f"


def test_tool_list_accepts_already_wrapped_shape():
    wrapped = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert _normalize_tool_list(wrapped)[0]["function"]["name"] == "f"


@pytest.mark.parametrize("bad", [None, "", "not json", 42, {}, []])
def test_tool_list_malformed_degrades_to_empty(bad):
    assert _normalize_tool_list(bad) == []


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


def test_eot_survives_chat_splitting():
    """Regression: stripping the sentinel here empties the corpus.

    ``<|endoftext|>`` is the terminator ``_CALL_BLOCK_RE`` matches on. Removing
    it during splitting means no call is ever found, every row is skipped as
    "no call", and the loader returns zero records while reporting success.
    """
    messages = _split_glaive_chat(GLAIVE_ROW["chat"])
    assistant_with_call = [m for m in messages if m["role"] == "assistant"][0]
    assert EOT in assistant_with_call["content"]


def test_unmarked_chat_is_treated_as_one_user_turn():
    messages = _split_glaive_chat("just some text with no role markers")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_empty_chat_yields_nothing():
    assert _split_glaive_chat("") == []
    assert _split_glaive_chat(None) == []


# --------------------------------------------------------------------------- #
# Row conversion
# --------------------------------------------------------------------------- #


def test_valid_row_converts():
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    assert record is not None
    assert record["expects_call"] is True
    assert record["gold_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Hanoi"}}
    ]
    assert record["messages"][-1]["role"] == "assistant"
    assert len(record["tools"]) == 1


def test_conversation_truncated_at_the_calling_turn():
    """The tool response and summary that follow are not part of the target."""
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]


def test_row_without_call_is_skipped():
    """The critical case: no call means this is not a positive example.

    Admitting it would train the model to emit a call where none belongs,
    which is the exact failure this project exists to measure.
    """
    row = dict(GLAIVE_ROW)
    row["chat"] = "USER: hi\nASSISTANT: hello there"
    assert convert_row(row, 0, "glaive") is None


def test_row_without_tools_is_skipped():
    """A call with nothing advertised cannot be rendered or scored."""
    row = dict(GLAIVE_ROW)
    row["system"] = "SYSTEM: You are a helpful assistant."
    assert convert_row(row, 0, "glaive") is None


def test_eot_stripped_from_final_messages():
    """It is the terminator during parsing, but must not survive to training."""
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    for message in record["messages"]:
        assert EOT not in (message.get("content") or "")


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

    from src.schema import render_assistant

    with_call = [m for m in record["messages"] if m.get("tool_calls")][0]
    assert render_assistant(with_call).count("<tool_call>") == 1


def test_leading_system_turn_is_dropped():
    record = convert_row(GLAIVE_ROW, 0, "glaive")
    assert record["messages"][0]["role"] != "system"


def test_closed_tag_still_accepted():
    """Backwards compatibility: a properly closed block must also parse."""
    row = dict(GLAIVE_ROW)
    row["chat"] = (
        "USER: weather?\nASSISTANT: <functioncall> "
        '{"name": "get_weather", "arguments": {"city": "Hanoi"}} </functioncall>'
    )
    record = convert_row(row, 0, "glaive")
    assert record is not None
    assert record["gold_calls"][0]["name"] == "get_weather"


def test_functions_column_still_used_as_fallback():
    """Some exports carry the schema as a column rather than in ``system``."""
    row = {
        "system": "SYSTEM: You are a helpful assistant.",
        "chat": GLAIVE_ROW["chat"],
        "functions": '[{"name": "get_weather", "parameters": {}}]',
    }
    record = convert_row(row, 0, "glaive")
    assert record is not None
    assert len(record["tools"]) == 1


def test_skip_stats_record_the_reason():
    from collections import Counter

    stats: Counter[str] = Counter()
    row = dict(GLAIVE_ROW)
    row["chat"] = "USER: hi\nASSISTANT: hello there"
    convert_row(row, 0, "glaive", stats)
    assert stats["skip:no_call"] == 1


# --------------------------------------------------------------------------- #
# Hermes
# --------------------------------------------------------------------------- #


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