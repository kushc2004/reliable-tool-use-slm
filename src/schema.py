"""Canonical tool-call format used across the whole project.

The model speaks exactly one protocol. Tools are advertised in the system turn:

    <tools>
    {"type": "function", "function": {"name": ..., "description": ...,
                                      "parameters": {...}}}
    </tools>

and the assistant either emits one or more

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Hanoi"}}
    </tool_call>

blocks, or replies in plain prose.

The plain-prose case is the whole point: there is an unambiguous, machine-
checkable difference between "called a tool" and "did not call a tool", which
is what makes negative (no-tool) supervision and a false-tool-call-rate metric
possible at all.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOLS_OPEN = "<tools>"
TOOLS_CLOSE = "</tools>"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"

# Legacy Glaive / Hermes-v0 spelling. Accepted on input, never emitted.
ALT_CALL_OPEN = "<functioncall>"
ALT_CALL_CLOSE = "</functioncall>"

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Call a function only when it is genuinely needed to satisfy the user's "
    "request; if no function is required, answer directly in plain text.\n"
    "{tools_block}"
)

__all__ = [
    "ToolCall",
    "ParseResult",
    "IM_START",
    "IM_END",
    "TOOLS_OPEN",
    "TOOLS_CLOSE",
    "CALL_OPEN",
    "CALL_CLOSE",
    "SYSTEM_TEMPLATE",
    "render_tools_block",
    "render_system",
    "render_assistant",
    "render_conversation",
    "build_segments",
    "parse_tool_calls",
    "has_tool_call",
    "normalize_value",
    "values_equal",
    "canonical_json",
    "canon_calls",
    "param_triples",
]


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolCall:
    """A single function invocation."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ParseResult:
    """Outcome of parsing one model completion."""

    calls: list[ToolCall] = field(default_factory=list)
    n_blocks: int = 0
    n_valid: int = 0
    malformed: list[str] = field(default_factory=list)

    @property
    def json_valid(self) -> bool:
        """True when every emitted block was well-formed.

        A completion that emits nothing is trivially valid: "I can't help with
        that" is a perfectly good structured output.
        """
        return not self.malformed and self.n_valid == self.n_blocks


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_tools_block(tools: Sequence[dict[str, Any]]) -> str:
    """Render a tool list into the ``<tools>`` block.

    Accepts either the wrapped OpenAI shape
    ``{"type": "function", "function": {...}}`` or a bare
    ``{"name": ..., "parameters": ...}`` dict, and normalizes to the former.
    """
    if not tools:
        return f"{TOOLS_OPEN}\n{TOOLS_CLOSE}"

    lines: list[str] = []
    for tool in tools:
        if "function" in tool:
            spec = tool
        else:
            spec = {"type": "function", "function": tool}
        lines.append(json.dumps(spec, ensure_ascii=False, sort_keys=True))
    return f"{TOOLS_OPEN}\n" + "\n".join(lines) + f"\n{TOOLS_CLOSE}"


def render_system(tools: Sequence[dict[str, Any]]) -> str:
    return SYSTEM_TEMPLATE.format(tools_block=render_tools_block(tools))


def render_assistant(message: dict[str, Any]) -> str:
    """Render one assistant turn, including any tool calls."""
    parts: list[str] = []
    content = (message.get("content") or "").strip()
    if content:
        parts.append(content)
    for call in message.get("tool_calls") or []:
        payload = {
            "name": call["name"],
            "arguments": call.get("arguments") or {},
        }
        parts.append(
            f"{CALL_OPEN}\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n{CALL_CLOSE}"
        )
    return "\n".join(parts)


def _render_tool_result(message: dict[str, Any]) -> str:
    name = message.get("name") or "tool"
    content = message.get("content")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return f"{name} returned: {content}"


def render_message(message: dict[str, Any]) -> str:
    role = message["role"]
    if role == "system":
        body = message.get("content") or ""
    elif role == "assistant":
        body = render_assistant(message)
    elif role in ("tool", "function"):
        body = _render_tool_result(message)
    else:
        body = message.get("content") or ""
    return f"{IM_START}{role}\n{body}{IM_END}\n"


def build_segments(record: dict[str, Any]) -> list[tuple[str, bool]]:
    """Flatten a record into ``(text, trainable)`` segments.

    ``trainable`` marks the spans that contribute to the loss: assistant turns
    only. Tool results, user turns and the system prompt are context, and
    training on them teaches the model to hallucinate tool output.
    """
    segments: list[tuple[str, bool]] = [
        (render_message({"role": "system", "content": render_system(record.get("tools") or [])}), False)
    ]
    for message in record.get("messages") or []:
        segments.append((render_message(message), message["role"] == "assistant"))
    return segments


def render_conversation(record: dict[str, Any], add_generation_prompt: bool = False) -> str:
    """Full ChatML string for a record, optionally ending on an assistant cue."""
    text = "".join(segment for segment, _ in build_segments(record))
    if add_generation_prompt:
        text += f"{IM_START}assistant\n"
    return text


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

_CLOSED_RE = re.compile(
    rf"(?:{re.escape(CALL_OPEN)}|{re.escape(ALT_CALL_OPEN)})\s*(.*?)\s*"
    rf"(?:{re.escape(CALL_CLOSE)}|{re.escape(ALT_CALL_CLOSE)})",
    re.DOTALL,
)
# An opening tag with no close: the model ran out of budget or stopped early.
_OPEN_ONLY_RE = re.compile(
    rf"(?:{re.escape(CALL_OPEN)}|{re.escape(ALT_CALL_OPEN)})\s*(.+)\Z",
    re.DOTALL,
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _loads_lenient(blob: str) -> Any:
    """Parse JSON, falling back to a Python literal and to brace trimming."""
    blob = blob.strip()
    for candidate in (blob, blob.rstrip(".,;")):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            pass
    match = _JSON_OBJECT_RE.search(blob)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    raise ValueError(f"unparseable tool call: {blob[:200]!r}")


def _coerce_call(obj: Any) -> ToolCall:
    """Turn a decoded JSON object into a ToolCall, or raise ValueError."""
    if not isinstance(obj, dict):
        raise ValueError(f"tool call is not an object: {type(obj).__name__}")

    # Tolerate {"function": {"name": ..., "arguments": ...}} nesting.
    if "name" not in obj and isinstance(obj.get("function"), dict):
        obj = obj["function"]

    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tool call has no usable name: {obj!r}")

    arguments = obj.get("arguments", obj.get("parameters", {}))
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = _loads_lenient(arguments)
        except ValueError:
            arguments = {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    return ToolCall(name=name.strip(), arguments=arguments)


def _extract_blobs(text: str) -> list[str]:
    blobs = [m.group(1).strip() for m in _CLOSED_RE.finditer(text)]
    if blobs:
        return blobs
    match = _OPEN_ONLY_RE.search(text)
    if match:
        return [match.group(1).strip()]
    return []


def parse_tool_calls(text: str) -> ParseResult:
    """Parse a completion into zero or more tool calls.

    Never raises. Malformed blocks are collected so the caller can charge them
    against JSON validity rather than silently treating them as "no call".
    """
    result = ParseResult()
    if not text:
        return result

    for blob in _extract_blobs(text):
        result.n_blocks += 1
        try:
            result.calls.append(_coerce_call(_loads_lenient(blob)))
            result.n_valid += 1
        except ValueError:
            result.malformed.append(blob)
    return result


def has_tool_call(text: str) -> bool:
    return bool(_extract_blobs(text))


# --------------------------------------------------------------------------- #
# Normalization and comparison
# --------------------------------------------------------------------------- #

_NUMERIC_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?$")
_TRUE = {"true", "yes"}
_FALSE = {"false", "no"}
_NULL = {"null", "none", "nil", ""}


def normalize_value(value: Any) -> Any:
    """Coerce a decoded argument into a comparable canonical form.

    Models emit ``"3"`` where the schema says integer, ``"true"`` where it says
    boolean, and ``"Hanoi "`` where the gold answer is ``"Hanoi"``. None of
    those are real errors, so we normalize before comparing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        if lowered in _NULL:
            return None
        if _NUMERIC_RE.match(stripped):
            try:
                return float(stripped.replace(",", ""))
            except ValueError:
                return stripped
        return stripped
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    return value


def values_equal(left: Any, right: Any, tol: float = 1e-6) -> bool:
    """Structural equality with numeric tolerance and order-insensitive lists."""
    left = normalize_value(left)
    right = normalize_value(right)

    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
        return math.isclose(left, right, rel_tol=tol, abs_tol=tol)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        # Order-insensitive: {"tags": ["a","b"]} == {"tags": ["b","a"]}.
        remaining = list(right)
        for item in left:
            for index, other in enumerate(remaining):
                if values_equal(item, other, tol):
                    remaining.pop(index)
                    break
            else:
                return False
        return True
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(values_equal(left[key], right[key], tol) for key in left)
    return left == right


def canonical_json(obj: Any) -> str:
    """Stable string form of a normalized object, for set/dict membership."""
    return json.dumps(normalize_value(obj), ensure_ascii=False, sort_keys=True)


def canon_calls(calls: Iterable[ToolCall]) -> list[str]:
    """Canonical, order-insensitive representation of a call set.

    Sorting makes the comparison insensitive to the order in which the model
    emitted parallel calls, which is not something we want to grade.
    """
    return sorted(canonical_json(call.to_dict()) for call in calls)


def param_triples(calls: Iterable[ToolCall]) -> set[tuple[str, str, str]]:
    """``(function_name, argument_name, canonical_value)`` triples."""
    triples: set[tuple[str, str, str]] = set()
    for call in calls:
        for key, value in call.arguments.items():
            triples.add((call.name, str(key), canonical_json(value)))
    return triples


def call_name_set(calls: Iterable[ToolCall]) -> set[str]:
    return {call.name for call in calls}