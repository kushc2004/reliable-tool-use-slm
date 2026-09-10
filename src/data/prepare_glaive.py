"""Load positive function-calling trajectories (Glaive, Hermes) into our format.

Glaive ships exactly two columns::

    system   "SYSTEM: You are a helpful assistant with access to the following
              functions. Use them if required -
              {"name": ..., "description": ..., "parameters": {...}}"
    chat     "USER: ...\\nASSISTANT: <functioncall> {...} <|endoftext|>\\n..."

Two properties of that shape cost the first version of this loader almost
everything, so they are stated plainly and pinned by tests:

1. **There is no ``functions`` column.** The tool schemas are embedded inside
   the system string. ``row.get("functions")`` is always ``None``, so every
   record failed validation with "no tools advertised" and the built corpus
   came out with zero training rows. Verified against the Hub: 13,776 of the
   first 20,000 rows carry a parseable schema inside ``system``.
2. **A call is terminated by ``<|endoftext|>``, not by ``</functioncall>``.**
   Every call block in the first 20,000 rows is unclosed. A parser that
   requires the closing tag matches nothing at all.

A third quirk shows up in a minority of rows: ``arguments`` is written as a
single-quoted string containing literal newlines, which is neither valid JSON
nor a valid Python literal. That repair lives in ``schema._loads_lenient`` so
the evaluator benefits from it too.

Usage:
    python -m src.data.prepare_glaive --input glaiveai/glaive-function-calling-v2 \\
        --out data/raw/glaive.jsonl --limit 5000
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..schema import (
    ALT_CALL_CLOSE,
    ALT_CALL_OPEN,
    CALL_CLOSE,
    CALL_OPEN,
    parse_tool_calls,
)

__all__ = [
    "DIALECTS",
    "EOT",
    "load_rows",
    "extract_tools_from_system",
    "convert_row",
    "convert",
    "main",
]

DIALECTS = ["glaive", "hermes"]

# Glaive's end-of-turn sentinel. It doubles as the terminator for an unclosed
# call block, which is how the dataset actually writes every call.
EOT = "<|endoftext|>"

# Role markers at the start of a line. Glaive uses these exact spellings.
_ROLE_RE = re.compile(
    r"^\s*(USER|ASSISTANT|FUNCTION RESPONSE|SYSTEM)\s*:\s?",
    re.IGNORECASE | re.MULTILINE,
)

_ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "function response": "tool",
    "system": "system",
}

# "Use them if required -" introduces the tool schema inside the system string.
_TOOLS_MARKER = re.compile(r"Use them if required\s*-?\s*", re.IGNORECASE)

# A whole call block -- tags *and* payload. Group 1 is the JSON body.
#
# Terminators, in priority order: the proper closing tag, the <|endoftext|>
# sentinel, the start of a following call block, then end-of-turn.
#
# * <|endoftext|> matters because every call block in Glaive is unclosed.
# * The lookahead for a following call-open matters for turns that issue two
#   calls: without it, the non-greedy body would run past the second opening
#   tag to the end-of-string fallback and the whole turn would fail to parse.
# * The end-of-string fallback is last so a real terminator always wins.
_CALL_BLOCK_RE = re.compile(
    rf"(?:{re.escape(CALL_OPEN)}|{re.escape(ALT_CALL_OPEN)})\s*(.*?)\s*"
    rf"(?:{re.escape(CALL_CLOSE)}|{re.escape(ALT_CALL_CLOSE)}|{re.escape(EOT)}"
    rf"|(?=(?:{re.escape(CALL_OPEN)}|{re.escape(ALT_CALL_OPEN)}))|$)",
    re.DOTALL | re.IGNORECASE,
)


def load_rows(source: str, limit: int | None = None, split: str = "train") -> list[dict[str, Any]]:
    """Read rows from a local .jsonl/.json file or from the Hub."""
    path = Path(source)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        from datasets import load_dataset  # lazy: needs network

        dataset = load_dataset(source, split=split)
        rows = [dict(row) for row in dataset]
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------- #
# Tool specs, extracted from the system string
# --------------------------------------------------------------------------- #


def _normalize_tool_list(items: Any) -> list[dict[str, Any]]:
    # Both real callers hand this a JSON *string*: Hermes stores ``tools`` as
    # one, and the ``functions`` fallback column does too. Not decoding it here
    # made every such row normalize to an empty tool list, which then failed
    # the "no tools advertised" check and dropped the row.
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(items, dict):
        for key in ("functions", "tools"):
            if key in items:
                return _normalize_tool_list(items[key])
        items = [items]
    if not isinstance(items, list):
        return []

    tools: list[dict[str, Any]] = []
    for tool in items:
        if isinstance(tool, str):
            try:
                tool = json.loads(tool)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(tool, dict):
            continue

        # Already in the OpenAI {"type", "function"} shape: keep as-is.
        # This must be checked *before* the bare-schema path, because a wrapped
        # spec has no top-level "name" and would otherwise be dropped as
        # malformed -- which is how Hermes rows lost every tool they declared.
        if isinstance(tool.get("function"), dict):
            if tool["function"].get("name"):
                tools.append(tool)
            continue

        if "name" not in tool:
            continue
        tools.append({"type": "function", "function": tool})
    return tools


def extract_tools_from_system(system: Any) -> list[dict[str, Any]]:
    """Pull the tool schema out of Glaive's ``system`` string.

    The schema is a bare JSON object (or array) appended after the phrase
    "Use them if required -". Returns an empty list when it is absent or
    unparseable, so the caller can count the row as skipped rather than guess.
    """
    if not isinstance(system, str) or not system.strip():
        return []

    marker = _TOOLS_MARKER.search(system)
    tail = system[marker.end():] if marker else system

    start = min(
        (index for index in (tail.find("{"), tail.find("[")) if index != -1),
        default=-1,
    )
    if start == -1:
        return []

    try:
        obj, _ = json.JSONDecoder().raw_decode(tail[start:])
    except (json.JSONDecodeError, ValueError):
        return []

    return _normalize_tool_list(obj)


# --------------------------------------------------------------------------- #
# Chat string
# --------------------------------------------------------------------------- #


def _strip_eot(text: str) -> str:
    return text.replace(EOT, " ").strip()


def _strip_eot_from_messages(messages: list[dict[str, Any]]) -> None:
    """Remove the ``<|endoftext|>`` sentinel from every turn, in place.

    This runs *after* call extraction, not before. Stripping it earlier deleted
    the terminator that ``_CALL_BLOCK_RE`` matches on, which silently reduced
    the corpus from thousands of rows to zero.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and EOT in content:
            message["content"] = " ".join(content.replace(EOT, " ").split())


def _split_glaive_chat(chat: str) -> list[dict[str, Any]]:
    """Split the role-marked chat string into message dicts.

    A ``FUNCTION RESPONSE`` turn becomes a ``tool`` message; everything else
    maps through to user/assistant. The ``<|endoftext|>`` sentinel is stripped
    from turn content so it never reaches the tokenizer as literal text.
    """
    if not isinstance(chat, str) or not chat.strip():
        return []

    matches = list(_ROLE_RE.finditer(chat))
    if not matches:
        # No markers at all: treat the whole thing as one user turn.
        return [{"role": "user", "content": _strip_eot(chat)}]

    messages: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        role = _ROLE_MAP.get(match.group(1).lower())
        if role is None:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
        # Content is kept verbatim here. The <|endoftext|> sentinel is the
        # terminator the call regex matches on, so it must survive until
        # extraction has run; _strip_eot_from_messages removes it afterwards.
        content = chat[start:end].strip()
        if content or role == "assistant":
            messages.append({"role": role, "content": content})
    return messages


def _parse_bodies(bodies: Iterable[str]) -> list[dict[str, Any]]:
    """Parse extracted call bodies by re-wrapping them in canonical tags.

    Re-wrapping means every lenient-parsing improvement in ``schema`` applies
    here for free, including the single-quoted ``arguments`` repair.
    """
    calls: list[dict[str, Any]] = []
    for body in bodies:
        wrapped = f"{CALL_OPEN}\n{body}\n{CALL_CLOSE}"
        calls.extend(call.to_dict() for call in parse_tool_calls(wrapped).calls)
    return calls


def _extract_calls_from_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect assistant calls into the structured ``tool_calls`` field.

    Two shapes exist in the wild and both must work:

    * Glaive embeds the call inline in the assistant's text, terminated by
      ``<|endoftext|>``. It is parsed out and the whole block is then stripped
      from ``content`` -- otherwise the renderer, which writes both the prose
      and the structured calls, would emit the same call twice and train the
      model to duplicate it.
    * Hermes already carries a structured ``tool_calls`` field. That is used
      as-is; there is nothing to strip.

    The assistant's surrounding prose is kept either way: a turn that says
    "let me check" and then calls a tool should train as both.
    """
    calls: list[dict[str, Any]] = []

    for message in messages:
        if message["role"] != "assistant":
            continue

        existing = message.get("tool_calls")
        if existing:
            calls.extend(existing)
            continue

        content = message.get("content") or ""
        bodies = [match.group(1).strip() for match in _CALL_BLOCK_RE.finditer(content)]
        if not bodies:
            continue

        message["tool_calls"] = _parse_bodies(bodies)
        calls.extend(message["tool_calls"])

        # Drop the now-redundant call block, keeping any surrounding prose.
        cleaned = _CALL_BLOCK_RE.sub(" ", content)
        message["content"] = " ".join(cleaned.split())

    return calls


# --------------------------------------------------------------------------- #
# Hermes
# --------------------------------------------------------------------------- #


def _split_hermes_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Hermes already carries a turn list; normalize its role spelling."""
    raw = row.get("conversations") or row.get("messages") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []

    messages: list[dict[str, Any]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or "user"
        role = {"human": "user", "gpt": "assistant", "system": "system"}.get(role, role)
        content = turn.get("content") or turn.get("value") or ""
        message: dict[str, Any] = {"role": role, "content": content}

        raw_calls = turn.get("tool_calls") or turn.get("function_call")
        if raw_calls:
            if isinstance(raw_calls, dict):
                raw_calls = [raw_calls]
            normalized: list[dict[str, Any]] = []
            for call in raw_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                name = fn.get("name")
                if not name:
                    continue
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                normalized.append({"name": name, "arguments": args or {}})
            if normalized:
                message["tool_calls"] = normalized
        messages.append(message)
    return messages


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def convert_row(
    row: dict[str, Any],
    index: int,
    dialect: str,
    stats: "Counter[str] | None" = None,
) -> dict[str, Any] | None:
    """Convert one source row into a project record, or None if unusable.

    ``stats`` is optional so existing callers (and the test suite) can keep
    using the three-argument form. When supplied, it collects per-reason skip
    counts -- a silent 0% keep-rate is exactly how an empty corpus shipped
    unnoticed, so the reasons are always available.
    """
    if dialect == "glaive":
        messages = _split_glaive_chat(row.get("chat") or row.get("conversations") or "")
    elif dialect == "hermes":
        messages = _split_hermes_row(row)
    else:
        raise ValueError(f"unknown dialect: {dialect}")

    if not messages:
        if stats is not None:
            stats["skip:no_messages"] += 1
        return None

    if dialect == "glaive":
        tools = extract_tools_from_system(row.get("system"))
        if not tools:
            tools = _normalize_tool_list(row.get("functions") or row.get("tools"))
    else:
        tools = _normalize_tool_list(row.get("functions") or row.get("tools"))

    _extract_calls_from_turns(messages)

    # Truncate at the first assistant turn that actually calls a tool, and use
    # that turn as the training target.
    #
    # Glaive conversations continue past the call -- the tool result follows,
    # then a summary turn -- so requiring the conversation to *end* on a call
    # rejects most of the corpus. Truncating instead guarantees a valid
    # user -> assistant(call) target and matches the shape the evaluator
    # scores. The discarded tail is context nothing is graded on.
    cut = next(
        (
            idx
            for idx, message in enumerate(messages)
            if message["role"] == "assistant" and message.get("tool_calls")
        ),
        None,
    )
    if cut is None:
        if stats is not None:
            stats["skip:no_call"] += 1
        return None

    messages = messages[: cut + 1]
    calls = [call for message in messages for call in (message.get("tool_calls") or [])]

    # A call with no advertised tool cannot be rendered or scored, so it is
    # dropped rather than admitted as an unusable training row.
    if not tools:
        if stats is not None:
            stats["skip:no_tools"] += 1
        return None

    # Drop a leading system turn: our renderer supplies its own.
    if messages and messages[0]["role"] == "system":
        messages = messages[1:]
    if not messages:
        if stats is not None:
            stats["skip:empty_after_system_strip"] += 1
        return None

    _strip_eot_from_messages(messages)

    return {
        "id": f"{dialect}-{index:06d}",
        "split": "train",
        "category": "tool_call",
        "source": dialect,
        "tools": tools,
        "messages": messages,
        "gold_calls": calls,
        "expects_call": True,
    }


def convert(
    rows: Iterable[dict[str, Any]], dialect: str = "glaive"
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert rows, counting why anything was skipped.

    The skip reasons are broken out rather than lumped into one counter: a
    silent 2% keep-rate is exactly the failure mode that produced an empty
    corpus without anyone noticing.
    """
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for index, row in enumerate(rows):
        stats["seen"] += 1
        try:
            record = convert_row(row, index, dialect, stats)
        except Exception:  # noqa: BLE001 - one bad row must not kill the run
            record = None
            stats["skip:exception"] += 1
        if record is None:
            stats["skipped"] += 1
            continue
        records.append(record)
        stats["kept"] += 1

    return records, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="glaiveai/glaive-function-calling-v2")
    parser.add_argument("--dialect", choices=DIALECTS, default="glaive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    rows = load_rows(args.input, limit=args.limit, split=args.split)
    records, stats = convert(rows, args.dialect)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    rate = (stats["kept"] / stats["seen"] * 100) if stats["seen"] else 0.0
    print(f"[{args.dialect}] seen={stats['seen']} kept={stats['kept']} "
          f"skipped={stats['skipped']} ({rate:.1f}% kept)")
    for key in sorted(stats):
        if key.startswith("skip:"):
            print(f"    {key:<32} {stats[key]}")
    print(f"wrote {len(records)} records to {out_path}")

    if not records:
        raise SystemExit(
            "no records converted -- this is a loader bug, not an empty dataset"
        )


if __name__ == "__main__":
    main()