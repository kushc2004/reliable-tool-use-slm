"""Load positive function-calling trajectories (Glaive, Hermes) into our format.

Glaive stores a conversation as one big string with role markers, plus a
separate JSON blob of tool schemas:

    {"system": "...", "chat": "USER: ...\\nASSISTANT: ...", "functions": "[...]"}

A call appears inline as ``<functioncall> {...} </functioncall>``. Hermes ships
the same content already as ChatML turns, so the two dialects differ only in
how a row is unpacked.

Everything here is defensive on purpose: this parses a third-party format that
could not be inspected while writing it, so unknown shapes are counted and
skipped rather than guessed at. A mis-parsed trajectory teaches the model
nonsense, which is worse than having fewer rows.

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

__all__ = ["DIALECTS", "load_rows", "convert", "convert_row", "main"]

DIALECTS = ["glaive", "hermes"]

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
# Tool specs
# --------------------------------------------------------------------------- #


def _normalize_tools(raw: Any) -> list[dict[str, Any]]:
    """Coerce the tools blob into the OpenAI ``{"type","function"}`` shape."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(raw, dict):
        # Some exports wrap the list in a key.
        for key in ("functions", "tools"):
            if key in raw:
                return _normalize_tools(raw[key])
        return []
    if not isinstance(raw, list):
        return []

    tools: list[dict[str, Any]] = []
    for tool in raw:
        if isinstance(tool, str):
            try:
                tool = json.loads(tool)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(tool, dict):
            continue
        tools.append(tool if "function" in tool else {"type": "function", "function": tool})
    return tools


# --------------------------------------------------------------------------- #
# Glaive chat string
# --------------------------------------------------------------------------- #


def _split_glaive_chat(chat: str) -> list[dict[str, Any]]:
    """Split the role-marked chat string into message dicts.

    A ``FUNCTION RESPONSE`` turn becomes a ``tool`` message; everything else
    maps through to user/assistant.
    """
    if not isinstance(chat, str) or not chat.strip():
        return []

    matches = list(_ROLE_RE.finditer(chat))
    if not matches:
        # No markers at all: treat the whole thing as one user turn.
        return [{"role": "user", "content": chat.strip()}]

    messages: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        role = _ROLE_MAP.get(match.group(1).lower())
        if role is None:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
        content = chat[start:end].strip()
        if content or role == "assistant":
            messages.append({"role": role, "content": content})
    return messages


# A whole inline call block -- tags *and* the JSON between them -- that must not
# survive into ``content`` once the call has been lifted into ``tool_calls``.
# Stripping only the tags would leave the payload behind as bare JSON, which
# still renders as a second, malformed call.
_CALL_BLOCK_RE = re.compile(
    rf"(?:{re.escape(CALL_OPEN)}|{re.escape(ALT_CALL_OPEN)}).*?"
    rf"(?:{re.escape(CALL_CLOSE)}|{re.escape(ALT_CALL_CLOSE)})",
    re.DOTALL | re.IGNORECASE,
)


def _extract_calls_from_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect assistant calls into the structured ``tool_calls`` field.

    Two shapes exist in the wild and both must work:

    * Glaive embeds the call inline in the assistant's text. It is parsed out
      and the markup is then stripped from ``content`` -- otherwise the
      renderer, which writes both the prose and the structured calls, would
      emit the same call twice and train the model to duplicate it.
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
        parsed = parse_tool_calls(content)
        if not parsed.calls:
            continue

        message["tool_calls"] = [call.to_dict() for call in parsed.calls]
        calls.extend(call.to_dict() for call in parsed.calls)

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


def convert_row(row: dict[str, Any], index: int, dialect: str) -> dict[str, Any] | None:
    """Convert one source row into a project record, or None if unusable."""
    if dialect == "glaive":
        messages = _split_glaive_chat(row.get("chat") or row.get("conversations") or "")
    elif dialect == "hermes":
        messages = _split_hermes_row(row)
    else:
        raise ValueError(f"unknown dialect: {dialect}")

    if not messages:
        return None

    tools = _normalize_tools(row.get("functions") or row.get("tools"))
    calls = _extract_calls_from_turns(messages)

    # A trajectory with no tool call at all is a no-tool example, not a
    # positive one. Leave it for the negative loader.
    if not calls:
        return None

    # Drop a leading system turn: our renderer supplies its own.
    if messages and messages[0]["role"] == "system":
        messages = messages[1:]
    if not messages:
        return None

    # The conversation must end on an assistant turn to be a training target.
    if messages[-1]["role"] != "assistant":
        return None

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
    """Convert rows, counting why anything was skipped."""
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for index, row in enumerate(rows):
        stats["seen"] += 1
        try:
            record = convert_row(row, index, dialect)
        except Exception:  # noqa: BLE001 - one bad row must not kill the run
            record = None
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
    print(f"wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()