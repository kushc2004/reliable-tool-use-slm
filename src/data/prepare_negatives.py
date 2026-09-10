"""Load should-not-call examples, primarily from NVIDIA When2Call.

When2Call is built around the decision of whether to invoke a function at all,
which is exactly the axis this project adds on top of a standard tool-call SFT
run. Records from it land in one of two shapes:

1. The row already says "no call" - it has no tool calls and a plain-text
   assistant turn. That becomes a ``no_tool_needed`` record.
2. The row asks for a function that is not among the advertised ones. That
   becomes an ``irrelevant_tool`` record: the tools are listed, none of them
   fits, and the correct behaviour is to say so rather than call the nearest
   one. This is the tool-hallucination case.

Rows whose gold label we cannot determine are skipped and counted, never
guessed at - a mislabelled negative teaches the model to refuse legitimate
calls, which is worse than having fewer negatives.

Usage:
    python -m src.data.prepare_negatives --input nvidia/When2Call \\
        --out data/raw/when2call.jsonl --limit 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ..schema import parse_tool_calls

__all__ = ["convert_when2call", "load_rows", "main"]

_FIELD_CANDIDATES = {
    "messages": ["messages", "conversations"],
    "tools": ["tools", "functions"],
    "answer": ["answer", "label", "should_call", "expected"],
    "query": ["query", "question", "prompt", "instruction"],
}


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def load_rows(source: str, limit: int | None = None, split: str = "train") -> list[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset  # lazy: needs network

        dataset = load_dataset(source, split=split)
        rows = [dict(row) for row in dataset]
    return rows[:limit] if limit else rows


def _normalize_tools(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
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


def _normalize_messages(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return [{"role": "user", "content": raw}]
    messages: list[dict[str, Any]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or "user"
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = turn.get("content") or turn.get("value") or ""
        messages.append({"role": role, "content": content})
    return messages


def convert_when2call(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Convert one When2Call row into a no-tool record, or return None."""
    tools = _normalize_tools(_first_present(row, _FIELD_CANDIDATES["tools"]))
    messages = _normalize_messages(_first_present(row, _FIELD_CANDIDATES["messages"]))

    if not messages:
        query = _first_present(row, _FIELD_CANDIDATES["query"])
        if not query:
            return None
        messages = [{"role": "user", "content": str(query)}]

    answer = _first_present(row, _FIELD_CANDIDATES["answer"])

    # Any assistant turn that already contains a call means this is a positive
    # example, not a negative one. Leave it for the positive loader.
    for message in messages:
        if message["role"] == "assistant" and parse_tool_calls(message.get("content") or "").calls:
            return None

    # Decide the category from the label if there is one, otherwise infer.
    calls_expected = None
    if isinstance(answer, bool):
        calls_expected = answer
    elif isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered in {"no", "false", "none", "no_call", "decline", "irrelevant"}:
            calls_expected = False
        elif lowered in {"yes", "true", "call", "invoke"}:
            calls_expected = True

    if calls_expected:
        return None

    # Distinguish "no tool was needed" from "the tools on offer do not fit".
    # A row that names a function which is not advertised is the second case.
    advertised = {
        tool["function"].get("name")
        for tool in tools
        if isinstance(tool.get("function"), dict)
    }
    requested = row.get("required_function") or row.get("function_name")
    category = "no_tool_needed"
    if requested and advertised and requested not in advertised:
        category = "irrelevant_tool"
    elif tools and not advertised:
        return None

    assistant = next(
        (m["content"] for m in messages if m["role"] == "assistant" and m.get("content")),
        "",
    )
    if not assistant:
        assistant = (
            "None of the available functions can help with that."
            if category == "irrelevant_tool"
            else "I can answer that without calling a function."
        )
        messages = messages + [{"role": "assistant", "content": assistant, "tool_calls": []}]
    else:
        for message in messages:
            if message["role"] == "assistant":
                message["tool_calls"] = []
                break

    return {
        "id": f"when2call-{index:06d}",
        "split": "train",
        "category": category,
        "source": "when2call",
        "tools": tools,
        "messages": messages,
        "gold_calls": [],
        "expects_call": False,
    }


def convert(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {"seen": 0, "kept": 0, "skipped": 0, "no_tool_needed": 0, "irrelevant_tool": 0}
    for index, row in enumerate(rows):
        stats["seen"] += 1
        try:
            record = convert_when2call(row, index)
        except Exception:  # noqa: BLE001
            record = None
        if record is None:
            stats["skipped"] += 1
            continue
        records.append(record)
        stats["kept"] += 1
        stats[record["category"]] += 1
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="nvidia/When2Call")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    rows = load_rows(args.input, limit=args.limit, split=args.split)
    records, stats = convert(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[when2call] seen={stats['seen']} kept={stats['kept']} skipped={stats['skipped']}")
    print(f"           no_tool_needed={stats['no_tool_needed']} "
          f"irrelevant_tool={stats['irrelevant_tool']}")
    print(f"wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()