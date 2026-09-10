"""Load NVIDIA When2Call into the two shapes this project needs.

When2Call is a *decision* benchmark, not a call-accuracy benchmark. Every row
has one of four correct behaviours:

    tool_call         sufficient info + a tool that fits -> call it
    direct            no tool needed -> just answer
    request_for_info  the right tool exists but an argument is missing ->
                      ask for it instead of inventing a value
    cannot_answer     nothing on offer can satisfy the request -> say so

That four-way label is what makes this project different from a plain
function-calling SFT run, so it is preserved verbatim rather than collapsed
into a binary should-call / should-not-call bit.

Two modes:

    --mode train   balanced subset -> SFT records (user + gold assistant turn)
    --mode eval    held-out test   -> records for four-way scoring

Usage:
    python -m src.data.prepare_when2call --mode train --out data/raw/w2c_train.jsonl
    python -m src.data.prepare_when2call --mode eval  --out data/raw/w2c_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..schema import parse_tool_calls

__all__ = ["DECISIONS", "convert_row", "balance", "load_rows", "main"]

DECISIONS = ["tool_call", "direct", "request_for_info", "cannot_answer"]
CALLING_DECISIONS = {"tool_call"}
NON_CALLING_DECISIONS = {"direct", "request_for_info", "cannot_answer"}

DEFAULT_HUB = "nvidia/When2Call"


def _as_obj(value: Any) -> Any:
    """When2Call stores tool specs and answer dicts as JSON strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _normalize_tools(raw: Any) -> list[dict[str, Any]]:
    raw = _as_obj(raw)
    if not raw or not isinstance(raw, list):
        return []
    tools: list[dict[str, Any]] = []
    for tool in raw:
        tool = _as_obj(tool)
        if not isinstance(tool, dict):
            continue
        tools.append(tool if "function" in tool else {"type": "function", "function": tool})
    return tools


def load_rows(source: str, limit: int | None = None, split: str = "test") -> list[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset  # lazy: needs network

        rows = [dict(row) for row in load_dataset(source, split=split)]
    return rows[:limit] if limit else rows


def _gold_calls(row: dict[str, Any], answers: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover the gold call for a ``tool_call`` row.

    The MCQ answer text for the tool_call category is the call itself, so it can
    be parsed back out. ``target_tool`` is the fallback when the text will not
    parse, which keeps function-name scoring possible even then.
    """
    parsed = parse_tool_calls(str(answers.get("tool_call") or ""))
    if parsed.calls:
        return [call.to_dict() for call in parsed.calls]
    name = row.get("target_tool")
    if isinstance(name, str) and name.strip():
        return [{"name": name.strip(), "arguments": {}}]
    return []


def convert_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Convert one When2Call row into a project record, or None if unusable."""
    question = row.get("question") or row.get("orig_question")
    if not question:
        return None

    label = (row.get("correct_answer") or "").strip()
    if label not in DECISIONS:
        return None

    answers = _as_obj(row.get("answers")) or {}
    if not isinstance(answers, dict):
        answers = {}

    tools = _normalize_tools(row.get("tools"))
    calls = _gold_calls(row, answers) if label == "tool_call" else []

    gold_response = str(answers.get(label) or "").strip()

    return {
        "id": str(row.get("uuid") or f"w2c-{index:06d}"),
        "split": "when2call",
        "decision": label,
        "source": row.get("source"),
        "source_id": row.get("source_id"),
        "tools": tools,
        "messages": [{"role": "user", "content": str(question)}],
        "gold_calls": calls,
        "expects_call": label == "tool_call",
        "gold_response": gold_response,
        "answers": {key: str(answers.get(key) or "") for key in DECISIONS},
        "target_tool": row.get("target_tool"),
        "held_out_param": row.get("held_out_param"),
    }


def convert(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for index, row in enumerate(rows):
        stats["seen"] += 1
        try:
            record = convert_row(row, index)
        except Exception:  # noqa: BLE001 - a bad row must not kill the run
            record = None
        if record is None:
            stats["skipped"] += 1
            continue
        records.append(record)
        stats["kept"] += 1
        stats[record["decision"]] += 1
    return records, dict(stats)


def balance(records: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Take a roughly class-balanced subset of ``n`` records.

    Balance matters here: the four categories are the experiment. A corpus that
    is 80% ``tool_call`` would teach the model to call a tool by default, which
    is precisely the failure mode being measured.
    """
    rng = random.Random(seed)
    by_decision: dict[str, list[dict[str, Any]]] = {key: [] for key in DECISIONS}
    for record in records:
        by_decision[record["decision"]].append(record)

    for bucket in by_decision.values():
        rng.shuffle(bucket)

    per_class = max(1, n // len(DECISIONS))
    picked: list[dict[str, Any]] = []
    shortfall = 0
    for key in DECISIONS:
        bucket = by_decision[key]
        take = min(per_class, len(bucket))
        shortfall += per_class - take
        picked.extend(bucket[:take])

    # Redistribute any shortfall into classes that still have rows left, so the
    # subset still lands near the requested size when one class is thin.
    if shortfall:
        for key in DECISIONS:
            extra = by_decision[key][per_class : per_class + shortfall]
            picked.extend(extra)
            shortfall -= len(extra)
            if shortfall <= 0:
                break

    rng.shuffle(picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_HUB)
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=1000, help="train mode: target subset size")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split = args.split or ("test" if args.mode == "eval" else "train")
    rows = load_rows(args.input, limit=args.limit, split=split)
    records, stats = convert(rows)

    if args.mode == "train":
        records = balance(records, args.n, seed=args.seed)
        for record in records:
            record["split"] = "train"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(record["decision"] for record in records)
    print(f"[when2call:{args.mode}] split={split} seen={stats.get('seen', 0)} "
          f"kept={stats.get('kept', 0)} skipped={stats.get('skipped', 0)}")
    print(f"           wrote {len(records)} records: {dict(counts)}")
    print(f"           -> {out_path}")


if __name__ == "__main__":
    main()