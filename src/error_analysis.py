"""Categorise failures into the buckets an interviewer will ask about.

Reads the ``failures.jsonl`` / ``predictions.jsonl`` written by the evaluators
and assigns each failure exactly one category, so the counts add up:

    wrong_tool              called a function, but not the right one
    malformed_json          emitted a call block that would not parse
    no_call_emitted         answered in prose when a call was required
    wrong_argument_value    right function, one or more values wrong
    missing_argument        right function, a required argument omitted
    hallucinated_argument   right function, invented an argument value
    unnecessary_tool_call   called a tool when none was warranted
    should_request_info     gold asked for clarification; model did not
    unsupported_request     gold declined; model did not

``missing_argument`` vs ``hallucinated_argument`` vs ``wrong_argument_value``
are separated because they imply different fixes: schema-aware decoding,
abstention training, or better grounding respectively.

Usage:
    python -m src.error_analysis --results results --out results/error_analysis.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import param_triples, parse_tool_calls

__all__ = ["CATEGORIES", "classify_failure", "analyse", "main"]

CATEGORIES = [
    "wrong_tool",
    "malformed_json",
    "wrong_argument_value",
    "missing_argument",
    "hallucinated_argument",
    "unnecessary_tool_call",
    "should_request_info",
    "unsupported_request",
]

DEFAULT_EXAMPLES = 20


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tool_call_failure(row: dict[str, Any]) -> str:
    """Categorise a miss on a tool-call example."""
    parsed = parse_tool_calls(row.get("prediction") or "")
    gold = row.get("gold_calls") or []

    if parsed.malformed and not parsed.calls:
        return "malformed_json"
    if not parsed.calls:
        return "missing_argument"  # abstained entirely; closest single bucket

    gold_names = {call["name"] for call in gold}
    pred_names = {call.name for call in parsed.calls}
    if gold_names != pred_names:
        return "wrong_tool"

    gold_triples = set()
    for call in gold:
        for key, value in (call.get("arguments") or {}).items():
            gold_triples.add((call["name"], str(key), json.dumps(value, sort_keys=True, default=str)))

    gold_args = {key for call in gold for key in (call.get("arguments") or {})}
    pred_args = {key for call in parsed.calls for key in call.arguments}

    missing = gold_args - pred_args
    if missing:
        return "missing_argument"

    extra = pred_args - gold_args
    if extra:
        return "hallucinated_argument"

    return "wrong_argument_value"


def classify_failure(row: dict[str, Any], gold_decision: str | None = None) -> str:
    """Assign one category to a failing row from either eval track."""
    if gold_decision is not None:
        if gold_decision == "request_for_info":
            return "should_request_info"
        if gold_decision == "cannot_answer":
            return "unsupported_request"
        return "unnecessary_tool_call"

    if not row.get("expects_call"):
        return "unnecessary_tool_call"
    return _tool_call_failure(row)


def analyse(results_dir: Path, max_examples: int = DEFAULT_EXAMPLES) -> dict[str, Any]:
    """Collect up to ``max_examples`` representative failures and count them."""
    examples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    # --- tool-call track -------------------------------------------------- #
    for run in ("base", "tool_sft", "reliable_tool_sft"):
        for row in load_jsonl(results_dir / f"{run}_failures.jsonl"):
            category = classify_failure(row)
            counts[category] += 1
            if len(examples) < max_examples:
                examples.append({
                    "track": "tool_call",
                    "checkpoint": run,
                    "id": row.get("id"),
                    "split": row.get("split"),
                    "category": category,
                    "prediction": (row.get("prediction") or "")[:500],
                })

    # --- decision track --------------------------------------------------- #
    for run in ("base", "tool_sft", "reliable_tool_sft"):
        for row in load_jsonl(results_dir / f"{run}_when2call_failures.jsonl"):
            category = classify_failure(row, gold_decision=row.get("gold"))
            counts[category] += 1
            if len(examples) < max_examples:
                examples.append({
                    "track": "when2call",
                    "checkpoint": run,
                    "id": row.get("id"),
                    "gold": row.get("gold"),
                    "predicted": row.get("predicted"),
                    "category": category,
                    "prediction": (row.get("prediction") or "")[:500],
                })

    return {
        "counts": {category: counts.get(category, 0) for category in CATEGORIES},
        "total_failures_observed": sum(counts.values()),
        "n_examples": len(examples),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/error_analysis.json")
    parser.add_argument("--max-examples", type=int, default=DEFAULT_EXAMPLES)
    args = parser.parse_args()

    report = analyse(Path(args.results), max_examples=args.max_examples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"total failures observed: {report['total_failures_observed']}")
    for category in CATEGORIES:
        print(f"  {category:<24} {report['counts'][category]}")
    print(f"\n{report['n_examples']} examples -> {out_path}")


if __name__ == "__main__":
    main()