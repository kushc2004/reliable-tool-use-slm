"""The six metrics, plus per-split breakdowns.

All six are derived from one pass over ``(record, prediction_text)`` pairs, so
adding a metric never means re-running generation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from .schema import (
    ParseResult,
    ToolCall,
    canon_calls,
    call_name_set,
    param_triples,
    parse_tool_calls,
)

__all__ = ["METRIC_NAMES", "score_one", "compute_metrics", "format_table"]

METRIC_NAMES = [
    "json_validity",
    "tool_selection_accuracy",
    "argument_accuracy",
    "exact_match",
    "unseen_function_accuracy",
    "false_tool_call_rate",
]


def _gold_calls(record: dict[str, Any]) -> list[ToolCall]:
    return [
        ToolCall(name=call["name"], arguments=call.get("arguments") or {})
        for call in record.get("gold_calls") or []
    ]


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def score_one(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    """Score a single prediction against its record.

    Returns per-example booleans and partial scores. Aggregation happens in
    :func:`compute_metrics`; keeping them separate means a failing example can
    be dumped verbatim for inspection.
    """
    parsed: ParseResult = parse_tool_calls(prediction)
    gold = _gold_calls(record)
    expects_call = bool(record.get("expects_call", bool(gold)))
    n_predicted = len(parsed.calls)

    selection_correct = call_name_set(parsed.calls) == call_name_set(gold)
    exact = canon_calls(parsed.calls) == canon_calls(gold) and n_predicted == len(gold)

    if expects_call and selection_correct and gold:
        argument_score = _jaccard(param_triples(gold), param_triples(parsed.calls))
    elif expects_call:
        argument_score = 0.0
    else:
        argument_score = None  # not applicable

    return {
        "id": record.get("id"),
        "split": record.get("split", "unknown"),
        "expects_call": expects_call,
        "json_valid": parsed.json_valid,
        "n_predicted": n_predicted,
        "n_gold": len(gold),
        "selection_correct": selection_correct if expects_call else None,
        "argument_score": argument_score,
        "exact_match": exact if expects_call else None,
        "emitted_call_when_none_expected": (not expects_call) and n_predicted > 0,
        "correctly_abstained": (not expects_call) and n_predicted == 0,
        "malformed": parsed.malformed,
        "prediction": prediction,
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        return {"n": 0}

    call_cases = [row for row in scored if row["expects_call"]]
    no_tool_cases = [row for row in scored if not row["expects_call"]]

    selection = [1.0 if row["selection_correct"] else 0.0 for row in call_cases]
    arguments = [row["argument_score"] for row in call_cases if row["argument_score"] is not None]
    exact = [1.0 if row["exact_match"] else 0.0 for row in call_cases]

    return {
        "n": len(scored),
        "n_call_expected": len(call_cases),
        "n_no_tool": len(no_tool_cases),
        # Metric 1 -- every emitted block parsed.
        "json_validity": _mean([1.0 if row["json_valid"] else 0.0 for row in scored]),
        # Metric 2 -- right function name(s).
        "tool_selection_accuracy": _mean(selection),
        # Metric 3 -- right parameters on correctly selected calls.
        "argument_accuracy": _mean(arguments),
        # Metric 4 -- entire call set identical.
        "exact_match": _mean(exact),
        # Metric 5 -- exact match on the held-out-function split only.
        "unseen_function_accuracy": _mean(
            [
                row["exact_match"] and 1.0 or 0.0
                for row in call_cases
                if row["split"] == "unseen_functions"
            ]
        ),
        # Metric 6 -- called a tool when none was warranted.
        "false_tool_call_rate": _mean(
            [1.0 if row["emitted_call_when_none_expected"] else 0.0 for row in no_tool_cases]
        ),
        # Supporting signals.
        "no_tool_accuracy": _mean(
            [1.0 if row["correctly_abstained"] else 0.0 for row in no_tool_cases]
        ),
        "call_recall": _mean(
            [1.0 if row["n_predicted"] > 0 else 0.0 for row in call_cases]
        ),
        "call_precision": _mean(
            [
                1.0 if row["n_predicted"] == row["n_gold"] else 0.0
                for row in call_cases
                if row["n_predicted"] > 0
            ]
        ),
        "malformed_blocks": sum(len(row["malformed"]) for row in scored),
        "examples": scored,
    }


def compute_metrics(pairs: Iterable[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    """Score every ``(record, prediction)`` pair, overall and per split."""
    scored = [score_one(record, prediction) for record, prediction in pairs]

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_split[row["split"]].append(row)

    overall = _aggregate(scored)
    overall.pop("examples", None)
    result = {"overall": overall}
    result["by_split"] = {
        split: {key: value for key, value in _aggregate(rows).items() if key != "examples"}
        for split, rows in sorted(by_split.items())
    }
    result["examples"] = scored
    return result


def format_table(metrics: dict[str, Any]) -> str:
    """Render the six headline metrics as a markdown table."""
    header = "| Metric | Value |"
    lines = [header, "|---|---|"]
    for name in METRIC_NAMES:
        value = metrics.get("overall", {}).get(name)
        lines.append(f"| `{name}` | {'n/a' if value is None else f'{value:.4f}'} |")
    return "\n".join(lines)