"""Four-way tool-decision scoring for When2Call.

Given a free-text completion, classify what the model *decided*:

    tool_call / direct / request_for_info / cannot_answer

and compare that against the gold label. ``tool_call`` is decided by the parser
in :mod:`src.schema` and is unambiguous -- either a well-formed call block was
emitted or it was not. The other three are prose, so they are separated by
ordered cue matching. That is a heuristic, and the module says so out loud:
``classification_source`` is recorded per example so a wrong heuristic call can
be audited rather than silently believed.

Reported metrics:

    decision_accuracy       four-way accuracy
    tool_call_precision     of the calls it made, how many were warranted
    tool_call_recall        of the warranted calls, how many it made
    false_tool_call_rate    non-tool rows where it emitted a call anyway
    missing_info_accuracy   on request_for_info rows, asked instead of guessing
    cannot_answer_accuracy  on cannot_answer rows, declined instead of inventing
    confusion_matrix        4x4, rows = gold, cols = predicted
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .schema import parse_tool_calls

__all__ = [
    "DECISIONS",
    "classify_prediction",
    "score_decision",
    "confusion_matrix",
    "aggregate_decisions",
    "format_decision_table",
]

DECISIONS = ["tool_call", "direct", "request_for_info", "cannot_answer"]

# Ordered: the first list that matches wins. Tool calls are handled separately
# and always take priority, because an emitted call is a decision no prose can
# contradict.
# Argument-shaped nouns. Used to recognise "what is the <arg> ..." as a
# clarifying question. Deliberately a closed list rather than a bare
# ``what is the \w+``: the latter also fires on "What is the air quality in
# Hanoi?", which is a cannot_answer row, and would trade one bug for another.
_ARG_NOUNS = (
    r"invoice|order|booking|parcel|tracking|service|company|domain|restaurant|"
    r"city|date|time|currency|language|artist|account|customer|reference|"
    r"identifier|id|ticker|amount|recipient|location|project|title"
)

_REQUEST_FOR_INFO = [
    r"\bcould you\b",
    r"\bcan you (?:please )?(?:specify|clarify|provide|tell|confirm)\b",
    r"\bplease (?:specify|provide|clarify|confirm|tell me|let me know)\b",
    r"\bi(?:'d| would)? need (?:more|additional|a few more) (?:information|details|info)\b",
    r"\bto (?:help|assist) you (?:better|further)\b",
    r"\bwhich (?:one|city|company|domain|restaurant|service|date|time|currency|language|artist)\b",
    r"\bwhat (?:city|company|domain|restaurant|service|date|time|currency|language|artist)\b",
    r"\bcould you (?:provide|specify|tell me)\b",
    # "what is the invoice identifier you would like me to look up?"
    rf"\bwhat (?:is|are) (?:the|your|a|an) (?:{_ARG_NOUNS})\b",
    r"\bwould you like me to\b",
    r"\blet me know\b",
    r"\bwhich one\b",
    # "what should I remind you about, and when?"
    r"\bwhat should i\b",
    r"\bwhat would you like\b",
    r"\bwhich (?:one|of those) (?:would|do) you\b",
]

_CANNOT_ANSWER = [
    r"\bi(?:'m| am) (?:unable|not able)\b",
    r"\bi (?:can(?:'t|not)|cannot)\b",
    r"\bi (?:don(?:'t| not)|do not) have (?:access|the ability|enough information)\b",
    r"\bnot (?:something )?i can (?:help|assist) with\b",
    r"\bno (?:available|provided) (?:function|tool)\b",
    r"\bnone of the (?:available )?(?:functions|tools)\b",
    r"\boutside (?:my|the) (?:capabilities|scope)\b",
    r"\bi (?:lack|do not have) the (?:information|data|ability)\b",
]

_RE_REQUEST = re.compile("|".join(_REQUEST_FOR_INFO), re.IGNORECASE)
_RE_CANNOT = re.compile("|".join(_CANNOT_ANSWER), re.IGNORECASE)


def classify_prediction(text: str) -> tuple[str, str]:
    """Map a completion onto one of the four decisions.

    Returns ``(decision, source)`` where ``source`` is one of ``tool_call``,
    ``request_cue``, ``cannot_cue`` or ``default_direct``, so the report can
    show how much of the score rests on the heuristic.
    """
    if not text or not text.strip():
        # An empty completion is a refusal to engage, not a direct answer.
        return "cannot_answer", "empty"

    parsed = parse_tool_calls(text)
    if parsed.calls or parsed.n_blocks:
        return "tool_call", "tool_call"

    # Strip the call syntax before cue matching: a malformed block should not
    # also make the surrounding prose look like a question.
    prose = re.sub(r"</?(?:tool_call|functioncall)>", " ", text, flags=re.IGNORECASE)

    if _RE_REQUEST.search(prose):
        return "request_for_info", "request_cue"
    if _RE_CANNOT.search(prose):
        return "cannot_answer", "cannot_cue"
    return "direct", "default_direct"


def score_decision(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    """Score one When2Call prediction."""
    gold = record.get("decision")
    predicted, source = classify_prediction(prediction)
    parsed = parse_tool_calls(prediction)

    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "gold": gold,
        "predicted": predicted,
        "classification_source": source,
        "correct": gold == predicted,
        "emitted_call": bool(parsed.calls or parsed.n_blocks),
        "json_valid": parsed.json_valid,
        "n_predicted_calls": len(parsed.calls),
        "target_tool": record.get("target_tool"),
        "predicted_tool_names": [call.name for call in parsed.calls],
        "prediction": prediction,
    }


def confusion_matrix(scored: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """4x4 counts, ``matrix[gold][predicted]``."""
    matrix = {gold: {pred: 0 for pred in DECISIONS} for gold in DECISIONS}
    for row in scored:
        gold, predicted = row.get("gold"), row.get("predicted")
        if gold in matrix and predicted in matrix[gold]:
            matrix[gold][predicted] += 1
    return matrix


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def aggregate_decisions(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """All decision metrics, plus the confusion matrix and cue provenance."""
    if not scored:
        return {"n": 0, "confusion_matrix": confusion_matrix([])}

    gold_calls = [row for row in scored if row["gold"] == "tool_call"]
    gold_non_calls = [row for row in scored if row["gold"] != "tool_call"]
    predicted_calls = [row for row in scored if row["predicted"] == "tool_call"]

    gold_request = [row for row in scored if row["gold"] == "request_for_info"]
    gold_cannot = [row for row in scored if row["gold"] == "cannot_answer"]

    cue_sources = {}
    for row in scored:
        cue_sources[row["classification_source"]] = cue_sources.get(row["classification_source"], 0) + 1

    return {
        "n": len(scored),
        "decision_accuracy": _mean([1.0 if row["correct"] else 0.0 for row in scored]),
        "macro_f1": _macro_f1(scored),
        "tool_call_precision": _rate(
            sum(1 for row in predicted_calls if row["gold"] == "tool_call"), len(predicted_calls)
        ),
        "tool_call_recall": _rate(
            sum(1 for row in gold_calls if row["predicted"] == "tool_call"), len(gold_calls)
        ),
        # The headline reliability number: how often the model acts when it
        # should not.
        "false_tool_call_rate": _rate(
            sum(1 for row in gold_non_calls if row["predicted"] == "tool_call"), len(gold_non_calls)
        ),
        "missing_info_accuracy": _mean(
            [1.0 if row["predicted"] == "request_for_info" else 0.0 for row in gold_request]
        ),
        "cannot_answer_accuracy": _mean(
            [1.0 if row["predicted"] == "cannot_answer" else 0.0 for row in gold_cannot]
        ),
        "direct_accuracy": _mean(
            [1.0 if row["predicted"] == "direct" else 0.0 for row in scored if row["gold"] == "direct"]
        ),
        "json_validity": _mean([1.0 if row["json_valid"] else 0.0 for row in scored]),
        "n_gold_tool_call": len(gold_calls),
        "n_gold_non_call": len(gold_non_calls),
        "classification_source_counts": cue_sources,
        "confusion_matrix": confusion_matrix(scored),
    }


def _macro_f1(scored: Sequence[dict[str, Any]]) -> float | None:
    """Unweighted mean of per-class F1, so rare categories still count."""
    scores: list[float] = []
    for label in DECISIONS:
        tp = sum(1 for row in scored if row["gold"] == label and row["predicted"] == label)
        fp = sum(1 for row in scored if row["gold"] != label and row["predicted"] == label)
        fn = sum(1 for row in scored if row["gold"] == label and row["predicted"] != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return _mean(scores)


def format_decision_table(metrics: dict[str, Any]) -> str:
    """Markdown summary, including the confusion matrix."""
    rows = [
        ("decision_accuracy", "Decision accuracy"),
        ("macro_f1", "Macro F1"),
        ("tool_call_precision", "Tool-call precision"),
        ("tool_call_recall", "Tool-call recall"),
        ("false_tool_call_rate", "False tool-call rate"),
        ("missing_info_accuracy", "Missing-info accuracy"),
        ("cannot_answer_accuracy", "Cannot-answer accuracy"),
        ("direct_accuracy", "Direct-answer accuracy"),
        ("json_validity", "JSON validity"),
    ]
    lines = ["| Metric | Value |", "|---|---|"]
    for key, label in rows:
        value = metrics.get(key)
        lines.append(f"| {label} | {'n/a' if value is None else f'{value * 100:.1f}%'} |")

    matrix = metrics.get("confusion_matrix") or {}
    if matrix:
        lines.append("")
        lines.append("| Gold \\ Pred | " + " | ".join(DECISIONS) + " |")
        lines.append("|---" * (len(DECISIONS) + 1) + "|")
        for gold in DECISIONS:
            cells = [str(matrix.get(gold, {}).get(pred, 0)) for pred in DECISIONS]
            lines.append(f"| **{gold}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def score_all(pairs: Iterable[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    return [score_decision(record, prediction) for record, prediction in pairs]