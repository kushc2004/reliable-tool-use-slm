"""Emit the one-page CV summary, computed from the actual result files.

Every number here is read from ``results/*.json``. Nothing is hardcoded, and
nothing is copied from the upstream repository. If a full run has not happened
yet the field is reported as null rather than guessed at, because a fabricated
CV bullet is worse than a missing one.

Usage:
    python -m src.cv_metrics --results results --out results/cv_metrics.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["collect", "render", "main"]

RUNS = ["base", "tool_sft", "reliable_tool_sft"]
LABELS = {"base": "Base", "tool_sft": "Tool-SFT", "reliable_tool_sft": "Reliable Tool-SFT"}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def collect(results_dir: Path) -> dict[str, Any]:
    """Pull the CV-relevant numbers out of the result files."""
    data: dict[str, Any] = {"runs": {}}

    for run in RUNS:
        tool = _load(results_dir / f"{run}_metrics.json") or {}
        decision = _load(results_dir / f"{run}_when2call.json") or {}
        overall = tool.get("overall", {})
        data["runs"][run] = {
            "exact_match": overall.get("exact_match"),
            "tool_selection_accuracy": overall.get("tool_selection_accuracy"),
            "argument_accuracy": overall.get("argument_accuracy"),
            "json_validity": overall.get("json_validity"),
            "unseen_function_accuracy": overall.get("unseen_function_accuracy"),
            "false_tool_call_rate": overall.get("false_tool_call_rate"),
            "when2call_decision_accuracy": decision.get("decision_accuracy"),
            "when2call_macro_f1": decision.get("macro_f1"),
            "when2call_tool_call_precision": decision.get("tool_call_precision"),
            "when2call_tool_call_recall": decision.get("tool_call_recall"),
            "when2call_false_tool_call_rate": decision.get("false_tool_call_rate"),
            "when2call_missing_info_accuracy": decision.get("missing_info_accuracy"),
            "when2call_cannot_answer_accuracy": decision.get("cannot_answer_accuracy"),
            "n_tool_eval": overall.get("n"),
            "n_when2call": decision.get("n"),
        }

    # Headline deltas. Base -> Reliable shows what post-training buys overall;
    # Tool-SFT -> Reliable isolates the effect of adding negative/abstention
    # supervision. Comparing false-call rate against Base is misleading because
    # Base almost never calls a tool, so its 0% false-call rate is degenerate.
    base = data["runs"]["base"]
    tool = data["runs"]["tool_sft"]
    final = data["runs"]["reliable_tool_sft"]
    tool_false = tool.get("when2call_false_tool_call_rate")
    final_false = final.get("when2call_false_tool_call_rate")
    false_drop = None if tool_false is None or final_false is None else tool_false - final_false
    false_drop_rel = None if not tool_false or final_false is None else false_drop / tool_false
    data["headline"] = {
        "exact_match_gain_abs": _delta(base.get("exact_match"), final.get("exact_match")),
        "exact_match_gain_rel": _relative(base.get("exact_match"), final.get("exact_match")),
        "tool_to_reliable_exact_match_change_abs": _delta(
            tool.get("exact_match"), final.get("exact_match")
        ),
        "tool_to_reliable_false_call_drop_abs": false_drop,
        "tool_to_reliable_false_call_drop_rel": false_drop_rel,
        "tool_to_reliable_decision_accuracy_gain_abs": _delta(
            tool.get("when2call_decision_accuracy"),
            final.get("when2call_decision_accuracy"),
        ),
        "tool_to_reliable_missing_info_gain_abs": _delta(
            tool.get("when2call_missing_info_accuracy"),
            final.get("when2call_missing_info_accuracy"),
        ),
        "tool_to_reliable_cannot_answer_gain_abs": _delta(
            tool.get("when2call_cannot_answer_accuracy"),
            final.get("when2call_cannot_answer_accuracy"),
        ),
    }

    # Trainable-parameter accounting, written by the trainer if it ran.
    for run in ("tool_sft", "reliable_tool_sft"):
        cfg = _load(results_dir.parent / "outputs" / run / "run_config.json")
        if cfg:
            data.setdefault("training", {})[run] = cfg

    stats = _load(results_dir.parent / "data" / "processed" / "stats.json")
    if stats:
        data["data"] = stats
    return data


def _delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _relative(before: Any, after: Any) -> float | None:
    if not before or after is None:
        return None
    return (after - before) / before


def render(data: dict[str, Any]) -> str:
    runs = data["runs"]
    head = data.get("headline", {})
    lines = ["# CV metrics (computed from `results/`)", ""]

    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Base exact tool-call accuracy | {_pct(runs['base'].get('exact_match'))} |")
    lines.append(
        f"| Reliable Tool-SFT exact tool-call accuracy | "
        f"{_pct(runs['reliable_tool_sft'].get('exact_match'))} |"
    )
    lines.append(f"| Absolute improvement | {_pct(head.get('exact_match_gain_abs'))} |")
    rel = head.get("exact_match_gain_rel")
    lines.append(f"| Relative improvement | {'n/a' if rel is None else f'{rel * 100:+.1f}%'} |")
    lines.append(
        f"| Held-out (unseen) function accuracy | "
        f"{_pct(runs['reliable_tool_sft'].get('unseen_function_accuracy'))} |"
    )
    lines.append(
        f"| Tool-SFT false tool-call rate | "
        f"{_pct(runs['tool_sft'].get('when2call_false_tool_call_rate'))} |"
    )
    lines.append(
        f"| Reliable Tool-SFT false tool-call rate | "
        f"{_pct(runs['reliable_tool_sft'].get('when2call_false_tool_call_rate'))} |"
    )
    lines.append(
        f"| False tool-call reduction (Tool-SFT → Reliable) | "
        f"{_pct(head.get('tool_to_reliable_false_call_drop_abs'))} |"
    )
    false_rel = head.get("tool_to_reliable_false_call_drop_rel")
    lines.append(
        f"| Relative false tool-call reduction | "
        f"{'n/a' if false_rel is None else f'{false_rel * 100:.1f}%'} |"
    )
    lines.append(
        f"| Decision-accuracy gain (Tool-SFT → Reliable) | "
        f"{_pct(head.get('tool_to_reliable_decision_accuracy_gain_abs'))} |"
    )
    lines.append("")

    lines.append("## Full comparison")
    lines.append("")
    lines.append("| Metric | Base | Tool-SFT | Reliable Tool-SFT |")
    lines.append("|---|---:|---:|---:|")
    rows = [
        ("exact_match", "Exact tool-call match"),
        ("tool_selection_accuracy", "Function-name accuracy"),
        ("argument_accuracy", "Argument accuracy"),
        ("json_validity", "JSON validity"),
        ("unseen_function_accuracy", "Held-out function EM"),
        ("when2call_decision_accuracy", "When2Call decision accuracy"),
        ("when2call_macro_f1", "When2Call macro F1"),
        ("when2call_tool_call_precision", "Tool-call precision"),
        ("when2call_tool_call_recall", "Tool-call recall"),
        ("when2call_false_tool_call_rate", "When2Call false tool-call rate"),
        ("when2call_missing_info_accuracy", "Missing-info accuracy"),
        ("when2call_cannot_answer_accuracy", "Cannot-answer accuracy"),
    ]
    for key, label in rows:
        cells = [_pct(runs[run].get(key)) for run in RUNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    counts = [
        ("n_tool_eval", "Tool-call eval examples"),
        ("n_when2call", "When2Call eval examples"),
    ]
    lines.append("| Count | Base | Tool-SFT | Reliable Tool-SFT |")
    lines.append("|---|---:|---:|---:|")
    for key, label in counts:
        cells = [str(runs[run].get(key) or "n/a") for run in RUNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    training = data.get("training")
    if training:
        lines.append("## Training")
        lines.append("")
        for run, cfg in training.items():
            lines.append(f"- **{LABELS.get(run, run)}**: {cfg.get('n_records')} records, "
                         f"variant `{cfg.get('variant')}`")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Numbers are read directly from `results/*.json`. Base's 0% false-call "
                 "rate is intentionally not used as a reliability headline because Base "
                 "almost never calls tools; the meaningful negative-supervision comparison "
                 "is Tool-SFT → Reliable Tool-SFT. GPU model, wall-clock training time and "
                 "peak memory should be taken from the run log when not present in run_config.json.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/cv_metrics.md")
    args = parser.parse_args()

    data = collect(Path(args.results))
    text = render(data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")

    with (Path(args.results) / "cv_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    print(text)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()