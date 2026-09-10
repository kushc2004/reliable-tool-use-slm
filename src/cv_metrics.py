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
            "when2call_false_tool_call_rate": decision.get("false_tool_call_rate"),
            "when2call_missing_info_accuracy": decision.get("missing_info_accuracy"),
            "n_tool_eval": overall.get("n"),
            "n_when2call": decision.get("n"),
        }

    # Headline deltas: base -> Reliable Tool-SFT.
    base = data["runs"]["base"]
    final = data["runs"]["reliable_tool_sft"]
    data["headline"] = {
        "exact_match_gain_abs": _delta(base.get("exact_match"), final.get("exact_match")),
        "exact_match_gain_rel": _relative(base.get("exact_match"), final.get("exact_match")),
        "false_call_drop_abs": _delta(
            base.get("when2call_false_tool_call_rate"),
            final.get("when2call_false_tool_call_rate"),
        ),
        "decision_accuracy_gain_abs": _delta(
            base.get("when2call_decision_accuracy"),
            final.get("when2call_decision_accuracy"),
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
        f"| Base false tool-call rate | "
        f"{_pct(runs['base'].get('when2call_false_tool_call_rate'))} |"
    )
    lines.append(
        f"| Reliable Tool-SFT false tool-call rate | "
        f"{_pct(runs['reliable_tool_sft'].get('when2call_false_tool_call_rate'))} |"
    )
    lines.append(
        f"| Absolute reduction in false tool calls | "
        f"{_pct(head.get('false_call_drop_abs'))} |"
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
        ("when2call_false_tool_call_rate", "When2Call false tool-call rate"),
        ("when2call_missing_info_accuracy", "Missing-info accuracy"),
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
    lines.append("Numbers are read directly from `results/*.json`. Fill in GPU model, "
                 "wall-clock training time and peak memory from your own run log — those "
                 "are not recorded by these scripts.")
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