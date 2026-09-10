"""Compare checkpoints into one table.

The comparison this project is built around is three rows:

    base            no training
    tool-sft        positive tool-call supervision
    tool-sft-neg    positive + negative supervision

Usage:
    python -m src.report --runs base=runs/base/metrics.json \\
                               tool-sft=runs/sft/metrics.json \\
                               tool-sft-neg=runs/sft-neg/metrics.json \\
                         --out runs/REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .metrics import METRIC_NAMES

__all__ = ["load_metrics", "render_report", "main"]

# For every metric except false_tool_call_rate, higher is better.
_LOWER_IS_BETTER = {"false_tool_call_rate"}

_LABELS = {
    "json_validity": "JSON validity",
    "tool_selection_accuracy": "Tool selection acc.",
    "argument_accuracy": "Argument acc.",
    "exact_match": "Exact tool-call match",
    "unseen_function_accuracy": "Unseen-function acc.",
    "false_tool_call_rate": "False tool-call rate",
}


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _delta(before: float | None, after: float | None, lower_is_better: bool) -> str:
    if before is None or after is None:
        return "n/a"
    change = (after - before) * 100
    if abs(change) < 0.05:
        return "0.0"
    good = change < 0 if lower_is_better else change > 0
    return f"{'+' if change > 0 else ''}{change:.1f} {'✅' if good else '❌'}"


def render_report(runs: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = ["# Checkpoint comparison", ""]

    header = "| Metric | " + " | ".join(name for name, _ in runs) + " |"
    lines.append(header)
    lines.append("|---" * (len(runs) + 1) + "|")

    for metric in METRIC_NAMES:
        label = _LABELS.get(metric, metric)
        cells = [_cell(metrics.get("overall", {}).get(metric)) for _, metrics in runs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Change vs. the first checkpoint")
    lines.append("")
    baseline_name, baseline = runs[0]
    for name, metrics in runs[1:]:
        lines.append(f"### {name} (vs {baseline_name})")
        lines.append("")
        lines.append("| Metric | Change |")
        lines.append("|---|---|")
        for metric in METRIC_NAMES:
            lines.append(
                f"| {_LABELS.get(metric, metric)} | "
                f"{_delta(baseline.get('overall', {}).get(metric), metrics.get('overall', {}).get(metric), metric in _LOWER_IS_BETTER)} |"
            )
        lines.append("")

    lines.append("## Per split")
    lines.append("")
    splits = sorted({split for _, metrics in runs for split in metrics.get("by_split", {})})
    for split in splits:
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| Metric | " + " | ".join(name for name, _ in runs) + " |")
        lines.append("|---" * (len(runs) + 1) + "|")
        for metric in METRIC_NAMES:
            cells = [
                _cell(metrics.get("by_split", {}).get(split, {}).get(metric))
                for _, metrics in runs
            ]
            lines.append(f"| {_LABELS.get(metric, metric)} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="one or more name=path/to/metrics.json pairs, in comparison order")
    parser.add_argument("--out", default="runs/REPORT.md")
    args = parser.parse_args()

    runs: list[tuple[str, dict[str, Any]]] = []
    for item in args.runs:
        if "=" not in item:
            raise SystemExit(f"expected name=path, got {item!r}")
        name, path = item.split("=", 1)
        runs.append((name, load_metrics(path)))

    report = render_report(runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()