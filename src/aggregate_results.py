"""Join every run's metrics into one comparison table plus the two figures.

Inputs are the ``metrics.json`` files written by the evaluators:

    results/base_metrics.json              tool-call metrics, no training
    results/tool_sft_metrics.json          tool-call metrics, positive-only SFT
    results/reliable_tool_sft_metrics.json tool-call metrics, + When2Call
    results/*_when2call.json               four-way decision metrics

Outputs:

    results/comparison.csv                 the CV table
    results/figures/exact_match.png        Figure 1
    results/figures/false_tool_call.png    Figure 2
    results/figures/confusion_matrix.png   optional, when decision data exists

Usage:
    python -m src.aggregate_results --results results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

__all__ = ["RUNS", "load_json", "build_rows", "write_csv", "make_figures", "main"]

# The three checkpoints, in comparison order.
RUNS = ["base", "tool_sft", "reliable_tool_sft"]

LABELS = {
    "base": "Base",
    "tool_sft": "Tool-SFT",
    "reliable_tool_sft": "Reliable Tool-SFT",
}

# Column order of comparison.csv, and how each cell is sourced.
COLUMNS = [
    "exact_match",
    "tool_selection_accuracy",
    "argument_accuracy",
    "json_validity",
    "unseen_function_accuracy",
    "when2call_decision_accuracy",
    "when2call_false_tool_call_rate",
    "when2call_missing_info_accuracy",
    "when2call_cannot_answer_accuracy",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(results_dir: Path) -> list[dict[str, Any]]:
    """One row per checkpoint, columns pulled from the two eval tracks."""
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        tool_metrics = load_json(results_dir / f"{run}_metrics.json")
        decision_metrics = load_json(results_dir / f"{run}_when2call.json")

        row: dict[str, Any] = {"checkpoint": LABELS.get(run, run)}
        overall = (tool_metrics or {}).get("overall", {})
        for key in ("exact_match", "tool_selection_accuracy", "argument_accuracy",
                    "json_validity", "unseen_function_accuracy"):
            row[key] = overall.get(key)

        row["when2call_decision_accuracy"] = (decision_metrics or {}).get("decision_accuracy")
        row["when2call_false_tool_call_rate"] = (decision_metrics or {}).get("false_tool_call_rate")
        row["when2call_missing_info_accuracy"] = (decision_metrics or {}).get("missing_info_accuracy")
        row["when2call_cannot_answer_accuracy"] = (decision_metrics or {}).get("cannot_answer_accuracy")
        row["n_tool_eval"] = overall.get("n")
        row["n_when2call"] = (decision_metrics or {}).get("n")
        rows.append(row)
    return rows


def _pct(value: Any) -> str:
    return "" if value is None else f"{value * 100:.1f}"


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + [row["checkpoint"] for row in rows])
        for column in COLUMNS:
            writer.writerow([column] + [_pct(row.get(column)) for row in rows])


def make_figures(rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    """Figure 1 (exact match) and Figure 2 (false tool-call rate)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    names = [row["checkpoint"] for row in rows]
    colors = ["#8c8c8c", "#4c72b0", "#2e7d32"]
    written: list[Path] = []

    def bar(values: list[float | None], title: str, ylabel: str, filename: str,
            lower_is_better: bool = False) -> None:
        if all(value is None for value in values):
            print(f"[skip] {filename}: no data")
            return
        plotted = [0.0 if value is None else value * 100 for value in values]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, plotted, color=colors[: len(names)])
        for rect, value in zip(bars, plotted):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.0,
                    f"{value:.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(plotted) * 1.25 + 5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        note = "lower is better" if lower_is_better else "higher is better"
        ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#666666")
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        print(f"[figure] {path}")

    bar([row.get("exact_match") for row in rows],
        "Exact Tool-Call Accuracy", "Exact match (%)", "exact_match.png")
    bar([row.get("when2call_false_tool_call_rate") for row in rows],
        "False Tool-Call Rate", "False call rate (%)", "false_tool_call.png",
        lower_is_better=True)

    return written


def make_confusion_figure(results_dir: Path, figures_dir: Path) -> Path | None:
    """Plot the Reliable Tool-SFT confusion matrix when it is available."""
    metrics = load_json(results_dir / "reliable_tool_sft_when2call.json")
    if not metrics or not metrics.get("confusion_matrix"):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["tool_call", "direct", "request_for_info", "cannot_answer"]
    matrix = np.array(
        [[metrics["confusion_matrix"].get(g, {}).get(p, 0) for p in labels] for g in labels],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Reliable Tool-SFT — When2Call decisions")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """The comparison table exactly as the project guide asks for it."""
    header = "| Metric | Base | Tool-SFT | Reliable Tool-SFT |"
    lines = [header, "|---|---:|---:|---:|"]
    pretty = {
        "exact_match": "Exact tool-call match",
        "tool_selection_accuracy": "Function-name accuracy",
        "argument_accuracy": "Argument accuracy",
        "json_validity": "JSON validity",
        "unseen_function_accuracy": "Held-out function EM",
        "when2call_decision_accuracy": "When2Call decision accuracy",
        "when2call_false_tool_call_rate": "False tool-call rate (lower better)",
        "when2call_missing_info_accuracy": "Missing-info accuracy",
        "when2call_cannot_answer_accuracy": "Cannot-answer accuracy",
    }
    for column in COLUMNS:
        cells = []
        for row in rows:
            value = row.get(column)
            cells.append("n/a" if value is None else f"{value * 100:.1f}%")
        lines.append(f"| {pretty.get(column, column)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results)
    rows = build_rows(results_dir)

    if all(row.get("exact_match") is None for row in rows):
        raise SystemExit(
            f"no tool-call metrics found in {results_dir}; run the evaluators first"
        )

    write_csv(rows, results_dir / "comparison.csv")
    print(f"[csv] wrote {results_dir / 'comparison.csv'}")

    figures_dir = results_dir / "figures"
    make_figures(rows, figures_dir)
    make_confusion_figure(results_dir, figures_dir)

    table = render_markdown(rows)
    (results_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print()
    print(table)


if __name__ == "__main__":
    main()