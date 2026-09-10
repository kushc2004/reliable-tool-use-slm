"""Evaluate a checkpoint on the When2Call four-way decision task.

Reuses the generation backends from :mod:`src.evaluate` so a checkpoint is
loaded exactly once per run and the same prompt format is used everywhere.

Usage:
    python -m src.evaluate_when2call --data data/raw/w2c_eval.jsonl \\
        --backend dummy --out results/base_when2call
    python -m src.evaluate_when2call --data data/raw/w2c_eval.jsonl \\
        --checkpoint Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/tool_sft \\
        --out results/tool_sft_when2call
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .decision_metrics import DECISIONS, aggregate_decisions, format_decision_table, score_all
from .schema import CALL_CLOSE, CALL_OPEN

__all__ = ["load_when2call", "dummy_when2call_backend", "run_eval", "main"]

BACKENDS = ["hf", "dummy", "oracle"]


def load_when2call(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[:limit] if limit else records


def _render_calls(calls: list[dict[str, Any]]) -> str:
    blocks = []
    for call in calls:
        payload = json.dumps(
            {"name": call["name"], "arguments": call.get("arguments") or {}}, ensure_ascii=False
        )
        blocks.append(f"{CALL_OPEN}\n{payload}\n{CALL_CLOSE}")
    return "\n".join(blocks)


def oracle_backend() -> Any:
    """Perfect-but-honest backend: reads the gold answer text verbatim.

    Used to validate the scorer itself. If the oracle does not score ~100%, the
    classifier -- not the model -- is what is broken, and that is worth knowing
    before trusting any real result.
    """

    def predict(record: dict[str, Any]) -> str:
        if record.get("decision") == "tool_call" and record.get("gold_calls"):
            return _render_calls(record["gold_calls"])
        return record.get("gold_response") or "I can answer that directly."

    return predict


def dummy_when2call_backend(failure_rate: float = 0.0, seed: int = 0) -> Any:
    """Degrades in the specific way positive-only tool SFT degrades.

    ``failure_rate=0`` reproduces a model that calls a tool whenever one is
    advertised, which is the failure this project exists to measure: perfect
    tool-call recall, terrible false-tool-call rate.
    """
    import random

    rng = random.Random(seed)

    def predict(record: dict[str, Any]) -> str:
        if record.get("decision") == "tool_call" and record.get("gold_calls"):
            return _render_calls(record["gold_calls"])
        if failure_rate and rng.random() < failure_rate:
            return record.get("gold_response") or "I can answer that directly."
        tools = record.get("tools") or []
        if tools:
            name = tools[0]["function"]["name"]
            return f'{CALL_OPEN}\n{{"name": "{name}", "arguments": {{}}}}\n{CALL_CLOSE}'
        return "I can answer that directly."

    return predict


def run_eval(
    records: list[dict[str, Any]],
    predict: Any,
    out_dir: Path,
    limit: int | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    if limit:
        records = records[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[dict[str, Any], str]] = []
    for index, record in enumerate(records):
        pairs.append((record, predict(record)))
        if (index + 1) % 50 == 0:
            print(f"  generated {index + 1}/{len(records)}")

    scored = score_all(pairs)
    metrics = aggregate_decisions(scored)
    if backend:
        metrics["backend"] = backend

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failures = [row for row in scored if not row["correct"]]
    with (out_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures[:200]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    gold_counts = Counter(row["gold"] for row in scored)
    print()
    print(format_decision_table(metrics))
    print()
    print(f"n={metrics['n']}  gold distribution: {dict(gold_counts)}")
    print(f"failures: {len(failures)}  -> {out_dir / 'failures.jsonl'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/w2c_eval.jsonl")
    # See src/evaluate.py: this defaulted to "dummy", which silently scored the
    # test fixture for all three arms because the notebook never passed
    # --backend. Default to the real backend so the failure mode is a loud
    # missing-torch error, not a fabricated comparison table.
    parser.add_argument("--backend", choices=BACKENDS, default="hf")
    parser.add_argument("--checkpoint", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--failure-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_when2call(args.data, limit=args.limit)
    print(f"loaded {len(records)} When2Call records from {args.data}")
    if not records:
        raise SystemExit("no records; run src.data.prepare_when2call first")

    present = {record.get("decision") for record in records}
    missing = [label for label in DECISIONS if label not in present]
    if missing:
        print(f"[warn] categories absent from this split: {missing}")

    if args.backend == "oracle":
        predict = oracle_backend()
    elif args.backend == "dummy":
        predict = dummy_when2call_backend(failure_rate=args.failure_rate, seed=args.seed)
    else:
        from .evaluate import hf_backend

        predict = hf_backend(
            args.checkpoint, args.adapter, max_new_tokens=args.max_new_tokens
        )

    run_eval(records, predict, Path(args.out), backend=args.backend)


if __name__ == "__main__":
    main()