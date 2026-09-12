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

#: The When2Call track carries the four-way decision task. Its prompts are much
#: longer than the tool-call track's (tool definitions + context + utterance),
#: so a batch size that fits a short-prompt eval (tool-call at 8) blows the
#: GPU's attention memory. ``1`` is unbatched and always safe; ``2`` is the
#: largest value that has been verified to fit alongside a loaded quantized
#: model. Tune up only on a card with more memory than a T4.
W2C_SAFE_BATCH_SIZE = 2

__all__ = ["load_when2call", "dummy_when2call_backend", "run_eval", "main"]

BACKENDS = ["hf", "dummy", "oracle"]

# ``direct`` is deliberately NOT required: the real ``mcq`` split carries all
# four answer strings but no row is gold-``direct`` (see prepare_when2call).
# The other three must be present or the "four-way" score is not four-way.
REQUIRED_DECISIONS = ["tool_call", "request_for_info", "cannot_answer"]


def stratified_sample(
    records: list[dict[str, Any]], limit: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Take a label-stratified sample of at most ``limit`` records.

    ``records[:limit]`` is the wrong operation here and it is not a subtle one.
    The real ``mcq`` split is *grouped by gold label* -- 1295 ``cannot_answer``,
    then 1062 ``request_for_info``, then 1295 ``tool_call`` -- so a 1200-row head
    is 1200 ``cannot_answer`` rows and nothing else. That silently collapsed the
    four-way decision track to one-class accuracy, and because the oracle ran
    without a limit it still passed and nobody noticed.

    Proportional per-class allocation, with one row guaranteed per class,
    keeps every class in the slice. Fixed seed, so the oracle and the arms
    sample the same rows.
    """
    import random

    if limit >= len(records):
        return list(records)

    rng = random.Random(seed)
    by_decision: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        by_decision.setdefault(record.get("decision"), []).append(record)
    for bucket in by_decision.values():
        rng.shuffle(bucket)

    # ``limit`` smaller than the number of classes: one row from each of the
    # first ``limit`` classes rather than an empty result.
    if limit < len(by_decision):
        picked = [by_decision[label][0] for label in sorted(by_decision)[:limit]]
        rng.shuffle(picked)
        return picked

    total = len(records)
    quota = {label: 1 for label in by_decision}
    remaining = limit - len(quota)
    for label in sorted(by_decision):
        quota[label] += int(remaining * len(by_decision[label]) / total)

    # Hand out the rounding remainder to the classes with the most unsampled
    # rows; trim it back from the largest quota if we overshot.
    while sum(quota.values()) < limit:
        label = max(by_decision, key=lambda k: len(by_decision[k]) - quota[k])
        quota[label] += 1
    while sum(quota.values()) > limit:
        label = max(quota, key=lambda k: quota[k])
        if quota[label] <= 1:
            break
        quota[label] -= 1

    picked = []
    for label in sorted(by_decision):
        picked.extend(by_decision[label][: quota[label]])
    rng.shuffle(picked)
    return picked


def load_when2call(
    path: str | Path, limit: int | None = None, seed: int = 0
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = stratified_sample(records, limit, seed=seed)
    return records


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
    batch_fn = getattr(predict, "batch", None)
    batch_size = getattr(predict, "batch_size", 1) or 1
    if batch_fn is not None and batch_size > 1:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            for record, prediction in zip(chunk, batch_fn(chunk)):
                pairs.append((record, prediction))
            done = min(start + batch_size, len(records))
            if done // 50 > start // 50:
                print(f"  generated {done}/{len(records)}")
    else:
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
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="seed for stratified --limit sampling. The oracle and the arms must "
             "use the same value, or the sanity check validates different rows "
             "than the ones the models are scored on.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="generate this many prompts at once; 1 disables batching. "
             "W2C prompts are long, so keep this small (see W2C_SAFE_BATCH_SIZE).",
    )
    parser.add_argument("--failure-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_when2call(args.data, limit=args.limit, seed=args.sample_seed)
    print(f"loaded {len(records)} When2Call records from {args.data}")
    if args.limit:
        print(
            f"[sample] stratified {len(records)} of the full split "
            f"(limit={args.limit}, seed={args.sample_seed})"
        )
    if not records:
        raise SystemExit("no records; run src.data.prepare_when2call first")

    # This used to be a ``print``. It fired on every arm run -- because a
    # 1200-row head of the label-sorted file is 1200 ``cannot_answer`` rows --
    # and scrolled past into a results table whose "four-way" decision_accuracy
    # was one-class accuracy. A missing required class is a broken experiment,
    # not a warning.
    present = {record.get("decision") for record in records}
    missing_required = [label for label in REQUIRED_DECISIONS if label not in present]
    if missing_required:
        raise SystemExit(
            f"decision split is missing required classes {missing_required} "
            f"(present: {sorted(present)}). A four-way score cannot be computed "
            "from this slice; check --limit / --sample-seed and the data file."
        )
    optional_missing = [label for label in DECISIONS if label not in present]
    if optional_missing:
        print(f"[warn] categories absent from this split (optional): {optional_missing}")

    if args.backend == "oracle":
        predict = oracle_backend()
    elif args.backend == "dummy":
        predict = dummy_when2call_backend(failure_rate=args.failure_rate, seed=args.seed)
    else:
        from .evaluate import hf_backend

        predict = hf_backend(
            args.checkpoint,
            args.adapter,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    run_eval(records, predict, Path(args.out), backend=args.backend)


if __name__ == "__main__":
    main()