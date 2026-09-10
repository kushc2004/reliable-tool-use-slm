"""Assemble the train / eval corpus and freeze the splits.

Sources can be mixed:

    --source synth                       offline synthetic corpus
    --source glaive,when2call            real corpora via prepare_*.py loaders
    --source /path/to/records.jsonl      anything already in our record format

Output layout:

    data/processed/train.jsonl
    data/processed/eval.jsonl
    data/processed/stats.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from . import synth

__all__ = ["load_jsonl", "write_jsonl", "build", "main"]

SPLITS = ["train", "heldout_tools", "unseen_functions", "no_tool"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _validate(record: dict[str, Any]) -> str | None:
    """Return an error string if the record is unusable."""
    if not record.get("messages"):
        return "no messages"
    if not record.get("tools"):
        return "no tools advertised"
    if record.get("expects_call") and not record.get("gold_calls"):
        return "expects_call but no gold_calls"
    if not record.get("expects_call") and record.get("gold_calls"):
        return "gold_calls present but expects_call is false"
    for message in record["messages"]:
        if message["role"] not in ("system", "user", "assistant", "tool"):
            return f"unknown role {message['role']!r}"
    return None


def _load_source(name: str, limit: int, neg_limit: int) -> list[dict[str, Any]]:
    """Load one named source into our record format."""
    path = Path(name)
    if path.suffix == ".jsonl" and path.exists():
        return load_jsonl(path)

    if name == "synth":
        # Synthetic data is generated whole; the split sizes are decided later.
        return []

    from .prepare_glaive import convert, load_rows as load_positive
    from .prepare_negatives import convert as convert_neg
    from .prepare_negatives import load_rows as load_negative

    records: list[dict[str, Any]] = []
    if name in ("glaive", "hermes"):
        dialect = name
        hub_id = (
            "glaiveai/glaive-function-calling-v2"
            if dialect == "glaive"
            else "NousResearch/hermes-function-calling-v1"
        )
        rows = load_positive(hub_id, limit=limit)
        converted, stats = convert(rows, dialect)
        print(f"[{name}] seen={stats['seen']} kept={stats['kept']} skipped={stats['skipped']}")
        records.extend(converted)
    elif name == "when2call":
        rows = load_negative("nvidia/When2Call", limit=neg_limit)
        converted, stats = convert_neg(rows)
        print(f"[{name}] seen={stats['seen']} kept={stats['kept']} skipped={stats['skipped']}")
        records.extend(converted)
    else:
        raise SystemExit(f"unknown source: {name}")
    return records


def _unseen_function_split(eval_records: list[dict[str, Any]], held_out_names: set[str]) -> list[dict[str, Any]]:
    """Retag eval records whose call targets a held-out function name."""
    for record in eval_records:
        names = {call["name"] for call in record.get("gold_calls") or []}
        if names & held_out_names:
            record["split"] = "unseen_functions"
    return eval_records


def build(
    sources: list[str],
    out_dir: Path,
    n_train: int,
    n_eval: int,
    neg_ratio: float,
    seed: int,
    limit: int,
    neg_limit: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    if sources == ["synth"]:
        corpus = synth.generate_records(
            n_train=n_train, n_eval=n_eval, neg_ratio=neg_ratio, seed=seed
        )
        train, eval_records = corpus["train"], corpus["eval"]
    else:
        records: list[dict[str, Any]] = []
        for source in sources:
            records.extend(_load_source(source, limit, neg_limit))

        rng.shuffle(records)
        train = [r for r in records if r.get("expects_call")]
        negatives = [r for r in records if not r.get("expects_call")]

        # Bring negatives in at the requested ratio, then apply the 3K budget:
        # the base repo's finding is that most of the benefit arrives by ~3K
        # examples, so the interesting variable is the negative ratio, not the
        # total row count.
        wanted_neg = int(n_train * neg_ratio)
        if len(negatives) < wanted_neg:
            print(f"[warn] only {len(negatives)} negatives available, wanted {wanted_neg}")
            wanted_neg = len(negatives)
        positives = train[: max(1, n_train - wanted_neg)]
        train = positives + negatives[:wanted_neg]
        rng.shuffle(train)

        # Held-out split: build eval sets out of whatever tools show up, and
        # mark a slice of function names as unseen.
        train_names = {
            call["name"] for record in train for call in (record.get("gold_calls") or [])
        }
        all_eval_pool = [r for r in records if r.get("expects_call")]
        eval_records = all_eval_pool[:n_eval] if all_eval_pool else []
        held_out = {name for name in train_names if name.endswith("_eval")}
        if held_out:
            eval_records = _unseen_function_split(eval_records, held_out)

    # Validate and drop anything unusable, reporting why.
    reasons: Counter[str] = Counter()
    clean_train: list[dict[str, Any]] = []
    for record in train:
        error = _validate(record)
        if error:
            reasons[error] += 1
        else:
            clean_train.append(record)
    clean_eval: list[dict[str, Any]] = []
    for record in eval_records:
        error = _validate(record)
        if error:
            reasons[error] += 1
        else:
            clean_eval.append(record)

    write_jsonl(out_dir / "train.jsonl", clean_train)
    write_jsonl(out_dir / "eval.jsonl", clean_eval)

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(records),
            "by_split": dict(Counter(r.get("split", "?") for r in records)),
            "by_category": dict(Counter(r.get("category", "?") for r in records)),
            "call_expected": sum(1 for r in records if r.get("expects_call")),
            "no_call_expected": sum(1 for r in records if not r.get("expects_call")),
        }

    stats = {
        "sources": sources,
        "seed": seed,
        "n_train_requested": n_train,
        "neg_ratio": neg_ratio,
        "train": summarize(clean_train),
        "eval": summarize(clean_eval),
        "dropped": dict(reasons),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="synth",
                        help="comma-separated: synth, glaive, hermes, when2call, or a .jsonl path")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--neg-ratio", type=float, default=0.0,
                        help="fraction of training rows that should be no-tool examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20000, help="cap on rows read per positive source")
    parser.add_argument("--neg-limit", type=int, default=5000, help="cap on rows read from When2Call")
    args = parser.parse_args()

    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    out_dir = Path(args.out)
    stats = build(
        sources=sources,
        out_dir=out_dir,
        n_train=args.n_train,
        n_eval=args.n_eval,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        limit=args.limit,
        neg_limit=args.neg_limit,
    )

    print(json.dumps(stats, indent=2))
    print(f"\nwrote {stats['train']['n']} train and {stats['eval']['n']} eval records to {out_dir}")


if __name__ == "__main__":
    main()