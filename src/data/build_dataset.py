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

# Fraction of distinct function names withheld from training entirely, so the
# unseen-function metric tests schema generalisation rather than memorisation.
UNSEEN_NAME_FRACTION = 0.15


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
    """Return an error string if the record is unusable.

    Tool list requirements are asymmetric on purpose:

    * ``expects_call`` -- the schema is mandatory. Without it the gold call
      cannot be rendered or scored, so the row is unusable.
    * no call expected -- an *empty* tool list is valid and in fact the
      sharpest case there is. When2Call builds its ``cannot_answer`` rows by
      removing the tools outright, and "no functions are available, so do not
      invent one" is exactly the behaviour being measured. Requiring tools here
      would discard those rows.
    """
    if not record.get("messages"):
        return "no messages"
    if record.get("expects_call") and not record.get("tools"):
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
        # Both training configs are needed and neither alone is sufficient:
        # train_sft (15000 rows) is entirely non-call, so it supplies only the
        # request_for_info / cannot_answer / direct decisions, while train_pref
        # (9000 rows) is the only place a tool_call gold behaviour exists.
        from .prepare_when2call import (
            DEFAULT_HUB as W2C_HUB,
            PREF_CONFIG,
            PREF_SPLIT,
            SFT_CONFIG,
            SFT_SPLIT,
        )
        from .prepare_when2call import convert as convert_w2c
        from .prepare_when2call import load_rows as load_w2c

        sft_rows = load_w2c(W2C_HUB, config=SFT_CONFIG, split=SFT_SPLIT, limit=neg_limit)
        pref_rows = load_w2c(W2C_HUB, config=PREF_CONFIG, split=PREF_SPLIT, limit=neg_limit)
        sft_records, sft_stats = convert_w2c(sft_rows, mode="train", origin=SFT_CONFIG)
        pref_records, pref_stats = convert_w2c(pref_rows, mode="train", origin=PREF_CONFIG)
        print(f"[when2call] sft: kept={sft_stats.get('kept')} "
              f"labels={ {k[6:]: v for k, v in sft_stats.items() if k.startswith('label:')} }")
        print(f"[when2call] pref: kept={pref_stats.get('kept')} "
              f"labels={ {k[6:]: v for k, v in pref_stats.items() if k.startswith('label:')} }")
        records.extend(sft_records + pref_records)
    else:
        raise SystemExit(f"unknown source: {name}")
    return records


def _retag(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Stamp a split name onto eval records, in place, and return them.

    Safe to mutate because every caller passes a slice that has already been
    removed from the pool training samples from.
    """
    for record in records:
        record["split"] = split
    return records


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
        positives = [r for r in records if r.get("expects_call")]
        negatives = [r for r in records if not r.get("expects_call")]

        # Hold out whole function NAMES before any training row is chosen.
        #
        # This replaces a tagging pass that keyed on a "_eval" name suffix no
        # real function carries, so it never fired and every "unseen" call had
        # in fact been trained on. Withholding the schemas up front is what
        # makes the metric mean anything.
        all_names = sorted(
            {call["name"] for record in positives for call in (record.get("gold_calls") or [])}
        )
        n_unseen_names = max(1, int(len(all_names) * UNSEEN_NAME_FRACTION))
        unseen_names = set(rng.sample(all_names, n_unseen_names))
        print(f"[split] holding out {len(unseen_names)}/{len(all_names)} function names as unseen")

        def targets_unseen(record: dict[str, Any]) -> bool:
            return bool({c["name"] for c in record.get("gold_calls") or []} & unseen_names)

        seen_pool = [r for r in positives if not targets_unseen(r)]
        unseen_pool = [r for r in positives if targets_unseen(r)]

        # Carve the eval sets out BEFORE training samples the pools, and slice
        # rather than copy: the rows used for eval are physically removed from
        # the lists training draws from, so overlap is impossible.
        per_split = max(1, n_eval // 3)
        eval_records = (
            _retag(seen_pool[:per_split], "heldout_tools")
            + _retag(unseen_pool[:per_split], "unseen_functions")
            + _retag(negatives[:per_split], "no_tool")
        )
        seen_train_pool = seen_pool[per_split:]
        negatives_train_pool = negatives[per_split:]

        # Bring negatives in at the requested ratio, then apply the 3K budget:
        # the base repo's finding is that most of the benefit arrives by ~3K
        # examples, so the interesting variable is the negative ratio, not the
        # total row count.
        wanted_neg = int(n_train * neg_ratio)
        if len(negatives_train_pool) < wanted_neg:
            print(f"[warn] only {len(negatives_train_pool)} negatives available, "
                  f"wanted {wanted_neg}")
            wanted_neg = len(negatives_train_pool)
        n_pos = max(1, n_train - wanted_neg)
        if len(seen_train_pool) < n_pos:
            print(f"[warn] only {len(seen_train_pool)} positive rows available, wanted {n_pos}")
            n_pos = len(seen_train_pool)
        train = seen_train_pool[:n_pos] + negatives_train_pool[:wanted_neg]
        rng.shuffle(train)
        print(f"[split] train={len(train)} "
              f"(pos={n_pos} neg={wanted_neg}) eval={len(eval_records)}")

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

    # Refuse to emit an empty training split. Writing zero rows and exiting 0 is
    # how a loader bug turns into a "successful" run that trains on nothing and
    # then republishes stale metrics.
    if not clean_train:
        raise SystemExit(
            f"refusing to write an empty training split to {out_dir}: all "
            f"{len(train)} candidate records were dropped ({dict(reasons) or 'none'}). "
            "This is a data-loader bug, not an empty dataset."
        )

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