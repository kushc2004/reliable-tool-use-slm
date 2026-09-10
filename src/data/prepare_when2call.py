"""Load NVIDIA When2Call into the two shapes this project needs.

When2Call is a *decision* benchmark, not a call-accuracy benchmark. Every row
has one of four correct behaviours::

    tool_call         sufficient info + a tool that fits -> call it
    direct            no tool needed -> just answer
    request_for_info  the right tool exists but an argument is missing ->
                      ask for it instead of inventing a value
    cannot_answer     nothing on offer can satisfy the request -> say so

Access pattern
--------------
The dataset is **config-scoped**, which the first version of this loader got
wrong by requesting a ``"train"`` split that does not exist::

    config        split        n      carries a gold label?
    test          mcq          3652   yes -- ``correct_answer``
    test          llm_judge     300   yes
    train_sft     train       15000   NO  -- ``tools`` + ``messages`` only
    train_pref    train        9000   chosen/rejected response pairs

Consequences that shape this module:

* Evaluation reads ``("test", "mcq")`` and uses ``correct_answer`` directly.
* ``train_sft`` contains **no tool calls at all** (verified: 0 of 15000 rows
  mention TOOLCALL anywhere). It supplies the non-call decisions only.
* ``train_pref`` is the only source of ``tool_call`` supervision: 3000 of its
  9000 ``chosen_response`` values contain a call. Its non-call rows are also
  usable, since ``chosen_response`` is the gold behaviour there.
* ``direct`` is rare in both -- 103 rows in ``train_sft``, 34 in ``train_pref``.
  A genuinely balanced four-way subset is therefore impossible; ``balance()``
  caps every class at the smallest available count rather than silently
  oversampling one, and reports what it did.
* The ``mcq`` eval set has all four keys in ``answers`` but only three gold
  labels -- no row is gold-``direct``. ``direct_accuracy`` is consequently not
  measurable on the real benchmark and is marked as such in the report.

Usage:
    python -m src.data.prepare_when2call --mode train --out data/raw/w2c_train.jsonl
    python -m src.data.prepare_when2call --mode eval  --out data/raw/w2c_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..schema import parse_tool_calls

__all__ = [
    "DECISIONS",
    "DEFAULT_HUB",
    "convert_row",
    "convert",
    "balance",
    "classify_gold_response",
    "load_rows",
    "main",
]

DECISIONS = ["tool_call", "direct", "request_for_info", "cannot_answer"]

DEFAULT_HUB = "nvidia/When2Call"

# config -> split, for each access path.
EVAL_CONFIG, EVAL_SPLIT = "test", "mcq"
SFT_CONFIG, SFT_SPLIT = "train_sft", "train"
PREF_CONFIG, PREF_SPLIT = "train_pref", "train"

_TOOLCALL_RE = re.compile(r"<TOOLCALL>.*?</TOOLCALL>", re.DOTALL | re.IGNORECASE)

# Ordered: a clarification beats a refusal when both cues appear, because a
# model that asks for the missing argument has done the harder thing.
_REQUEST_CUES = re.compile(
    r"\b(could you|can you (please )?(specify|clarify|provide|tell|confirm)|"
    r"please (specify|provide|clarify|confirm|tell me|let me know)|"
    r"would you like|which one|which (city|province|company|order|invoice|page)|"
    r"what should i|i(?:'d| would)? need (more|additional)|"
    r"to (help|assist) you|what (is|are) the|how many|for which)\b",
    re.IGNORECASE,
)

_CANNOT_CUES = re.compile(
    r"\b(i'?m (unable|not able|sorry)|i (can'?t|cannot)|"
    r"i (don'?t|do not) have (access|the ability)|"
    r"none of the (available )?(functions|tools)|"
    r"outside (my|the) (capabilities|scope)|"
    r"not something i can|no (available|provided) (function|tool)|apologies)\b",
    re.IGNORECASE,
)


def _as_obj(value: Any) -> Any:
    """When2Call stores tool specs and answer dicts as JSON strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _normalize_tools(raw: Any) -> list[dict[str, Any]]:
    raw = _as_obj(raw)
    if not raw or not isinstance(raw, list):
        return []
    tools: list[dict[str, Any]] = []
    for tool in raw:
        tool = _as_obj(tool)
        if not isinstance(tool, dict):
            continue
        tools.append(tool if "function" in tool else {"type": "function", "function": tool})
    return tools


def classify_gold_response(text: str) -> str:
    """Infer the four-way decision from a gold assistant turn.

    Needed because ``train_sft`` and ``train_pref`` carry no
    ``correct_answer`` field -- only the behaviour itself. A call is detected
    by the ``<TOOLCALL>`` tag; the three non-call behaviours are separated by
    ordered cue matching, with clarification taking priority over refusal.
    """
    if not text or not text.strip():
        return "cannot_answer"
    if _TOOLCALL_RE.search(text):
        return "tool_call"
    if _REQUEST_CUES.search(text):
        return "request_for_info"
    if _CANNOT_CUES.search(text):
        return "cannot_answer"
    return "direct"


def _extract_gold_calls(text: str) -> list[dict[str, Any]]:
    """Pull calls out of a ``<TOOLCALL>[{...}]</TOOLCALL>`` block.

    The payload is a JSON *array* of call objects, which is why the generic
    ``parse_tool_calls`` (built for one object per block) is not used directly:
    re-wrapping each element as its own block keeps a single parser in play.
    """
    match = _TOOLCALL_RE.search(text or "")
    if not match:
        return []
    body = match.group(0)
    body = re.sub(r"</?TOOLCALL>", "", body, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return list(parse_tool_calls(body).calls and
                    [c.to_dict() for c in parse_tool_calls(body).calls])

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    calls: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        args = item.get("arguments", item.get("parameters", {})) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"value": args}
        calls.append({"name": name, "arguments": args})
    return calls


def load_rows(
    source: str = DEFAULT_HUB,
    config: str | None = None,
    split: str = "train",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read rows from a local file or from the Hub.

    ``config`` is passed through to ``load_dataset`` when set. Omitting it is
    what caused the original ``Unknown split "train"`` failure: without a
    config the loader is handed the default ``['llm_judge', 'mcq']`` split
    list, and ``"train"`` is not among them.
    """
    path = Path(source)
    if path.exists():
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows[:limit] if limit else rows

    from datasets import load_dataset  # lazy: needs network

    if config:
        dataset = load_dataset(source, config, split=split)
    else:
        dataset = load_dataset(source, split=split)
    rows = [dict(row) for row in dataset]
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------- #
# Eval (mcq): labels are given
# --------------------------------------------------------------------------- #


def convert_eval_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Convert one ``mcq`` row, whose ``correct_answer`` is authoritative."""
    question = row.get("question") or row.get("orig_question")
    if not question:
        return None

    label = (row.get("correct_answer") or "").strip()
    if label not in DECISIONS:
        return None

    answers = _as_obj(row.get("answers")) or {}
    if not isinstance(answers, dict):
        answers = {}

    tools = _normalize_tools(row.get("tools"))
    gold_text = str(answers.get(label) or "").strip()
    calls = _extract_gold_calls(gold_text) if label == "tool_call" else []

    return {
        "id": str(row.get("uuid") or f"w2c-eval-{index:06d}"),
        "split": "when2call",
        "decision": label,
        "category": label,
        "source": row.get("source"),
        "source_id": row.get("source_id"),
        "tools": tools,
        "messages": [{"role": "user", "content": str(question)}],
        "gold_calls": calls,
        "expects_call": label == "tool_call",
        "gold_response": gold_text,
        "answers": {key: str(answers.get(key) or "") for key in DECISIONS},
        "target_tool": row.get("target_tool"),
        "held_out_param": row.get("held_out_param"),
    }


# --------------------------------------------------------------------------- #
# Training: labels are inferred
# --------------------------------------------------------------------------- #


def convert_sft_row(row: dict[str, Any], index: int, origin: str) -> dict[str, Any] | None:
    """Convert a ``train_sft`` or ``train_pref`` row into a training record.

    ``train_sft`` supplies the gold turn as the last assistant message;
    ``train_pref`` supplies it as ``chosen_response``, which is the field that
    makes the 3000 ``tool_call`` rows reachable at all.
    """
    messages = row.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None

    if origin == PREF_CONFIG:
        chosen = row.get("chosen_response")
        if isinstance(chosen, dict):
            gold_text = chosen.get("content") or ""
        else:
            gold_text = str(chosen or "")
    else:
        assistant_turns = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_turns:
            return None
        gold_text = assistant_turns[-1].get("content") or ""

    if not gold_text.strip():
        return None

    decision = classify_gold_response(gold_text)
    tools = _normalize_tools(row.get("tools"))
    calls = _extract_gold_calls(gold_text) if decision == "tool_call" else []

    # A tool_call with no recoverable call cannot supervise anything.
    if decision == "tool_call" and not calls:
        return None

    # Truncate at the user turn that precedes the decision: the training
    # target is the gold behaviour, and any earlier turns are context.
    user_turns = [m for m in messages if m.get("role") == "user"]
    if not user_turns:
        return None

    return {
        "id": f"w2c-{origin}-{index:06d}",
        "split": "train",
        "decision": decision,
        # Mirrors ``decision`` so the shared stats summary, which counts by
        # ``category``, reports the four-way decision mix rather than "?".
        "category": decision,
        "source": origin,
        "tools": tools,
        "messages": [
            {"role": "user", "content": user_turns[-1].get("content") or ""},
            {"role": "assistant", "content": gold_text, "tool_calls": calls},
        ],
        "gold_calls": calls,
        "expects_call": decision == "tool_call",
        "gold_response": gold_text,
    }


def convert(
    rows: Iterable[dict[str, Any]],
    mode: str = "eval",
    origin: str = SFT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert rows, counting why anything was skipped and the label mix."""
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for index, row in enumerate(rows):
        stats["seen"] += 1
        try:
            if mode == "eval":
                record = convert_eval_row(row, index)
            else:
                record = convert_sft_row(row, index, origin)
        except Exception:  # noqa: BLE001 - a bad row must not kill the run
            record = None
            stats["skip:exception"] += 1
        if record is None:
            stats["skipped"] += 1
            continue
        records.append(record)
        stats["kept"] += 1
        stats[f"label:{record['decision']}"] += 1

    return records, dict(stats)


def balance(records: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Take a class-balanced subset of at most ``n`` records.

    Balance matters because the four categories *are* the experiment: a corpus
    that is 80% ``tool_call`` teaches the model to call a tool by default,
    which is the exact failure being measured.

    ``direct`` is genuinely rare in When2Call (137 rows across both training
    configs), so a true four-way balance is not achievable. Every class is
    capped at the smallest available count and the shortfall is reported rather
    than hidden by oversampling a class the benchmark barely contains.
    """
    rng = random.Random(seed)
    by_decision: dict[str, list[dict[str, Any]]] = {key: [] for key in DECISIONS}
    for record in records:
        by_decision[record["decision"]].append(record)
    for bucket in by_decision.values():
        rng.shuffle(bucket)

    target = max(1, n // len(DECISIONS))
    available = {key: len(bucket) for key, bucket in by_decision.items()}
    per_class = min(target, min(available.values())) if available else 0

    picked: list[dict[str, Any]] = []
    for key in DECISIONS:
        picked.extend(by_decision[key][:per_class])

    rng.shuffle(picked)
    return picked


def balance_report(records: list[dict[str, Any]], picked: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe what ``balance`` did, including any shortfall."""
    before = Counter(r["decision"] for r in records)
    after = Counter(r["decision"] for r in picked)
    return {
        "available": {key: before.get(key, 0) for key in DECISIONS},
        "selected": {key: after.get(key, 0) for key in DECISIONS},
        "limiting_class": min(before, key=lambda k: before.get(k, 0)) if before else None,
        "note": (
            "'direct' is rare in When2Call; a true four-way balance is not "
            "achievable and every class is capped at the smallest count."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_HUB)
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=400, help="train mode: subset size")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "eval":
        rows = load_rows(args.input, config=EVAL_CONFIG, split=EVAL_SPLIT, limit=args.limit)
        records, stats = convert(rows, mode="eval")
        report = {"config": EVAL_CONFIG, "split": EVAL_SPLIT}
    else:
        # Both training configs contribute: train_pref for tool_call, both for
        # the non-call decisions.
        sft_rows = load_rows(args.input, config=SFT_CONFIG, split=SFT_SPLIT)
        pref_rows = load_rows(args.input, config=PREF_CONFIG, split=PREF_SPLIT)
        sft_records, sft_stats = convert(sft_rows, mode="train", origin=SFT_CONFIG)
        pref_records, pref_stats = convert(pref_rows, mode="train", origin=PREF_CONFIG)
        pool = sft_records + pref_records
        picked = balance(pool, args.n, seed=args.seed)
        records = picked
        stats = {"kept": len(picked), **{f"sft_{k}": v for k, v in sft_stats.items()},
                 **{f"pref_{k}": v for k, v in pref_stats.items()}}
        report = balance_report(pool, picked)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[when2call:{args.mode}] {report}")
    counts = Counter(r["decision"] for r in records)
    print(f"  wrote {len(records)} records: {dict(counts)}")
    print(f"  -> {out_path}")

    if not records:
        raise SystemExit("no When2Call records converted -- loader or config is wrong")


if __name__ == "__main__":
    main()