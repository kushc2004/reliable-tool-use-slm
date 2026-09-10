"""Offline tests for corpus construction, splitting and masking."""
from __future__ import annotations

import json
from pathlib import Path

from src.data import synth
from src.data.build_dataset import build, load_jsonl
from src.schema import build_segments, parse_tool_calls, render_conversation


def test_synth_is_deterministic():
    first = synth.generate_records(n_train=50, n_eval=12, neg_ratio=0.2, seed=7)
    second = synth.generate_records(n_train=50, n_eval=12, neg_ratio=0.2, seed=7)
    assert [r["id"] for r in first["train"]] == [r["id"] for r in second["train"]]


def test_unseen_tools_never_appear_in_training_targets():
    corpus = synth.generate_records(n_train=200, n_eval=30, neg_ratio=0.2, seed=1)
    unseen = {tool.name for tool in synth.UNSEEN_TOOLS}
    train_targets = {
        call["name"] for record in corpus["train"] for call in (record.get("gold_calls") or [])
    }
    assert not (train_targets & unseen), "held-out functions leaked into training"

    eval_unseen = {
        call["name"]
        for record in corpus["eval"]
        if record["split"] == "unseen_functions"
        for call in record["gold_calls"]
    }
    assert eval_unseen, "unseen-functions split is empty"
    assert eval_unseen <= unseen


def test_negatives_are_present_and_balanced_by_ratio():
    corpus = synth.generate_records(n_train=100, n_eval=30, neg_ratio=0.2, seed=2)
    negatives = [r for r in corpus["train"] if not r["expects_call"]]
    assert len(negatives) == 20
    assert all(not r["gold_calls"] for r in negatives)


def test_every_generated_record_round_trips_through_the_parser():
    corpus = synth.generate_records(n_train=60, n_eval=30, neg_ratio=0.1, seed=3)
    for record in corpus["train"] + corpus["eval"]:
        for message in record["messages"]:
            if message["role"] != "assistant":
                continue
            rendered = render_conversation(
                {"tools": record["tools"], "messages": [message]}
            )
            parsed = parse_tool_calls(rendered)
            assert len(parsed.calls) == len(message.get("tool_calls") or [])
            assert parsed.json_valid


def test_build_writes_splits_and_stats(tmp_path: Path):
    stats = build(
        sources=["synth"],
        out_dir=tmp_path,
        n_train=90,
        n_eval=30,
        neg_ratio=0.2,
        seed=0,
        limit=10,
        neg_limit=10,
    )
    train = load_jsonl(tmp_path / "train.jsonl")
    eval_records = load_jsonl(tmp_path / "eval.jsonl")

    assert len(train) == stats["train"]["n"]
    assert stats["train"]["no_call_expected"] > 0
    assert stats["eval"]["no_call_expected"] > 0
    assert set(stats["eval"]["by_split"]) >= {"heldout_tools", "unseen_functions", "no_tool"}
    assert json.loads((tmp_path / "stats.json").read_text())["train"]["n"] == len(train)
    assert eval_records


def test_only_assistant_spans_are_trainable():
    corpus = synth.generate_records(n_train=1, n_eval=1, seed=0)
    record = corpus["train"][0]
    segments = build_segments(record)
    trainable = [text for text, is_trainable in segments if is_trainable]
    assert len(trainable) == 1
    # The system prompt advertises tools but must never be trained on.
    assert all("<tools>" not in text for text in trainable)