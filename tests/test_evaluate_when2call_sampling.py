"""Regression tests for the label-sorted ``--limit`` bug.

The real When2Call ``mcq`` split is stored grouped by gold label. ``records[:limit]``
therefore returns one class, and because the class that lands first is
``cannot_answer``, the failure mode is invisible: the scorer still reports a
``decision_accuracy``, it is just one-class accuracy wearing a four-way label.
"""
from __future__ import annotations

import json

from src.evaluate_when2call import (
    REQUIRED_DECISIONS,
    load_when2call,
    stratified_sample,
)


def _make_records(n_cannot: int, n_request: int, n_tool: int) -> list[dict]:
    """A file in the real layout: grouped by label, not shuffled."""
    records = []
    for label, count in (
        ("cannot_answer", n_cannot),
        ("request_for_info", n_request),
        ("tool_call", n_tool),
    ):
        for i in range(count):
            records.append({"id": f"{label}-{i}", "decision": label})
    return records


def test_head_slice_is_single_class():
    """The bug being fixed, stated as a test.

    Counts mirror the real file (1295 / 1062 / 1295) so the head is genuinely
    single-class, which is what made the original bug invisible.
    """
    records = _make_records(1295, 1062, 1295)
    head = records[:1200]
    assert {r["decision"] for r in head} == {"cannot_answer"}


def test_stratified_sample_keeps_every_class():
    records = _make_records(1295, 1062, 1295)
    picked = stratified_sample(records, 1200, seed=0)
    assert len(picked) == 1200
    assert {r["decision"] for r in picked} == set(REQUIRED_DECISIONS)


def test_stratified_sample_is_roughly_proportional():
    records = _make_records(1295, 1062, 1295)
    picked = stratified_sample(records, 1200, seed=0)
    counts: dict[str, int] = {}
    for record in picked:
        counts[record["decision"]] = counts.get(record["decision"], 0) + 1
    assert abs(counts["cannot_answer"] / 1200 - 1295 / 3652) < 0.02
    assert abs(counts["request_for_info"] / 1200 - 1062 / 3652) < 0.02
    assert abs(counts["tool_call"] / 1200 - 1295 / 3652) < 0.02


def test_stratified_sample_is_deterministic():
    records = _make_records(50, 50, 50)
    first = stratified_sample(records, 90, seed=0)
    second = stratified_sample(records, 90, seed=0)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_stratified_sample_seed_changes_the_sample():
    records = _make_records(50, 50, 50)
    a = stratified_sample(records, 90, seed=0)
    b = stratified_sample(records, 90, seed=1)
    assert [r["id"] for r in a] != [r["id"] for r in b]
    assert {r["decision"] for r in b} == {r["decision"] for r in a}


def test_limit_above_length_returns_everything():
    records = _make_records(10, 10, 10)
    assert len(stratified_sample(records, 1000, seed=0)) == 30


def test_limit_below_class_count_still_returns_rows():
    records = _make_records(10, 10, 10)
    assert len(stratified_sample(records, 2, seed=0)) == 2


def test_load_when2call_no_limit_is_a_passthrough(tmp_path):
    path = tmp_path / "w2c.jsonl"
    records = _make_records(5, 4, 5)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert len(load_when2call(path)) == 14


def test_load_when2call_with_limit_is_stratified(tmp_path):
    path = tmp_path / "w2c.jsonl"
    records = _make_records(200, 150, 200)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    loaded = load_when2call(path, limit=120, seed=0)
    assert len(loaded) == 120
    assert {r["decision"] for r in loaded} == set(REQUIRED_DECISIONS)


def test_required_decisions_excludes_direct():
    """``direct`` is genuinely absent from the real mcq split."""
    assert "direct" not in REQUIRED_DECISIONS
    assert set(REQUIRED_DECISIONS) == {
        "tool_call",
        "request_for_info",
        "cannot_answer",
    }
