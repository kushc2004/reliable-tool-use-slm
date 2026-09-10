"""Offline When2Call-shaped generator, for smoke-testing the decision scorer.

The real When2Call test set lives on the Hub. This module produces a small
stand-in with the same four-way label distribution and the same record shape,
so ``evaluate_when2call`` and ``decision_metrics`` can be exercised with no
network and no GPU.

It is deliberately *not* a substitute for the real benchmark. Any number
computed against this corpus is a pipeline check, not a research result, and
the smoke run says so in its output.

Usage:
    python -m src.data.synth_when2call --out data/processed/w2c_eval.jsonl \\
        --n-per-class 40 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ..schema import CALL_CLOSE, CALL_OPEN

__all__ = ["DECISIONS", "generate", "main"]

DECISIONS = ["tool_call", "direct", "request_for_info", "cannot_answer"]

# A small tool set, distinct in name from synth.TRAIN_TOOLS so nothing here can
# be mistaken for training data by accident.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_invoice",
            "description": "Look up an invoice by its identifier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string", "description": "Invoice identifier"},
                    "currency": {"type": "string", "description": "Currency to quote in"},
                },
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_id",
            "description": "Retrieve the identifier for a service in a province.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_id": {"type": "integer", "description": "Service type id"},
                    "province_id": {"type": "integer", "description": "Province id"},
                },
                "required": ["service_id", "province_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the status of a customer order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order identifier"},
                },
                "required": ["order_id"],
            },
        },
    },
]

_TOOL_CALL_CASES = [
    ("Where is order ORD-4471 right now?", "check_order_status", {"order_id": "ORD-4471"}),
    ("Look up invoice INV-2210 for me.", "lookup_invoice", {"invoice_id": "INV-2210"}),
    ("What is the status of order ORD-9930?", "check_order_status", {"order_id": "ORD-9930"}),
    ("Find invoice INV-8801 please.", "lookup_invoice", {"invoice_id": "INV-8801"}),
]

_DIRECT_CASES = [
    "What is 17 times 24?",
    "Explain the difference between TCP and UDP.",
    "Give me a synonym for 'tenacious'.",
    "How many sides does a hexagon have?",
]

_REQUEST_CASES = [
    ("Where is my order?", "order_id"),
    ("Look up my invoice.", "invoice_id"),
    ("Check the order status.", "order_id"),
    ("Find the invoice for me.", "invoice_id"),
]

_REQUEST_REPLIES = [
    "To help you with that, could you please provide the order ID?",
    "Sure - what is the invoice identifier you would like me to look up?",
    "Happy to check. Could you tell me which order ID you mean?",
    "I can do that. Could you specify the invoice ID?",
]

_CANNOT_CASES = [
    "Book me a flight to Lisbon next Tuesday.",
    "What is the air quality in Hanoi right now?",
    "Track my parcel SF1122334455 for me.",
    "Find me a recipe for pho for four people.",
]

_CANNOT_REPLIES = [
    "Apologies, I'm unable to help with that - I don't have access to flight booking.",
    "I can't provide that; none of the available functions cover air quality data.",
    "Sorry, that is outside my capabilities - I have no parcel tracking function.",
    "I don't have access to recipe data, so I can't help with that request.",
]


def _call_block(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{CALL_OPEN}\n{payload}\n{CALL_CLOSE}"


def generate(n_per_class: int = 40, seed: int = 0) -> list[dict[str, Any]]:
    """Build a balanced four-way corpus with the real record shape."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    def add(index: int, decision: str, question: str, gold_response: str,
            calls: list[dict[str, Any]], tools: list[dict[str, Any]],
            target_tool: str | None = None,
            held_out_param: str | None = None) -> None:
        records.append({
            "id": f"synth-w2c-{decision}-{index:04d}",
            "split": "when2call",
            "decision": decision,
            "source": "synthetic",
            "source_id": None,
            "tools": tools,
            "messages": [{"role": "user", "content": question}],
            "gold_calls": calls,
            "expects_call": decision == "tool_call",
            "gold_response": gold_response,
            "answers": {},
            "target_tool": target_tool,
            "held_out_param": held_out_param,
        })

    # tool_call: everything needed is present and a tool fits.
    for index in range(n_per_class):
        question, name, args = _TOOL_CALL_CASES[index % len(_TOOL_CALL_CASES)]
        add(index, "tool_call", question, _call_block(name, args),
            [{"name": name, "arguments": args}], rng.sample(_TOOLS, len(_TOOLS)),
            target_tool=name)

    # direct: a tool is advertised but none is needed.
    for index in range(n_per_class):
        question = _DIRECT_CASES[index % len(_DIRECT_CASES)]
        add(index, "direct", question,
            "I can answer that directly without calling a function.",
            [], rng.sample(_TOOLS, 2))

    # request_for_info: a tool fits, but a required argument is missing.
    for index in range(n_per_class):
        question, missing = _REQUEST_CASES[index % len(_REQUEST_CASES)]
        reply = _REQUEST_REPLIES[index % len(_REQUEST_REPLIES)]
        add(index, "request_for_info", question, reply, [],
            rng.sample(_TOOLS, len(_TOOLS)), held_out_param=missing)

    # cannot_answer: the tools on offer cannot satisfy the request.
    for index in range(n_per_class):
        question = _CANNOT_CASES[index % len(_CANNOT_CASES)]
        reply = _CANNOT_REPLIES[index % len(_CANNOT_REPLIES)]
        add(index, "cannot_answer", question, reply, [], [])

    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = generate(n_per_class=args.n_per_class, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[synth-when2call] wrote {len(records)} records "
          f"({args.n_per_class} per class) -> {out_path}")
    print("                  SYNTHETIC: pipeline validation only, not a benchmark result")


if __name__ == "__main__":
    main()