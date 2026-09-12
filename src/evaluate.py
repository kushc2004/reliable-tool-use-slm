"""Generate predictions for the eval splits and compute the six metrics.

Backends:

``hf``      a real transformers model, optionally with a PEFT adapter
``dummy``   a scripted oracle backend that degrades on purpose

The dummy backend exists so the metric layer can be exercised end to end with no
GPU, no weights and no download. It reads the gold answer and degrades it
deliberately, which is what you want when the question is "is my
false-tool-call metric actually measuring what I think it is".

Usage:
    python -m src.evaluate --data data/processed --split all --backend dummy --out runs/dummy
    python -m src.evaluate --data data/processed --split all \\
        --checkpoint Qwen/Qwen2.5-1.5B-Instruct --adapter runs/sft-neg --out runs/sft-neg
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

from .metrics import compute_metrics, format_table
from .schema import CALL_CLOSE, CALL_OPEN, IM_END, render_conversation

__all__ = ["load_eval", "dummy_backend", "hf_backend", "run_eval", "main"]

BACKENDS = ["hf", "dummy"]


# --------------------------------------------------------------------------- #
# Eval data
# --------------------------------------------------------------------------- #


def load_eval(data_dir: Path, split: str = "all") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with (data_dir / "eval.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if split != "all":
        records = [r for r in records if r.get("split") == split]
    return records


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _render_gold(record: dict[str, Any]) -> str:
    if not record.get("gold_calls"):
        content = ""
        for message in record.get("messages") or []:
            if message["role"] == "assistant" and message.get("content"):
                content = message["content"]
        return content or "I can answer that directly."
    blocks = []
    for call in record["gold_calls"]:
        payload = json.dumps(
            {"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False
        )
        blocks.append(f"{CALL_OPEN}\n{payload}\n{CALL_CLOSE}")
    return "\n".join(blocks)


def dummy_backend(failure_rate: float = 0.0, seed: int = 0) -> Callable[[dict[str, Any]], str]:
    """Oracle backend that degrades itself in a controlled way.

    With ``failure_rate=0`` it never abstains on no-tool prompts: it always
    emits a call. That is the point of the fixture. A model that always calls a
    tool looks perfect on tool-call accuracy and terrible on
    ``false_tool_call_rate``, and the report should show exactly that.
    """
    rng = random.Random(seed)

    def predict(record: dict[str, Any]) -> str:
        if record.get("expects_call"):
            if failure_rate and rng.random() < failure_rate:
                return "I'm not sure which function to use."
            return _render_gold(record)

        # The characteristic failure of a positive-only SFT run.
        if failure_rate and rng.random() < failure_rate:
            return "I can answer that directly."
        tools = record.get("tools") or []
        if not tools:
            return "I can answer that directly."
        name = tools[0]["function"]["name"]
        return f'{CALL_OPEN}\n{{"name": "{name}", "arguments": {{}}}}\n{CALL_CLOSE}'

    return predict


def hf_backend(
    checkpoint: str,
    adapter: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 1,
) -> Callable[[dict[str, Any]], str]:
    """Real generation backend: 4-bit base plus optional LoRA adapter.

    ``batch_size > 1`` generates several prompts per forward pass. The returned
    callable then carries ``.batch`` (list -> list) and ``.batch_size``
    attributes, which the evaluators pick up when present.

    Batching is the difference between a decision eval that costs 7 hours and
    one that costs 40 minutes, and it was the single largest term in a 10h47m
    run. Left padding is mandatory for batched decoder-only generation: with
    right padding every sequence after the first is conditioned on pad tokens.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    eos_id = tokenizer.convert_tokens_to_ids(IM_END)
    prev_side = tokenizer.padding_side

    def batch_predict(records: list[dict[str, Any]]) -> list[str]:
        prompts = [render_conversation(r, add_generation_prompt=True) for r in records]
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = prev_side
        inputs = inputs.to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature or None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
        return [text.split(IM_END)[0] for text in decoded]

    def predict(record: dict[str, Any]) -> str:
        return batch_predict([record])[0]

    predict.batch = batch_predict  # type: ignore[attr-defined]
    predict.batch_size = max(1, batch_size)  # type: ignore[attr-defined]
    return predict


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_eval(
    records: list[dict[str, Any]],
    predict: Callable[[dict[str, Any]], str],
    out_dir: Path,
    limit: int | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    if limit:
        records = records[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[dict[str, Any], str]] = []
    predictions: list[dict[str, Any]] = []
    batch_fn = getattr(predict, "batch", None)
    batch_size = getattr(predict, "batch_size", 1) or 1
    if batch_fn is not None and batch_size > 1:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            for record, prediction in zip(chunk, batch_fn(chunk)):
                pairs.append((record, prediction))
                predictions.append(
                    {
                        "id": record.get("id"),
                        "split": record.get("split"),
                        "prediction": prediction,
                    }
                )
            if min(start + batch_size, len(records)) % 100 < batch_size:
                print(f"  generated {min(start + batch_size, len(records))}/{len(records)}")
    else:
        for index, record in enumerate(records):
            prediction = predict(record)
            pairs.append((record, prediction))
            predictions.append(
                {"id": record.get("id"), "split": record.get("split"), "prediction": prediction}
            )
            if (index + 1) % 100 == 0:
                print(f"  generated {index + 1}/{len(records)}")

    metrics = compute_metrics(pairs)
    if backend:
        # Stamped into metrics.json so a result carries the conditions that
        # produced it. Without this, "which backend made these numbers" is
        # unanswerable after the fact -- which is how the fixture results got
        # as far as a comparison table.
        metrics["backend"] = backend
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    examples = metrics.pop("examples", [])
    failures = [
        row
        for row in examples
        if (row["expects_call"] and not row["exact_match"])
        or row["emitted_call_when_none_expected"]
    ]
    with (out_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures[:200]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print()
    print(format_table(metrics))
    print()
    print(f"n={metrics['overall']['n']}  failures dumped: {len(failures)}")
    print(f"wrote {out_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--split", default="all",
                        help="all, heldout_tools, unseen_functions or no_tool")
    # Defaults to hf, NOT dummy. A bare `--checkpoint ... --out ...` invocation
    # must not silently score the test fixture: that is exactly how a full
    # training run once produced three byte-identical "results", because the
    # notebook passed --checkpoint and --adapter but never --backend. Failing
    # loudly on a missing torch install is strictly better than faking a table.
    parser.add_argument("--backend", choices=BACKENDS, default="hf")
    parser.add_argument("--checkpoint", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="generate this many prompts at once; 1 disables batching")
    parser.add_argument("--failure-rate", type=float, default=0.0,
                        help="dummy backend only: how often to degrade a prediction")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_eval(Path(args.data), args.split)
    print(f"loaded {len(records)} eval records from {args.data} (split={args.split})")
    if not records:
        raise SystemExit("no eval records; run src/data/build_dataset.py first")

    if args.backend == "dummy":
        predict = dummy_backend(failure_rate=args.failure_rate, seed=args.seed)
    else:
        predict = hf_backend(
            args.checkpoint,
            args.adapter,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    run_eval(records, predict, Path(args.out), limit=args.limit, backend=args.backend)


if __name__ == "__main__":
    main()