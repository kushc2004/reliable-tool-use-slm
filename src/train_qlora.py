"""4-bit QLoRA supervised fine-tuning for tool use.

Deliberately plain: bitsandbytes for the 4-bit base, PEFT for the adapter, and a
transformers ``Trainer`` for the loop. No TRL, no Axolotl, no flash-attn build
step - the point is that this runs on a single L4 or a 24GB card without
fighting an install.

Two variants, and the difference between them is the whole experiment:

``sft``
    positive tool-call trajectories only. Teaches the model to call tools.

``sft-neg``
    the same positives plus no-tool examples. Teaches the model *not* to.

Loss is computed on assistant spans only (see ``schema.build_segments``), so the
model never learns to generate tool output or user turns.

Usage:
    python -m src.train_qlora --config configs/qwen2.5-1.5b.yaml \\
        --data data/processed --variant sft-neg --out runs/sft-neg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset

from .schema import IM_END, build_segments

__all__ = ["ChatMLDataset", "load_config", "train", "main"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class ChatMLDataset(Dataset):
    """Tokenizes records and masks every non-assistant token with -100."""

    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        input_ids: list[int] = []
        labels: list[int] = []

        for text, trainable in build_segments(record):
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(ids)
            labels.extend(ids if trainable else [-100] * len(ids))

        # Always end on the assistant terminator so the model learns to stop.
        end_ids = self.tokenizer(IM_END, add_special_tokens=False)["input_ids"]
        input_ids.extend(end_ids)
        labels.extend(end_ids)

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


def _collate(pad_token_id: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in batch)
        out: dict[str, list[torch.Tensor]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for item in batch:
            pad = width - len(item["input_ids"])
            out["input_ids"].append(
                torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
            )
            out["labels"].append(
                torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)])
            )
            out["attention_mask"].append(
                torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
        return {key: torch.stack(value) for key, value in out.items()}

    return collate


def _load_records(data_dir: Path, variant: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if variant == "sft":
        # Positive-only ablation: drop every no-tool example.
        kept = [r for r in records if r.get("expects_call")]
        print(f"[sft] dropping {len(records) - len(kept)} no-tool records for the positive-only ablation")
        records = kept
    elif variant == "sft-neg":
        negatives = [r for r in records if not r.get("expects_call")]
        positives = [r for r in records if r.get("expects_call")]
        # Oversample negatives 2x: they are the minority and the behaviour we
        # are actually trying to install.
        records = positives + negatives * 2
        print(f"[sft-neg] {len(positives)} positive + {len(negatives)} negative (x2 oversampled)")
    else:
        raise SystemExit(f"unknown variant: {variant} (expected sft or sft-neg)")

    return records


def train(config: dict[str, Any], data_dir: Path, variant: str, out_dir: Path) -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    model_name = config["model_name"]
    records = _load_records(data_dir, variant)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ChatMLDataset(records, tokenizer, max_length=config.get("max_length", 2048))

    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=config.get("num_train_epochs", 2),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2.0e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        logging_steps=config.get("logging_steps", 10),
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        report_to=[],
        seed=config.get("seed", 0),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=_collate(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "variant": variant, "n_records": len(records)}, handle, indent=2)

    print(f"saved adapter to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--variant", choices=["sft", "sft-neg"], default="sft")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out or config.get("output_dir", f"runs/{args.variant}"))
    train(config, Path(args.data), args.variant, out_dir)


if __name__ == "__main__":
    main()