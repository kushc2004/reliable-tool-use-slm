# Reliable Tool-Use SLMs via QLoRA Post-Training

Post-train a small instruct model (Qwen2.5-1.5B) with 4-bit QLoRA so it **calls tools correctly when it should, and does not call them when it shouldn't.**

The distinguishing axis is not "fine-tune a model for function calling" — that is well covered. It is the **negative-supervision** axis: most tool-calling data teaches a model to call tools, and a model that calls a tool on every prompt looks excellent on tool-call accuracy while being unusable in production.

## Problem

Small LLMs frequently make incorrect tool calls, and — more damagingly — call tools when no tool is warranted. Standard function-calling benchmarks score only whether the right function was called with the right arguments, so they do not penalise a model that fires a tool at "what is 17 times 24?". This project measures that failure directly.

## Method

Three checkpoints, trained/evaluated from the same base so the comparison is clean:

| Checkpoint | Training data | Isolates |
|---|---|---|
| **Base** | none | zero-shot behaviour |
| **Tool-SFT** | ~3K function-calling examples | what positive supervision buys |
| **Reliable Tool-SFT** | ~3K function-calling + ~1K balanced When2Call decisions | what negative supervision buys on top |

Training is 4-bit NF4 QLoRA (rank 16, alpha 32, LR 2e-4, 3 epochs, max seq len 512, effective batch 16) via PEFT + `transformers.Trainer`. Tool-SFT and Reliable Tool-SFT are trained **independently from the same base**, not sequentially, so the only difference is the data.

## Evaluation

Two independent tracks, both scored from raw generations:

**Tool-call track** — JSON validity, function-name accuracy, argument accuracy, exact call-set match, and held-out-function exact match (whole function *names* excluded from training, so this measures schema generalisation rather than phrasing memorisation).

**Tool-decision track (When2Call)** — four-way decision accuracy over `tool_call` / `direct` / `request_for_info` / `cannot_answer`, plus the metrics that give the project its identity:

- **false tool-call rate** — of all samples where no tool was warranted, how often a call was emitted anyway
- **missing-info accuracy** — on under-specified prompts, did it ask rather than invent an argument
- **cannot-answer accuracy** — did it decline rather than call the nearest available function

## Results

> **These numbers come from `MODE=smoke`, which uses scripted backends, not a trained model.** They validate the pipeline and the metric layer end to end; they are *not* research results. Re-run with `MODE=full` on a GPU to replace them. See `results/cv_metrics.md`.

| Metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Exact tool-call match | 30.0% | 100.0% | 81.0% |
| Function-name accuracy | 30.0% | 100.0% | 81.0% |
| Argument accuracy | 30.0% | 100.0% | 81.0% |
| JSON validity | 100.0% | 100.0% | 100.0% |
| Held-out function EM | 32.0% | 100.0% | 78.0% |
| When2Call decision accuracy | 72.5% | 25.0% | 35.6% |
| False tool-call rate (lower better) | 22.5% | 66.7% | 57.5% |
| Missing-info accuracy | 70.0% | 0.0% | 22.5% |
| Cannot-answer accuracy | 70.0% | 0.0% | 20.0% |

The smoke run reproduces the *shape* of the failure the project is built to detect: the positive-only backend (Tool-SFT) drives tool-call accuracy to 100% while its false tool-call rate climbs from 22.5% to 66.7%. That is the trade-off the real experiment is designed to measure.

## Key finding

**Not yet written.** The smoke backends are scripted, so any conclusion drawn from them would be circular. Run `MODE=full` and let `src/cv_metrics.py` compute the deltas; write the finding from those.

## Reproduction

Smoke run — no GPU, no downloads, validates the whole pipeline:

```bash
MODE=smoke ./scripts/run_experiments.sh
```

Full run — needs a GPU (Colab L4/A100), network, and the Hub datasets:

```bash
MODE=full ./scripts/run_experiments.sh
```

Individual stages:

```bash
python -m src.data.build_dataset --source glaive --out data/processed --n-train 3000 --n-eval 500 --seed 0
python -m src.data.prepare_when2call --mode train --n 1000 --out data/raw/w2c_train.jsonl
python -m src.data.prepare_when2call --mode eval --out data/raw/w2c_eval.jsonl
python -m src.train_qlora --config configs/tool_sft.yaml --data data/processed --variant sft --out outputs/tool_sft
python -m src.train_qlora --config configs/reliable_tool_sft.yaml --data data/reliable --variant sft-neg --out outputs/reliable_tool_sft
```

The scorer is verified against a perfect oracle before any model is trusted — a step that caught a real classifier bug during development (see below).

```bash
python -m pytest tests/ -q
python -m src.evaluate_when2call --data data/raw/w2c_eval.jsonl --backend oracle --out results/oracle_when2call
```

## Layout

```
configs/            QLoRA hyperparameters for the two trained checkpoints
src/schema.py       canonical <tool_call> format: render, parse, normalize
src/metrics.py      the six tool-call metrics + per-split breakdown
src/decision_metrics.py   four-way When2Call classifier and metrics
src/train_qlora.py  4-bit QLoRA SFT on a plain transformers Trainer
src/evaluate.py     tool-call generation + metrics, pluggable backend
src/evaluate_when2call.py  decision-track evaluation
src/aggregate_results.py   comparison CSV, markdown table, figures
src/error_analysis.py       failure categorisation
src/cv_metrics.py           one-page CV summary from the result files
src/data/           synth, glaive/hermes, when2call, dataset builder
scripts/run_experiments.sh  end-to-end driver (smoke | full)
tests/              offline tests for the parser, metrics, classifier
```

## A note on the classifier

`direct`, `request_for_info` and `cannot_answer` are prose, so they are separated by ordered cue matching rather than by parsing. That is a heuristic, and it is recorded per example in `classification_source` so a misclassification can be audited instead of silently trusted.

During development the oracle sanity check scored 93.75% instead of 100%, which exposed a genuine gap: the cue list matched "could you provide..." but not "what is the identifier you would like me to look up?". Ten legitimate `request_for_info` rows were falling through to `direct`, and the missing-info metric read 75% instead of 100%. The patterns were widened with a bounded argument-noun list (deliberately not a bare `what is the \w+`, which would have swallowed `cannot_answer` rows like "what is the air quality in Hanoi?") and pinned by `tests/test_decision_metrics.py`.

## Limitations

- The full training run has **not** been executed in this environment: `torch`, `transformers` and `peft` are not installed and no GPU is available. Everything up to and including the metric layer is exercised; the model numbers are not yet produced.
- The synthetic corpus under `data/processed` is a pipeline fixture, not a substitute for Glaive/When2Call.
- `prepare_when2call.py` parses the published When2Call field shapes defensively and logs skip counts; run it once on the real data and check the stats.

## Acknowledgements

The experimental design follows **NVIDIA/When2Call** (Ross, Mahabaleshwarkar & Suhara, NAACL 2025) for the tool-decision benchmark, and the QLoRA configuration follows **XIECHENG6/small-llms-tool-use**. The implementation here was written against those designs; it is not a fork of either, and no benchmark numbers are copied from them. Glaive Function Calling v2 and Salesforce xlam-function-calling-60k are used as upstream training corpora.