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
| **Tool-SFT** | ~3K function-calling positives | what positive supervision buys |
| **Reliable Tool-SFT** | the same ~3K positives **plus** ~1K no-tool / clarification / refusal examples | what negative supervision buys on top |

Both trained arms read **one shared corpus** and are separated by `--variant`, not by a separately-built dataset. That distinction matters: with two independently-built corpora the arms disagreed on 40 of 166 eval rows, so one model was graded on rows it had already trained on, and the "negative" corpus silently resolved to **zero** negatives because it was sourced from Glaive alone (Glaive contains no no-tool rows). Sharing the corpus makes the positive sets identical and leaves the negatives as the only variable.

Training is 4-bit NF4 QLoRA (rank 16, alpha 32, LR 2e-4, 3 epochs, max seq len 512, effective batch 16) via PEFT + `transformers.Trainer`. The two checkpoints are trained **independently from the same base**, not sequentially.

## Evaluation

Two independent tracks, both scored from raw generations:

**Tool-call track** — JSON validity, function-name accuracy, argument accuracy, exact call-set match, and held-out-function exact match (whole function *names* excluded from training, so this measures schema generalisation rather than phrasing memorisation).

**Tool-decision track (When2Call)** — decision accuracy over `tool_call` / `request_for_info` / `cannot_answer`, plus the metrics that give the project its identity:

- **false tool-call rate** — of all samples where no tool was warranted, how often a call was emitted anyway
- **missing-info accuracy** — on under-specified prompts, did it ask rather than invent an argument
- **cannot-answer accuracy** — did it decline rather than call the nearest available function

Two caveats on this track, both measured against the real dataset rather than assumed:

- The `mcq` test split carries all four answer strings on every row, but **no row is gold-`direct`** — the label distribution is `tool_call` 35.5%, `cannot_answer` 35.5%, `request_for_info` 29.1%. `direct` accuracy is therefore **not measurable on the real benchmark**; the scorer still reports it, and it is meaningful only on the synthetic fixture.
- The training negatives are **not class-balanced and cannot be made so**. When2Call contributes 481 `request_for_info`, 509 `cannot_answer` and just **10 `direct`** rows, so any balanced subset is capped at ten examples per class. The word "balanced" was removed rather than quietly oversampling a class the benchmark barely contains.

## Results

> **No model numbers exist yet.** Every figure in `results/` was produced by `MODE=smoke`, which uses *scripted backends* rather than a trained model. They validate the pipeline and the metric layer end to end; they are **not** research results. Run `MODE=full` on a GPU to produce the real ones.

The generated table lives in `results/comparison.md` and `results/comparison.csv`, written by `src/aggregate_results.py` on every run. It is deliberately **not** reproduced here: the smoke numbers move with `N_EVAL` and with the backend's `--failure-rate`, so a hand-copied table in this file goes stale the moment anyone re-runs — which is how the earlier version of this README came to disagree with its own results directory.

What the smoke run does establish is that the harness detects the failure the project is built around. The positive-only backend drives tool-call accuracy to 100% while its false tool-call rate rises from 22.5% to 66.7% — the exact trade-off the real experiment is designed to measure on a trained model.

## Key finding

**Not yet written.** The smoke backends are scripted, so any conclusion drawn from them would be circular. Run `MODE=full` and let `src/cv_metrics.py` compute the deltas; write the finding from those.

## Reproduction

Smoke run — no GPU, no downloads, validates the whole pipeline. `N_EVAL` controls how many eval rows each split contributes and therefore the exact percentages:

```bash
MODE=smoke N_EVAL=150 ./scripts/run_experiments.sh
```

Full run — needs a GPU (Colab L4/A100), network, and the Hub datasets:

```bash
MODE=full ./scripts/run_experiments.sh
```

Individual stages — note that both training arms read the same `data/processed` corpus:

```bash
# 1. one shared corpus: ~3000 positives + ~1000 negatives, all three eval splits
python -m src.data.build_dataset --source glaive,when2call --out data/processed \
    --n-train 4000 --n-eval 250 --neg-ratio 0.25 --seed 0

# 2. the When2Call test split, for the decision track
python -m src.data.prepare_when2call --mode eval --out data/raw/w2c_eval.jsonl

# 3. both arms from the same corpus; --variant selects which rows are used
python -m src.train_qlora --config configs/tool_sft.yaml \
    --data data/processed --variant sft --out outputs/tool_sft
python -m src.train_qlora --config configs/reliable_tool_sft.yaml \
    --data data/processed --variant sft-neg --out outputs/reliable_tool_sft
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

- The full training run has **not** been executed in this environment: `torch`, `transformers` and `peft` are not installed and no GPU is available. Everything up to and including the data layer is exercised against the real Hub corpora; the model numbers are not yet produced.
- Under `MODE=smoke` the corpus is synthetic and is a pipeline fixture only. Under `MODE=full`, `data/processed` is the real shared Glaive + When2Call corpus, verified to build 4000 train / 249 eval rows across all three splits (`heldout_tools`, `unseen_functions`, `no_tool`).
- The two arms differ in record count (3000 vs 4000) as well as in content, because the added negatives *are* the treatment. Both run the same 3 epochs, so gradient-step count differs slightly — that is part of the intervention, not a controlled quantity.
- `direct` accuracy is not measurable on the real When2Call test split (no gold-`direct` rows), and the training negatives are capped at 10 `direct` rows by the benchmark itself. See the Evaluation section.
- The `direct` / `request_for_info` / `cannot_answer` split is heuristic cue matching over prose. `classification_source` is stored per example so the share of decisions resting on the heuristic can be audited; on real When2Call prose the cue lists may still need widening.

## Acknowledgements

The experimental design follows **NVIDIA/When2Call** (Ross, Mahabaleshwarkar & Suhara, NAACL 2025) for the tool-decision benchmark, and the QLoRA configuration follows **XIECHENG6/small-llms-tool-use**. The implementation here was written against those designs; it is not a fork of either, and no benchmark numbers are copied from them. Glaive Function Calling v2 and Salesforce xlam-function-calling-60k are used as upstream training corpora.