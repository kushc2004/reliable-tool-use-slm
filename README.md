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

Three complementary evaluations are reported. The first two use raw generations;
the third is the standardized NVIDIA When2Call multiple-choice benchmark.

**Tool-call track** — JSON validity, function-name accuracy, argument accuracy, exact call-set match, and held-out-function exact match (whole function *names* excluded from training, so this measures schema generalisation rather than phrasing memorisation).

**Tool-decision track (When2Call)** — decision accuracy over `tool_call` / `request_for_info` / `cannot_answer`, plus the metrics that give the project its identity:

- **false tool-call rate** — of all samples where no tool was warranted, how often a call was emitted anyway
- **missing-info accuracy** — on under-specified prompts, did it ask rather than invent an argument
- **cannot-answer accuracy** — did it decline rather than call the nearest available function

**Official When2Call MCQ** — all 3,652 examples from NVIDIA's published
`when2call-qwen2_5` LM-Eval task, reporting raw accuracy, length-normalized
accuracy, macro-F1, the benchmark-defined hallucination rate, and the answer
category confusion matrix. This evaluation does not depend on the project's
free-form cue classifier.

Two caveats on this track, both measured against the real dataset rather than assumed:

- The `mcq` test split carries all four answer strings on every row, but **no row is gold-`direct`** — the label distribution is `tool_call` 35.5%, `cannot_answer` 35.5%, `request_for_info` 29.1%. `direct` accuracy is therefore **not measurable on the real benchmark**; the scorer still reports it, and it is meaningful only on the synthetic fixture.
- The training negatives are **not class-balanced and cannot be made so**. When2Call contributes 481 `request_for_info`, 509 `cannot_answer` and just **10 `direct`** rows, so any balanced subset is capped at ten examples per class. The word "balanced" was removed rather than quietly oversampling a class the benchmark barely contains.

## Results

The strongest decision result is the full official When2Call MCQ evaluation.
It was run on a Modal L40S using the already-trained public Kaggle adapters;
**no retraining was performed**. All three checkpoints see the same 3,652
benchmark examples.

| Official When2Call MCQ metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Accuracy | 47.21% | 43.81% | **69.17%** |
| Length-normalized accuracy | 52.79% | 49.81% | **71.03%** |
| Macro F1 | 30.63% | 26.26% | **52.01%** |
| Hallucination rate ↓ | 24.42% | 40.70% | **8.14%** |

Relative to positive-only Tool-SFT, Reliable Tool-SFT improves normalized
accuracy by **21.22 percentage points**, improves macro-F1 by **25.75 points**,
and reduces the official hallucination rate by **32.56 points (80.0% relative)**.
From the official confusion matrix it still recalls **79.0%** of gold tool-call
cases (1023/1295), so the standardized evaluation shows a substantially better
reliability/recall balance than the earlier free-form diagnostic.

The original generation-based evaluation remains useful because it tests actual
emitted JSON/function calls rather than MCQ choice likelihoods. Those real-model
results were run with the Hugging Face backend on a Kaggle Tesla T4:

| Metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Exact tool-call match | 0.0% | **94.6%** | 86.1% |
| Function-name accuracy | 0.0% | **95.2%** | 87.3% |
| Argument accuracy | 0.0% | **94.9%** | 86.7% |
| JSON validity | 100.0% | 99.2% | **100.0%** |
| Held-out function EM | 0.0% | **96.4%** | 78.3% |
| When2Call decision accuracy | 23.2% | 37.3% | **62.1%** |
| False tool-call rate ↓ | 0.0%* | 89.9% | **11.2%** |
| Missing-info accuracy | 66.2% | 10.0% | **73.4%** |
| Cannot-answer accuracy | 11.3% | 0.0% | **66.7%** |

The tool-call track contains 249 examples (166 call-required and 83 no-tool); the When2Call decision track uses a fixed, label-stratified 1,200-example subset (425 `tool_call`, 349 `request_for_info`, 426 `cannot_answer`). The real `mcq` split has no gold `direct` rows, so direct-answer accuracy is not reported as a model claim.

Additional decision metrics expose the reliability/recall trade-off:

| Metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Macro F1 | 16.7% | 17.9% | **47.4%** |
| Tool-call precision | n/a | 37.2% | **70.2%** |
| Tool-call recall | 0.0% | **97.2%** | 48.2% |

\* Base's 0% false-tool-call rate is degenerate: the base model essentially never emits tool calls, so it cannot be treated as a strong reliability baseline. The meaningful negative-supervision comparison is Tool-SFT → Reliable Tool-SFT.

## Key finding

Positive-only function-calling SFT teaches the model **how to call tools** but
does not teach it reliably **when** to call them. On the full official When2Call
MCQ benchmark, Tool-SFT is actually worse than Base on macro-F1 (26.26% vs
30.63%) and has a 40.70% hallucination rate. Adding ~1K
clarification/refusal/no-tool examples raises macro-F1 to **52.01%**, normalized
accuracy to **71.03%**, and cuts hallucination to **8.14%**.

The generation-based track independently shows the same mechanism from another
angle: Tool-SFT reaches 94.6% exact call match but over-calls heavily, while
Reliable Tool-SFT reduces the free-form false-tool-call rate from 89.9% to
11.2%. The reliability gain is not free — exact call match falls from 94.6% to
86.1% and unseen-function EM from 96.4% to 78.3% — but the official benchmark
shows that call recall remains **79.0%** under standardized scoring rather than
collapsing completely.

The evaluated adapters use Qwen2.5-1.5B-Instruct with 4-bit QLoRA. The run metadata records 18,464,768 trainable parameters (2.036% of 907,081,216) and 4.86 GB peak GPU memory on a Tesla T4. The final evaluation run restored already-trained adapters, so the original training wall-clock duration is not reconstructed from this log.

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

### Official When2Call MCQ validation on Modal

The final standardized validation is **evaluation-only**: it does not retrain
either QLoRA adapter. `modal_jobs/when2call_mcq.py` downloads version 2 of the
public Kaggle adapter dataset (`kushchaudhari/reliable-tool-use-slm-adapters`),
caches `Qwen/Qwen2.5-1.5B-Instruct` on a persistent Modal Volume, and evaluates:

1. Base Qwen2.5-1.5B-Instruct
2. Tool-SFT adapter
3. Reliable Tool-SFT adapter

using NVIDIA's official `when2call-qwen2_5` LM-Eval-Harness MCQ task. The
When2Call and LM-Eval repositories are pinned to exact commits, and the runner
records those commits, GPU, precision, dataset version and timings in
`provenance.json`.

The asset download/model cache happens in a CPU-only Modal function so GPU time
is spent only on evaluation. The default GPU is an L40S; LM-Eval uses automatic
batch sizing with a configurable cap.

```bash
# The Modal CLI can also be invoked as `uvx modal ...` if it is not on PATH.

# First verify the complete stack cheaply.
uvx modal run modal_jobs/when2call_mcq.py --limit 32

# Then run all 3,652 official MCQ examples for all three checkpoints.
uvx modal run modal_jobs/when2call_mcq.py

# Download a completed run (the command printed by the job contains the exact path).
uvx modal volume get reliable-tool-use-slm-eval \
    results/<run-name> modal-results/<run-name>
```

The official MCQ run reports macro-F1, raw accuracy, length-normalized accuracy,
tool-hallucination rate and the answer-category confusion matrix. These metrics
are kept separate from the project's free-form 1,200-example decision track;
the latter measures generated behavior, while the MCQ benchmark provides the
standardized published comparison.

Completed full-run results are tracked under
`results/official_when2call_mcq/`. Provenance for the published run records an
NVIDIA L40S, BF16 + SDPA, all 3,652 examples, the exact When2Call and LM-Eval
commits, Kaggle adapter dataset version 2, and `training_performed: false`.

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

- The committed model metrics come from a real GPU evaluation. `MODE=smoke` still exists only as an offline pipeline fixture and must never be interpreted as model performance.
- The free-form When2Call diagnostic still uses a deterministic 1,200-example stratified subset and an auditable cue-based prose classifier. Its gold-answer oracle reaches 99.3%, so those diagnostic absolute scores retain a small measured classifier ceiling. The primary standardized decision result now comes from the full 3,652-example official MCQ evaluation and does not use that classifier.
- The official MCQ hallucination rate is NVIDIA's benchmark-defined metric (tool calls on a particular `cannot_answer`/no-tool subset). It is not numerically interchangeable with this project's broader free-form false-tool-call rate.
- The two arms differ in record count (3000 vs 4000) as well as in content, because the added negatives *are* the treatment. Both run the same 3 epochs, so gradient-step count differs slightly — that is part of the intervention, not a controlled quantity.
- `direct` accuracy is not measurable on the real When2Call test split (no gold-`direct` rows), and the training negatives are capped at 10 `direct` rows by the benchmark itself. See the Evaluation section.
- Results are from one QLoRA configuration/seed. The project deliberately avoids a large hyperparameter sweep, but the reported deltas do not include multi-seed confidence intervals.
- The held-out-function split withholds whole function names before training sampling (verified: 0 of 511 held-out names leak into training). Re-verify this invariant if the data-building logic changes.

## Acknowledgements

The experimental design follows **NVIDIA/When2Call** (Ross, Mahabaleshwarkar & Suhara, NAACL 2025) for the tool-decision benchmark, and the QLoRA configuration follows **XIECHENG6/small-llms-tool-use**. The implementation here was written against those designs; it is not a fork of either, and no benchmark numbers are copied from them. Glaive Function Calling v2 and Salesforce xlam-function-calling-60k are used as upstream training corpora.
