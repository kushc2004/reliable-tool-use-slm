# Technical report

## What was built, and what changed from upstream

`XIECHENG6/small-llms-tool-use` studies QLoRA function calling across five model families and reports exact-match improvements from adding an EOS token, first-object-only JSON parsing, and stop sequences. This project keeps that QLoRA recipe and its data-preprocessing approach, and narrows the scope to the axis the upstream work does not test.

**Kept from upstream**
- 4-bit NF4 QLoRA with double quantization, LoRA rank 16 / alpha 32, LR 2e-4, 3 epochs, max seq len 512, effective batch 16.
- The Qwen2.5-Instruct target family, reduced to one model (1.5B) so three checkpoints can be compared instead of five architectures.
- The held-out-function idea: whole function names excluded from training to test schema generalisation.
- `transformers.Trainer` + PEFT rather than TRL/Axolotl, so the whole thing installs and runs on a single Colab GPU.

**Added here**
- A four-way tool-decision benchmark built on NVIDIA When2Call, with `tool_call` / `direct` / `request_for_info` / `cannot_answer` preserved as distinct categories rather than collapsed into a binary.
- A third checkpoint (Reliable Tool-SFT) trained with balanced When2Call supervision on top of the same positive corpus, trained independently from the same base as Tool-SFT.
- The false-tool-call rate as a first-class metric, plus missing-info and cannot-answer accuracy.
- A perfect-oracle self-check that validates the scorer before any model output is trusted.
- An offline synthetic corpus so the pipeline is runnable and testable with no network and no GPU.

**Not done** — no hyperparameter sweep, no RLHF/DPO/GRPO, no RAG, no agent framework, no deployment surface. None of those affect the core comparison.

## Experimental design

Three checkpoints from the same base model `Qwen/Qwen2.5-1.5B-Instruct`:

| Checkpoint | Data | Variant |
|---|---|---|
| Base | — | no training |
| Tool-SFT | ~3K Glaive function-calling examples | `sft` (positives only) |
| Reliable Tool-SFT | ~3K Glaive + ~1K balanced When2Call decisions | `sft-neg` |

Training Tool-SFT and Reliable Tool-SFT independently — rather than continuing from the same adapter — is deliberate: a sequential run would confound "more data" with "more training", and the whole claim rests on the difference being the *data*.

When2Call training data is class-balanced before use. A corpus that is 80% `tool_call` would teach the model to call a tool by default, which is precisely the failure being measured.

## Metrics

**Tool-call track** (six metrics, `src/metrics.py`): JSON validity, tool-selection accuracy, argument accuracy (Jaccard over `(name, arg, value)` triples), exact call-set match, unseen-function accuracy, false-tool-call rate on the no-tool split.

**Decision track** (`src/decision_metrics.py`): four-way decision accuracy, macro F1, tool-call precision and recall, false tool-call rate, missing-info accuracy, cannot-answer accuracy, and a 4×4 confusion matrix.

## Verification

Two checks run before any model is trusted:

1. **Oracle check.** A backend that echoes the gold answer must score 100% on every decision metric. If it does not, the scorer is broken and no model number means anything.
2. **Unit tests.** `tests/test_decision_metrics.py` pins classification for each of the four categories and asserts the oracle property directly.

Both caught a real defect. The first oracle run scored 93.75% decision accuracy with missing-info accuracy at 75%. Diagnosis: the `request_for_info` cue list covered "could you provide/specify..." but not "what is the identifier you would like me to look up?", so ten genuine clarification rows were classified as `direct`.

The fix required care. The obvious widening — matching any `what is the <word>` — would have broken `cannot_answer`, because rows like "what is the air quality in Hanoi?" are unsupported requests, not clarifications. The patterns were widened with a bounded list of argument-shaped nouns (`invoice`, `order`, `identifier`, `ticker`, …) plus explicit frames (`what should I`, `what would you like`). Oracle returned to 100%, and the cases are now pinned by tests so the bug cannot silently return.

## Current status

The pipeline runs end to end. `MODE=smoke ./scripts/run_experiments.sh` builds the offline corpus, runs both evaluation tracks with scripted backends, writes `comparison.csv`, generates three figures, and produces `error_analysis.json` and `cv_metrics.md`.

**The full training run has not been executed.** `torch`, `transformers` and `peft` are absent from this environment and there is no GPU, so `MODE=full` cannot run here. Consequently:

- The numbers currently in `results/` come from scripted backends and are **pipeline-validation artifacts, not model results.**
- The "key finding" section of the README is intentionally unwritten. It should be written from a real run, not from fixtures.

What the smoke run does establish is that the measurement apparatus works and reproduces the expected failure shape: the positive-only backend reaches 100% exact tool-call match while its false tool-call rate rises from 22.5% to 66.7%.

## How to finish

```bash
MODE=full ./scripts/run_experiments.sh
```

On a Colab L4 this trains both adapters and evaluates all three checkpoints. Then:

- `results/comparison.csv` — the three-way table
- `results/figures/exact_match.png`, `false_tool_call.png`, `confusion_matrix.png`
- `results/error_analysis.json` — 20 categorised failures
- `results/cv_metrics.md` — headline deltas, computed from the result files

GPU model, wall-clock training time and peak memory are not recorded by these scripts; capture those from the run log to complete the CV summary.

## Threats to validity

- The `direct` / `request_for_info` / `cannot_answer` split is heuristic cue matching. `classification_source` is stored per example so the share of decisions resting on the heuristic can be audited, and on real When2Call prose the cue lists will need widening again — the synthetic corpus is cleaner than real data.
- The synthetic corpus is small and templated, so smoke-run numbers overstate how separable the four classes are.
- Held-out-function accuracy depends on whole function names being excluded; if a real corpus reuses a name across splits the metric is meaningless. Verify this on the real data rather than assuming it.