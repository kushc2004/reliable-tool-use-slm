# Technical report

## What was built, and what changed from upstream

`XIECHENG6/small-llms-tool-use` studies QLoRA function calling across five model families and reports exact-match improvements from adding an EOS token, first-object-only JSON parsing, and stop sequences. This project keeps that QLoRA recipe and its data-preprocessing approach, and narrows the scope to the axis the upstream work does not test.

**Kept from upstream**
- 4-bit NF4 QLoRA with double quantization, LoRA rank 16 / alpha 32, LR 2e-4, 3 epochs, max seq len 512, effective batch 16.
- The Qwen2.5-Instruct target family, reduced to one model (1.5B) so three checkpoints can be compared instead of five architectures.
- The held-out-function idea: whole function names excluded from training to test schema generalisation.
- `transformers.Trainer` + PEFT rather than TRL/Axolotl, so the whole thing installs and runs on a single Colab GPU.

**Added here**
- A tool-decision track built on NVIDIA When2Call, preserving `tool_call` / `direct` / `request_for_info` / `cannot_answer` as distinct categories rather than collapsing them to a binary. The taxonomy is four-way; the *test split* only carries three gold labels (see Metrics).
- A third checkpoint (Reliable Tool-SFT) that reads the **same corpus** as Tool-SFT and differs only by `--variant`, so the positives are identical between arms and the negatives are the sole variable.
- The false-tool-call rate as a first-class metric, plus missing-info and cannot-answer accuracy.
- A perfect-oracle self-check that validates the scorer before any model output is trusted.
- An offline synthetic corpus so the pipeline is runnable and testable with no network and no GPU.
- A single-corpus data pipeline, which removes the cross-arm eval leakage and the zero-negatives failure documented under "Defects found while building".

**Not done** — no hyperparameter sweep, no RLHF/DPO/GRPO, no RAG, no agent framework, no deployment surface. None of those affect the core comparison.

## Experimental design

Three checkpoints from the same base model `Qwen/Qwen2.5-1.5B-Instruct`:

| Checkpoint | Data | Variant |
|---|---|---|
| Base | — | no training |
| Tool-SFT | ~3K tool-call positives (Glaive) | `sft` (positives only) |
| Reliable Tool-SFT | the same ~3K positives + ~1K no-tool / clarification / refusal examples | `sft-neg` |

Training the two arms independently — rather than continuing from one adapter — is deliberate: a sequential run would confound "more data" with "more training".

Both arms read **one shared corpus** built in a single pass. `--variant` then selects rows: `sft` keeps only `expects_call` records (~3000), `sft-neg` keeps everything (~4000). An earlier design built the two corpora separately and was replaced because it produced (a) 40 of 166 eval rows shared between one arm's eval set and the other arm's *training* set, and (b) a "negative" corpus built from Glaive alone — which contains no no-tool rows at all — so its negatives silently resolved to zero and the two arms would have trained on identical data.

The negatives are **not** class-balanced, and cannot be: When2Call contributes 481 `request_for_info`, 509 `cannot_answer` and only **10 `direct`** rows. Every class is capped at the smallest count rather than oversampling a category the benchmark barely contains.

## Metrics

**Tool-call track** (six metrics, `src/metrics.py`): JSON validity, tool-selection accuracy, argument accuracy (Jaccard over `(name, arg, value)` triples), exact call-set match, unseen-function accuracy, false-tool-call rate on the no-tool split.

**Decision track** (`src/decision_metrics.py`): decision accuracy, macro F1, tool-call precision and recall, false tool-call rate, missing-info accuracy, cannot-answer accuracy, and a 4×4 confusion matrix.

One caveat, measured rather than assumed: the `mcq` test split carries all four answer strings on every row but **no row is gold-`direct`** (`tool_call` 35.5%, `cannot_answer` 35.5%, `request_for_info` 29.1%). `direct` accuracy is therefore not measurable on the real benchmark; the metric is reported but is only meaningful on the synthetic fixture. The matrix keeps four columns regardless, because a model can wrongly *predict* `direct`.

## Verification

Two checks run before any model is trusted:

1. **Oracle check.** A backend that echoes the gold answer must score 100% on every decision metric. If it does not, the scorer is broken and no model number means anything.
2. **Unit tests.** `tests/test_decision_metrics.py` pins classification for each of the four categories and asserts the oracle property directly.

Both caught a real defect. The first oracle run scored 93.75% decision accuracy with missing-info accuracy at 75%. Diagnosis: the `request_for_info` cue list covered "could you provide/specify..." but not "what is the identifier you would like me to look up?", so ten genuine clarification rows were classified as `direct`.

The fix required care. The obvious widening — matching any `what is the <word>` — would have broken `cannot_answer`, because rows like "what is the air quality in Hanoi?" are unsupported requests, not clarifications. The patterns were widened with a bounded list of argument-shaped nouns (`invoice`, `order`, `identifier`, `ticker`, …) plus explicit frames (`what should I`, `what would you like`). Oracle returned to 100%, and the cases are now pinned by tests so the bug cannot silently return.

## Defects found while building

All of these were caught by running against the real Hub corpora, not by reading the code. Each would otherwise have silently produced a wrong or empty result.

1. **`prepare_glaive` read a column that does not exist.** Glaive ships `['system', 'chat']`; there is no `functions` column, so `row.get("functions")` was always `None` and every record was dropped as "no tools advertised" — the corpus built to **zero** training rows while reporting success. The schemas are embedded inside the `system` string.
2. **Glaive call blocks are unclosed.** They terminate with `<|endoftext|>`, not `</functioncall>`. A parser requiring the closing tag matched almost nothing: 410 of 11,100 call rows parsed. Fixing the terminator, plus a repair for single-quoted `arguments` containing literal newlines, took it to 11,099 of 20,000 rows.
3. **When2Call is config-scoped, not split-scoped.** `load_dataset(..., split="train")` raises `Unknown split "train"`; the configs are `test` / `train_sft` / `train_pref`. Worse, `train_sft` (15,000 rows) contains **no tool calls at all** — `train_pref` is the only source of `tool_call` supervision, with 3,000 of 9,000 `chosen_response` values carrying a call.
4. **Two corpora caused cross-arm eval leakage and zero negatives.** Separate builds disagreed on 40 of 166 eval rows (one arm graded on rows the other had trained on), and the reliable corpus — sourced from Glaive alone — resolved `--neg-ratio 0.25` to `wanted_neg = 0`. Both arms would have trained on identical positives-only data, making the headline claim vacuous. Replaced by a single shared corpus.
5. **The unseen-function split never fired.** It keyed on function names ending in `_eval`, a suffix no real function carries, so every "held-out" call had in fact been trained on. Whole names are now withheld before training rows are sampled; verified 0 of 511 names leak.
6. **bf16 was selected on a P100.** `torch.cuda.is_bf16_supported()` returned `True` on compute capability 6.0, which has no native bf16, so training silently ran with a dtype the hardware cannot do. The gate is now the compute capability (`>= 8`).
7. **A crashed run republished stale numbers.** `results/` is tracked in git, so a fresh clone arrives pre-populated; when the Kaggle eval steps crashed, `aggregate_results` re-read the committed table and printed it as though freshly measured. The notebook now deletes `results/` after cloning, so a failed run yields nothing rather than a misleading table.

Two guards cover the same class of failure: `build_dataset` refuses to write an empty training split and names the drop reasons, and `train_qlora` refuses to train on zero records. Both conditions were previously silent.

## Current status

The pipeline runs end to end. `MODE=smoke ./scripts/run_experiments.sh` builds the offline corpus, runs both evaluation tracks with scripted backends, writes `comparison.csv`, generates three figures, and produces `error_analysis.json` and `cv_metrics.md`.

The **data layer is verified against the real Hub corpora**: `build_dataset --source glaive,when2call` converts 11,099 of 20,000 Glaive rows and builds a 4000-row training corpus (3000 positives + 1000 negatives) with 249 eval rows across all three splits. The **model layer is not** — `torch`, `transformers` and `peft` are absent from this environment and there is no GPU, so `MODE=full` cannot run here. Consequently:

- The numbers currently in `results/` come from scripted backends and are **pipeline-validation artifacts, not model results.**
- The "key finding" section of the README is intentionally unwritten. It should be written from a real run, not from fixtures.

What the smoke run does establish is that the measurement apparatus works and reproduces the expected failure shape: the positive-only backend reaches 100% exact tool-call match while its false tool-call rate rises from 22.5% to 66.7%.

## How to finish

```bash
MODE=full ./scripts/run_experiments.sh
```

On a Kaggle T4 (or Colab L4) this trains both adapters and evaluates all three checkpoints. `kaggle/` holds a notebook that drives it, cloning this repo so the code has one source of truth rather than a copy that drifts. Then:

- `results/comparison.csv` — the three-way table
- `results/figures/exact_match.png`, `false_tool_call.png`, `confusion_matrix.png`
- `results/error_analysis.json` — 20 categorised failures
- `results/cv_metrics.md` — headline deltas, computed from the result files

GPU model, wall-clock training time and peak memory are not recorded by these scripts; capture those from the run log to complete the CV summary.

## Threats to validity

- The `direct` / `request_for_info` / `cannot_answer` split is heuristic cue matching. `classification_source` is stored per example so the share of decisions resting on the heuristic can be audited; on real When2Call prose the cue lists may need further widening — the synthetic fixture is cleaner than real data.
- The two arms differ in record count (3000 vs 4000) as well as in content. The added negatives *are* the treatment, and both arms run the same 3 epochs, so gradient-step count differs slightly. That is part of the intervention rather than a controlled quantity.
- `direct` accuracy is not measurable on the real When2Call test split, and the training negatives contain only 10 `direct` rows. Any claim about direct-answer behaviour rests on very little real data.
- Held-out-function accuracy now rests on whole names being withheld before training sampling (verified: 0 of 511 names leak). That holds for the current corpus; re-verify if the sampling logic changes.
- `results/` is tracked in git by choice, so a run that fails partway can still republish the previous run's numbers. The notebook mitigates this by deleting `results/` after cloning; any other consumer of this repo must do the same.