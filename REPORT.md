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

1. **Oracle check.** A backend that echoes the gold answer measures the ceiling of the prose classifier. Tool-call decisions are exact; the free-form clarification/refusal categories are cue-classified, so the real full-split oracle is expected to be close to rather than literally 100%.
2. **Unit tests.** `tests/test_decision_metrics.py` pins classification for each of the four categories and asserts the oracle property directly.

Both caught a real defect. The first oracle run scored 93.75% decision accuracy with missing-info accuracy at 75%. Diagnosis: the `request_for_info` cue list covered "could you provide/specify..." but not "what is the identifier you would like me to look up?", so ten genuine clarification rows were classified as `direct`.

The fix required care. The obvious widening — matching any `what is the <word>` — would have broken `cannot_answer`, because rows like "what is the air quality in Hanoi?" are unsupported requests, not clarifications. The patterns were widened with a bounded list of argument-shaped nouns (`invoice`, `order`, `identifier`, `ticker`, …) plus explicit frames (`what should I`, `what would you like`). On the complete 3,652-row `mcq` split the resulting gold-answer oracle scores **99.3% decision accuracy**, with the residual errors retained as an explicit measurement of the heuristic ceiling. The covered cases are pinned by tests so the classifier cannot silently regress.

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

## Final model results

The real GPU experiment has completed. The evaluated checkpoints use the Hugging Face backend rather than the scripted smoke fixture, and the real metrics are now promoted into the committed `results/` directory.

| Metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Exact tool-call match | 0.0% | **94.6%** | 86.1% |
| Function-name accuracy | 0.0% | **95.2%** | 87.3% |
| Argument accuracy | 0.0% | **94.9%** | 86.7% |
| JSON validity | 100.0% | 99.2% | **100.0%** |
| Held-out function EM | 0.0% | **96.4%** | 78.3% |
| When2Call decision accuracy | 23.2% | 37.3% | **62.1%** |
| When2Call macro F1 | 16.7% | 17.9% | **47.4%** |
| Tool-call precision | n/a | 37.2% | **70.2%** |
| Tool-call recall | 0.0% | **97.2%** | 48.2% |
| False tool-call rate ↓ | 0.0%* | 89.9% | **11.2%** |
| Missing-info accuracy | 66.2% | 10.0% | **73.4%** |
| Cannot-answer accuracy | 11.3% | 0.0% | **66.7%** |

\* Base almost never emits calls, so its 0% false-call rate is degenerate. The meaningful reliability ablation is Tool-SFT → Reliable Tool-SFT.

The tool-call track contains 249 examples. The decision track uses a deterministic 1,200-example label-stratified subset of When2Call: 425 `tool_call`, 349 `request_for_info`, and 426 `cannot_answer` cases. The final evaluated adapters use Qwen2.5-1.5B-Instruct with 4-bit QLoRA; the run metadata records 18,464,768 trainable parameters (2.036% of 907,081,216) and 4.86 GB peak GPU memory on a Tesla T4.

### Main finding

Tool-SFT demonstrates that positive supervision is enough to teach highly accurate function execution (94.6% exact match) but creates severe over-calling (89.9% false-tool-call rate on When2Call). Adding ~1K negative/clarification/refusal examples reduces false calls to **11.2%** (−78.7 percentage points; 87.5% relative reduction), raises decision accuracy from **37.3% to 62.1%**, and improves missing-information and unsupported-request handling.

The intervention also makes the model more conservative: exact call match falls to 86.1%, held-out-function EM to 78.3%, and tool-call recall to 48.2%. This is the central empirical result: reliable tool use is a precision/abstention versus recall trade-off, not merely a function-calling-format problem.

### Run provenance

The downloaded final evaluation log completed successfully on a Tesla T4 and wrote both the results and adapter archives. The evaluation run restored already-trained adapters, so it does not provide the original wall-clock training duration. The archived adapter metadata records the model/configuration, trainable parameter count, training-record count, dtype and peak GPU memory. The results ZIP passes an integrity check.

## Recommended next validation

The core project is complete. The highest-value remaining experiment is not more infrastructure or another model family; it is a standardized full When2Call evaluation that removes the custom prose-classification heuristic and uses the complete benchmark. After that, a single step-matched positive-only control would isolate negative supervision from the extra optimizer steps introduced by 4000 versus 3000 training records.

## Threats to validity

- The `direct` / `request_for_info` / `cannot_answer` split is heuristic cue matching over generated prose. `classification_source` is stored per example so the score can be audited. On the full 3,652-row `mcq` split, the gold-answer oracle reaches 99.3% decision accuracy rather than 100%, which measures the heuristic ceiling directly.
- The reported model decision metrics use a deterministic 1,200-example stratified subset rather than all 3,652 When2Call `mcq` examples.
- The two arms differ in record count (3000 vs 4000) as well as in content. The added negatives *are* the treatment, and both arms run the same 3 epochs, so gradient-step count differs slightly. That is part of the intervention rather than a controlled quantity.
- `direct` accuracy is not measurable on the real When2Call test split, and the training negatives contain only 10 `direct` rows. Any claim about direct-answer behaviour rests on very little real data.
- Held-out-function accuracy now rests on whole names being withheld before training sampling (verified: 0 of 511 names leak). That holds for the current corpus; re-verify if the sampling logic changes.
- `results/` is tracked in git by choice, so a run that fails partway can still republish the previous run's numbers. The notebook mitigates this by deleting `results/` after cloning; any other consumer of this repo must do the same.
- The final comparison is one QLoRA configuration/seed, so there are no multi-seed confidence intervals.
