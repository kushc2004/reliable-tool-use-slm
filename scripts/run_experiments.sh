#!/usr/bin/env bash
# End-to-end driver: data -> train -> evaluate -> aggregate -> figures.
#
# Two ways to run it:
#
#   MODE=smoke ./scripts/run_experiments.sh
#       CPU only, no model download. Builds the offline synthetic corpus and
#       drives the whole pipeline with the scripted backends. This validates the
#       code path and the metric layer; it does NOT produce model numbers.
#
#   MODE=full ./scripts/run_experiments.sh
#       Real run. Needs a GPU, network access, and the Hub datasets. Trains
#       Tool-SFT and Reliable Tool-SFT from the same base, then evaluates base,
#       both adapters on the tool-call splits and the When2Call decision set.
#
# Everything is seeded. Every stage writes its config next to its output.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Pick an interpreter that exists. `python` is absent on many macOS installs.
PY="${PY:-}"
if [ -z "${PY}" ]; then
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then PY="${candidate}"; break; fi
  done
fi
if [ -z "${PY}" ]; then
  echo "error: no python interpreter found; set PY=/path/to/python3" >&2
  exit 1
fi

MODE="${MODE:-smoke}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-0}"
N_TRAIN="${N_TRAIN:-3000}"
N_W2C_TRAIN="${N_W2C_TRAIN:-1000}"
N_EVAL="${N_EVAL:-500}"
# Decision-track sample size and the seed its stratified sample is drawn with.
# The oracle and the three arms MUST use the same seed: they are compared on
# the same rows, and a sanity check run on different rows validates nothing.
N_W2C_EVAL="${N_W2C_EVAL:-1200}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
# Batched generation. 1 = the old one-row-at-a-time path.
BATCH_SIZE="${BATCH_SIZE:-8}"
W2C_MAX_NEW_TOKENS="${W2C_MAX_NEW_TOKENS:-96}"

RESULTS="${ROOT}/results"
DATA="${ROOT}/data"
OUTPUTS="${ROOT}/outputs"

echo "== run_experiments.sh =="
echo "   mode=${MODE} base=${BASE_MODEL} seed=${SEED}"
echo "   root=${ROOT}"
echo

mkdir -p "${RESULTS}" "${DATA}" "${OUTPUTS}"

# --------------------------------------------------------------------------- #
# 1. Data preparation
# --------------------------------------------------------------------------- #
echo "-- [1/6] data preparation"

if [ "${MODE}" = "smoke" ]; then
  "${PY}" -msrc.data.build_dataset \
      --source synth \
      --out "${DATA}/processed" \
      --n-train "${N_TRAIN}" \
      --n-eval "${N_EVAL}" \
      --seed "${SEED}"

  # A small synthetic When2Call-shaped eval set so the decision scorer runs
  # offline. Replace with data/raw/w2c_eval.jsonl on a real run.
  "${PY}" -msrc.data.synth_when2call \
      --out "${DATA}/processed/w2c_eval.jsonl" \
      --n-per-class 40 \
      --seed "${SEED}"
else
  # ONE corpus, shared by both training variants.
  #
  # This replaces two separately-built corpora, which carried two defects that
  # each invalidated the comparison on their own:
  #
  #  * Cross-contamination. Each corpus carved its own eval set out of the same
  #    shuffled Glaive pool, so 40 of 166 tool-call eval rows were training rows
  #    for the OTHER variant. All three checkpoints are scored on one eval set,
  #    so one model was graded on rows it had already memorised.
  #  * No negatives. The reliable corpus was built with --source glaive alone,
  #    and Glaive contains no no-tool rows at all. --neg-ratio 0.25 therefore
  #    resolved to zero negatives ("only 0 negatives available"), so
  #    Reliable Tool-SFT would have trained on exactly the same positives-only
  #    data as Tool-SFT. The two checkpoints would be identical and the headline
  #    claim about negative supervision vacuous.
  #
  # Building once and letting --variant choose the rows means both checkpoints
  # see an identical positive set and differ only by the negatives, which is the
  # experimental variable. It also gives the tool-call track a no_tool split to
  # measure a false-call rate against, which a Glaive-only corpus cannot do.
  "${PY}" -msrc.data.build_dataset \
      --source glaive,when2call \
      --out "${DATA}/processed" \
      --n-train "$((N_TRAIN + N_W2C_TRAIN))" \
      --n-eval "${N_EVAL}" \
      --neg-ratio 0.25 \
      --seed "${SEED}"

  # When2Call test split, for the four-way decision track.
  "${PY}" -msrc.data.prepare_when2call \
      --mode eval \
      --out "${DATA}/raw/w2c_eval.jsonl"
fi

"${PY}" -mpytest tests/ -q

# --------------------------------------------------------------------------- #
# 2. Training
# --------------------------------------------------------------------------- #
if [ "${MODE}" = "full" ]; then
  echo "-- [2/6] training Tool-SFT"
  "${PY}" -msrc.train_qlora \
      --config configs/tool_sft.yaml \
      --data "${DATA}/processed" \
      --variant sft \
      --out "${OUTPUTS}/tool_sft"

  echo "-- [3/6] training Reliable Tool-SFT"
  # Same corpus as Tool-SFT. --variant sft-neg keeps the negatives and drops
  # nothing, so the only difference between the two checkpoints is the presence
  # of the ~1K no-tool / clarification / refusal examples.
  "${PY}" -msrc.train_qlora \
      --config configs/reliable_tool_sft.yaml \
      --data "${DATA}/processed" \
      --variant sft-neg \
      --out "${OUTPUTS}/reliable_tool_sft"
else
  echo "-- [2/6] training skipped (MODE=${MODE})"
  echo "-- [3/6] training skipped (MODE=${MODE})"
fi

# --------------------------------------------------------------------------- #
# 4. Evaluation -- tool-call track
# --------------------------------------------------------------------------- #
echo "-- [4/6] tool-call evaluation"

if [ "${MODE}" = "full" ]; then
  "${PY}" -msrc.evaluate --data "${DATA}/processed" --split all \
      --batch-size "${BATCH_SIZE}" \
      --checkpoint "${BASE_MODEL}" --out "${RESULTS}/base"
  "${PY}" -msrc.evaluate --data "${DATA}/processed" --split all \
      --batch-size "${BATCH_SIZE}" \
      --checkpoint "${BASE_MODEL}" --adapter "${OUTPUTS}/tool_sft" \
      --out "${RESULTS}/tool_sft"
  "${PY}" -msrc.evaluate --data "${DATA}/processed" --split all \
      --batch-size "${BATCH_SIZE}" \
      --checkpoint "${BASE_MODEL}" --adapter "${OUTPUTS}/reliable_tool_sft" \
      --out "${RESULTS}/reliable_tool_sft"
else
  # Scripted backends: base abstains, Tool-SFT over-calls, Reliable is balanced.
  "${PY}" -msrc.evaluate --data "${DATA}/processed" --split all \
      --backend dummy --failure-rate 0.75 --out "${RESULTS}/base"
  "${PY}" -msrc.evaluate --data "${DATA}/processed" --split all \
      --backend dummy --failure-rate 0.0 --out "${RESULTS}/tool_sft"
  "${PY}" -msrc.evaluate --data "${DATA}/processed" --split all \
      --backend dummy --failure-rate 0.25 --out "${RESULTS}/reliable_tool_sft"
fi

for run in base tool_sft reliable_tool_sft; do
  cp "${RESULTS}/${run}/metrics.json" "${RESULTS}/${run}_metrics.json"
  cp "${RESULTS}/${run}/failures.jsonl" "${RESULTS}/${run}_failures.jsonl" 2>/dev/null || true
done

# --------------------------------------------------------------------------- #
# 5. Evaluation -- When2Call decision track
# --------------------------------------------------------------------------- #
echo "-- [5/6] When2Call decision evaluation"

if [ "${MODE}" = "smoke" ]; then
  W2C_EVAL="${DATA}/processed/w2c_eval.jsonl"
else
  W2C_EVAL="${DATA}/raw/w2c_eval.jsonl"
fi

# Sanity check the scorer against a perfect backend before trusting any model.
#
# --limit and --sample-seed are passed HERE TOO. The previous version ran the
# oracle on the full file while the arms ran on the first 1200 rows, so a
# passing oracle proved nothing about the rows actually being scored -- and
# those rows were all one label.
"${PY}" -msrc.evaluate_when2call --data "${W2C_EVAL}" \
    --backend oracle --limit "${N_W2C_EVAL}" --sample-seed "${SAMPLE_SEED}" \
    --out "${RESULTS}/oracle_when2call"

if [ "${MODE}" = "full" ]; then
  "${PY}" -msrc.evaluate_when2call --data "${W2C_EVAL}" \
      --limit "${N_W2C_EVAL}" --sample-seed "${SAMPLE_SEED}" \
      --max-new-tokens "${W2C_MAX_NEW_TOKENS}" --batch-size "${BATCH_SIZE}" \
      --checkpoint "${BASE_MODEL}" --out "${RESULTS}/base_when2call"
  "${PY}" -msrc.evaluate_when2call --data "${W2C_EVAL}" \
      --limit "${N_W2C_EVAL}" --sample-seed "${SAMPLE_SEED}" \
      --max-new-tokens "${W2C_MAX_NEW_TOKENS}" --batch-size "${BATCH_SIZE}" \
      --checkpoint "${BASE_MODEL}" --adapter "${OUTPUTS}/tool_sft" \
      --out "${RESULTS}/tool_sft_when2call"
  "${PY}" -msrc.evaluate_when2call --data "${W2C_EVAL}" \
      --limit "${N_W2C_EVAL}" --sample-seed "${SAMPLE_SEED}" \
      --max-new-tokens "${W2C_MAX_NEW_TOKENS}" --batch-size "${BATCH_SIZE}" \
      --checkpoint "${BASE_MODEL}" --adapter "${OUTPUTS}/reliable_tool_sft" \
      --out "${RESULTS}/reliable_tool_sft_when2call"
else
  "${PY}" -msrc.evaluate_when2call --data "${W2C_EVAL}" \
      --limit "${N_W2C_EVAL}" --sample-seed "${SAMPLE_SEED}" \
      --backend dummy --failure-rate 0.75 --out "${RESULTS}/base_when2call"
  "${PY}" -msrc.evaluate_when2call --data "${W2C_EVAL}" \
      --limit "${N_W2C_EVAL}" --sample-seed "${SAMPLE_SEED}" \
      --backend dummy --failure-rate 0.0 --out "${RESULTS}/tool_sft_when2call"
  "${PY}" -msrc.evaluate_when2call --data "${W2C_EVAL}" \
      --limit "${N_W2C_EVAL}" --sample-seed "${SAMPLE_SEED}" \
      --backend dummy --failure-rate 0.25 --out "${RESULTS}/reliable_tool_sft_when2call"
fi

for run in base tool_sft reliable_tool_sft; do
  cp "${RESULTS}/${run}_when2call/metrics.json" "${RESULTS}/${run}_when2call.json"
  cp "${RESULTS}/${run}_when2call/failures.jsonl" "${RESULTS}/${run}_when2call_failures.jsonl" 2>/dev/null || true
done

# --------------------------------------------------------------------------- #
# 6. Aggregate, analyse, plot
# --------------------------------------------------------------------------- #
echo "-- [6/6] aggregation, error analysis, figures"

# The aggregator refuses to build a comparison table from dummy-backed runs,
# because that is how a full training run once produced three byte-identical
# arms. The smoke mode is deliberately fixture-backed, so it opts out
# explicitly rather than the guard being weakened for everyone.
AGG_ARGS=()
if [ "${MODE}" = "smoke" ]; then
  AGG_ARGS+=(--allow-fixture)
fi
"${PY}" -msrc.aggregate_results --results "${RESULTS}" "${AGG_ARGS[@]}"
"${PY}" -msrc.error_analysis --results "${RESULTS}" \
    --out "${RESULTS}/error_analysis.json" --max-examples 20
"${PY}" -msrc.cv_metrics --results "${RESULTS}" --out "${RESULTS}/cv_metrics.md"

echo
echo "== done =="
echo "  comparison : ${RESULTS}/comparison.csv"
echo "  figures    : ${RESULTS}/figures/"
echo "  errors     : ${RESULTS}/error_analysis.json"
echo
echo "NOTE: MODE=${MODE}. A 'smoke' run uses scripted backends and does NOT"
echo "      produce model numbers. Use MODE=full for reportable results."