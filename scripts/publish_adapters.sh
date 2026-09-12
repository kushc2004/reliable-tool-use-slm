#!/usr/bin/env bash
# Publish the trained LoRA adapters as a Kaggle Dataset, so a re-run of the
# notebook does not pay for QLoRA training again.
#
# The notebook already writes reliable_tool_use_adapters.zip at the end of every
# run. This script takes that zip (or rebuilds it from an extracted outputs/
# tree) and pushes it to Kaggle as a Dataset. Attach that Dataset to the kernel
# and the cache cell in the notebook unpacks it; the two training cells then
# skip themselves.
#
# One-time setup: get an API token from https://www.kaggle.com/settings
# (Account -> Create New API Token) and save it to ~/.kaggle/kaggle.json,
# then `chmod 600 ~/.kaggle/kaggle.json`.
#
# Usage:
#   scripts/publish_adapters.sh                 # use the existing zip
#   scripts/publish_adapters.sh path/to/zip
#   KAGGLE_DATASET=user/name scripts/publish_adapters.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

KAGGLE_DATASET="${KAGGLE_DATASET:-kushc2004/reliable-tool-use-slm-adapters}"
STAGE="${ROOT}/.kaggle-adapters-stage"

# Where the notebook's zip ends up after a run. The Kaggle output download
# names it with a trailing " (1)" when a previous download exists, so accept
# both spellings rather than failing on a cosmetic difference.
ZIP="${1:-}"
if [ -z "${ZIP}" ]; then
  for candidate in \
      "${ROOT}/kaggle-outputs/reliable_tool_use_adapters.zip" \
      "${ROOT}/kaggle-outputs/reliable_tool_use_adapters (1).zip" \
      "${ROOT}/outputs/reliable_tool_use_adapters.zip"; do
    if [ -f "${candidate}" ]; then ZIP="${candidate}"; break; fi
  done
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: kaggle CLI not found. pip install kaggle" >&2
  exit 1
fi

if [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
  echo "error: no ${HOME}/.kaggle/kaggle.json -- see the header of this script." >&2
  exit 1
fi

rm -rf "${STAGE}"
mkdir -p "${STAGE}"

if [ -n "${ZIP}" ] && [ -f "${ZIP}" ]; then
  echo "== staging adapters from ${ZIP}"
  # Extract inside the stage dir so the archive's own "outputs/..." paths
  # become the dataset's paths. The notebook extracts at the repo root, so the
  # layout must stay outputs/<arm>/....
  unzip -q -o "${ZIP}" -d "${STAGE}"
elif [ -d "${ROOT}/outputs/tool_sft" ]; then
  echo "== staging adapters from ${ROOT}/outputs"
  mkdir -p "${STAGE}/outputs"
  for arm in tool_sft reliable_tool_sft; do
    src="${ROOT}/outputs/${arm}"
    [ -d "${src}" ] || continue
    mkdir -p "${STAGE}/outputs/${arm}"
    # Only what PEFT needs to load, plus the provenance record. Checkpoints and
    # optimizer state are ~440 MB each and are not needed to evaluate.
    for f in adapter_config.json adapter_model.safetensors run_config.json \
             tokenizer.json tokenizer_config.json chat_template.jinja; do
      [ -f "${src}/${f}" ] && cp "${src}/${f}" "${STAGE}/outputs/${arm}/${f}"
    done
  done
else
  echo "error: no adapter zip and no outputs/ tree found." >&2
  echo "       Run the notebook once (or pass a zip path) first." >&2
  exit 1
fi

# A dataset without these is useless: PEFT cannot load an adapter whose config
# or weights are missing, and the failure would only surface hours later inside
# the Kaggle evaluation.
for arm in tool_sft reliable_tool_sft; do
  for f in adapter_config.json adapter_model.safetensors; do
    if [ ! -f "${STAGE}/outputs/${arm}/${f}" ]; then
      echo "error: staging is missing outputs/${arm}/${f}" >&2
      exit 1
    fi
  done
done
echo "== staged:"
find "${STAGE}" -type f | sed "s|${STAGE}/||" | sort

# Kaggle requires the metadata file at the dataset root.
SLUG="${KAGGLE_DATASET##*/}"
cat > "${STAGE}/dataset-metadata.json" <<JSON
{
  "title": "Reliable Tool-Use SLM - QLoRA adapters",
  "id": "${KAGGLE_DATASET}",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

if kaggle datasets list --user "${KAGGLE_DATASET%%/*}" 2>/dev/null \
     | grep -q "${SLUG}"; then
  echo "== dataset exists -> new version"
  kaggle datasets version -p "${STAGE}" -m "Adapters re-upload" --dir-mode zip
else
  echo "== creating dataset ${KAGGLE_DATASET}"
  kaggle datasets create -p "${STAGE}" --dir-mode zip
fi

echo
echo "== done. Attach ${KAGGLE_DATASET} to the notebook:"
echo "   Kaggle UI -> your notebook -> Add Input -> Your Datasets -> ${SLUG}"
echo "   The cache cell finds it under /kaggle/input/ and both training"
echo "   cells skip themselves."