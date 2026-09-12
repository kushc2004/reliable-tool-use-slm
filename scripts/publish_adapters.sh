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
# (Account -> Create New API Token) and save it to ~/.kaggle/kaggle.json
# (`chmod 600 ~/.kaggle/kaggle.json`), or place an OAuth token at
# ~/.kaggle/access_token. Either is accepted.
#
# Usage:
#   scripts/publish_adapters.sh                 # use the existing zip
#   scripts/publish_adapters.sh path/to/zip
#   KAGGLE_DATASET=user/name scripts/publish_adapters.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# NOTE: the dataset owner must match the Kaggle username of the credentials in
# use (kaggle config view -> username), not the GitHub handle.
KAGGLE_DATASET="${KAGGLE_DATASET:-kushchaudhari/reliable-tool-use-slm-adapters}"
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

# The CLI accepts either the legacy kaggle.json or an OAuth access_token file.
# Requiring kaggle.json specifically rejected a working setup whose only
# credential was ~/.kaggle/access_token.
if [ ! -f "${HOME}/.kaggle/kaggle.json" ] \
   && [ ! -f "${HOME}/.kaggle/access_token" ]; then
  echo "error: no Kaggle credentials found in ${HOME}/.kaggle/ -- see the header of this script." >&2
  exit 1
fi

rm -rf "${STAGE}"
mkdir -p "${STAGE}"

# The dataset holds ONE zip, not an unpacked tree.
#
# This is not a style choice. Kaggle's uploader takes --dir-mode skip/zip/tar,
# and NONE of them preserve a leading "outputs/" for a folder upload: "skip"
# drops the directory outright ("Skipping folder: outputs"), and "zip" flattens
# it into outputs.zip whose root is tool_sft/, not outputs/tool_sft/. The first
# attempt used zip and the notebook's cache cell matched neither shape, so it
# silently retrained both arms.
#
# A single FILE is uploaded verbatim regardless of --dir-mode, so building the
# archive here is the only way to control the internal paths. The notebook's
# cache cell looks for exactly this filename and extracts it at the repo root.
ARCHIVE="${STAGE}/reliable_tool_use_adapters.zip"

if [ -n "${ZIP}" ] && [ -f "${ZIP}" ]; then
  echo "== staging adapters from ${ZIP}"
  cp "${ZIP}" "${ARCHIVE}"
elif [ -d "${ROOT}/outputs/tool_sft" ]; then
  echo "== staging adapters from ${ROOT}/outputs"
  # Build the archive in a scratch tree so the paths inside it are outputs/<arm>/.
  SCRATCH="$(mktemp -d)"
  trap 'rm -rf "${SCRATCH}"' EXIT
  for arm in tool_sft reliable_tool_sft; do
    src="${ROOT}/outputs/${arm}"
    [ -d "${src}" ] || continue
    mkdir -p "${SCRATCH}/outputs/${arm}"
    # Only what PEFT needs to load, plus the provenance record. Checkpoints and
    # optimizer state are ~440 MB each and are not needed to evaluate.
    for f in adapter_config.json adapter_model.safetensors run_config.json \
             tokenizer.json tokenizer_config.json chat_template.jinja; do
      [ -f "${src}/${f}" ] && cp "${src}/${f}" "${SCRATCH}/outputs/${arm}/${f}"
    done
  done
  (cd "${SCRATCH}" && zip -q -r "${ARCHIVE}" outputs)
else
  echo "error: no adapter zip and no outputs/ tree found." >&2
  echo "       Run the notebook once (or pass a zip path) first." >&2
  exit 1
fi

# A dataset without these is useless: PEFT cannot load an adapter whose config
# or weights are missing, and the failure would only surface hours later inside
# the Kaggle evaluation. Check the CONTENT of the archive, not a staging tree.
for arm in tool_sft reliable_tool_sft; do
  for f in adapter_config.json adapter_model.safetensors; do
    if ! unzip -l "${ARCHIVE}" | grep -q "outputs/${arm}/${f}$"; then
      echo "error: ${ARCHIVE} is missing outputs/${arm}/${f}" >&2
      exit 1
    fi
  done
done
echo "== staged:"
unzip -l "${ARCHIVE}" | awk 'NR>3 && $4 {print $4}' | sed '/^$/d'

# Kaggle requires the metadata file at the dataset root.
SLUG="${KAGGLE_DATASET##*/}"
cat > "${STAGE}/dataset-metadata.json" <<JSON
{
  "title": "Reliable Tool-Use SLM - QLoRA adapters",
  "id": "${KAGGLE_DATASET}",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

# --dir-mode zip, not skip: the stage holds one archive, and "skip" would drop
# it if the CLI ever decided to treat it as a directory. For a plain file the
# mode is irrelevant, which is the point -- the archive's own paths are what
# the notebook reads.
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
echo
echo "   Or declare it in kaggle/kernel-metadata.json so it is attached on push:"
echo "     \"dataset_sources\": [\"${KAGGLE_DATASET}\"]"