"""Run the official NVIDIA When2Call MCQ benchmark on Modal.

This is evaluation-only. It does NOT retrain either adapter. The two PEFT
adapters are downloaded from the public Kaggle dataset produced by the earlier
Kaggle training run and evaluated on top of Qwen/Qwen2.5-1.5B-Instruct.

Examples:

    # Cheap smoke test first.
    modal run modal_jobs/when2call_mcq.py --limit 32

    # Full 3,652-example official MCQ benchmark on all three checkpoints.
    modal run modal_jobs/when2call_mcq.py

Results are persisted in the ``reliable-tool-use-slm-eval`` Modal Volume under
``results/<run-name>/``. Download a completed run with:

    modal volume get reliable-tool-use-slm-eval results/<run-name> modal-results/<run-name>
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "reliable-tool-use-slm-when2call"
VOLUME_NAME = "reliable-tool-use-slm-eval"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KAGGLE_ADAPTER_DATASET = "kushchaudhari/reliable-tool-use-slm-adapters/versions/2"

# Pin both evaluation implementations so a later upstream change cannot move the
# benchmark underneath an already-published result.
WHEN2CALL_COMMIT = "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"
LM_EVAL_COMMIT = "ad8737ae7fad24cf64e50fc7fc31397bff586b9e"

CACHE_ROOT = Path("/cache")
ADAPTER_ROOT = CACHE_ROOT / "adapters"
HF_ROOT = CACHE_ROOT / "hf"
RESULTS_ROOT = CACHE_ROOT / "results"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.10.0",
        "transformers==5.0.0",
        "peft==0.19.1",
        "accelerate==1.13.0",
        "datasets>=4.0",
        "huggingface_hub>=0.34",
        "safetensors>=0.4",
        "sentencepiece>=0.2",
        "scikit-learn>=1.5",
        "pandas>=2.2",
        "kagglehub>=1.0.0",
        f"git+https://github.com/EleutherAI/lm-evaluation-harness.git@{LM_EVAL_COMMIT}",
    )
    .run_commands(
        "git init /opt/When2Call",
        "git -C /opt/When2Call remote add origin https://github.com/NVIDIA/When2Call.git",
        f"git -C /opt/When2Call fetch --depth 1 origin {WHEN2CALL_COMMIT}",
        "git -C /opt/When2Call checkout --detach FETCH_HEAD",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

common_env = {
    "HF_HOME": str(HF_ROOT),
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _adapter_path(name: str) -> Path:
    """Resolve an adapter directory and fail loudly if the Kaggle layout moved."""
    expected = ADAPTER_ROOT / "outputs" / name
    if (expected / "adapter_model.safetensors").exists():
        return expected

    matches = list(ADAPTER_ROOT.rglob(f"{name}/adapter_model.safetensors"))
    if len(matches) == 1:
        return matches[0].parent
    raise FileNotFoundError(
        f"could not uniquely resolve adapter {name!r} under {ADAPTER_ROOT}; "
        f"found {len(matches)} candidates"
    )


@app.function(
    image=image,
    volumes={str(CACHE_ROOT): volume},
    env=common_env,
    timeout=30 * 60,
)
def prepare_assets() -> dict[str, Any]:
    """Populate the persistent cache without paying for a GPU."""
    import kagglehub
    from huggingface_hub import snapshot_download

    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    if not (ADAPTER_ROOT / "outputs" / "tool_sft" / "adapter_model.safetensors").exists():
        print(f"Downloading public Kaggle adapters: {KAGGLE_ADAPTER_DATASET}", flush=True)
        kagglehub.dataset_download(KAGGLE_ADAPTER_DATASET, output_dir=str(ADAPTER_ROOT))

    tool_adapter = _adapter_path("tool_sft")
    reliable_adapter = _adapter_path("reliable_tool_sft")

    print(f"Caching base model: {BASE_MODEL}", flush=True)
    snapshot_download(BASE_MODEL, cache_dir=str(HF_ROOT / "hub"))

    volume.commit()
    return {
        "base_model": BASE_MODEL,
        "adapter_dataset": KAGGLE_ADAPTER_DATASET,
        "tool_sft": str(tool_adapter),
        "reliable_tool_sft": str(reliable_adapter),
    }


def _prepare_official_task() -> Path:
    source = Path("/opt/When2Call/evaluation/mcq/lm_eval_harness/when2call")
    dataset = Path("/opt/When2Call/data/test/when2call_test_mcq.jsonl")
    task_dir = Path("/tmp/when2call-task")
    shutil.rmtree(task_dir, ignore_errors=True)
    shutil.copytree(source, task_dir)

    # NVIDIA's YAML expects the task to have been copied into lm_eval/tasks.
    # We use lm-eval's supported --include_path instead, so point its JSON loader
    # at the pinned official dataset by absolute path.
    qwen_yaml = task_dir / "when2call-qwen2_5.yaml"
    text = qwen_yaml.read_text(encoding="utf-8")
    text = text.replace(
        "data_files: lm_eval/tasks/when2call/when2call_test_mcq.jsonl",
        f"data_files: {dataset}",
    )
    qwen_yaml.write_text(text, encoding="utf-8")
    return task_dir


def _load_official_additional_metrics(task_dir: Path):
    module_path = task_dir / "additional_metrics.py"
    spec = importlib.util.spec_from_file_location("when2call_additional_metrics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(metrics: dict[str, Any], prefix: str) -> Any:
    for key, value in metrics.items():
        if key == prefix or key.startswith(prefix + ","):
            return value
    return None


def _read_lm_eval_summary(out_dir: Path, task_dir: Path) -> dict[str, Any]:
    result_files = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    sample_files = sorted(out_dir.rglob("samples_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not result_files or not sample_files:
        raise RuntimeError(
            f"lm-eval did not produce expected results/samples under {out_dir}: "
            f"results={len(result_files)} samples={len(sample_files)}"
        )

    result_file = result_files[-1]
    sample_file = sample_files[-1]
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    task_metrics = payload["results"]["when2call-qwen2_5"]

    additional = _load_official_additional_metrics(task_dir)
    hallucination_rate = additional.calculate_hallucination_rate(
        str(sample_file), print_rate=False
    )
    confusion_matrix = additional.calculate_confusion_matrix(
        str(sample_file), print_matrix=False
    )

    with sample_file.open("r", encoding="utf-8") as handle:
        n_samples = sum(1 for _ in handle)

    return {
        "n": n_samples,
        "macro_f1": _metric(task_metrics, "macro_f1"),
        "accuracy": _metric(task_metrics, "acc"),
        "accuracy_norm": _metric(task_metrics, "acc_norm"),
        "hallucination_rate": hallucination_rate,
        "confusion_matrix": confusion_matrix,
        "result_file": str(result_file),
        "samples_file": str(sample_file),
    }


@app.function(
    image=image,
    gpu="L40S",
    volumes={str(CACHE_ROOT): volume},
    env=common_env,
    timeout=4 * 60 * 60,
    memory=32768,
)
def evaluate_mcq(
    run_name: str,
    batch_size: str = "auto",
    max_batch_size: int = 64,
    limit: int = 0,
) -> dict[str, Any]:
    """Evaluate Base, Tool-SFT and Reliable Tool-SFT on official When2Call MCQ."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal GPU function")

    task_dir = _prepare_official_task()
    run_dir = RESULTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    arms = [
        ("base", None),
        ("tool_sft", _adapter_path("tool_sft")),
        ("reliable_tool_sft", _adapter_path("reliable_tool_sft")),
    ]

    provenance = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "base_model": BASE_MODEL,
        "kaggle_adapter_dataset": KAGGLE_ADAPTER_DATASET,
        "when2call_commit": WHEN2CALL_COMMIT,
        "lm_eval_commit": LM_EVAL_COMMIT,
        "task": "when2call-qwen2_5",
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "limit": limit or None,
        "training_performed": False,
        "precision": "bfloat16",
        "attention": "sdpa",
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    summaries: dict[str, Any] = {}
    for arm, adapter in arms:
        out_dir = run_dir / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        response_cache = CACHE_ROOT / "lm_eval_cache" / run_name / f"{arm}_"
        response_cache.parent.mkdir(parents=True, exist_ok=True)

        model_args = (
            f"pretrained={BASE_MODEL},dtype=bfloat16,attn_implementation=sdpa"
        )
        if adapter is not None:
            model_args += f",peft={adapter}"

        cmd = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            "when2call-qwen2_5",
            "--include_path",
            str(task_dir),
            "--batch_size",
            batch_size,
            "--max_batch_size",
            str(max_batch_size),
            "--num_fewshot",
            "0",
            "--device",
            "cuda:0",
            "--output_path",
            str(out_dir),
            "--use_cache",
            str(response_cache),
            "--log_samples",
            "--write_out",
            "--trust_remote_code",
        ]
        if limit > 0:
            cmd.extend(["--limit", str(limit)])

        print("$ " + " ".join(map(str, cmd)), flush=True)
        started = time.perf_counter()
        subprocess.run(cmd, check=True, cwd="/tmp")
        elapsed = time.perf_counter() - started

        summary = _read_lm_eval_summary(out_dir, task_dir)
        summary["elapsed_seconds"] = round(elapsed, 2)
        summary["adapter"] = None if adapter is None else str(adapter)
        summaries[arm] = summary
        (run_dir / f"{arm}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        # Persist each completed arm independently, so an interruption in a
        # later arm cannot erase earlier evaluation work.
        volume.commit()

    final = {"provenance": provenance, "runs": summaries}
    (run_dir / "summary.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    volume.commit()
    return {
        "volume": VOLUME_NAME,
        "remote_dir": str(run_dir.relative_to(CACHE_ROOT)),
        "summary": final,
    }


@app.local_entrypoint()
def main(
    gpu: str = "L40S",
    batch_size: str = "auto",
    max_batch_size: int = 64,
    limit: int = 0,
    run_name: str = "",
):
    """Prepare assets on CPU, then run the benchmark on the requested GPU."""
    if not run_name:
        suffix = f"limit-{limit}" if limit else "full"
        run_name = datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%SZ-{suffix}")

    assets = prepare_assets.remote()
    print("Prepared assets:")
    print(json.dumps(assets, indent=2))

    result = evaluate_mcq.with_options(gpu=gpu).remote(
        run_name=run_name,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        limit=limit,
    )
    print("\nOfficial When2Call MCQ result:")
    print(json.dumps(result, indent=2))
    print(
        "\nDownload full artifacts with:\n"
        f"modal volume get {VOLUME_NAME} {result['remote_dir']} "
        f"modal-results/{run_name}"
    )

