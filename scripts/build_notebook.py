#!/usr/bin/env python3
"""Generate the Kaggle notebook.

Kept as a script rather than hand-edited JSON so the cell list is reviewable in
a diff. Two things it enforces that the previous hand-written notebook got
wrong:

1. **Every command runs through ``subprocess.run(..., check=True)``.** Notebook
   ``!cmd`` syntax does not propagate a non-zero exit code -- a failing stage
   prints a traceback, the cell is still marked successful, and the next cell
   runs against whatever is on disk. Worse, ``!`` performs *shell* variable
   expansion, so ``$((N_TRAIN + N_W2C_TRAIN))`` is arithmetic over unset shell
   variables and evaluates to 0. That combination trained on an empty corpus and
   then republished a previous run's metrics.

2. **Explicit unique cell ids.** Without them nbformat warns and Kaggle's
   cell-id handling is undefined.
"""
from __future__ import annotations

import json
import pathlib

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({
        "cell_type": "markdown",
        "id": "md-%02d" % len(CELLS),
        "metadata": {},
        "source": text,
    })


def code(text: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "id": "code-%02d" % len(CELLS),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


md(r'''# Reliable Tool-Use SLM: QLoRA post-training with tool-decision supervision

Fine-tune Qwen2.5-1.5B-Instruct so it **calls tools when it should and does not when it shouldn't**.

| Checkpoint | Training data |
|---|---|
| **Base** | none |
| **Tool-SFT** | ~3K tool-call positives |
| **Reliable Tool-SFT** | the same ~3K positives + ~1K no-tool / clarification / refusal examples |

Both trained arms read **one shared corpus** and are separated by `--variant`, so the
positives are identical and the negatives are the only variable.

Code: [github.com/kushc2004/reliable-tool-use-slm](https://github.com/kushc2004/reliable-tool-use-slm)

**Runtime:** ~2-3h on a T4.

> Every stage runs through `subprocess.run(..., check=True)` rather than notebook `!cmd`.
> `!cmd` does not propagate a non-zero exit code, so a failing stage silently succeeds and
> the next cell runs against missing data.''')


md("## 1. Environment")
code(r'''import os, sys, subprocess, json, shutil, time, zipfile
from collections import Counter
import torch

print('python  :', sys.version.split()[0])
print('torch   :', torch.__version__)
print('cuda    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device  :', torch.cuda.get_device_name(0))
    print('vram    : %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def run(*args, **kwargs):
    # Run a command and RAISE on non-zero exit.
    #
    # Notebook !cmd syntax does not propagate the exit code: a failing stage
    # prints a traceback, the cell is still marked successful, and the next cell
    # runs against whatever happens to be on disk. !cmd also does SHELL variable
    # expansion, so $((A + B)) is arithmetic over unset shell variables and
    # evaluates to 0 -- which is how a corpus build once silently ran with
    # --n-train 0. Everything below uses subprocess with check=True instead.
    print('$ ' + ' '.join(str(a) for a in args), flush=True)
    return subprocess.run([str(a) for a in args], check=True, **kwargs)''')


md("## 2. Clone the repo")
code(r'''REPO = 'https://github.com/kushc2004/reliable-tool-use-slm.git'
PROJECT = '/kaggle/working/reliable-tool-use-slm'

if not os.path.exists(PROJECT):
    run('git', 'clone', '--depth', '1', REPO, PROJECT)
else:
    run('git', '-C', PROJECT, 'pull')

os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
head = run('git', 'rev-parse', 'HEAD', capture_output=True, text=True).stdout.strip()
print('commit:', head)
print(run('ls', capture_output=True, text=True).stdout)''')

code(r'''# Discard the committed results/ before running anything.
#
# results/ is tracked in git, so a fresh clone arrives with the previous run's
# metrics already in place. If a later stage fails, aggregate_results would
# re-read those files and print them as though just measured -- which is how a
# crashed run once republished the previous run's table. Deleting the directory
# first means a failed run produces NO report rather than a misleading one.
shutil.rmtree('results', ignore_errors=True)
print('results/ cleared - only freshly generated metrics will appear below')''')


md("## 3. Dependencies")
code(r'''run(sys.executable, '-m', 'pip', 'install', '-q',
    'bitsandbytes>=0.43', 'peft>=0.11', 'accelerate>=0.33',
    'datasets>=2.20', 'PyYAML>=6.0')

import importlib
for m in ['transformers', 'peft', 'bitsandbytes', 'datasets', 'accelerate']:
    mod = importlib.import_module(m)
    print('%-15s %s' % (m, getattr(mod, '__version__', '?')))''')


md(r'''## 4. Sanity-check the scorer

Before trusting any model number, verify the measurement apparatus. A backend that echoes
the gold answer must score 100% on every decision metric. If it does not, the scorer is
broken and nothing downstream means anything.''')
code(r'''run(sys.executable, '-m', 'pytest', 'tests/', '-q')''')


md("## 5. Build the corpora")
code(r'''N_TRAIN, N_W2C_TRAIN, N_EVAL = 3000, 1000, 250

# ONE shared corpus for both training variants.
#
# Two separately-built corpora caused two defects that each invalidated the
# comparison: 40 of 166 eval rows leaked into the other variant's training set,
# and the reliable corpus was built from Glaive alone -- which contains no
# no-tool rows -- so its negatives silently resolved to zero and both arms
# would have trained on identical data.
#
# 4000 rows at neg_ratio 0.25 = ~3000 positives + ~1000 negatives. --variant
# then selects rows: "sft" drops the negatives, "sft-neg" keeps everything.
run(sys.executable, '-m', 'src.data.build_dataset',
    '--source', 'glaive,when2call',
    '--out', 'data/processed',
    '--n-train', N_TRAIN + N_W2C_TRAIN,
    '--n-eval', N_EVAL,
    '--neg-ratio', 0.25,
    '--seed', 0)''')

code(r'''# When2Call test split, for the decision track.
#
# The dataset is CONFIG-scoped, not split-scoped: train_sft and train_pref each
# live under their own config, and asking the default config for split="train"
# raises ValueError. prepare_when2call resolves the configs internally.
run(sys.executable, '-m', 'src.data.prepare_when2call',
    '--mode', 'eval',
    '--out', 'data/raw/w2c_eval.jsonl')''')

code(r'''# Inspect the shared corpus before training on it.
#
# The two numbers that matter: train.call_expected vs train.no_call_expected
# (the arms differ by the latter), and eval.by_split -- a missing no_tool split
# means the tool-call track cannot report a false-call rate at all.
s = json.load(open('data/processed/stats.json'))
print('=== shared corpus ===')
print('  sources:', s['sources'])
print('  train  :', s['train'])
print('  eval   :', s['eval'])
print('  dropped:', s.get('dropped'))

assert s['train']['call_expected'] > 0, 'no positive training rows'
assert s['train']['no_call_expected'] > 0, 'no negatives -- sft-neg would equal sft'
assert 'no_tool' in s['eval']['by_split'], 'no no_tool eval split'

rows = [json.loads(line) for line in open('data/processed/eval.jsonl') if line.strip()]
print('  eval by split:', dict(Counter(r['split'] for r in rows)))
print('OK - corpus supports the full comparison')''')


md("## 6. Verify the oracle")
code(r'''run(sys.executable, '-m', 'src.evaluate_when2call',
    '--data', 'data/raw/w2c_eval.jsonl',
    '--backend', 'oracle',
    '--out', 'results/oracle_when2call')

m = json.load(open('results/oracle_when2call/metrics.json'))

# A floor, not equality. The cue classifier is a heuristic over prose, and after
# inspecting every remaining mismatch it settles at ~99.4% on the 3,652-row mcq
# split. Those last ~0.6% are genuinely ambiguous phrasings, not missing
# patterns, so demanding 100% would mean fitting the cues to this eval set.
# What this check is FOR is catching a catastrophically broken scorer -- and it
# did: an earlier version scored 59.7% because tool-call extraction only handled
# the tagged gold shape and every eval call parsed to nothing.
ORACLE_FLOOR = 0.99
acc = m['decision_accuracy']
assert acc >= ORACLE_FLOOR, f'ORACLE BELOW FLOOR ({acc:.3f} < {ORACLE_FLOOR}) - scorer is broken, stop here'
print(f'Oracle at {acc:.1%} (floor {ORACLE_FLOOR:.0%}). Scorer is trustworthy.')
print('NOTE: this is the measurement ceiling for every number below.')''')


md(r'''## 6b. Restore cached adapters (optional)

Kaggle wipes `/kaggle/working/` between runs and does not carry `outputs/` forward. So
without a cache, a single bug anywhere downstream costs the full ~2.5h of QLoRA training
again -- which is exactly what happened when the evaluators silently scored the test
fixture.

To use the cache: publish the adapters as a Kaggle Dataset with
`scripts/publish_adapters.sh`, attach that Dataset to this kernel (either through the
UI or via `dataset_sources` in `kaggle/kernel-metadata.json`), and the cell below copies
both arms out of `/kaggle/input/` into `outputs/`. The two training cells then skip
themselves. With no Dataset attached it is a no-op.

The cell finds the arms by searching for `adapter_config.json` rather than by matching a
fixed path, because the mount layout depends on how the Dataset was uploaded and an
earlier exact-path version silently matched nothing.''')
code(r'''import glob, tempfile

CACHE_HITS = []

# Locate the arms by NAME, not by a fixed path.
#
# The previous version globbed two exact layouts -- /kaggle/input/*/outputs and
# the notebook's own reliable_tool_use_adapters.zip. This notebook does not
# control how the Dataset was uploaded, and the publish script uses
# `--dir-mode zip`, which stores the contents of outputs/ at the archive root
# (tool_sft/..., reliable_tool_sft/...). Neither pattern matched that, so the
# cache was ignored in silence and both arms retrained -- exactly the 2.5 GPU
# hours the cache exists to save.
#
# Searching for adapter_config.json instead is layout-agnostic: whatever shape
# the mount arrives in, if the arm is there it is found. /kaggle/input is
# read-only, so matches are copied into the writable working directory.
SEARCH_ROOTS = ['/kaggle/input']

# The zip this notebook writes at the end, if that is what was attached.
for zpath in sorted(glob.glob('/kaggle/input/*/reliable_tool_use_adapters.zip')):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(tmp)
    SEARCH_ROOTS.append(tmp)

for root in SEARCH_ROOTS:
    for cfg in sorted(glob.glob(os.path.join(root, '**', 'adapter_config.json'),
                                recursive=True)):
        arm = os.path.basename(os.path.dirname(cfg))
        if arm not in ('tool_sft', 'reliable_tool_sft'):
            continue
        dst = os.path.join('outputs', arm)
        if os.path.isdir(dst):
            continue
        shutil.copytree(os.path.dirname(cfg), dst)
        CACHE_HITS.append(os.path.dirname(cfg))


def cached(arm):
    # adapter_config.json is what PEFT writes alongside adapter_model.safetensors,
    # so its presence means a usable adapter -- not a half-written directory.
    # adapter_model.safetensors is checked too: an earlier version looked for
    # the config alone, and a truncated upload would have passed the check and
    # then failed inside PeftModel.from_pretrained during evaluation.
    d = os.path.join('outputs', arm)
    return (os.path.isfile(os.path.join(d, 'adapter_config.json'))
            and os.path.isfile(os.path.join(d, 'adapter_model.safetensors')))


for arm in ['tool_sft', 'reliable_tool_sft']:
    print('%-20s cached=%s' % (arm, cached(arm)))
print('cache hits:', CACHE_HITS or 'none - will train from scratch')''')


md(r'''## 7. Train Tool-SFT

4-bit NF4 QLoRA, rank 16 / alpha 32, 3 epochs. fp16 on T4 -- Turing has no bf16, and the
trainer gates on compute capability rather than on `is_bf16_supported()`, which returns
True on hardware that cannot do it.''')
code(r'''# SKIP if the adapter is already present.
#
# The cache cell above restores the arms from an attached Kaggle Dataset.
# Without this guard the cell trained anyway -- the markdown claimed the cells
# "skip themselves" but no code ever checked, so attaching the cache saved
# nothing and a re-run still cost the full ~1h of QLoRA.
if cached('tool_sft'):
    print('tool_sft adapter already present - skipping training')
    print(json.load(open('outputs/tool_sft/run_config.json')))
else:
    t0 = time.time()
    run(sys.executable, '-m', 'src.train_qlora',
        '--config', 'configs/tool_sft.yaml',
        '--data', 'data/processed',
        '--variant', 'sft',
        '--out', 'outputs/tool_sft')
    print('Tool-SFT training took %.1f min' % ((time.time() - t0) / 60))''')


md(r'''## 8. Train Reliable Tool-SFT

Same `--data` as Tool-SFT. The arms differ by `--variant`, not by corpus.''')
code(r'''# Same guard as Tool-SFT. The two arms are cached independently, so a dataset
# holding only one of them still saves that one.
if cached('reliable_tool_sft'):
    print('reliable_tool_sft adapter already present - skipping training')
    print(json.load(open('outputs/reliable_tool_sft/run_config.json')))
else:
    t0 = time.time()
    run(sys.executable, '-m', 'src.train_qlora',
        '--config', 'configs/reliable_tool_sft.yaml',
        '--data', 'data/processed',
        '--variant', 'sft-neg',
        '--out', 'outputs/reliable_tool_sft')
    print('Reliable Tool-SFT training took %.1f min' % ((time.time() - t0) / 60))''')


md("## 9. Evaluate - tool-call track")
code(r'''BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
ARMS = [('base', None),
        ('tool_sft', 'outputs/tool_sft'),
        ('reliable_tool_sft', 'outputs/reliable_tool_sft')]

# --backend hf IS REQUIRED. Both evaluators default to the "dummy" scripted
# fixture, which reads the gold answer and degrades it on purpose -- so omitting
# this flag produces a table that looks plausible and is entirely fake. An
# earlier version of this notebook did exactly that, and all three arms came out
# byte-identical because none of them loaded an adapter.
for arm, adapter in ARMS:
    args = [sys.executable, '-m', 'src.evaluate',
            '--data', 'data/processed', '--split', 'all',
            '--backend', 'hf',
            '--checkpoint', BASE, '--out', 'results/' + arm]
    if adapter:
        args += ['--adapter', adapter]
    run(*args)''')


md("## 10. Evaluate - When2Call decision track")
code(r'''# --backend hf again, and a bounded sample.
#
# The generation backend is batch-of-1, so scoring all 3,652 mcq rows for three
# arms is ~11,000 sequential generations -- hours on a T4, and the reason a
# full-corpus decision eval is the thing most likely to hit Kaggle's wall clock.
# N_W2C_EVAL is set well above the point where the per-class rates stabilise:
# at 1,200 rows the no-tool split still contributes ~770 examples, which is
# ample for a false-tool-call rate. Widen it if you have the budget.
N_W2C_EVAL = 1200

for arm, adapter in ARMS:
    args = [sys.executable, '-m', 'src.evaluate_when2call',
            '--data', 'data/raw/w2c_eval.jsonl',
            '--backend', 'hf',
            '--limit', N_W2C_EVAL,
            '--checkpoint', BASE, '--out', 'results/' + arm + '_when2call']
    if adapter:
        args += ['--adapter', adapter]
    run(*args)''')


md("## 11. Aggregate, analyse, plot")
code(r'''# Flatten the per-run metrics into the names aggregate_results expects.
for arm, _ in ARMS:
    for src, dst in [
        ('results/%s/metrics.json' % arm, 'results/%s_metrics.json' % arm),
        ('results/%s/failures.jsonl' % arm, 'results/%s_failures.jsonl' % arm),
        ('results/%s_when2call/metrics.json' % arm, 'results/%s_when2call.json' % arm),
        ('results/%s_when2call/failures.jsonl' % arm, 'results/%s_when2call_failures.jsonl' % arm),
    ]:
        if os.path.exists(src):
            shutil.copy(src, dst)

os.environ['MPLCONFIGDIR'] = '/kaggle/working/mpl'
run(sys.executable, '-m', 'src.aggregate_results', '--results', 'results')
run(sys.executable, '-m', 'src.error_analysis', '--results', 'results',
    '--out', 'results/error_analysis.json', '--max-examples', 20)
run(sys.executable, '-m', 'src.cv_metrics', '--results', 'results',
    '--out', 'results/cv_metrics.md')''')


md("## 12. Results")
code(r'''from IPython.display import Markdown, Image, display
display(Markdown(open('results/cv_metrics.md').read()))''')
code(r'''for fig in ['exact_match.png', 'false_tool_call.png', 'confusion_matrix.png']:
    p = 'results/figures/' + fig
    if os.path.exists(p):
        display(Markdown('**%s**' % fig))
        display(Image(p))''')


md("## 13. Error analysis")
code(r'''ea = json.load(open('results/error_analysis.json'))
print('Failure counts by category:')
for k, v in ea['counts'].items():
    print('  %-24s %d' % (k, v))
print('%d representative failures:' % ea['n_examples'])
for ex in ea['examples'][:5]:
    print('[%s/%s] %s' % (ex['checkpoint'], ex['track'], ex['category']))
    print('  %s' % ex['prediction'][:200])''')


md(r'''## 14. Training provenance

Hardware, dtype and parameter counts recorded at train time.''')
code(r'''for arm in ['tool_sft', 'reliable_tool_sft']:
    p = 'outputs/%s/run_config.json' % arm
    if os.path.exists(p):
        m = json.load(open(p))
        print('=== %s ===' % arm)
        for k in ['n_records', 'trainable_params', 'total_params', 'trainable_pct',
                  'compute_dtype', 'device', 'peak_gpu_mem_gb']:
            print('  %-18s %s' % (k, m.get(k)))''')

code(r'''# Bundle results AND the trained adapters.
#
# results/ alone is not enough: if this kernel dies during evaluation, an
# archive without outputs/ means the 2.5 GPU-hours of training are gone and the
# only way back is to re-train. The adapters are ~35 MB each.
shutil.make_archive('/kaggle/working/reliable_tool_use_results', 'zip', 'results')

adapter_paths = []
for arm in ['tool_sft', 'reliable_tool_sft']:
    d = 'outputs/%s' % arm
    if os.path.isdir(d):
        for name in os.listdir(d):
            if name.startswith('adapter_') or name == 'run_config.json':
                adapter_paths.append(os.path.join(d, name))
if adapter_paths:
    with zipfile.ZipFile('/kaggle/working/reliable_tool_use_adapters.zip', 'w',
                         zipfile.ZIP_DEFLATED) as zf:
        for p in adapter_paths:
            zf.write(p, p)
    print('wrote /kaggle/working/reliable_tool_use_adapters.zip')

print('wrote /kaggle/working/reliable_tool_use_results.zip')
print(subprocess.run(['du', '-sh', 'results', 'outputs'],
                     capture_output=True, text=True).stdout)''')


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "reliable_tool_use_slm.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in reloaded["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    assert all(c["id"] for c in reloaded["cells"]), "missing cell id"
    print("wrote", path)
    print("cells:", len(reloaded["cells"]), "| ids unique:", len(ids) == len(set(ids)))
    for i, c in enumerate(reloaded["cells"]):
        src = "".join(c["source"]).strip()
        first = src.split("\n")[0][:76] if src else "(empty)"
        print("%2d [%s] %-8s %s" % (i, c["cell_type"][:4], c["id"], first))


if __name__ == "__main__":
    main()