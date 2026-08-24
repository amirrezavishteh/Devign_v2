#!/usr/bin/env bash
# Phase 1 matrix, run sequentially. The GPU is shared and already near capacity, so running these
# concurrently would just make them fight each other and would make every timing meaningless.
# Each stage logs separately and a failure in one does not stop the rest.
set -u
cd "$(dirname "$0")/.."
# Override PY to point at whichever interpreter has the deps; the default is the A100 box's
# conda env. Note `conda run` is deliberately NOT used: it buffers stdout, so a long training
# log stays empty until the process exits, which makes a running job indistinguishable from a
# hung one.
PY="${PY:-/media/external20/amirreza_vishteh/anaconda3/envs/devigen/bin/python}"
# GPU selection lives in config_a100.yaml (`project.cuda_device: 1`), NOT here. Setting
# CUDA_VISIBLE_DEVICES as well would remap the indices underneath that setting, so the config
# would be describing a card it is not actually using -- and the manifest would record the lie.

# Wait for ANY of this project's training entry points, not just run_seeds. The first version
# watched run_seeds alone, so a Phase 1 leakage check (scripts.run_leakage_check) and a Phase 3
# sweep started training concurrently on the same card. It did no harm at the time because 78 GB
# were free, but on a full card it would have meant OOM-skipped batches in both -- silently
# training each on less data than configured.
while pgrep -u "$USER" -f "scripts\.(run_seeds|train|run_leakage_check|run_ablation|reproduce)" >/dev/null; do sleep 60; done

stage () {                      # stage <logname> <args...>
  local name="$1"; shift
  if [ -f "logs/DONE_${name}" ]; then echo "[queue] ${name}: already done, skipping"; return; fi
  echo "[queue] $(date +%H:%M:%S) starting ${name}"
  if "$@" > "logs/${name}.log" 2>&1; then
    touch "logs/DONE_${name}"
    echo "[queue] $(date +%H:%M:%S) ${name} OK"
  else
    echo "[queue] $(date +%H:%M:%S) ${name} FAILED (exit $?) -- see logs/${name}.log"
  fi
}

stage seeds_ggrn $PY -u -m scripts.run_seeds \
  --config configs/a100_codexglue.yaml --model ggrn --seeds 1 2 3 \
  --out artifacts/seeds/ggrn_codexglue.json

stage seeds_devign_paperfaithful $PY -u -m scripts.run_seeds \
  --config configs/a100_paper_faithful.yaml --model devign --seeds 1 2 3 \
  --out artifacts/seeds/devign_paper_faithful.json

stage leakage_check $PY -u -m scripts.run_leakage_check \
  --config configs/a100_codexglue.yaml --models devign

echo "[queue] $(date +%H:%M:%S) ALL STAGES ATTEMPTED"
