#!/usr/bin/env bash
# Phase 3: the three-arm readout comparison, at 5 seeds per arm.
#
# Sequential by design. The GPU is shared and already near capacity, so concurrent runs would
# fight each other and make every timing meaningless. Each stage logs separately, marks itself
# done so a re-run resumes rather than restarts, and one stage failing does not stop the others.
#
# Seeds 1-3 already exist for `devign` and `ggrn` from Phase 1; this extends both to 5 so every
# arm is compared on the SAME five seeds. Comparing a 5-seed arm against a 3-seed arm would make
# the paired test meaningless -- the pairs would not line up.
#
# GPU selection lives in config_a100.yaml (`project.cuda_device: 1`), NOT here. Exporting
# CUDA_VISIBLE_DEVICES as well would remap indices underneath that key, so the config would
# describe a card it is not using and the manifest would record that.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/media/external20/amirreza_vishteh/anaconda3/envs/devigen/bin/python}"
CFG="${CFG:-configs/a100_codexglue.yaml}"
SEEDS="${SEEDS:-1 2 3 4 5}"

mkdir -p logs artifacts/seeds

# Wait for ANY of this project's training entry points, not just run_seeds. The first version
# watched run_seeds alone, so a Phase 1 leakage check (scripts.run_leakage_check) and a Phase 3
# sweep started training concurrently on the same card. It did no harm at the time because 78 GB
# were free, but on a full card it would have meant OOM-skipped batches in both -- silently
# training each on less data than configured.
while pgrep -u "$USER" -f "scripts\.(run_seeds|train|run_leakage_check|run_ablation|reproduce)" >/dev/null; do sleep 60; done

stage () {                      # stage <logname> <args...>
  local name="$1"; shift
  if [ -f "logs/DONE_${name}" ]; then echo "[phase3] ${name}: already done, skipping"; return; fi
  echo "[phase3] $(date +%H:%M:%S) starting ${name}"
  if "$@" > "logs/${name}.log" 2>&1; then
    touch "logs/DONE_${name}"
    echo "[phase3] $(date +%H:%M:%S) ${name} OK"
  else
    echo "[phase3] $(date +%H:%M:%S) ${name} FAILED (exit $?) -- see logs/${name}.log"
  fi
}

# The new arm first: it is the one the phase exists to test, so if compute runs out it is the one
# that must have completed.
stage p3_mil_k1 $PY -u -m scripts.run_seeds --config "$CFG" --model mil --seeds $SEEDS \
  --out artifacts/seeds/mil.json

# Extend the two Phase 1 arms to the same five seeds. run_seeds writes each seed into its own
# artifacts subdirectory, so seeds 1-3 are recomputed rather than reused -- slower, but it keeps
# every arm on one code revision instead of silently mixing two.
stage p3_devign_5seed $PY -u -m scripts.run_seeds --config "$CFG" --model devign --seeds $SEEDS \
  --out artifacts/seeds/devign.json

stage p3_ggrn_5seed $PY -u -m scripts.run_seeds --config "$CFG" --model ggrn --seeds $SEEDS \
  --out artifacts/seeds/ggrn.json

# The paper's readout exactly as written, as the "conv, logit_affine FALSE" row. Expected to train
# badly; that is the evidence, not a failure.
stage p3_conv_affine_off $PY -u -m scripts.run_seeds --config configs/a100_conv_affine_off.yaml \
  --model devign --seeds $SEEDS --out artifacts/seeds/devign_affine_off.json

echo "[phase3] $(date +%H:%M:%S) building the comparison table"
$PY -u -m scripts.compare_readouts --dir artifacts/seeds --split test_at_tuned \
  --baseline devign --out artifacts/readout_comparison.json 2>&1 | tee logs/compare_readouts.log

echo "[phase3] $(date +%H:%M:%S) ALL STAGES ATTEMPTED"
