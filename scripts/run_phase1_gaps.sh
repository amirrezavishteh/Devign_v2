#!/usr/bin/env bash
# The Phase 1 work that was never run: the four Table-2 baselines, and the per-project columns.
#
# Section 1.3 of the brief asked for six models across three project cells. What existed was two
# models on Combined alone, which means there was no Table 2 to compare against the paper and no
# figure that lines up with any single cell the paper prints.
#
# Order is by what is missing most: baselines first (the paper's entire comparison is against
# them, and sequence models train in minutes rather than hours), then the per-project graph runs.
#
# Sequential, resumable, isolated -- same contract as the other queue scripts. GPU selection comes
# from config_a100.yaml (`project.cuda_device: 1`), never from CUDA_VISIBLE_DEVICES here.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/media/external20/amirreza_vishteh/anaconda3/envs/devigen/bin/python}"
CFG="${CFG:-configs/a100_codexglue.yaml}"
# Three seeds for the per-project cells rather than five: six extra cells at five seeds is ~10
# hours of shared GPU, and 3 seeds still yields a std. The Combined headline stays at 5.
SEEDS="${SEEDS:-1 2 3}"

mkdir -p logs artifacts/seeds

while pgrep -u "$USER" -f "scripts\.(run_seeds|train|run_leakage_check|run_ablation|reproduce)" >/dev/null; do sleep 60; done

stage () {
  local name="$1"; shift
  if [ -f "logs/DONE_${name}" ]; then echo "[gaps] ${name}: already done, skipping"; return; fi
  echo "[gaps] $(date +%H:%M:%S) starting ${name}"
  if "$@" > "logs/${name}.log" 2>&1; then
    touch "logs/DONE_${name}"
    echo "[gaps] $(date +%H:%M:%S) ${name} OK"
  else
    echo "[gaps] $(date +%H:%M:%S) ${name} FAILED (exit $?) -- see logs/${name}.log"
  fi
}

# ---- 1. The four Table-2 baselines, on every project cell -------------------------------------
# scripts.train_baselines trains all four for one project in a single invocation.
for project in qemu ffmpeg combined; do
  stage "baselines_${project}" $PY -u -m scripts.train_baselines --config "$CFG" --project "$project"
done

# ---- 2. Per-project graph models --------------------------------------------------------------
# The paper reports QEMU and FFmpeg as separate columns. Every graph figure so far is pooled.
for project in qemu ffmpeg; do
  for model in devign ggrn mil; do
    stage "seeds_${model}_${project}" $PY -u -m scripts.run_seeds \
      --config "$CFG" --model "$model" --seeds $SEEDS --project "$project" \
      --out "artifacts/seeds_per_project/${model}_${project}.json"
  done
done

echo "[gaps] $(date +%H:%M:%S) ALL STAGES ATTEMPTED"
