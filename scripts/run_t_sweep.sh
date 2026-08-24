#!/usr/bin/env bash
# Phase 3.3: sensitivity to the number of GGNN message-passing steps T.
#
# The paper fixes T = 6. Median AST depth in this corpus is 11 (Q3 14, p90 18), so at T = 6 the
# upper half of a typical tree is unreachable from its leaves -- information from a token cannot
# arrive at the function root at all, however many epochs it trains for. If detection or
# localisation improves with T, over-squashing is bounding the reported result and that belongs in
# the write-up as a limitation with evidence rather than as a hunch.
#
# Sequential, resumable, one stage per T. Detection only; the localisation sweep reuses these
# checkpoints rather than retraining.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/media/external20/amirreza_vishteh/anaconda3/envs/devigen/bin/python}"
CFG="${CFG:-configs/a100_codexglue.yaml}"
MODEL="${MODEL:-mil}"
SEEDS="${SEEDS:-1 2 3}"

mkdir -p logs artifacts/t_sweep

while pgrep -u "$USER" -f "scripts.run_seeds" >/dev/null; do sleep 60; done

for T in 4 6 8 12; do
  name="t_sweep_${MODEL}_T${T}"
  if [ -f "logs/DONE_${name}" ]; then echo "[tsweep] ${name}: done, skipping"; continue; fi
  echo "[tsweep] $(date +%H:%M:%S) starting ${name}"
  if $PY -u -m scripts.run_seeds --config "$CFG" --model "$MODEL" --seeds $SEEDS \
       --set "model.time_steps=${T}" \
       --out "artifacts/t_sweep/${MODEL}_T${T}.json" > "logs/${name}.log" 2>&1; then
    touch "logs/DONE_${name}"
    echo "[tsweep] $(date +%H:%M:%S) ${name} OK"
  else
    echo "[tsweep] $(date +%H:%M:%S) ${name} FAILED -- see logs/${name}.log"
  fi
done
echo "[tsweep] $(date +%H:%M:%S) ALL T VALUES ATTEMPTED"
