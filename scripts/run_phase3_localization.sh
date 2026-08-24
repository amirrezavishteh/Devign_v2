#!/usr/bin/env bash
# Phase 3.2 / 3.4: localisation evaluation and the qualitative page.
#
# Runs AFTER the detection sweep, because it loads the trained checkpoints rather than retraining.
#
# Two preconditions it checks rather than assumes:
#   1. PrimeVul paired split present -- CodeXGLUE cannot substitute, it has no func_after.
#   2. Processed data at format v3 -- v2 carries no per-node line spans, so attention cannot be
#      projected onto lines. Training is unaffected by the difference, which is exactly why it
#      would otherwise go unnoticed until the localisation numbers came out empty.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/media/external20/amirreza_vishteh/anaconda3/envs/devigen/bin/python}"
CFG="${CFG:-configs/a100_codexglue.yaml}"
PRIMEVUL="${PRIMEVUL:-data/primevul/primevul_test_paired.jsonl}"
SEED_DIR="${SEED_DIR:-artifacts/seed1}"
AGG="${AGG:-max}"

mkdir -p logs artifacts/localization

if ! $PY -m scripts.fetch_primevul --check --path "$PRIMEVUL"; then
  echo "[loc] PrimeVul missing or unusable -- see the instructions above. Nothing else to do."
  exit 1
fi

# Format v3 check. `prepare` is ~15 minutes; a silently line-span-less evaluation costs more.
if ! $PY - <<'PYEOF'
import sys, os
from devign_data.dataset import DevignDataset, FORMAT_VERSION
from training.utils import load_config
cfg = load_config(os.environ.get("CFG", "configs/a100_codexglue.yaml"))
path = os.path.join(cfg["data"]["processed_dir"], "test.pkl")
ds = DevignDataset.load(path)
missing = sum(1 for s in ds.samples[:200] if getattr(s, "node_lines", None) is None)
print(f"[loc] {path}: {missing}/200 sampled graphs lack line spans")
sys.exit(1 if missing else 0)
PYEOF
then
  echo "[loc] processed data predates format v3 (no per-node line spans)."
  echo "[loc] re-preparing -- training results are unaffected, only localisation needs this."
  $PY -u -m scripts.prepare_data --config "$CFG" 2>&1 | tail -6
fi

for model in mil devign; do
  dir="${SEED_DIR}/${model}/combined"
  if [ ! -f "${dir}/model.pt" ]; then
    echo "[loc] ${model}: no checkpoint at ${dir}, skipping"
    continue
  fi
  echo "[loc] $(date +%H:%M:%S) localisation for ${model}"
  $PY -u -m scripts.run_localization --config "$CFG" --model "$model" --model-dir "$dir" \
      --primevul "$PRIMEVUL" --aggregation "$AGG" \
      --out "artifacts/localization/${model}_${AGG}.json" 2>&1 | tee "logs/loc_${model}.log"

  # Both aggregations, because the choice moves the numbers and must never be implicit.
  other=$([ "$AGG" = "max" ] && echo sum || echo max)
  $PY -u -m scripts.run_localization --config "$CFG" --model "$model" --model-dir "$dir" \
      --primevul "$PRIMEVUL" --aggregation "$other" \
      --out "artifacts/localization/${model}_${other}.json" > "logs/loc_${model}_${other}.log" 2>&1
done

if [ -f "${SEED_DIR}/mil/combined/model.pt" ]; then
  echo "[loc] $(date +%H:%M:%S) rendering qualitative examples"
  $PY -u -m scripts.render_attention --config "$CFG" --model mil \
      --model-dir "${SEED_DIR}/mil/combined" --primevul "$PRIMEVUL" \
      --aggregation "$AGG" --successes 4 --failures 1 \
      --out artifacts/attention_examples.html 2>&1 | tail -3
fi
echo "[loc] $(date +%H:%M:%S) DONE"
