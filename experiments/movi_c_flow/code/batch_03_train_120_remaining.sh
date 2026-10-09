#!/usr/bin/env bash
# Remaining exploratory comparisons on the existing 120/30/30 pilot split.
# The new runner has a changed protocol: rerun no-flow and aligned-flow as anchors.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/autodl-fs/data/movi_c_flow_v1}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/oc-causal-wm/out/movi_c_flow/pilot_120/movi_c_flow_120_v2_remaining}"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/oc-causal-wm/models/movi_c_flow/pilot_120}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEEDS="${SEEDS:-0 1 2}"

[[ -f "$SCRIPT_DIR/train_movi_flow_suite.py" ]] || { echo 'Missing train_movi_flow_suite.py' >&2; exit 1; }
for entry in 'train 120' 'validation 30' 'test 30'; do
  read -r split expected <<< "$entry"
  actual="$(find "$DATA_ROOT/$split" -maxdepth 1 -type f -name '*.npz' | wc -l)"
  [[ "$actual" -eq "$expected" ]] || { echo "$split: expected $expected, got $actual" >&2; exit 1; }
done
"$TRAIN_PYTHON" -c 'import torch, numpy; assert torch.cuda.is_available(), "Enable the 4090 GPU"; print("GPU:", torch.cuda.get_device_name(0))'
mkdir -p "$OUT_ROOT"

if [[ ! -s "$OUT_ROOT/heuristics/heuristic_metrics.json" ]]; then
  mkdir -p "$OUT_ROOT/heuristics"
  "$TRAIN_PYTHON" -u "$SCRIPT_DIR/train_movi_flow_suite.py" \
    --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT/heuristics" \
    --mode no_flow --flow-order auto --heuristics \
    2>&1 | tee "$OUT_ROOT/heuristics/run.log"
fi

run_training() {
  local mode="$1"
  local seed="$2"
  local run_dir="$OUT_ROOT/${mode}_seed${seed}"
  if [[ -s "$run_dir/metrics.json" ]]; then echo "SKIP completed: $mode seed=$seed"; return; fi
  mkdir -p "$run_dir"
  echo "TRAIN $mode seed=$seed"
  "$TRAIN_PYTHON" -u "$SCRIPT_DIR/train_movi_flow_suite.py" \
    --data-root "$DATA_ROOT" --output-dir "$run_dir" \
    --mode "$mode" --seed "$seed" --flow-order auto \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$run_dir/run.log"
  if [[ -f "$run_dir/checkpoint_best.pt" ]]; then
    mkdir -p "$MODEL_ROOT/${mode}_seed${seed}"
    mv -f "$run_dir/checkpoint_best.pt" "$MODEL_ROOT/${mode}_seed${seed}/checkpoint_best.pt"
    ln -s "$MODEL_ROOT/${mode}_seed${seed}/checkpoint_best.pt" "$run_dir/checkpoint_best.pt"
  fi
}

run_perturbation() {
  local perturb="$1"
  local seed="$2"
  local run_dir="$OUT_ROOT/aligned_flow_eval_${perturb}_seed${seed}"
  local checkpoint="$MODEL_ROOT/aligned_flow_seed${seed}/checkpoint_best.pt"
  if [[ -s "$run_dir/metrics.json" ]]; then echo "SKIP completed: eval $perturb seed=$seed"; return; fi
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  mkdir -p "$run_dir"
  echo "EVAL aligned checkpoint with $perturb flow, seed=$seed"
  "$TRAIN_PYTHON" -u "$SCRIPT_DIR/train_movi_flow_suite.py" \
    --data-root "$DATA_ROOT" --output-dir "$run_dir" \
    --mode aligned_flow --seed "$seed" --flow-order auto \
    --checkpoint "$checkpoint" --eval-perturb "$perturb" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$run_dir/run.log"
}

for seed in $SEEDS; do
  for mode in no_flow aligned_flow flow_aux input_aux flow_bottleneck visual_no_flow visual_flow; do
    run_training "$mode" "$seed"
  done
  for perturb in zero shuffled negated; do
    run_perturbation "$perturb" "$seed"
  done
done
echo "All pilot follow-up runs finished. Results: $OUT_ROOT"
