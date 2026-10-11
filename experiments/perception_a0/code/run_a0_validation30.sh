#!/usr/bin/env bash
set -euo pipefail

# Full A0 validation audit. Run only after the n8 smoke/audit succeeds.
REPO="/root/autodl-tmp/oc-causal-wm"
PY="$REPO/experiments/perception_a0/code/a0_perception_audit.py"
DATA="/autodl-fs/data/movi_c_flow_v1"
SAM2_REPO="/root/autodl-tmp/third_party/sam2"
SAM2_CKPT="/autodl-fs/data/models/stage_a/sam2/sam2.1_hiera_base_plus.pt"
RAFT_CKPT="/autodl-fs/data/models/stage_a/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
DINO_CKPT="/root/autodl-tmp/models/dinov2_small/model.safetensors"
OUT="$REPO/out/perception_a0/wm_percept_005/a0_validation_n30"
LOG_DIR="$REPO/out/perception_a0/wm_percept_005/logs"
LOG="$LOG_DIR/a0_validation_n30.log"

mkdir -p "$LOG_DIR"

python "$PY" \
  --data-root "$DATA" \
  --split validation \
  --num-scenes 30 \
  --obs-frames 4 \
  --seed 20261011 \
  --output-dir "$OUT" \
  --sam2-repo "$SAM2_REPO" \
  --sam2-checkpoint "$SAM2_CKPT" \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --raft-checkpoint "$RAFT_CKPT" \
  --dino-checkpoint "$DINO_CKPT" \
  --device cuda \
  --flow-order auto \
  --points-per-side 16 \
  --points-per-batch 64 \
  --pred-iou-thresh 0.72 \
  --stability-thresh 0.82 \
  --min-mask-area 18 \
  --max-mask-area-ratio 0.45 \
  --duplicate-iou 0.80 \
  --max-masks 20 \
  --match-iou 0.50 \
  --track-score-thresh 0.22 \
  --visualize-scenes 12 \
  --overwrite \
  2>&1 | tee "$LOG"

echo "A0 validation metrics: $OUT/metrics.json"
