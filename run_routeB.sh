#!/usr/bin/env bash
# Route B 一键运行：CLEVRER DINOv2 特征预计算 + 按 video ID 切分训练。
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLEVRER_ROOT="${CLEVRER_ROOT:-/root/autodl-tmp/clevrer}"
DINOV2_CKPT="${DINOV2_CKPT:-/root/autodl-tmp/models/dinov2_small/model.safetensors}"
FEATURE_ROOT="${FEATURE_ROOT:-/autodl-fs/data/oc-causal-wm/clevrer_features}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/out/clevrer/route_b_image/v7_routeB_image}"
NTRAIN="${NTRAIN:-80000}"
NVAL="${NVAL:-10000}"
PRE_BATCH="${PRE_BATCH:-64}"
TRAIN_BATCH="${TRAIN_BATCH:-384}"
OBS="${OBS:-4}"; PRED="${PRED:-8}"; M="${M:-6}"
EPOCHS="${EPOCHS:-40}"; SEEDS="${SEEDS:-3}"

export DINOV2_CKPT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$FEATURE_ROOT" "$RESULT_ROOT"

echo "==== [1/3] precompute train features (n=$NTRAIN) ===="
python precompute_clevrer_feats.py --root "$CLEVRER_ROOT" --split train \
  --n "$NTRAIN" --obs "$OBS" --pred "$PRED" --M "$M" --stride 8 --crop 64 \
  --batch "$PRE_BATCH" --device cuda --out "$FEATURE_ROOT/clevrer_feats_train.npz"

echo "==== [2/3] precompute validation features (n=$NVAL) ===="
python precompute_clevrer_feats.py --root "$CLEVRER_ROOT" --split validation \
  --n "$NVAL" --obs "$OBS" --pred "$PRED" --M "$M" --stride 8 --crop 64 \
  --batch "$PRE_BATCH" --device cuda --out "$FEATURE_ROOT/clevrer_feats_val.npz"

echo "==== [3/3] train (dev/test split by video ID) ===="
python run_clevrer_dino.py \
  --train_feats "$FEATURE_ROOT/clevrer_feats_train.npz" \
  --val_feats "$FEATURE_ROOT/clevrer_feats_val.npz" \
  --obs "$OBS" --pred "$PRED" --epochs "$EPOCHS" --batch "$TRAIN_BATCH" \
  --d 128 --nlayers 4 --nhead 4 --pos_w 5 --seeds "$SEEDS" \
  --device cuda --out "$RESULT_ROOT/clevrer_dino_clean.json"

echo "==== done -> $RESULT_ROOT/clevrer_dino_clean.json ===="
