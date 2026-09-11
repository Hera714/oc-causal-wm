#!/usr/bin/env bash
# 路线 B 一键跑通：预计算 DINOv2 特征（train + val）→ 训练（按 video id 切分）。
# 用法：
#   export DINOV2_CKPT=/root/autodl-tmp/models/dinov2_small/model.safetensors
#   export HF_HUB_OFFLINE=1
#   export CLEVRER_ROOT=/root/autodl-tmp/clevrer
#   bash run_routeB.sh
set -e

CLEVRER_ROOT="${CLEVRER_ROOT:-/root/autodl-tmp/clevrer}"
DINOV2_CKPT="${DINOV2_CKPT:-/root/autodl-tmp/models/dinov2_small/model.safetensors}"
NTRAIN="${NTRAIN:-80000}"
NVAL="${NVAL:-10000}"
PRE_BATCH="${PRE_BATCH:-64}"      # 预计算 DINOv2 的 batch（4090 可到 128）
TRAIN_BATCH="${TRAIN_BATCH:-384}" # 训练 batch（4090 24G 可到 512）
OBS="${OBS:-4}"; PRED="${PRED:-8}"; M="${M:-6}"
EPOCHS="${EPOCHS:-40}"; SEEDS="${SEEDS:-3}"

export DINOV2_CKPT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p out

echo "==== [1/3] precompute train features (n=$NTRAIN) ===="
python precompute_clevrer_feats.py --root "$CLEVRER_ROOT" --split train \
  --n "$NTRAIN" --obs "$OBS" --pred "$PRED" --M "$M" --stride 8 --crop 64 \
  --batch "$PRE_BATCH" --device cuda --out out/clevrer_feats_train.npz

echo "==== [2/3] precompute val features (n=$NVAL) ===="
python precompute_clevrer_feats.py --root "$CLEVRER_ROOT" --split validation \
  --n "$NVAL" --obs "$OBS" --pred "$PRED" --M "$M" --stride 8 --crop 64 \
  --batch "$PRE_BATCH" --device cuda --out out/clevrer_feats_val.npz

echo "==== [3/3] train (dev/test split BY video id) ===="
python run_clevrer_dino.py \
  --train_feats out/clevrer_feats_train.npz \
  --val_feats out/clevrer_feats_val.npz \
  --obs "$OBS" --pred "$PRED" --epochs "$EPOCHS" --batch "$TRAIN_BATCH" \
  --d 128 --nlayers 4 --nhead 4 --pos_w 5 --seeds "$SEEDS" \
  --device cuda --out out/clevrer_dino_clean.json

echo "==== done -> out/clevrer_dino_clean.json ===="
