#!/usr/bin/env bash
# AutoDL 环境检查/初始化（路线 B）。可重复运行。
set -e

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=== python ==="
python --version
python - <<'PY'
mods = ["torch", "torchvision", "timm", "cv2", "pycocotools", "numpy", "huggingface_hub"]
missing = []
for m in mods:
    try:
        __import__(m)
        print(f"  OK  {m}")
    except Exception as e:
        print(f"  MISSING  {m} ({type(e).__name__})")
        missing.append(m)
import sys
sys.exit(0)
PY

echo "=== install missing (若上一步有 MISSING，手动执行) ==="
echo "pip install -i $PIP_MIRROR numpy timm pycocotools opencv-python-headless huggingface_hub tqdm matplotlib"
echo "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"

echo "=== DINOv2 weight ==="
CKPT="${DINOV2_CKPT:-/root/autodl-tmp/models/dinov2_small/model.safetensors}"
if [ -f "$CKPT" ]; then
  echo "  found: $CKPT"
else
  echo "  not found at $CKPT ; downloading from hf-mirror ..."
  mkdir -p "$(dirname "$CKPT")"
  python - <<PY
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download("timm/vit_small_patch14_dinov2.lvd142m", "model.safetensors")
dst = "$CKPT"
import shutil, os
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(p, dst)
print("copied ->", dst)
PY
fi

echo "=== data root check ==="
CR="${CLEVRER_ROOT:-/root/autodl-tmp/clevrer}"
for d in train validation video_train video_validation derender questions; do
  n=$(find "$CR/$d" -maxdepth 2 \( -name "annotation_*.json" -o -name "video_*.mp4" -o -name "proposal_*.json" -o -name "*.json" \) 2>/dev/null | wc -l)
  echo "  $CR/$d : $n files"
done
echo "done. 记得 export DINOV2_CKPT=$CKPT HF_HUB_OFFLINE=1 CLEVRER_ROOT=$CR"
