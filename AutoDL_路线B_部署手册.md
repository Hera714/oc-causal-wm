# AutoDL 部署手册 —— 路线 B（CLEVRER 图像级 Δ 世界模型）

> 目标：在一台 AutoDL 服务器上，从零环境到**跑通训练**并得到干净数字。
> 仓库：`git@github.com:Hera714/oc-causal-wm.git`（仓库根 = 原 `exp/`）。

---

## 0. 实例与目录约定

- **GPU**：3090 / 4090 / A5000（≥24G 足够；DINOv2 预计算是瓶颈在 CPU 解码）。
- **系统盘**：`/root/`；**数据盘**：`/root/autodl-tmp/`（大，放 CLEVRER 数据）。
- 约定：
  - 代码：`/root/oc-causal-wm`
  - 数据：`/root/autodl-tmp/clevrer/{train,validation,video_train,video_validation,derender,questions}`
  - DINOv2 权重：`/root/models/dinov2_small/model.safetensors`

---

## 1. 环境配置

```bash
# 1.1 拉代码
cd /root
git clone git@github.com:Hera714/oc-causal-wm.git
#    （若没配 SSH key，用 HTTPS：git clone https://github.com/Hera714/oc-causal-wm.git）
cd oc-causal-wm

# 1.2 conda 环境
conda create -y -n ocwm python=3.11
source activate ocwm          # 或 conda activate ocwm

# 1.3 PyTorch（按实例 CUDA 选；下面 cu121 通用，若 3090/4090 亦可 cu118）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 1.4 其余依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy timm pycocotools opencv-python-headless huggingface_hub tqdm

# 1.5 验证
python -c "import torch, timm, cv2, pycocotools; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

> 国内网络建议长期设置：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 2. 下载数据（CLEVRER）

```bash
export CR=/root/autodl-tmp/clevrer
mkdir -p $CR && cd $CR

# 2.1 标注 / 问题（小）
wget http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip
wget http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip
wget http://data.csail.mit.edu/clevrer/questions/train.json
wget http://data.csail.mit.edu/clevrer/questions/validation.json
mkdir -p train validation questions
unzip -q annotation_train.zip -d train
unzip -q annotation_validation.zip -d validation
mv train.json validation.json questions/

# 2.2 de-render 物体 mask（397MB，抠 crop 用）
wget http://data.csail.mit.edu/clevrer/derender_proposals.zip
mkdir -p derender && unzip -q derender_proposals.zip -d derender

# 2.3 视频（关键，较大）
wget http://data.csail.mit.edu/clevrer/videos/train/video_train.zip          # 12.35GB
wget http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip # 6.21GB
mkdir -p video_train video_validation
unzip -q video_train.zip -d video_train
unzip -q video_validation.zip -d video_validation
```

> 目录层级不用管：`precompute_clevrer_feats.py` 会递归找 `annotation_*.json` / `video_*.mp4` / `proposal_*.json`。

---

## 3. DINOv2 权重（~85MB）

```bash
mkdir -p /root/models/dinov2_small && cd /root/models/dinov2_small
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("timm/vit_small_patch14_dinov2.lvd142m", "model.safetensors")
print("saved:", p)
PY
# 也可直接 wget 直链：
# wget https://hf-mirror.com/timm/vit_small_patch14_dinov2.lvd142m/resolve/main/model.safetensors
```

---

## 4. 环境变量（每次开新 shell 都要设）

```bash
export DINOV2_CKPT=/root/models/dinov2_small/model.safetensors   # 或 HF 缓存里的路径
export HF_HUB_OFFLINE=1                                          # 权重就绪后离线加载
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLEVRER_ROOT=/root/autodl-tmp/clevrer
```

---

## 5. 一键跑通（预计算 + 训练）

```bash
cd /root/oc-causal-wm
bash server_setup.sh          # 可选：检查依赖/权重（见脚本）
bash run_routeB.sh            # 预计算 train+val 特征，然后训练
```

`run_routeB.sh` 做的事：
1. `precompute_clevrer_feats.py --split train` → `out/clevrer_feats_train.npz`
2. `precompute_clevrer_feats.py --split validation` → `out/clevrer_feats_val.npz`
3. `run_clevrer_dino.py --train_feats ... --val_feats ...`（**dev/test 按 video id 切分**，无泄漏）

可调环境变量：`NTRAIN`（训练窗口数，默认 80000）、`NVAL`（默认 10000）。

---

## 6. 单步命令（自行控制）

```bash
cd /root/oc-causal-wm

# 6.1 训练集特征（10000 视频；~1.5–2h；npz 约 1–1.5GB）
python precompute_clevrer_feats.py --root $CLEVRER_ROOT --split train \
  --n 80000 --obs 4 --pred 8 --M 6 --stride 8 --crop 64 --batch 32 \
  --device cuda --out out/clevrer_feats_train.npz

# 6.2 验证集特征（5000 视频）
python precompute_clevrer_feats.py --root $CLEVRER_ROOT --split validation \
  --n 10000 --obs 4 --pred 8 --M 6 --stride 8 --crop 64 --batch 32 \
  --device cuda --out out/clevrer_feats_val.npz

# 6.3 训练（干净基线：训练集=train，dev/test=validation 按视频切分）
python run_clevrer_dino.py \
  --train_feats out/clevrer_feats_train.npz \
  --val_feats out/clevrer_feats_val.npz \
  --obs 4 --pred 8 --epochs 40 --batch 256 --d 128 --nlayers 4 --nhead 4 \
  --pos_w 5 --seeds 3 --device cuda --out out/clevrer_dino_clean.json
```

产物：`out/clevrer_dino_clean.json`（含 absolute / delta_fixed / delta_rec 的
future_mse、target_mse、逐 step 曲线）+ `out/recursion_curves_dinov2.png`。

---

## 7. 预期耗时 / 显存

| 步骤 | 耗时（估） | 显存 |
|---|---|---|
| 训练集预计算（10000 视频，n=80000） | 1.5–2 h（CPU 解码为主） | <4G |
| 验证集预计算（5000 视频） | ~15–25 min | <4G |
| 训练（3 config × 3 seed） | ~10–20 min | <4G |

> 若嫌预计算慢：减小 `--n`、增大 `--stride`、或只取部分视频（`--max_videos` 若需要可加）。

---

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| `pycocotools` 装不上 | `pip install pycocotools`（有 wheel）；或 `pip install pycocotools-windows`（Win） |
| DINOv2 卡在下载 | 设 `HF_ENDPOINT=https://hf-mirror.com`；或手动 wget 后设 `DINOV2_CKPT` |
| 训练一开始极慢 | 确认没落在旧版"惰性 npz 解压"分支——本仓库 `run_clevrer_dino.py` 已一次性载入内存 |
| 显存不足 | 预计算 `--batch 16`；训练 `--batch 128` |
| `cv2` 读视频失败 | 用 `opencv-python-headless`；确认 mp4 完整解压 |
| `CausalSpatial` 无关 | 本手册只跑路线 B；阶段 A/CausalSpatial 见另一份手册 |
