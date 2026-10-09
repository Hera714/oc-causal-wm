# AutoDL 部署手册 —— 路线 B（CLEVRER 图像级 Δ 世界模型）· 4090 24G 专版

> 目标：在 **RTX 4090 24G** 实例上，从零环境到**跑通训练**并得到干净数字。
> 仓库：`git@github.com:Hera714/oc-causal-wm.git`（仓库根 = 原 `exp/`）。
> 4090 是 Ada（sm_89），用标准 `torch cu121` 即可（**不需要** cu128）。
>
> **所有内容都放 `/root/autodl-tmp/`（50G 数据盘）下**：代码、数据、权重。

---

## 0. 目录约定与磁盘

```bash
/root/autodl-tmp/
├─ oc-causal-wm/          # 代码
├─ clevrer/               # 数据 {train,validation,video_train,video_validation,derender,questions}
└─ models/dinov2_small/model.safetensors   # DINOv2 权重
```

- **GPU**：RTX 4090 24G。
- **镜像**：PyTorch 2.x + CUDA 12.1 官方镜像即可。
- **磁盘（autodl-tmp 50G）**：
  - 数据解压后约：视频 18.5G + derender 2.9G + 标注 ~2G + 问题 0.13G ≈ **24G**
  - 特征 npz：train ~1.2–1.5G + val ~0.2G；代码与权重 <1G
  - 合计约 **26–27G → 50G 充足**；下载的 zip **解压后立即删除**，避免峰值翻倍。

---

## 1. 环境配置

```bash
# 1.1 拉代码（放在数据盘）
mkdir -p /root/autodl-tmp && cd /root/autodl-tmp
git clone git@github.com:Hera714/oc-causal-wm.git      # 或 https 地址
cd oc-causal-wm

# 1.2 conda 环境（Python 3.11）
conda create -y -n ocwm python=3.11
source activate ocwm

# 1.3 PyTorch（4090 → cu121）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 1.4 其余依赖（含 matplotlib，否则不出曲线图）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy timm pycocotools opencv-python-headless huggingface_hub tqdm matplotlib

# 1.5 验证：务必确认在 ocwm 环境里（python 路径应含 envs/ocwm）
python -c "import sys,torch,timm,cv2,pycocotools,matplotlib; print(sys.executable); print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```

> ⚠️ **一定要在 `ocwm` 环境里跑**：非交互 shell 里 `conda activate` 可能不生效，导致落到 base（缺依赖）。
> 保险写法：`conda run -n ocwm python ...`，或每次先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate ocwm`。

---

## 2. 下载数据（边下边解、删 zip 省盘）

```bash
export CR=/root/autodl-tmp/clevrer
mkdir -p $CR && cd $CR

# 2.1 标注 / 问题（小）
wget http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip
wget http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip
wget http://data.csail.mit.edu/clevrer/questions/train.json
wget http://data.csail.mit.edu/clevrer/questions/validation.json
mkdir -p train validation questions
unzip -q annotation_train.zip -d train && rm annotation_train.zip
unzip -q annotation_validation.zip -d validation && rm annotation_validation.zip
mv train.json validation.json questions/

# 2.2 de-render 物体 mask（397MB → 解压 2.9G）
wget http://data.csail.mit.edu/clevrer/derender_proposals.zip
mkdir -p derender && unzip -q derender_proposals.zip -d derender && rm derender_proposals.zip

# 2.3 视频（较大，逐个下→解压→删 zip）
wget http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
mkdir -p video_train && unzip -q video_train.zip -d video_train && rm video_train.zip

wget http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
mkdir -p video_validation && unzip -q video_validation.zip -d video_validation && rm video_validation.zip

# 2.4 检查
df -h /root/autodl-tmp
find $CR -name "annotation_*.json" | wc -l   # 期望 15000
find $CR -name "video_*.mp4" | wc -l         # 期望 15000
find $CR/derender -name "proposal_*.json" | wc -l  # 期望 20000
```

---

## 3. DINOv2 权重（~85MB）

```bash
mkdir -p /root/autodl-tmp/models/dinov2_small
cd /root/autodl-tmp/models/dinov2_small
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import hf_hub_download
print("saved:", hf_hub_download("timm/vit_small_patch14_dinov2.lvd142m", "model.safetensors"))
PY
# 或：wget https://hf-mirror.com/timm/vit_small_patch14_dinov2.lvd142m/resolve/main/model.safetensors
```

---

## 4. 环境变量（每次新 shell）

```bash
export DINOV2_CKPT=/root/autodl-tmp/models/dinov2_small/model.safetensors
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLEVRER_ROOT=/root/autodl-tmp/clevrer
```

---

## 5. 一键跑通（推荐）

```bash
cd /root/autodl-tmp/oc-causal-wm
bash server_setup.sh     # 检查依赖/权重/数据
bash run_routeB.sh       # 预计算 train+val 特征 → 训练（按 video id 切分）
```

`run_routeB.sh` 会用 4090 友好的默认值，可用环境变量覆盖：

| 变量 | 默认 | 说明 |
|---|---|---|
| `NTRAIN` | 80000 | 训练窗口数 |
| `NVAL` | 10000 | 验证窗口数 |
| `PRE_BATCH` | 64 | 预计算 DINOv2 的 batch（4090 可到 128） |
| `TRAIN_BATCH` | 384 | 训练 batch（4090 24G 可到 512） |
| `EPOCHS` / `SEEDS` | 40 / 3 | |

示例（拉满 4090）：
```bash
PRE_BATCH=128 TRAIN_BATCH=512 NTRAIN=120000 bash run_routeB.sh
```

---

## 6. 单步命令（自行控制）

```bash
cd /root/autodl-tmp/oc-causal-wm

# 6.1 训练集特征（10000 视频）
python precompute_clevrer_feats.py --root $CLEVRER_ROOT --split train \
  --n 80000 --obs 4 --pred 8 --M 6 --stride 8 --crop 64 --batch 128 \
  --device cuda --out out/clevrer_feats_train.npz

# 6.2 验证集特征（5000 视频）
python precompute_clevrer_feats.py --root $CLEVRER_ROOT --split validation \
  --n 10000 --obs 4 --pred 8 --M 6 --stride 8 --crop 64 --batch 128 \
  --device cuda --out out/clevrer_feats_val.npz

# 6.3 训练（干净：train 训，dev/test 按 video id 切）
python run_clevrer_dino.py \
  --train_feats out/clevrer_feats_train.npz \
  --val_feats out/clevrer_feats_val.npz \
  --obs 4 --pred 8 --epochs 40 --batch 512 --d 128 --nlayers 4 --nhead 4 \
  --pos_w 5 --seeds 3 --device cuda --out out/clevrer_dino_clean.json
```

产物：`out/clevrer_dino_clean.json` + `out/recursion_curves_dinov2.png`。

---

## 7. 4090 预期耗时 / 显存

| 步骤 | 耗时（估） | 显存 |
|---|---|---|
| 训练集预计算（10000 视频, n=80000, batch 128） | **主要看 CPU**，约 1–1.5 h | ~4–6G |
| 验证集预计算（5000 视频） | ~15–25 min | ~4–6G |
| 训练（3 config × 3 seed, batch 512） | ~8–15 min | ~3–5G |

> 4090 24G 对**当前路线 B 完全过剩**；真正的瓶颈是**视频解码（CPU）**。
> 因此选实例时优先 **多核 CPU + 大内存**；GPU 甚至 3090 就够。

---

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| `pycocotools` 装不上 | 用 wheels：`pip install pycocotools`（Linux 有 wheel） |
| DINOv2 卡下载 | `export HF_ENDPOINT=https://hf-mirror.com`；或手动 wget 后设 `DINOV2_CKPT` |
| 训练一开始很慢 | 确认用了本仓库 `run_clevrer_dino.py`（已一次性载入内存，避免 npz 反复解压） |
| 显存不足 | 降 `PRE_BATCH`/`TRAIN_BATCH`（4090 一般不会） |
| `cv2` 读视频失败 | 用 `opencv-python-headless`；确认 mp4 完整解压 |
| 磁盘不足 | 按 §2 边下边解删 zip；50G 一般够；不够再租更大数据盘 |
| 想用 GPU 更省时 | 无解——解码在 CPU；可增大 `--stride` 或减小 `--n` 来少抽窗口 |
