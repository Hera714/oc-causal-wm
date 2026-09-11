# AutoDL 上手命令清单（CausalSpatial A 阶段：复现弱基线）

> 目标：在 AutoDL 单卡 3090/4090（32-48GB）上，复现 CausalSpatial 论文的 MLLM baseline
> （纯文本 CoT 在 Collision 上接近随机 vs 人类 78-86% 的 gap）。
> 只需 **collision（+可选 physics）** 子集，约 1~2GB，跑 MLLM 评测。
>
> 日期 2026-09-04 ｜ 配套 `技术框架记录.md §9.4`

---

## 0. 实例配置建议

- **GPU**：单卡 3090 / 4090 皆可（32-48GB）。
- **镜像**：建议 PyTorch 2.1+ / CUDA 11.8+，Python 3.10。AutoDL 官方 pytorch 镜像即可。
- **磁盘**：数据 ~2GB + 模型权重（Qwen3-VL-8B 约 16GB fp16 / 8bit 约 8GB），预留 30-50GB 保险。

---

## 1. 克隆仓库（跳过 ATI submodule，避免下载大模型）

```bash
cd /root/autodl-tmp
git clone https://github.com/CausalSpatial/CausalSpatial.git
cd CausalSpatial
# 阶段 A 不需要 COW，跳过大体积 submodule（map-anything / ati 都不需要）
# 若想保留仓库结构完整可：git submodule update --init --recursive（但会拉 ATI，很大，建议跳过）
```

> 注意：git clone 后 `sub_module/` 目录会是空的（因为没有 update）。`eval.py` / `pipeline.py`
> 里 import 了 submodule 的内容，但**阶段 A 只用到 `eval.py` 的 eval 路径**。若 import 报错，
> 可临时注释掉 `pipeline.py` 顶部对 ATI/map-anything 的 import（或用 `--subset collision` 时
> 不会走到 COW 分支）。**阶段 A 我们只用官方 `eval.py` 的一个子集，无需 submodule。**

---

## 2. 安装依赖

仓库 requirements 只有 `transformers` / `uniception`，其余需要自己补。执行：

```bash
pip install -r requirements.txt
pip install torch torchvision   # 若镜像已带可跳过；torch>=2.1
pip install -U "datasets>=2.19" "huggingface_hub>=0.24"   # load_dataset + hf_hub_download
pip install "qwen_vl_utils"     # eval.py 里 from qwen_vl_utils import process_vision_info
pip install "accelerate"        # device_map="auto"
pip install "flash-attn"        # eval.py 默认 attn_implementation="flash_attention_2"；装不上可跳过(见第4步)
pip install openai anthropic google-genai tqdm pillow opencv-python   # API 与工具类
```

> `flash-attn` 编译较慢/易失败。若失败，**不影响**——`eval.py` 在 except 里会自动降级到
> `torch_dtype=float16, device_map="cuda:0"`。但 flash-attn 失败会导致 fp16 版本可能更吃显存。

---

## 3.（可选）先检查数据集 schema —— 强烈建议第一步做

在跑评测前先确认字段，尤其看有没有 `depth` / 3D 朝向框（决定后续阶段 B "几何锚定"可行性）。

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Mwxinnn/CausalSpatial', 'collision', split='train')
print('样本数:', len(ds))
print('columns:', ds.column_names)
print('id 示例:', ds[0]['id'])
print('question 示例:', ds[0]['question'][:120])
print('answer:', ds[0]['answer'], '| not_sure:', ds[0].get('not_sure'))
print('image type:', type(ds[0]['image']))
# 若有 depth 字段，打印其 shape / 范围
if 'depth' in ds.column_names:
    print('depth:', ds[0]['depth'])
"
```

**重点看**：`image` 字段编码方式（HF `Image` 或 bytes）、有没有 `depth`/`not_sure`/
其它几何字段。这决定阶段 B 能不能用"深度图/3D 朝向框锚定"。

```bash
输出：
Generating train split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 826/826 [00:10<00:00, 82.05 examples/s]
样本数: 82
columns: ['id', 'question', 'answer', 'image', 'not_sure']
id 示例: Collision_Level_1_0
question 示例: You can see a toy retro vehicle with a classic design on the floor. Is there a chance that it will hit something if it m
answer: B | not_sure: C
image type: <class 'dict'>---
```
## 4. 跑基线评测（纯文本 CoT）

单卡 8B（建议 8bit 以省显存；若 flash-attn 无则 4B 更稳妥）：

```bash
cd /root/autodl-tmp/CausalSpatial  
  python eval.py \
  --model_path ./models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master \
  --output_file ./output/qwen3vl-8b.jsonl \
  --subset collision physics \
  --batch_size 2

```

备选（显存紧张时）：
```bash
python eval.py --model_path Qwen/Qwen3-VL-4B-Instruct \
  --output_file ./output/qwen3vl-4b.jsonl --subset collision physics
```

> 说明：
> - `eval.py` 会在开头 `load_dataset` 拉数据（collision+physics 约 1.7GB），首次较慢，之后走缓存。
> - 由于 `eval.py` 用 `load_dataset(..., split="train")`，**默认只测 train split**。CausalSpatial 的
>   parquet 均以 train 命名，按官方用法即可。
> - 输出到 `output/*.jsonl`，打分逻辑在脚本末尾自动打印 `L1/L2 × collision/physics + Overall`。
> - 纯文本 CoT：`eval.py` 默认 prompt 就是"写进 JSON: {'Reasoning':..., 'Answer':...}，
>   且不带 COW 的模拟帧提示，正是我们要的基线。

---

## 5. 结果判读（对照论文表 2）

| 指标 | 目标（论文） |
|---|---|
| Qwen3-VL-8B 纯文本 CoT | Collision L1 ≈ 41-48%，L2 ≈ 22-36%；Overall L1 ≈ 47%，L2 ≈ 36% |
| random | ≈ 24-31% |
| Human | Collision L1 ≈ 86.8%，L2 ≈ 78.6%；Overall L1 ≈ 85% L2 ≈ 84% |

**重点确认**：你的 8B 分数是否落在论文附近、Collision 是否显著低于人类（体现 gap）。
分数接近 random 也可接受（说明该子集难），只要脚本能正确打分。

---

## 6. 常见报错速查

| 报错 | 原因 / 解决 |
|---|---|
| `ImportError: qwen_vl_utils` | `pip install qwen_vl_utils` |
| `ModuleNotFoundError: sub_module`（ATI/map-anything） | 阶段 A 用不到 COW，可忽略或用 `--subset collision` 单子集避免触发；必要时注释 `pipeline.py` 顶部 import |
| flash_attention_2 安装失败 | 跳过装 flash-attn；`eval.py` 会自动降级 fp16。若 OOM 换 4B 或加 `--load_in_8bit` |
| OOM（CUDA out of memory） | 换 Qwen3-VL-4B，或用 vLLM 部署 + `--node/--port` |
| `load_dataset` 网络超时 | 先 `huggingface-cli login`，或设 `HF_ENDPOINT=https://hf-mirror.com` |
| 数据下载慢 | `export HF_ENDPOINT=https://hf-mirror.com`（国内镜像，AutoDL 推荐） |
| 打分输出 `NaN (0/0)` | 说明该子集 0 样本或跑错了 config；检查 `--subset` 是否匹配 config 名 |

---

## 7. 下一步（阶段 B 衔接）

阶段 A 跑通后：
- 确认数据是否含 `depth` / 3D 朝向框（第 3 步）。
- 若有 → 写"几何锚定 prompt 构造器"（把 3D 空间信息拼进 prompt），对比 A vs B。
- 若无 → 需从 `asset` 或其它字段找几何信息，或用图像级增强（把深度 overlay 到图上当第二张图）。

> 完整 A→B→C 计划见 `技术框架记录.md §9.4`。
