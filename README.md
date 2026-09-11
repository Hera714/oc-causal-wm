# 实验 1：动作条件注入 predictor 的"联合式 vs 分离式"验证

> ⚠️ 重要定位修正（v2，2026-09-04）：本实验只是**机制性 sanity check**（验证 pipeline
> 跑通 + A/B 开关正确），**不能**推广为"显式动作条件 > C-JEPA 式隐式干预"的结论。
> 原因与重定位见"局限与研究方向"章。

---

## v3 更新（2026-09-10）：思路 1（TDV 式 Δ 动力学）+ 逐帧编码 + 双编码器

在 Windows 侧用 **WSL2 Ubuntu 22.04** 的 `tdv` conda 环境（torch 2.10+cu128，RTX 5060 Ti 16G）运行。

**代码升级**
- `models.py` v2：观测改为**逐帧物体 token**（真时序 block）；动作条件注入所有观测帧**并注入未来查询 token**；新增
  `delta_mode`（`future_latent = base + Δ`）与 `delta_recursive`（`base_k=base_{k-1}+Δ_k`）；编码器可选
  `cnn` / `dinov2`（timm `vit_small_patch14_dinov2`，用 `DINOV2_CKPT` 指向本地权重）/ `dinov2_feat`（冻结 backbone+预计算特征）。
- `data.py`：无界缓存改为**有界 LRU**（`max_cache`），以适配 m=4/size=48/n=4000。
- `run.py`：collate 输出逐帧 `obs_posn`；`evaluate` 增加 `delta_mag` 与干预响应指标
  `align`（首步位移与动作方向余弦）、`align_flip`、`cons`（翻转后符号翻转率）。
- 新增 `run_delta.py`：思路 1 主实验脚本（absolute / delta_fixed / delta_rec × cnn/dinov2）。

**配置**：`m=4, pred=6, obs=2, d=128, 4 层, size=48, n=4000, epochs=60, batch=256, seeds=3`。

**结果（test，3 seed 均值；越低越好，align/cons 越高越好）**

| encoder | config | future_mse | target_mse | dMAG | align | alignF | cons | acc |
|---|---|---|---|---|---|---|---|---|
| cnn | absolute | 0.03457 | 0.02506 | 0 | 0.709 | 0.590 | 0.737 | 0.788 |
| cnn | **delta_fixed** | **0.00585** | **0.00936** | 0.0205 | 0.938 | 0.563 | 0.823 | 0.813 |
| cnn | delta_rec | 0.00599 | 0.01121 | 0.0081 | 0.913 | 0.078 | 0.473 | 0.807 |
| dinov2 | absolute | 0.00830 | 0.00997 | 0 | 0.945 | 0.891 | 0.996 | 0.826 |
| dinov2 | delta_fixed | 0.00850 | 0.01317 | 0.0258 | 0.928 | 0.779 | 0.992 | 0.851 |
| dinov2 | delta_rec | 0.00831 | 0.01030 | 0.0101 | 0.974 | 0.240 | 0.578 | 0.859 |

**结论**
1. 编码器弱（cnn）时，**Δ 建模收益巨大**：future_mse 0.0346→0.0059（-83%），target_mse 0.0251→0.0094（-63%）。
2. 编码器强（冻结 DINOv2）时，Δ 的 MSE 收益变小（特征已含运动信息），但分类 acc 仍提升；`delta_fixed` 略差于 `absolute` 的 target_mse。
3. **固定 base 一致优于递归累加**（`delta_fixed` > `delta_rec`，尤其 target_mse / alignF / cons），符合"递归累积误差"的预期。
4. ⚠️ **已知混淆**：toy 的动作方向由几何（朝/背伙伴）决定，模型可绕过动作输入靠几何猜，故翻转响应 `alignF/cons` 在部分设定下偏低；且绝对模式会周期性出现坏 seed（见 cnn/absolute seed2）。
   下一步建议：**在 `data.py` 里把动作方向随机化**（与几何解耦）后再验证干预指标，或直接迁移到真实动态数据。

## v4 更新（2026-09-10）：解耦动作方向 + `follow` 指标

v3 发现 toy 的动作方向由几何（朝/背伙伴）决定，模型可绕过动作输入靠几何猜。v4 修正：

- `data.py`：动作方向改为**随机打乱 4 个方向**后挑选（与几何解耦），模型必须真正读动作输入；
  碰撞窗口对齐预测步长 `win_end = obs + pred_steps`；为平衡标签把 `radius_scale 0.06→0.12`、`push_vel 3→4`
  （正样本率 ~0.57、真反事实率 ~0.40）。
- `run.py`：新增 **`follow`** 指标——对 4 个输入方向各前向一次，取 target 首步位移与输入方向的余弦再平均。
  真正用动作的模型 ≈ 1；只靠几何忽略动作的模型 ≈ 0（正负方向对消）。同时报告 `align`/`align_flip`/`cons`。

**结果（test，3 seed 均值；`*.json` = `out/delta_results_decoupled.json`）**

| encoder | config | future_mse | target_mse | dMAG | follow | align | alignF | cons | acc |
|---|---|---|---|---|---|---|---|---|---|
| cnn | absolute | 0.02328 | 0.02551 | 0 | 0.775 | 0.774 | 0.775 | 0.819 | 0.590 |
| cnn | **delta_fixed** | **0.01118** | **0.02182** | 0.0195 | **0.847** | 0.849 | 0.852 | 0.866 | 0.684 |
| cnn | delta_rec | 0.01512 | 0.03419 | 0.0091 | 0.774 | 0.757 | 0.787 | 0.860 | 0.719 |
| dinov2 | absolute | **0.01470** | **0.03610** | 0 | 0.795 | 0.803 | 0.810 | 0.944 | 0.781 |
| dinov2 | delta_fixed | 0.01665 | 0.04587 | 0.0243 | **0.973** | **0.974** | **0.975** | **1.000** | 0.845 |
| dinov2 | delta_rec | 0.02082 | 0.06708 | 0.0077 | 0.741 | 0.738 | 0.771 | 0.837 | 0.855 |

**结论（v4，解耦后更可信）**
1. 弱编码器（cnn）：`delta_fixed` 在**所有指标**上胜过 absolute（future_mse -52%、target_mse -14%、follow 0.78→0.85）。
2. 强编码器（冻结 DINOv2）：`delta_fixed` 的 MSE 不比 absolute 好（特征已含运动信息），但**干预遵循度大幅提升**
   （follow 0.80→0.97、cons 0.94→1.00），分类 acc 也更高（0.78→0.85）。即：**Δ 的价值在"因果/干预"而非纯拟合**。
3. `align ≈ align_flip ≈ follow`（dinov2/delta_fixed 三者都 ≈0.97）正是"真用动作、非几何捷径"的签名，解耦生效。
4. **固定 base 一致优于递归累加**（`delta_fixed` > `delta_rec`，尤其 target_mse/follow），符合递归累积误差预期。

## v5 更新（2026-09-10）：delta_recursive 逐步误差分析

新增逐预测步指标（`future/target/nontarget_mse_step`、`delta_mag_step`），出图 `out/recursion_curves_{cnn,dinov2}.png`。

**逐预测步 target MSE（test，3 seed 均值）**

| encoder | config | k0 | k1 | k2 | k3 | k4 | k5 |
|---|---|---|---|---|---|---|---|
| cnn | absolute | .0165 | .0292 | .0222 | .0249 | .0288 | .0314 |
| cnn | delta_fixed | .0132 | .0208 | .0138 | .0189 | .0273 | .0328 |
| cnn | delta_rec | .0187 | .0259 | .0135 | .0226 | .0278 | **.0444** |
| dinov2 | absolute | .0353 | .0283 | .0284 | .0362 | .0448 | .0436 |
| dinov2 | delta_fixed | **.0085** | .0250 | .0490 | .0587 | .0633 | .0708 |
| dinov2 | delta_rec | .0191 | .0370 | .0349 | .0705 | .0955 | **.1456** |

**逐预测步 |Δ_k|**

| encoder | config | k0..k5 |
|---|---|---|
| cnn | delta_fixed | .023 .021 .015 .019 .019 .018 |
| cnn | delta_rec | .0085 .0075 .0098 .0098 .0089 .0093 |
| dinov2 | delta_fixed | .024 .023 .023 .025 .025 .026 |
| dinov2 | delta_rec | .0083 .0067 .0075 .0076 .0080 .0084 |

**结论**
1. `delta_rec` **每个 horizon 都差于 `delta_fixed`**，误差随 k 快速发散（dinov2 k5 0.146 vs fixed 0.071，k3 后反超 absolute）。
2. `delta_rec` 学到的 **|Δ_k| 仅 `delta_fixed` 的 ~1/3 且几乎平坦**——模型**自我衰减 Δ** 以抑制 cumsum 漂移，正是文档预判的"Δ≈0 退化"现象。
3. Δ 的优势是**前置的**：k0 最强（dinov2 delta_fixed 0.0085，比 absolute 0.035 好 ~4 倍）；horizon 越长越被 absolute 追平——因为 toy 的干预是**一次性冲量**，固定 base 天然匹配，递归不适合。
4. 实践结论：**用 delta_fixed**；若要用递归，需加 **Δ 一致性/多样性约束**防衰减，并限制 horizon。

## v6 更新（2026-09-10）：路线 A —— CLEVRER 轨迹级 Δ 门禁

**目的**：把思路1（Δ 动力学）从 2D 玩具搬到**真实碰撞动力学**（CLEVRER 20k 视频的多物体接触/弹性碰撞），
但只吃**标注里的物体状态**（位置/速度/属性），不吃图像 → 隔离动力学核心，作为廉价门禁。

**新增文件**
- `clevrer_dataset.py`：读 CLEVRER annotation，输出与 `data.make_sample` 相同接口（`enc_feats (obs,M,18)`、
  `traj_yes`、`label_yes/no` 等）。状态向量 18 维 = pos2+vel2+color8+material2+shape3+view1，位置/速度按数据集均值方差标准化。
- `models.py`：`ObjectEncoder` 新增 `kind="state"`（纯 MLP，无 backbone），`Model` 加 `state_dim`。
- `run_clevrer_delta.py`：路线 A 主脚本（absolute / delta_fixed / delta_rec，逐 step 曲线 + 图）。

**数据（只需标注，不用视频）**
```
http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip        # 157MB
http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip  # 78MB
# 解压到 ~/datasets/clevrer/{train,validation}
```

**运行（WSL）**
```bash
~/anaconda3/envs/tdv/bin/python run_clevrer_delta.py \
  --root ~/datasets/clevrer --n 20000 --ne 500 --nt 1000 \
  --obs 4 --pred 8 --M 5 --stride 8 --epochs 40 --batch 256 --d 128 --seeds 3 --device cuda
```

> 已用伪造 annotation 冒烟通过（`out/clevrer_fake_smoke.json`）；真实数据到位后直接换 `--root` 即可。

**真实 CLEVRER 结果（test，3 seed 均值；10000 train / 5000 val，obs=4/pred=8/M=6）**

| config | future_mse | target_mse | dMAG | acc |
|---|---|---|---|---|
| absolute | 0.03648 | 0.04430 | 0 | 0.848 |
| **delta_fixed** | **0.00513** | **0.00541** | 0.0138 | **0.902** |
| delta_rec | 0.00544 | 0.00584 | 0.0052 | 0.888 |

**逐 step target MSE**：absolute 平（0.052→0.042）；delta_fixed 在 k2–k3 最优（~0.0015–0.002）后升至 k7 0.008；
delta_rec 同样 k2–k3 最优但**后段上升更快**（k7 0.0156）。
**逐 step |Δ_k|**：delta_fixed 0.024→0.016；delta_rec 仅 ~0.004→0.007（约 1/4，自我衰减）。

**结论**：真实碰撞动力学上复现 toy 结论——Δ(fixed) 大幅胜出（future_mse ×7、target_mse ×8、acc +5.4pt）；
`delta_rec` 略差且 Δ 被压缩（累积误差 + 衰减），与 toy 一致。→ **Δ 门禁在真实数据通过**；下一步接图像（路线 B）。

## v7 更新（2026-09-10）：路线 B —— 图像级 Δ（CLEVRER 视频 → 物体 crop → DINOv2）

**目的**：在真实视频上验证"像素→物体 token→Δ"，接口与 toy/路线 A 一致，只把编码器换成 DINOv2 特征。

**新增文件**
- `clevrer_video.py`：用 de-render 的 COCO mask 定位物体、抠 crop（按 color/material/shape 匹配到 annotation object_id），
  输出 `crops (M,obs,3,64,64)` + 位置/标签。
- `precompute_clevrer_feats.py`：**每个视频只解码一次**，抽窗口 crop → 冻结 DINOv2 → 存 `(obs,M,384)` 特征到 `.npz`。
- `run_clevrer_dino.py`：读特征 npz 训练 absolute / delta_fixed / delta_rec（走 `encoder="dinov2_feat"`）。
- 依赖：`pycocotools`（已装到 tdv 环境）。

**数据（需视频；mask/标注已在）**
```
http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip   # 6.2GB
# 解压到 ~/datasets/clevrer/video_validation
```
> 先用 val 5000 视频切 train/dev/test 即可；训练集视频（12GB）需要时再下。

**运行（WSL）**
```bash
# 1) 预计算 DINOv2 特征（一次，CPU 解码是瓶颈）
~/anaconda3/envs/tdv/bin/python precompute_clevrer_feats.py \
  --root ~/datasets/clevrer --split validation --n 20000 \
  --obs 4 --pred 8 --M 6 --stride 8 --crop 64 --batch 32 --device cuda \
  --out out/clevrer_feats.npz
# 2) 训练/对照
~/anaconda3/envs/tdv/bin/python run_clevrer_dino.py \
  --feats out/clevrer_feats.npz --epochs 40 --batch 256 --d 128 --seeds 3 --device cuda
```
> 已用**合成视频 + 真实 proposal/annotation** 冒烟通过（`out/fakeb_feats.npz`、`out/fakeb_dino.json`）。

**真实 CLEVRER 结果（路线 B，val 5000 视频，20000 窗口，3 seed 均值）**

| config | future_mse | target_mse | dMAG | acc |
|---|---|---|---|---|
| absolute | 0.33136 | 0.75045 | 0 | 0.841 |
| **delta_fixed** | **0.00667** | **0.00970** | 0.0075 | 0.834 |
| delta_rec | 0.00730 | 0.01023 | 0.0027 | 0.830 |

→ **delta_fixed 的 future_mse 比 absolute 好 ~50×、target_mse 好 ~77×**。图像级世界里 absolute（直接回归未来 latent）几乎学不动，
而 Δ 极其有效。`delta_rec` 略差 + |Δ| 仅固定版的 ~1/3（同样的衰减现象）。
逐 step：delta 的 target MSE 从 k0 0.004 升到 k7 0.024（长时程/碰撞变难），absolute 全程 ~0.7。
产物：`out/clevrer_feats.npz`、`out/clevrer_dino.json`、`out/recursion_curves_dinov2.png`。

---

运行：
```bash
# WSL
export DINOV2_CKPT=$HOME/models/dinov2_small/model.safetensors HF_HUB_OFFLINE=1
~/anaconda3/envs/tdv/bin/python run_delta.py --encoders cnn,dinov2 --device cuda
```

---

验证论文框架里的一个设计取舍：

> **把"假设动作"（language action）注入到 object-centric JEPA predictor 里
> （联合式，model B），是否能让 predictor 对未来物体状态的预测（尤其是被
> 干预的 target 物体）更准？** 对比：动作只在答案头融合、predictor 对动作
> 无感知（分离式，model A）。

## 观测结果（3 seed 平均，合成物理数据）

| 指标（越低越好） | SEPARATE (A) | JOINT (B) | 相对改善 |
|---|---|---|---|
| 全物体未来位置 MSE | 0.0548 | 0.0529 | **-3.5%** |
| 被干预物体 (target) 未来位置 MSE | 0.0421 | 0.0330 | **-21.6%** |

在该简化世界里，动作注入 predictor 改善被干预物体未来轨迹预测；分类 acc 被"总是 no"
捷径饱和（0.69），不作主指标。

## 局限与研究方向（must read）

### 为什么不能推广成"显式 > 隐式干预"
1. **目标错位**：C-JEPA 的目标是学"物体间相互作用**关系**"，把形状/接触点/质量等
   **语言无法穷尽描述**的因素作为**隐变量**。本实验为了跑通把场景简化成"圆形、无形状差、
   动作仅 4 方向"——**恰好删掉了语言之外的所有因素**，所以"显式语言条件"才能完美刻画
   干预。两者探讨的**不是同一目标**。
2. 在真实 3D/机器人场景，"推某物"的结果由**接触点、形状受力、质量、摩擦**等决定，
   **无法用自然语言穷尽描述**。因此"语言可描述 ≠ 干预有效"。

### 真正的研究方向：CausalSpatial / COW 那条线
- **CausalSpatial**（bencSmark，arXiv:2601.13304）：物体级、真实 3D（Blender）因果空间推理，
  4 类任务 Collision / Compatibility / Occlusion / Trajectory。核心发现：**MLLM 纯文本 CoT
  推理会空间漂移幻觉**（human 84% vs GPT-5 54%）。
- **COW**：把"假设动作"外部化为**轨迹条件化的仿真视频**，用显式视觉证据锚定 MLLM
  （3 帧就有 2.4%/2.2% 提升）。

**我们可定位的差异化**：不用**外部视频生成**（COW 依赖 ATI 扩散、贵且物体级一致性仍开放），
而是用**内部、可学习、物体级因果世界模型**在 latent 域做干预 + VLM 融合——更紧凑、可学习、
对象级一致，且抑制文本 CoT 空间幻觉。

## 怎么跑（已在本机 GPU 2GB 验证通过）

```bash
cd exp
python run.py --n 320 --ne 110 --epochs 40 --batch 48 --seeds 3 --lr 2e-4 --d 96 --pos_w 5 --device auto
```

`--device auto` 会自动用 cuda。远程强 GPU 可加大：
```bash
python run.py --n 5000 --ne 600 --epochs 80 --batch 256 --seeds 5 --d 256 --pos_w 5 --device cuda
```

## 文件说明
- `data.py`：合成物理视频生成器。确定性因果场景：m 个物体 + 每次随机布局，动作
  "push the <color> object to the <direction>" 给 target 一个定向速度增量；输出真值框、
  颜色、动作、未来位置、碰撞标签（保证"动作确实改变结果"=真反事实）。
- `models.py`：物体编码器（CNN on crops + 位置嵌入）→ block-causal Transformer
  predictor（物体间 self-attention，跨帧因果 mask）→ 位置头 + 答案头。
  `joint=True` 时动作经 bind 注入 predictor；`joint=False` 时不注入（A/B 开关）。
- `run.py`：训练 + 评估（比较未来位置 MSE / 被干预物体 MSE / 分类）。

## 局限（如实交代）
- toy：物体少(3)、运动简单、合成渲染，主要用于**验证机制**，不代表真实数据集性能。
- 后续应转向真实 3D（CausalSpatial/Blender）、真实物体表征(DINOv2/V-JEPA2)、VLM 融合后
  的 causal spatial 问答，并与 COW 的外部视频基线对比。
