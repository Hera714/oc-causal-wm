# 实验 1：动作条件注入 predictor 的"联合式 vs 分离式"验证

> **分工约定**：本 README 记录**细节/代码级更新**（模块实现、命令、逐 step、冒烟等）；
> **标准格式的实验记录**（可跨版本比对）见 `../技术框架记录.md` §11。

> ⚠️ 重要定位修正（v2，2026-09-04）：该 toy 实验只是**机制性 sanity check**（验证 pipeline
> 跑通 + A/B 开关正确），**不能**推广为"显式动作条件 > C-JEPA 式隐式干预"的结论。
> 原因与重定位见 `../技术框架记录.md` §6。

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

**目的**：在真实视频上验证“像素→物体 token→Δ”，接口与 toy/路线 A 一致，只把编码器换成 DINOv2 特征。

**本次记录（标准格式见 `../技术框架记录.md` §11）**
- 数据：CLEVRER（train 训 / validation 评）
- 输入：de-render 的 COCO mask 取 bbox 中心 → 抠 64×64 物体 crop → **冻结 DINOv2-small**（384 维）
- 配置：`obs=4, pred=8, M=6, d=128, nlayers=4, nhead=4, pos_w=5, epochs=40, batch=384, seeds=3`
- 损失：`CrossEntropy(答案) + 5.0 × MSE(未来位置)`

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

## 数据集规模（读自 clevrer_feats_{train,val}.npz）

| 集 | 窗口数 | 视频数 | 切分方式 |
|---|---|---|---|
| train | 79,998 | 10,000 | CLEVRER train |
| val（总） | 10,000 | 4,426 | CLEVRER validation |
| ├ dev | 4,993 | 2,213 | 按 video id 切 |
| └ test | 5,007 | 2,213 | 按 video id 切 |

说明：train/val 位置归一化**统一**（统计量取自全部标注）；dev/test **按 video id 切分**，无同视频窗口泄漏。


**真实 CLEVRER 结果（路线 B，干净重训：train 10000 视频训练、dev/test 按 video id 切分，3 seed 均值）**

| config | future_mse | target_mse | dMAG | acc |
|---|---|---|---|---|
| absolute | 0.30955 | 0.68941 | 0 | 0.871 |
| **delta_fixed** | **0.00559** | **0.00871** | 0.0127 | 0.859 |
| delta_rec | 0.00626 | 0.00944 | 0.0042 | 0.857 |

→ **delta_fixed 的 future_mse 比 absolute 好 ~55×、target_mse 好 ~79×**。图像级世界里 absolute（直接回归未来 latent）几乎学不动，
而 Δ 极其有效。`delta_rec` 略差 + |Δ| 仅固定版的 ~1/3（同样的衰减现象）。
（此结果取代初版"用 validation 训练 + window 切分"的泄漏结果 `out/clevrer_dino.json`。）
产物：`out/clevrer_feats_train.npz` / `clevrer_feats_val.npz`、`out/clevrer_dino_clean.json`、`out/recursion_curves_dinov2.png`。

## predictor 在预测什么

- 预测**所有 M=6 个物体**的未来 8 步位置（`pred_pos` 形状 `(B, 8, 6, 2)`）；`target_mse` 只是额外挑出"最后观测帧速度最大的物体(mover)"单独统计。
- **mask 用于定位抠框**；跨帧物体身份按 `(color, material, shape)` 匹配到 annotation 的 `object_id`（**无跟踪网络**）。
- 未来位置**监督来自 annotation `motion_trajectory`**（GT），图像只提供外观（DINOv2 特征）+ 位置嵌入。
- 本质：**object-centric、无动作、图像条件的多物体多步未来轨迹预测器**（Δ 动力学建在它之上）。

---

> 早期 toy（合成 2D）实验的定位与局限（为何不能与 C-JEPA 直接对比）：见 `../技术框架记录.md` §6。
