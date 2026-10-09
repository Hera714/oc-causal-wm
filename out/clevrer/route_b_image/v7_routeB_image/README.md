# v7 · 路线 B（CLEVRER 图像级 Δ）—— 干净重训记录

- 时间：2026-09-12（AutoDL 4090 24G）
- 数据：CLEVRER（train 训 / validation 评）
- 输入：de-render 的 COCO mask 取 bbox 中心 → 抠 64×64 物体 crop → **冻结 DINOv2-small**（384 维）
- 配置：`obs=4, pred=8, M=6, d=128, nlayers=4, nhead=4, pos_w=5, epochs=40, batch=384, seeds=3`
- 损失：`CrossEntropy(答案) + 5.0 × MSE(未来位置)`

## 数据集规模（读自 clevrer_feats_{train,val}.npz）

| 集 | 窗口数 | 视频数 | 切分方式 |
|---|---|---|---|
| train | 79,998 | 10,000 | CLEVRER train |
| val（总） | 10,000 | 4,426 | CLEVRER validation |
| ├ dev | 4,993 | 2,213 | 按 video id 切 |
| └ test | 5,007 | 2,213 | 按 video id 切 |

说明：train/val 位置归一化**统一**（统计量取自全部标注）；dev/test **按 video id 切分**，无同视频窗口泄漏。

## 结果（test，3 seed 均值）

| config | future_mse | target_mse | dMAG | acc |
|---|---|---|---|---|
| absolute | 0.30955 | 0.68941 | 0 | 0.871 |
| **delta_fixed** | **0.00559** | **0.00871** | 0.0127 | 0.859 |
| delta_rec | 0.00626 | 0.00944 | 0.0042 | 0.857 |

→ Δ(fixed) 相对 absolute：**future ×55、target ×79**；`delta_rec` 略差且 |Δ| 被压到 ~1/3（累积误差+自我衰减）。

## predictor 到底在预测什么（澄清）

- 预测**所有 M=6 个物体**的未来 8 步位置（`pred_pos` 形状 `(B, 8, 6, 2)`）；`target_mse` 只是额外挑出"最后观测帧速度最大的物体(mover)"单独统计。
- **mask 用于定位抠框**；跨帧物体身份按 `(color, material, shape)` 匹配到 annotation 的 `object_id`（**无跟踪网络**）。
- 未来位置**监督来自 annotation `motion_trajectory`**（GT），图像只提供外观（DINOv2 特征）+ 位置嵌入。
- 本质：**object-centric、无动作、图像条件的多物体多步未来轨迹预测器**（Δ 动力学建在它之上）。

## 本目录文件

- `clevrer_dino_clean.json` —— **干净重训结果**（含逐 step 数组）
- `recursion_curves_dinov2.png` —— 由该 JSON 重绘（服务器 CPU 无 matplotlib 时的补救）
- `clevrer_dino.json` / `clevrer_dino.log` —— 初版（用 validation 训练 + window 切分，**已弃用**，仅存对照）
- `clevrer_precompute.log` —— 特征预计算日志
