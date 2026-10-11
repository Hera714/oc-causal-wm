# WM-PERCEPT-005 / A0 感知前端审计

本目录只新增 A0 感知审计，不修改 Route-B。模型输入严格为 MOVi-C 前 4 帧 RGB；GT segmentation、visibility 和 flow 仅用于计算审计指标。

## 第一轮链路

- SAM2.1 Base+：每个观测帧自动生成实例 mask；过滤背景、大区域和重复 mask。
- DINOv2-small：从预测 mask crop 提取冻结外观特征。
- Tracking：相邻帧以 mask IoU、中心距离、DINO cosine 做 Hungarian matching。
- RAFT-Large：估计 0→1、1→2、2→3 dense flow。
- 输出：IoU=0.5 的 mask precision/recall、匹配 mask IoU、centroid MAE、ID switch、dense/object flow EPE，以及逐场景图。

## 运行

必须等 Route-B 结束、GPU 空闲后执行：

```bash
conda activate ocwm
cd /root/autodl-tmp/oc-causal-wm
bash experiments/perception_a0/code/run_a0_perception_audit.sh
```

默认固定随机抽取 validation 8 个场景，只用于 A0 smoke/audit，不训练 predictor。结果位于：

```text
out/perception_a0/wm_percept_005/a0_validation_n8/
  config.json
  per_scene.jsonl
  metrics.json
  figures/*.png

out/perception_a0/wm_percept_005/logs/a0_validation_n8.log
```

若第一轮正常，再另建输出目录扩大到 validation 30；不要覆盖 n8 结果。

完整 validation 30 命令：

```bash
bash experiments/perception_a0/code/run_a0_validation30.sh
```
