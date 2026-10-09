# WM-FLOW-004：光流 predictor 可解释性分析

本目录保存 `WM-FLOW-003` 冻结 checkpoint 的分析结果，不包含数据或模型权重。

## 设置

- 数据：MOVi-C v2，train/validation/test = 1200/300/300。
- 模型：`no_flow`、`aligned_flow`、`input_aux`，每类 seeds 0/1/2。
- checkpoint：`/root/autodl-tmp/oc-causal-wm/models/movi_c_flow/full_1200/`。
- predictor 完全冻结；线性探针只在 train 拟合、validation 选 ridge 系数、test 只评一次。

## 主要文件

- `metrics.json`：完整汇总。
- `motion_metrics.json`：位置、速度、加速度、分 horizon 和运动强度分组。
- `probe_metrics.json`：输入投影、Transformer L1/L2 的冻结线性探针。
- `flow_intervention_metrics.json`：幅值、旋转和时间片反事实干预。
- `object_flow_ablation.json`：输入对象到输出对象的影响矩阵。
- `figures/`：汇总图和 6 个轨迹案例。
- `run.log`：正式运行日志。

## 结果边界

结果支持：正确对齐的 object flow 被 predictor 以对象局部、方向和尺度敏感的方式使用，并改善未来位移、速度和加速度误差。

结果不支持：通用物理理解、动作理解或因果问答能力。当前输入仍依赖 GT mask、identity、visibility 和 GT flow；部分高阶动态在深层表示中的线性可读性没有超过 no-flow。

