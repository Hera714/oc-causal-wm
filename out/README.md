# `out/` 实验输出索引

输出按数据集与实验 ID 归档；`metrics.json`、日志和图表保留在服务器/代码仓库，训练权重另存于 `models/`。

## CLEVRER

- `clevrer/route_a_state/v6_routeA_trajectory/`：状态输入 Route A。
- `clevrer/route_b_image/v7_routeB_image/`：图像输入 Route B，含 DINOv2 clean 重训结果。
- `clevrer/toy_action/v3_toy_delta/`：toy delta 机制实验。
- `clevrer/toy_action/v4_toy_decoupled/`：动作方向解耦实验。
- `clevrer/toy_action/v5_recursive_analysis/`：递归误差分析。
- `clevrer/qa_data/`：CLEVRER QA 产物。
- `clevrer/smoke/`：冒烟检查产物。
- CLEVRER 特征数据不放在 `out/`；服务器路径见根目录 `README.md`。

## MOVi-C 光流

- `movi_c_flow/mini/`：最小 pilot 及归一化光流版本。
- `movi_c_flow/pilot_120/movi_c_flow_120_v2_remaining/`：120/30/30，完成 21 个训练指标、9 个扰动评估指标和启发式基线。
- `movi_c_flow/full_1200/movi_c_flow_1200_v2/`：1200/300/300，完成 27 个训练指标、9 个扰动评估指标和启发式基线。
- `movi_c_flow/controls/`：早期 zero/shuffled flow 控制实验。

MOVi-C 数据路径、权重路径及其用途见根目录 `README.md` 与 `docs/EXPERIMENT_INDEX.md`。
