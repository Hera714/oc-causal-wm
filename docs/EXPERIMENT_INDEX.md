# 实验与服务器资源索引

这是代码整理索引，不作实验结果分析。

| 实验 ID | 代码 | 结果 | 数据 | 模型/权重 |
|---|---|---|---|---|
| CLEVRER Route A | 根目录历史 CLEVRER 脚本，主结果为 `run_clevrer_delta.py` | `out/clevrer/route_a_state/v6_routeA_trajectory/` | `/root/autodl-tmp/clevrer` | 运行时生成，未纳入 GitHub |
| CLEVRER Route B | 根目录 CLEVRER 图像/DINOv2 脚本，入口 `run_routeB.sh` | `out/clevrer/route_b_image/v7_routeB_image/` | `/root/autodl-tmp/clevrer` | DINOv2：`/root/autodl-tmp/models/dinov2_small/model.safetensors` |
| CLEVRER 特征 | `precompute_clevrer_feats.py` | `/autodl-fs/data/oc-causal-wm/clevrer_features` | `/root/autodl-tmp/clevrer` | 外部数据，不提交 |
| MOVi-C mini | `experiments/movi_c_flow/code/train_movi_flow_mini.py` | `out/movi_c_flow/mini/` | 早期 pilot 数据 | `models/movi_c_flow/mini/` |
| MOVi-C pilot_120 | `batch_03_train_120_remaining.sh` | `out/movi_c_flow/pilot_120/` | `/autodl-fs/data/movi_c_flow_v1` | `models/movi_c_flow/pilot_120/` |
| MOVi-C full_1200 | `batch_02_train_1200_all.sh` | `out/movi_c_flow/full_1200/` | `/autodl-fs/data/movi_c_flow_v2_1200` | `models/movi_c_flow/full_1200/` |
| MOVi-C controls | `train_movi_flow_controls.py` | `out/movi_c_flow/controls/` | 早期 pilot 数据 | `models/movi_c_flow/controls/` |

GitHub 仅保存代码、轻量日志/指标和文档。`*.npz`、`*.pt`、`*.pth`、`*.ckpt`、原始视频/TFRecord 均不提交。
