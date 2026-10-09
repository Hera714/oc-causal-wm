# MOVi-C 光流实验

## 代码

`code/` 中包含：

- `prepare_movi_c_1200.py`：准备 1200/300/300 的 per-video NPZ。
- `train_movi_flow_suite.py`：统一训练器和控制实验入口。
- `batch_02_train_1200_all.sh`：full_1200 实验矩阵。
- `batch_03_train_120_remaining.sh`：pilot_120 剩余实验矩阵。
- `train_movi_flow_mini.py`、`train_movi_flow_controls.py`：早期 mini/control 版本。

## 数据和输出

- pilot 数据：`/autodl-fs/data/movi_c_flow_v1`（120/30/30）。
- full 数据：`/autodl-fs/data/movi_c_flow_v2_1200`（1200/300/300）。
- 结果：`/root/autodl-tmp/oc-causal-wm/out/movi_c_flow/`。
- 权重：`/root/autodl-tmp/oc-causal-wm/models/movi_c_flow/`。

## 运行

```bash
conda activate ocwm
cd /root/autodl-tmp/oc-causal-wm
bash experiments/movi_c_flow/code/batch_03_train_120_remaining.sh
bash experiments/movi_c_flow/code/batch_02_train_1200_all.sh
```

两个脚本均支持 `OUT_ROOT`、`MODEL_ROOT`、`DATA_ROOT`、`EPOCHS`、`BATCH_SIZE`、`SEEDS` 环境变量覆盖默认值。
