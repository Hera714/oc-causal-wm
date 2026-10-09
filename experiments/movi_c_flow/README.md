# MOVi-C 光流实验

## 代码

`code/` 中包含：

- `prepare_movi_c_1200.py`：准备 1200/300/300 的 per-video NPZ。
- `train_movi_flow_suite.py`：统一训练器和控制实验入口。
- `batch_02_train_1200_all.sh`：full_1200 实验矩阵。
- `batch_03_train_120_remaining.sh`：pilot_120 剩余实验矩阵。
- `analyze_movi_motion_repr.py`：冻结 checkpoint 的运动指标、线性探针、光流反事实干预和对象归因分析。
- `train_movi_flow_mini.py`、`train_movi_flow_controls.py`：早期 mini/control 版本。

## 数据和输出

- pilot 数据：`/autodl-fs/data/movi_c_flow_v1`（120/30/30）。
- full 数据：`/autodl-fs/data/movi_c_flow_v2_1200`（1200/300/300）。
- 结果：`/root/autodl-tmp/oc-causal-wm/out/movi_c_flow/`。
- 权重：`/root/autodl-tmp/oc-causal-wm/models/movi_c_flow/`。
- 可解释性正式结果：`out/movi_c_flow/interpretability/wm_flow_004/`。

数据集、预计算大文件和模型权重不提交到 GitHub。

## 训练

```bash
conda activate ocwm
cd /root/autodl-tmp/oc-causal-wm
bash experiments/movi_c_flow/code/batch_03_train_120_remaining.sh
bash experiments/movi_c_flow/code/batch_02_train_1200_all.sh
```

两个脚本均支持 `OUT_ROOT`、`MODEL_ROOT`、`DATA_ROOT`、`EPOCHS`、`BATCH_SIZE`、`SEEDS` 环境变量覆盖默认值。

## WM-FLOW-004 可解释性分析

```bash
conda activate ocwm
cd /root/autodl-tmp/oc-causal-wm
export OMP_NUM_THREADS=8

python -u experiments/movi_c_flow/code/analyze_movi_motion_repr.py \
  --data-root /autodl-fs/data/movi_c_flow_v2_1200 \
  --checkpoint-root /root/autodl-tmp/oc-causal-wm/models/movi_c_flow/full_1200 \
  --output-dir out/movi_c_flow/interpretability/wm_flow_004 \
  --seeds 0 1 2 \
  --modes no_flow aligned_flow input_aux \
  --batch-size 128 \
  --device cuda \
  --case-count 6 \
  --flow-order xy
```

该分析不重新训练 predictor。它复用 full_1200 的 9 个 checkpoint，并输出：

- 位置、速度、加速度和分 horizon 误差；
- train-only ridge probe（validation 选正则，test 只评一次）；
- flow 幅值、旋转和时间片干预；
- 单物体 flow 消融与跨物体影响矩阵；
- 汇总图和轨迹案例。

