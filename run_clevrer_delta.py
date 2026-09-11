"""Route A: delta-dynamics gate on CLEVRER object trajectories (no images).

Trains the same object-centric predictor as the synthetic toy, but the object
encoder is replaced by an MLP over annotated states (position/velocity/attributes).
Compares absolute / delta_fixed / delta_rec future-position prediction.

Usage (WSL):
  python run_clevrer_delta.py --root ~/datasets/clevrer --n 20000 --seeds 3
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from clevrer_dataset import ClevrerDataset, STATE_DIM
from models import Model
from run import train_one, evaluate
from run_delta import CONFIGS, aggregate, print_step_report


def build_datasets(args):
    root = os.path.expanduser(args.root)
    all_files = sorted(glob.glob(os.path.join(root, "**", "annotation_*.json"),
                                 recursive=True))
    if not all_files:
        raise FileNotFoundError(f"no annotation_*.json under {root}")
    # 自动按路径区分 train / validation（兼容解压后的各种目录层级）
    val_files = [f for f in all_files if "validation" in f.lower()]
    train_files = [f for f in all_files if f not in set(val_files)]
    if not val_files:                      # 没有验证集就切训练文件
        val_files = train_files[::5]
        train_files = train_files[1::5]
    dev_files = val_files[0::2]
    test_files = val_files[1::2]

    def sub(n, seed, files):
        return ClevrerDataset(root, obs=args.obs, pred=args.pred, M=args.M,
                              stride=args.stride, max_windows=n, seed=seed,
                              files=files, cache_files=args.cache_files)

    train_ds = sub(args.n, 0, train_files)
    dev_ds = sub(args.ne, 1, dev_files)
    test_ds = sub(args.nt, 2, test_files)
    if args.materialize:
        for name, ds in [("train", train_ds), ("dev", dev_ds), ("test", test_ds)]:
            t0 = time.time()
            ds.materialize()
            print(f"  materialized {name}: {len(ds)} windows in {time.time()-t0:.1f}s",
                  flush=True)
    return train_ds, dev_ds, test_ds


def run_one(cfg, seed, args, device, train_ds, dev_ds, test_ds):
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)
    model = Model(d_model=args.d, m=args.M, obs=args.obs, pred=args.pred,
                  nhead=args.nhead, nlayers=args.nlayers, joint=True,
                  encoder="state", state_dim=STATE_DIM,
                  delta_mode=cfg["delta_mode"],
                  delta_recursive=cfg["delta_recursive"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    train_one(model, train_ds, dev_ds, device, args.epochs, args.lr, args.batch,
              pos_w=args.pos_w, pred=args.pred, obs=args.obs, consistency=False)
    dt = time.time() - t0
    dev = evaluate(model, dev_ds, device, args.batch, pred=args.pred,
                   obs=args.obs, consistency=False)
    test = evaluate(model, test_ds, device, args.batch, pred=args.pred,
                    obs=args.obs, consistency=False)
    row = {"encoder": "state", "config": cfg["name"], "seed": seed,
           "n_params": n_params, "train_s": round(dt, 1)}
    for k in dev:
        row["dev_" + k] = dev[k]
        row["test_" + k] = test[k]
    print(f"[state/{cfg['name']} seed{seed}] "
          f"test future_mse {test['future_mse']:.5f} target_mse {test['target_mse']:.5f} "
          f"dmag {test['delta_mag']:.5f} acc {test['acc']:.3f} ({dt:.1f}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/datasets/clevrer")
    ap.add_argument("--n", type=int, default=20000, help="train windows")
    ap.add_argument("--ne", type=int, default=500, help="dev windows")
    ap.add_argument("--nt", type=int, default=1000, help="test windows")
    ap.add_argument("--obs", type=int, default=4)
    ap.add_argument("--pred", type=int, default=8)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--materialize", type=int, default=1)
    ap.add_argument("--cache_files", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--pos_w", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="out/clevrer_delta.json")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device={device} root={args.root} obs={args.obs} pred={args.pred} M={args.M} "
          f"n={args.n} d={args.d} seeds={args.seeds}")
    train_ds, dev_ds, test_ds = build_datasets(args)
    print(f"train windows={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    rows = []
    t_all = time.time()
    for cfg in CONFIGS:
        for seed in range(args.seeds):
            rows.append(run_one(cfg, seed, args, device, train_ds, dev_ds, test_ds))
    agg = aggregate(rows)

    print("\n=== SUMMARY (test, mean over seeds) ===")
    print(f"{'encoder':8s} {'config':12s} {'future_mse':>11s} {'target_mse':>11s} "
          f"{'dMAG':>8s} {'acc':>6s}")
    for a in agg:
        print(f"{a['encoder']:8s} {a['config']:12s} "
              f"{a['future_mse_mean']:11.5f} {a['target_mse_mean']:11.5f} "
              f"{a['delta_mag_mean']:8.4f} {a['acc_mean']:6.3f}")
    print_step_report(agg, args.pred)

    payload = {"args": vars(args), "rows": rows, "summary": agg,
               "total_s": round(time.time() - t_all, 1)}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
