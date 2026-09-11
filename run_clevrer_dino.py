"""Route B training: delta dynamics on precomputed CLEVRER DINOv2 object features.

Reads an .npz produced by precompute_clevrer_feats.py and trains the same
object-centric predictor (encoder='dinov2_feat'), comparing absolute /
delta_fixed / delta_rec. Uses run.py's train/eval machinery (consistency off).
"""

import argparse
import json
import time

import numpy as np
import torch

from models import Model
from run import train_one, evaluate
from run_delta import CONFIGS, aggregate, print_step_report


class FeatDataset:
    def __init__(self, data, indices):
        self.d = data
        self.idx = np.asarray(indices)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        M = self.d["feats"].shape[2]
        tgt = int(self.d["target"][j])
        return {
            "m": M,
            "enc_feats": self.d["feats"][j].astype(np.float32),
            "color_idx": self.d["color"][j],
            "direction_idx": 0,
            "target_idx": tgt,
            "partner_idx": (tgt + 1) % M,
            "label_yes": int(self.d["label_yes"][j]),
            "label_no": int(self.d["label_no"][j]),
            "label": int(self.d["label_yes"][j]),
            "traj_yes": self.d["traj"][j].astype(np.float32),
            "traj_no": self.d["traj"][j].astype(np.float32),
            "box": [1.0, 1.0],
            "action_text": "",
            "question_text": "",
        }


def run_one(cfg, seed, args, device, train_ds, dev_ds, test_ds, M):
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)
    model = Model(d_model=args.d, m=M, obs=args.obs, pred=args.pred,
                  nhead=args.nhead, nlayers=args.nlayers, joint=True,
                  encoder="dinov2_feat", delta_mode=cfg["delta_mode"],
                  delta_recursive=cfg["delta_recursive"]).to(device)
    t0 = time.time()
    train_one(model, train_ds, dev_ds, device, args.epochs, args.lr, args.batch,
              pos_w=args.pos_w, pred=args.pred, obs=args.obs, consistency=False)
    dt = time.time() - t0
    test = evaluate(model, test_ds, device, args.batch, pred=args.pred,
                    obs=args.obs, consistency=False)
    row = {"encoder": "dinov2", "config": cfg["name"], "seed": seed,
           "train_s": round(dt, 1)}
    for k in test:
        row["test_" + k] = test[k]
    print(f"[dinov2/{cfg['name']} seed{seed}] test future_mse {test['future_mse']:.5f} "
          f"target_mse {test['target_mse']:.5f} dmag {test['delta_mag']:.5f} "
          f"acc {test['acc']:.3f} ({dt:.1f}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="out/clevrer_feats.npz")
    ap.add_argument("--obs", type=int, default=4)
    ap.add_argument("--pred", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--pos_w", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split", default="0.7,0.15,0.15")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="out/clevrer_dino.json")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.feats, allow_pickle=False)
    data = {k: z[k] for k in z.files}   # 一次性读入内存，避免每次采样重复解压
    z.close()
    N = len(data["target"])
    M = data["feats"].shape[2]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    fr = [float(x) for x in args.split.split(",")]
    a = int(fr[0] * N)
    b = int((fr[0] + fr[1]) * N)
    tr_idx, dv_idx, te_idx = perm[:a], perm[a:b], perm[b:]
    train_ds = FeatDataset(data, tr_idx)
    dev_ds = FeatDataset(data, dv_idx)
    test_ds = FeatDataset(data, te_idx)
    print(f"device={device} N={N} M={M} split train/dev/test={len(tr_idx)}/{len(dv_idx)}/{len(te_idx)}")

    rows = []
    t_all = time.time()
    for cfg in CONFIGS:
        for seed in range(args.seeds):
            rows.append(run_one(cfg, seed, args, device, train_ds, dev_ds, test_ds, M))
    agg = aggregate(rows)

    print("\n=== SUMMARY (test, mean over seeds) ===")
    print(f"{'encoder':8s} {'config':12s} {'future_mse':>11s} {'target_mse':>11s} "
          f"{'dMAG':>8s} {'acc':>6s}")
    for a_ in agg:
        print(f"{a_['encoder']:8s} {a_['config']:12s} "
              f"{a_['future_mse_mean']:11.5f} {a_['target_mse_mean']:11.5f} "
              f"{a_['delta_mag_mean']:8.4f} {a_['acc_mean']:6.3f}")
    print_step_report(agg, args.pred)

    payload = {"args": vars(args), "rows": rows, "summary": agg,
               "total_s": round(time.time() - t_all, 1)}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
