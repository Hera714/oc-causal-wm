"""Idea 1 experiment: TDV-style delta dynamics in the object-centric predictor.

We keep the action-conditioning fixed (joint=True, action injected into the
predictor) and only change *how the future latent is produced*:

  absolute     : future_latent = predictor_output            (v1 baseline)
  delta_fixed  : future_latent = base + Delta                (TDV, fixed base)
  delta_rec    : future_latent = base + cumsum(Delta)       (TDV, recursive)

We compare them on the counterfactual collision toy, reporting:
  future_mse / target_mse (trajectory quality), delta_mag (degeneracy check),
  cons / align (intervention response: flipping / following the action).

Two object encoders are compared:
  cnn     : trained end-to-end (tiny conv net)
  dinov2  : frozen DINOv2-small backbone, features precomputed once, then a
            trained linear projection + predictor (fast, strong features).

Example:
  python run_delta.py --encoders cnn --seeds 3 --device cuda
  python run_delta.py --encoders cnn,dinov2 --seeds 3 --device cuda
"""

import argparse
import gc
import json
import time

import numpy as np
import torch

from data import make_dataset
from models import Model, ObjectEncoder
from run import train_one, evaluate


CONFIGS = [
    {"name": "absolute", "delta_mode": False, "delta_recursive": False},
    {"name": "delta_fixed", "delta_mode": True, "delta_recursive": False},
    {"name": "delta_rec", "delta_mode": True, "delta_recursive": True},
]


def precompute_features(ds, device, obs, batch=64):
    """用冻结的 DINOv2 backbone 把整个数据集预计算成特征列表。

    返回的每个样本 dict 与原来相同，但去掉 "crops"，改成 "enc_feats"
    (obs, m, 384)。这样后续训练不再跑 ViT，只训练 proj + predictor。
    """
    extractor = ObjectEncoder(obs=obs, d_model=64, kind="dinov2").to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, len(ds), batch):
            samples = [ds[j] for j in range(i0, min(i0 + batch, len(ds)))]
            crops = torch.from_numpy(np.stack([s["crops"] for s in samples])).float()
            crops = crops.permute(0, 1, 2, 5, 3, 4)   # (b,m,obs,C,H,W)
            feats = extractor.backbone_features(crops.to(device)).cpu().numpy()
            for s, f in zip(samples, feats):
                d = dict(s)
                d.pop("crops", None)
                d["enc_feats"] = f.astype(np.float32)
                out.append(d)
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  precomputed {len(out)} feats in {time.time()-t0:.1f}s", flush=True)
    return out


def run_one(cfg, encoder, model_encoder, seed, args, device,
            train_ds, dev_ds, test_ds):
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)
    model = Model(d_model=args.d, m=args.m, obs=args.obs, pred=args.pred,
                  nhead=args.nhead, nlayers=args.nlayers, joint=True,
                  size=args.size, encoder=model_encoder,
                  delta_mode=cfg["delta_mode"],
                  delta_recursive=cfg["delta_recursive"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    train_one(model, train_ds, dev_ds, device, args.epochs, args.lr, args.batch,
              pos_w=args.pos_w, pred=args.pred, obs=args.obs)
    dt = time.time() - t0
    dev = evaluate(model, dev_ds, device, args.batch, pred=args.pred, obs=args.obs)
    test = evaluate(model, test_ds, device, args.batch, pred=args.pred, obs=args.obs)
    row = {"encoder": encoder, "config": cfg["name"], "seed": seed,
           "n_params": n_params, "train_s": round(dt, 1)}
    for k in dev:
        row["dev_" + k] = dev[k]
        row["test_" + k] = test[k]
    print(f"[{encoder}/{cfg['name']} seed{seed}] "
          f"test future_mse {test['future_mse']:.5f} target_mse {test['target_mse']:.5f} "
          f"dmag {test['delta_mag']:.5f} cons {test['cons']:.3f} "
          f"follow {test['follow']:.3f} align {test['align']:.3f}/{test['align_flip']:.3f} "
          f"acc {test['acc']:.3f} ({dt:.1f}s)", flush=True)
    return row


def aggregate(rows):
    """把同一 (encoder, config) 的多种子结果求均值/标准差。"""
    groups = {}
    for r in rows:
        groups.setdefault((r["encoder"], r["config"]), []).append(r)
    agg = []
    for (enc, cfg), rs in groups.items():
        out = {"encoder": enc, "config": cfg, "seeds": len(rs)}
        for k in ["future_mse", "target_mse", "delta_mag", "cons", "cons_strict",
                  "align", "align_flip", "follow", "acc", "flip_acc"]:
            vals = [r["test_" + k] for r in rs]
            out[k + "_mean"] = float(np.mean(vals))
            out[k + "_std"] = float(np.std(vals))
        # 逐预测步曲线（跨种子按元素平均）
        for k in ["future_mse_step", "target_mse_step", "nontarget_mse_step",
                  "delta_mag_step"]:
            arrs = np.array([r["test_" + k] for r in rs], dtype=float)
            out[k + "_mean"] = arrs.mean(axis=0).tolist()
        agg.append(out)
    return agg


def print_step_report(agg, pred):
    """打印逐预测步曲线，并保存对比图（若 matplotlib 可用）。"""
    print("\n=== 逐预测步 target MSE (mean over seeds) ===")
    print(f"{'encoder':8s} {'config':12s} " + " ".join(f"k{i}" .rjust(8) for i in range(pred)))
    for a in agg:
        vals = a.get("target_mse_step_mean")
        if vals is None:
            continue
        print(f"{a['encoder']:8s} {a['config']:12s} "
              + " ".join(f"{v:8.5f}" for v in vals))
    print("\n=== 逐预测步 Δ 幅值 mean|Δ_k| ===")
    for a in agg:
        vals = a.get("delta_mag_step_mean")
        if vals is None:
            continue
        print(f"{a['encoder']:8s} {a['config']:12s} "
              + " ".join(f"{v:8.4f}" for v in vals))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(matplotlib 不可用，跳过绘图:", e, ")")
        return
    encoders = sorted({a["encoder"] for a in agg})
    for enc in encoders:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for a in agg:
            if a["encoder"] != enc:
                continue
            xs = range(len(a["target_mse_step_mean"]))
            axes[0].plot(xs, a["target_mse_step_mean"], marker="o", label=a["config"])
            axes[1].plot(xs, a["delta_mag_step_mean"], marker="s", label=a["config"])
        axes[0].set_title(f"{enc}: target MSE vs future step")
        axes[0].set_xlabel("future step k"); axes[0].set_ylabel("MSE"); axes[0].legend()
        axes[1].set_title(f"{enc}: mean|Delta_k| vs step")
        axes[1].set_xlabel("future step k"); axes[1].set_ylabel("|Delta|"); axes[1].legend()
        fig.tight_layout()
        p = f"out/recursion_curves_{enc}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"saved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="train samples")
    ap.add_argument("--ne", type=int, default=300, help="dev samples")
    ap.add_argument("--nt", type=int, default=500, help="test samples")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--precompute_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--obs", type=int, default=2)
    ap.add_argument("--pred", type=int, default=6)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--pos_w", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cache", type=int, default=5000)
    ap.add_argument("--encoders", default="cnn", help="comma list: cnn,dinov2")
    ap.add_argument("--only", default="", help="comma list of config names to run")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="out/delta_results.json")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    configs = CONFIGS
    if args.only:
        keep = {c.strip() for c in args.only.split(",")}
        configs = [c for c in CONFIGS if c["name"] in keep]

    print(f"device={device} m={args.m} pred={args.pred} d={args.d} "
          f"nlayers={args.nlayers} n={args.n} seeds={args.seeds}")
    print(f"encoders={encoders} configs={[c['name'] for c in configs]}")

    def make_raw(split, n):
        return make_dataset(split, n, m=args.m, size=args.size,
                            max_cache=args.cache, pred_steps=args.pred)

    rows = []
    t_all = time.time()
    for encoder in encoders:
        if encoder == "dinov2":
            # 冻结 backbone + 预计算特征（只算一次，所有 config/seed 复用）
            model_encoder = "dinov2_feat"
            raw_tr, raw_dv, raw_te = make_raw("train", args.n), \
                make_raw("dev", args.ne), make_raw("test", args.nt)
            print("precomputing DINOv2 features ...")
            tr = precompute_features(raw_tr, device, args.obs, args.precompute_batch)
            dv = precompute_features(raw_dv, device, args.obs, args.precompute_batch)
            te = precompute_features(raw_te, device, args.obs, args.precompute_batch)
            del raw_tr, raw_dv, raw_te
            gc.collect()
        else:
            model_encoder = "cnn"
            tr = make_raw("train", args.n)
            dv = make_raw("dev", args.ne)
            te = make_raw("test", args.nt)

        for cfg in configs:
            for seed in range(args.seeds):
                rows.append(run_one(cfg, encoder, model_encoder, seed, args,
                                    device, tr, dv, te))

    agg = aggregate(rows)
    print("\n=== SUMMARY (test, mean over seeds) ===")
    header = (f"{'encoder':8s} {'config':12s} {'future_mse':>11s} "
              f"{'target_mse':>11s} {'dMAG':>8s} {'follow':>7s} {'align':>7s} "
              f"{'alignF':>7s} {'cons':>6s} {'acc':>6s}")
    print(header)
    for a in agg:
        print(f"{a['encoder']:8s} {a['config']:12s} "
              f"{a['future_mse_mean']:11.5f} {a['target_mse_mean']:11.5f} "
              f"{a['delta_mag_mean']:8.4f} {a['follow_mean']:7.3f} "
              f"{a['align_mean']:7.3f} {a['align_flip_mean']:7.3f} "
              f"{a['cons_mean']:6.3f} {a['acc_mean']:6.3f}")

    print_step_report(agg, args.pred)

    payload = {"args": vars(args), "rows": rows, "summary": agg,
               "total_s": round(time.time() - t_all, 1)}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {args.out} (total {payload['total_s']}s)")


if __name__ == "__main__":
    main()
