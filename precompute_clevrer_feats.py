"""Precompute frozen DINOv2 features for CLEVRER route B windows.

Decoding video is the bottleneck, so we do it once: for each sampled window we
extract object crops (obs frames) then run the frozen DINOv2 backbone, storing
(obs, M, 384) features + normalized future positions + labels to an .npz.
Training then reads only features (fast), reusing models.py encoder="dinov2_feat".

Usage:
  python precompute_clevrer_feats.py --root ~/datasets/clevrer \
    --split validation --n 20000 --obs 4 --pred 8 --M 6 --crop 64 --out out/clevrer_feats.npz
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from clevrer_video import build_sample, load_ann, load_proposal, _decode_frames
from models import ObjectEncoder


def video_index(root):
    """{video_index(int): mp4 path} by scanning for video_*.mp4."""
    idx = {}
    for p in glob.glob(os.path.join(root, "**", "video_*.mp4"), recursive=True):
        base = os.path.basename(p)
        try:
            i = int(base.replace("video_", "").replace(".mp4", ""))
            idx[i] = p
        except ValueError:
            continue
    return idx


def ann_index(root):
    """{video_index(int): annotation json path}."""
    idx = {}
    for p in glob.glob(os.path.join(root, "**", "annotation_*.json"), recursive=True):
        base = os.path.basename(p)
        try:
            i = int(base.replace("annotation_", "").replace(".json", ""))
            idx[i] = p
        except ValueError:
            continue
    return idx


def prop_index(root):
    """{video_index(int): proposal json path}."""
    idx = {}
    for p in glob.glob(os.path.join(root, "**", "proposal_*.json"), recursive=True):
        base = os.path.basename(p)
        try:
            i = int(base.replace("proposal_", "").replace(".json", ""))
            idx[i] = p
        except ValueError:
            continue
    return idx


def compute_pos_stats(ann_files, n_sample=120, seed=0):
    files = sorted(ann_files)             # 固定顺序，保证 train/val 两次运行统计量完全一致
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(files), size=min(n_sample, len(files)), replace=False)
    pos = []
    for i in sel:
        a = load_ann(files[int(i)])
        for fr in a["motion_trajectory"]:
            for o in fr["objects"]:
                pos.append(o["location"][:2])
    pos = np.asarray(pos, dtype=np.float64)
    return pos.mean(0).astype(np.float32), (pos.std(0) + 1e-6).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/datasets/clevrer")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--obs", type=int, default=4)
    ap.add_argument("--pred", type=int, default=8)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--expand", type=float, default=1.6)
    ap.add_argument("--batch", type=int, default=32, help="windows per DINOv2 batch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="out/clevrer_feats.npz")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    anns = ann_index(root)
    vids = video_index(root)
    props = prop_index(root)
    # prefer a subfolder matching the split, but fall back to all
    split_key = args.split.lower()
    ann_files = [p for p in anns.values() if split_key in p.lower()] or list(anns.values())
    ann_files = sorted(ann_files)
    print(f"annotations={len(ann_files)} videos={len(vids)} proposals={len(props)} split={args.split}")

    # map each annotation to its indices
    items = []
    for ap_ in ann_files:
        base = os.path.basename(ap_)
        try:
            vi = int(base.replace("annotation_", "").replace(".json", ""))
        except ValueError:
            continue
        if vi in vids and vi in props:
            items.append((vi, ap_, vids[vi], props[vi]))
    print(f"usable (annotation+video+proposal)={len(items)}")
    if not items:
        raise SystemExit("no usable videos found; did you unzip video_validation.zip?")

    mean, std = compute_pos_stats(list(anns.values()))   # 用全部标注，train/val 一致
    print("pos mean", mean, "std", std)

    T = args.obs + args.pred
    n_frames = 128
    # sample windows
    rng = np.random.default_rng(0)
    all_windows = [(ii, t0) for ii in range(len(items))
                   for t0 in range(0, n_frames - T + 1, args.stride)]
    if len(all_windows) > args.n:
        sel = rng.choice(len(all_windows), size=args.n, replace=False)
        windows = [all_windows[i] for i in sorted(sel)]
    else:
        windows = all_windows
    print(f"windows={len(windows)}")

    extractor = ObjectEncoder(obs=args.obs, d_model=64, kind="dinov2").to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    feats, trajs, colors, targets = [], [], [], []
    lyes, lno = [], []
    videos = []                      # 每个窗口的来源 video_index（用于按视频切分）
    buf_crops, buf_meta = [], []
    t_start = time.time()
    done = 0
    failed = 0
    first_err = None

    def flush():
        if not buf_crops:
            return
        crops = torch.from_numpy(np.stack(buf_crops)).float().to(device)  # (B,M,obs,3,S,S)
        with torch.no_grad():
            f = extractor.backbone_features(crops).cpu().numpy().astype(np.float16)
        for k, (vi_, tr, ci, tg, ly, ln) in enumerate(buf_meta):
            feats.append(f[k])
            trajs.append(tr)
            colors.append(ci)
            targets.append(tg)
            lyes.append(ly)
            lno.append(ln)
            videos.append(vi_)
        buf_crops.clear()
        buf_meta.clear()

    # group windows by video so each video is decoded only once
    by_item = {}
    for ii, t0 in windows:
        by_item.setdefault(ii, []).append(t0)

    for ii, t0s in by_item.items():
        vi, ap_, vp, pp_ = items[ii]
        try:
            ann = load_ann(ap_)
            prop = load_proposal(pp_)
            need = set()
            for t0 in t0s:
                need.update(range(t0, t0 + args.obs))
            frames_cache = _decode_frames(vp, sorted(need))
        except Exception as e:
            failed += len(t0s)
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            continue
        for t0 in t0s:
            try:
                s = build_sample(vp, ann, prop, t0, args.obs, args.pred, args.M,
                                 out_size=args.crop, expand=args.expand,
                                 pos_stats=(mean, std), frames_cache=frames_cache)
            except Exception as e:
                failed += 1
                if first_err is None:
                    first_err = f"{type(e).__name__}: {e}"
                continue
            buf_crops.append(s["crops"])          # (M, obs, 3, S, S)
            buf_meta.append((vi, s["traj_yes"], s["color_idx"], s["target_idx"],
                             s["label_yes"], s["label_no"]))
            if len(buf_crops) >= args.batch:
                flush()
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - t_start, 1e-9)
                print(f"  {done}/{len(windows)} windows  ({rate:.1f}/s)", flush=True)
    flush()
    if failed:
        print(f"  WARNING: {failed}/{len(windows)} windows failed. first error: {first_err}",
              flush=True)

    feats = np.stack(feats)           # (N, obs, M, 384) fp16
    trajs = np.stack(trajs)           # (N, T, M, 2)
    colors = np.stack(colors)         # (N, M)
    targets = np.asarray(targets)     # (N,)
    lyes = np.asarray(lyes)
    lno = np.asarray(lno)
    videos = np.asarray(videos, dtype=np.int64)   # (N,) 来源 video_index
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, feats=feats, traj=trajs, color=colors,
                        target=targets, label_yes=lyes, label_no=lno,
                        video=videos, pos_mean=mean, pos_std=std)
    print(f"saved {len(feats)} windows -> {args.out}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
