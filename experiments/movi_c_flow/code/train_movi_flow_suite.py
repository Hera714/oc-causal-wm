#!/usr/bin/env python3
"""Unified, reproducible MOVi-C flow comparison on prepared per-video NPZs.

This is a NEW protocol. Rerun its no-flow and aligned-flow baselines; do not
combine its numbers with checkpoints/metrics from the earlier pilot script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

MODES = ("no_flow", "aligned_flow", "zero_flow", "shuffled_flow",
         "flow_aux", "input_aux", "flow_bottleneck",
         "visual_no_flow", "visual_flow")
FLOW_INPUT = {"aligned_flow", "zero_flow", "shuffled_flow", "input_aux",
              "flow_bottleneck", "visual_flow"}
AUX = {"flow_aux", "input_aux", "flow_bottleneck"}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--aux-weight", type=float, default=0.25)
    p.add_argument("--consistency-weight", type=float, default=0.10)
    p.add_argument("--flow-order", choices=("auto", "xy", "row_col"), default="auto")
    p.add_argument("--eval-perturb", choices=("none", "zero", "shuffled", "negated"), default="none")
    p.add_argument("--checkpoint", type=Path, help="Evaluate a trained model without training")
    p.add_argument("--heuristics", action="store_true", help="Evaluate copy-last and constant velocity only")
    return p.parse_args()


def paths_for(root: Path, split: str) -> list[Path]:
    aliases = ("validation", "val", "dev") if split == "validation" else (split,)
    for alias in aliases:
        folder = root / alias
        paths = sorted(folder.rglob("*.npz")) if folder.is_dir() else []
        if paths:
            return paths
    raise FileNotFoundError(f"No per-video NPZ files for {split} in {root}")


def auto_flow_order(paths: list[Path]) -> str:
    votes = [0.0, 0.0]
    n = 0
    for path in paths[:min(24, len(paths))]:
        with np.load(path, allow_pickle=False) as z:
            if "flow_component_order" in z:
                return "xy" if str(z["flow_component_order"].item()) == "xy" else "row_col"
            seg, flow, pos, visible = z["seg"], z["flow_fw"], z["mask_center_norm"], z["visible"]
            for obj in range(min(10, pos.shape[0])):
                for t in range(3):
                    mask = (seg[t] == obj + 1) & np.isfinite(flow[t]).all(-1)
                    if mask.sum() < 16 or not (visible[obj, t] and visible[obj, t + 1]):
                        continue
                    if not np.isfinite(pos[obj, t:t + 2]).all():
                        continue
                    pooled = flow[t][mask].astype(np.float32).mean(0) / 127.0
                    delta = pos[obj, t + 1] - pos[obj, t]
                    votes[0] += float(np.square(pooled - delta).sum())
                    votes[1] += float(np.square(pooled[::-1] - delta).sum())
                    n += 1
    if n < 5:
        raise RuntimeError("Cannot infer flow order; pass --flow-order xy or row_col")
    chosen = "xy" if votes[0] <= votes[1] else "row_col"
    print(f"flow order auto-check: xy_error={votes[0]/n:.6g}, "
          f"row_col_error={votes[1]/n:.6g}; chosen={chosen}", flush=True)
    return chosen


def build_visual(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    grid = np.zeros((10, 4, 8, 8, 4), dtype=np.float32)
    for obj in range(10):
        for t in range(4):
            mask = seg[t] == obj + 1
            counts = mask.reshape(8, 16, 8, 16).sum((1, 3))
            sums = (rgb[t].astype(np.float32) * mask[..., None]).reshape(
                8, 16, 8, 16, 3).sum((1, 3))
            grid[obj, t, :, :, :3] = sums / np.maximum(counts[..., None], 1) / 255.0
            grid[obj, t, :, :, 3] = counts / 256.0
    return grid


class SceneDataset(Dataset):
    def __init__(self, paths: list[Path], mode: str, order: str, seed: int,
                 perturb: str = "none"):
        self.paths = paths
        self.mode = mode
        self.records = []
        self.ids = []
        self.hashes = []
        self.nonfinite_invisible_positions = 0
        self.nonfinite_flow_pixels = 0
        need_visual = mode.startswith("visual_")
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                for key in ("mask_center_norm", "visible", "object_valid", "seg", "flow_fw"):
                    if key not in z:
                        raise ValueError(f"{path}: missing {key}")
                pos = np.asarray(z["mask_center_norm"], dtype=np.float32)
                vis = np.asarray(z["visible"], dtype=bool)
                obj_valid = np.asarray(z["object_valid"], dtype=bool)
                seg = np.asarray(z["seg"], dtype=np.uint8)
                flow = np.asarray(z["flow_fw"], dtype=np.float32)
                if pos.shape != (10, 12, 2) or vis.shape != (10, 12) or \
                   obj_valid.shape != (10,) or seg.shape != (12, 128, 128) or \
                   flow.shape != (11, 128, 128, 2):
                    raise ValueError(f"{path}: unexpected array shapes")
                bad_pos = ~np.isfinite(pos).all(-1)
                if np.any(bad_pos & vis & obj_valid[:, None]):
                    raise ValueError(f"{path}: non-finite position for a visible valid object")
                self.nonfinite_invisible_positions += int(bad_pos.sum())
                pos[bad_pos] = 0.0  # Never use invisible/padded coordinates as input or target.
                finite_flow = np.isfinite(flow).all(-1)
                self.nonfinite_flow_pixels += int((~finite_flow).sum())
                if order == "row_col":
                    flow = flow[..., ::-1]
                pooled = np.zeros((10, 11, 2), dtype=np.float32)
                valid_flow = np.zeros((10, 11), dtype=bool)
                for obj in range(10):
                    if not obj_valid[obj]:
                        continue
                    for t in range(11):
                        mask = (seg[t] == obj + 1) & finite_flow[t]
                        if mask.any() and vis[obj, t] and vis[obj, t + 1]:
                            pooled[obj, t] = flow[t][mask].mean(0) / 127.0
                            valid_flow[obj, t] = True
                if need_visual:
                    visual = np.asarray(z["visual_grid"], dtype=np.float32) \
                        if "visual_grid" in z else build_visual(z["rgb"], seg)
                    if visual.shape != (10, 4, 8, 8, 4):
                        raise ValueError(f"{path}: unexpected visual_grid shape")
                else:
                    visual = np.zeros((10, 4, 8, 8, 4), dtype=np.float32)
                video_id = str(z["video_id"].item()) if "video_id" in z else path.stem
                rgb_hash = str(z["rgb_frame0_sha256"].item()) if "rgb_frame0_sha256" in z \
                    else hashlib.sha256(np.asarray(z["rgb"][0]).tobytes()).hexdigest()
                record = {
                    "obs_pos": pos[:, :4], "obs_vis": vis[:, :4].astype(np.float32),
                    "obs_flow": pooled[:, :3], "obs_flow_valid": valid_flow[:, :3].astype(np.float32),
                    "future_pos": pos[:, 4:12], "future_vis": vis[:, 4:12].astype(np.float32),
                    "future_flow": pooled[:, 3:11],
                    "future_flow_valid": valid_flow[:, 3:11].astype(np.float32),
                    "object_valid": obj_valid.astype(np.float32), "visual": visual,
                }
                if not (record["future_vis"] * record["object_valid"][:, None]).any():
                    raise ValueError(f"{path}: no visible future object")
                self.records.append(record)
                self.ids.append(video_id)
                self.hashes.append(rgb_hash)
        if len(set(self.ids)) != len(self.ids) or len(set(self.hashes)) != len(self.hashes):
            raise ValueError("Duplicate video IDs or RGB frames within split")
        transform = perturb
        if mode == "zero_flow":
            transform = "zero"
        elif mode == "shuffled_flow":
            transform = "shuffled"
        if transform == "zero":
            for r in self.records:
                r["obs_flow"][:] = 0
        elif transform == "negated":
            for r in self.records:
                r["obs_flow"] *= -1
        elif transform == "shuffled":
            if len(self.records) < 2:
                raise ValueError("Shuffled flow needs at least two scenes")
            rng = np.random.default_rng(seed)
            sequence = rng.permutation(len(self.records))
            donor = np.empty_like(sequence)
            donor[sequence] = np.roll(sequence, 1)
            flow_copy = [r["obs_flow"].copy() for r in self.records]
            valid_copy = [r["obs_flow_valid"].copy() for r in self.records]
            for i, r in enumerate(self.records):
                r["obs_flow"] = flow_copy[donor[i]]
                r["obs_flow_valid"] = valid_copy[donor[i]]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {key: torch.from_numpy(value.copy()) for key, value in self.records[index].items()}


class Predictor(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.visual = mode.startswith("visual_")
        self.use_flow = mode in FLOW_INPUT
        self.geo = nn.Sequential(nn.Linear(21, 128), nn.GELU(), nn.Linear(128, 128))
        self.vis = nn.Sequential(nn.Linear(4 * 8 * 8 * 4, 128), nn.GELU(), nn.Linear(128, 128))
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.position_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 16))
        self.flow_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 16))

    def forward(self, b):
        batch, objects = b["object_valid"].shape
        flow = b["obs_flow"] if self.use_flow else torch.zeros_like(b["obs_flow"])
        flow_valid = b["obs_flow_valid"] if self.use_flow else torch.zeros_like(b["obs_flow_valid"])
        if self.visual:
            base = self.vis(b["visual"].reshape(batch, objects, -1))
            # Visual ablation never receives GT centroid as a model input.
            geo = torch.cat((torch.zeros_like(b["obs_pos"]).flatten(2),
                             b["obs_vis"], flow.flatten(2), flow_valid), dim=-1)
        else:
            geo = torch.cat((b["obs_pos"].flatten(2), b["obs_vis"],
                             flow.flatten(2), flow_valid), dim=-1)
            base = 0
        h = self.encoder(self.geo(geo) + base,
                         src_key_padding_mask=(b["object_valid"] == 0))
        predicted_flow = self.flow_head(h).reshape(batch, objects, 8, 2)
        if self.mode == "flow_bottleneck":
            predicted_position = b["obs_pos"][:, :, -1:, :] + predicted_flow.cumsum(dim=2)
        elif self.visual:
            predicted_position = self.position_head(h).reshape(batch, objects, 8, 2)
        else:
            predicted_position = b["obs_pos"][:, :, -1:, :] + \
                self.position_head(h).reshape(batch, objects, 8, 2)
        return predicted_position, predicted_flow


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def losses(model, batch, aux_weight, consistency_weight):
    pred, pred_flow = model(batch)
    mask = batch["future_vis"] * batch["object_valid"][:, :, None]
    position_loss = masked_mean((pred - batch["future_pos"]).square(),
                                mask[:, :, :, None].expand_as(pred))
    loss = position_loss
    flow_loss = torch.zeros((), device=pred.device)
    consistency = torch.zeros((), device=pred.device)
    if model.mode in AUX:
        fmask = batch["future_flow_valid"] * batch["object_valid"][:, :, None]
        flow_loss = masked_mean((pred_flow - batch["future_flow"]).square(),
                                fmask[:, :, :, None].expand_as(pred_flow))
        delta = torch.cat((pred[:, :, :1] - batch["obs_pos"][:, :, -1:, :],
                           pred[:, :, 1:] - pred[:, :, :-1]), dim=2)
        consistency = masked_mean((delta - pred_flow).square(),
                                  fmask[:, :, :, None].expand_as(pred_flow))
        loss = loss + aux_weight * flow_loss + consistency_weight * consistency
    return loss, position_loss, flow_loss, consistency


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sse = np.zeros(8, dtype=np.float64)
    sae = np.zeros(8, dtype=np.float64)
    count = np.zeros(8, dtype=np.float64)
    flow_epe_sum = 0.0
    flow_count = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred, pred_flow = model(batch)
        mask = (batch["future_vis"] * batch["object_valid"][:, :, None]).cpu().numpy()
        diff = (pred - batch["future_pos"]).cpu().numpy()
        sse += (np.square(diff).sum(-1) * mask).sum((0, 1))
        sae += (np.abs(diff).sum(-1) * mask).sum((0, 1))
        count += mask.sum((0, 1)) * 2
        if model.mode in AUX:
            fm = (batch["future_flow_valid"] * batch["object_valid"][:, :, None]).cpu().numpy()
            fdiff = (pred_flow - batch["future_flow"]).cpu().numpy()
            flow_epe_sum += float((np.linalg.norm(fdiff, axis=-1) * fm).sum()) * 127.0
            flow_count += float(fm.sum())
    if not np.all(count > 0):
        raise RuntimeError("At least one future horizon has no valid targets")
    return {"position_mse": float(sse.sum() / count.sum()),
            "mae_px": float(sae.sum() / count.sum() * 127.0),
            "rmse_px": float(np.sqrt(sse.sum() / count.sum()) * 127.0),
            "per_horizon_mse": (sse / count).tolist(),
            "valid_scalar_count": int(count.sum()),
            "future_flow_epe_px": float(flow_epe_sum / flow_count) if flow_count else None}


def heuristic_scores(ds):
    output = {}
    for name in ("copy_last", "constant_velocity_lsq"):
        sse = np.zeros(8); sae = np.zeros(8); count = np.zeros(8)
        for r in ds.records:
            pred = np.repeat(r["obs_pos"][:, -1:, :], 8, axis=1)
            if name == "constant_velocity_lsq":
                for obj in range(10):
                    ts = np.flatnonzero(r["obs_vis"][obj] > 0)
                    if len(ts) >= 2:
                        x = np.column_stack((ts.astype(np.float32), np.ones(len(ts))))
                        slope, intercept = np.linalg.lstsq(x, r["obs_pos"][obj, ts], rcond=None)[0]
                        pred[obj] = np.arange(4, 12)[:, None] * slope + intercept
                pred = np.clip(pred, 0.0, 1.0)
            mask = r["future_vis"] * r["object_valid"][:, None]
            diff = pred - r["future_pos"]
            sse += (np.square(diff).sum(-1) * mask).sum(0)
            sae += (np.abs(diff).sum(-1) * mask).sum(0)
            count += mask.sum(0) * 2
        output[name] = {"position_mse": float(sse.sum() / count.sum()),
                        "mae_px": float(sae.sum() / count.sum() * 127.0),
                        "per_horizon_mse": (sse / count).tolist(),
                        "valid_scalar_count": int(count.sum())}
    return output


def main():
    a = args_parser()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not a.heuristics:
        raise RuntimeError("Training requires CUDA; enable the 4090 before running")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: paths_for(a.data_root, s) for s in ("train", "validation", "test")}
    order = auto_flow_order(paths["train"]) if a.flow_order == "auto" else a.flow_order
    if a.heuristics:
        test = SceneDataset(paths["test"], "no_flow", order, a.seed)
        result = {"kind": "heuristics", "data_root": str(a.data_root),
                  "test_count": len(test), "flow_order": order, "results": heuristic_scores(test)}
        (a.output_dir / "heuristic_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    if a.checkpoint and a.eval_perturb == "none":
        raise ValueError("--checkpoint requires an --eval-perturb")
    if a.eval_perturb != "none" and not a.checkpoint:
        raise ValueError("--eval-perturb requires --checkpoint")
    if a.eval_perturb != "none" and a.mode not in FLOW_INPUT:
        raise ValueError("Input perturbation requires a mode with observed flow")
    split_ids = {s: set() for s in paths}
    split_hashes = {s: set() for s in paths}
    datasets = {}
    for offset, split in enumerate(("train", "validation", "test")):
        perturb = a.eval_perturb if (a.checkpoint and split == "test") else "none"
        datasets[split] = SceneDataset(paths[split], a.mode, order,
                                       a.seed + offset * 1000, perturb)
        split_ids[split] = set(datasets[split].ids)
        split_hashes[split] = set(datasets[split].hashes)
        print(f"{split}: masked {datasets[split].nonfinite_invisible_positions} "
              f"non-finite invisible positions; ignored "
              f"{datasets[split].nonfinite_flow_pixels} non-finite flow pixels", flush=True)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_ids[left] & split_ids[right] or split_hashes[left] & split_hashes[right]:
            raise RuntimeError(f"Video leakage between {left} and {right}")
    model = Predictor(a.mode).to(device)
    loaders = {s: DataLoader(datasets[s], batch_size=a.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True) for s in datasets}
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data_hash = hashlib.sha256("\n".join(sorted(
        f"{s}:{i}:{h}" for s in datasets for i, h in
        zip(datasets[s].ids, datasets[s].hashes))).encode()).hexdigest()
    config = {"mode": a.mode, "seed": a.seed, "epochs": a.epochs,
              "batch_size": a.batch_size, "lr": a.lr, "patience": a.patience,
              "aux_weight": a.aux_weight, "consistency_weight": a.consistency_weight,
              "flow_order": order, "data_root": str(a.data_root),
              "split_counts": {s: len(datasets[s]) for s in datasets},
              "nonfinite_data": {s: {
                  "invisible_positions": datasets[s].nonfinite_invisible_positions,
                  "flow_pixels_ignored": datasets[s].nonfinite_flow_pixels,
              } for s in datasets},
              "data_sha256": data_hash, "source_sha256": source_hash,
              "torch": torch.__version__, "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (a.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_path = a.output_dir / "checkpoint_best.pt"
    if a.checkpoint:
        state = torch.load(a.checkpoint, map_location=device, weights_only=False)
        if state["config"]["mode"] != a.mode or state["config"]["flow_order"] != order \
           or state["config"]["data_sha256"] != data_hash:
            raise RuntimeError("Checkpoint mode, flow order or dataset does not match")
        model.load_state_dict(state["model"])
        result = {"kind": "evaluation_perturbation", "config": config,
                  "perturbation": a.eval_perturb,
                  "test": evaluate(model, loaders["test"], device)}
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        best_val = float("inf"); best_epoch = 0; stale = 0
        for epoch in range(1, a.epochs + 1):
            model.train()
            generator = torch.Generator().manual_seed(a.seed * 100000 + epoch)
            train_loader = DataLoader(datasets["train"], batch_size=a.batch_size,
                                      shuffle=True, generator=generator, num_workers=0,
                                      pin_memory=True)
            total = 0.0; steps = 0
            for batch in train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, _, _, _ = losses(model, batch, a.aux_weight,
                                       a.consistency_weight)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.item()); steps += 1
            val = evaluate(model, loaders["validation"], device)
            print(f"epoch={epoch:03d} train_loss={total/max(steps,1):.6f} "
                  f"val_mse={val['position_mse']:.6f} val_mae_px={val['mae_px']:.3f}",
                  flush=True)
            if val["position_mse"] < best_val - 1e-8:
                best_val = val["position_mse"]; best_epoch = epoch; stale = 0
                torch.save({"model": model.state_dict(), "config": config,
                            "best_epoch": best_epoch}, best_path)
            else:
                stale += 1
            if stale >= a.patience:
                break
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        result = {"kind": "trained", "config": config, "best_epoch": best_epoch,
                  "validation": evaluate(model, loaders["validation"], device),
                  "test": evaluate(model, loaders["test"], device)}
    metric_path = a.output_dir / "metrics.json"
    metric_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
