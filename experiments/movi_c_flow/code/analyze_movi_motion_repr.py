#!/usr/bin/env python3
"""Interpretability audit for MOVi-C predictors with and without observed flow.

The script never trains or changes the main predictor. It evaluates existing
checkpoints, performs controlled flow interventions, extracts layerwise object
tokens, fits train-only linear probes, and saves quantitative/visual evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_movi_flow_suite import Predictor, SceneDataset, paths_for


PIXEL_SCALE = 127.0
LAYERS = ("input_projection", "transformer_1", "transformer_2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--modes", nargs="+", default=["no_flow", "aligned_flow", "input_aux"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--max-scenes", type=int, default=0,
                   help="0 uses the full split; positive values are for smoke tests")
    p.add_argument("--case-count", type=int, default=6)
    p.add_argument("--flow-order", choices=("xy", "row_col"), default="xy")
    return p.parse_args()


def set_deterministic(seed: int = 20261009) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_datasets(data_root: Path, order: str, max_scenes: int):
    datasets = {}
    for split in ("train", "validation", "test"):
        paths = paths_for(data_root, split)
        if max_scenes:
            paths = paths[:max_scenes]
        datasets[split] = SceneDataset(paths, "aligned_flow", order, seed=0)
        print(f"loaded {split}: {len(datasets[split])} scenes", flush=True)
    return datasets


def loader_for(dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def load_model(mode: str, seed: int, checkpoint_root: Path,
               device: torch.device) -> Predictor:
    path = checkpoint_root / f"{mode}_seed{seed}" / "checkpoint_best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=False)
    saved_mode = state.get("config", {}).get("mode")
    if saved_mode and saved_mode != mode:
        raise RuntimeError(f"{path}: expected mode={mode}, checkpoint has {saved_mode}")
    model = Predictor(mode).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def forward_with_features(model: Predictor, batch: dict[str, torch.Tensor]):
    batch_size, objects = batch["object_valid"].shape
    flow = batch["obs_flow"] if model.use_flow else torch.zeros_like(batch["obs_flow"])
    flow_valid = (batch["obs_flow_valid"] if model.use_flow
                  else torch.zeros_like(batch["obs_flow_valid"]))
    if model.visual:
        base = model.vis(batch["visual"].reshape(batch_size, objects, -1))
        geo = torch.cat((torch.zeros_like(batch["obs_pos"]).flatten(2),
                         batch["obs_vis"], flow.flatten(2), flow_valid), dim=-1)
    else:
        base = 0
        geo = torch.cat((batch["obs_pos"].flatten(2), batch["obs_vis"],
                         flow.flatten(2), flow_valid), dim=-1)
    hidden = model.geo(geo) + base
    features = [hidden]
    padding_mask = batch["object_valid"] == 0
    for layer in model.encoder.layers:
        hidden = layer(hidden, src_key_padding_mask=padding_mask)
        features.append(hidden)
    if model.encoder.norm is not None:
        hidden = model.encoder.norm(hidden)
        features[-1] = hidden
    predicted_flow = model.flow_head(hidden).reshape(batch_size, objects, 8, 2)
    if model.mode == "flow_bottleneck":
        predicted_position = (batch["obs_pos"][:, :, -1:, :]
                              + predicted_flow.cumsum(dim=2))
    elif model.visual:
        predicted_position = model.position_head(hidden).reshape(batch_size, objects, 8, 2)
    else:
        predicted_position = (batch["obs_pos"][:, :, -1:, :]
                              + model.position_head(hidden).reshape(batch_size, objects, 8, 2))
    return predicted_position, predicted_flow, features


@torch.no_grad()
def collect_split(model: Predictor, loader: DataLoader, device: torch.device,
                  keep_features: bool = True):
    keys = ("obs_pos", "obs_vis", "obs_flow", "obs_flow_valid",
            "future_pos", "future_vis", "object_valid")
    arrays = {key: [] for key in keys}
    predictions = []
    features = [[] for _ in LAYERS]
    checked = False
    for cpu_batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
        pred, _, hidden = forward_with_features(model, batch)
        if not checked:
            reference, _ = model(batch)
            if not torch.allclose(pred, reference, atol=1e-6, rtol=1e-5):
                raise RuntimeError("Feature-forward path does not match Predictor.forward")
            checked = True
        predictions.append(pred.cpu().numpy())
        for key in keys:
            arrays[key].append(cpu_batch[key].numpy())
        if keep_features:
            for index, tensor in enumerate(hidden):
                features[index].append(tensor.cpu().numpy())
    output = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    output["prediction"] = np.concatenate(predictions, axis=0)
    if keep_features:
        output["features"] = [np.concatenate(parts, axis=0) for parts in features]
    return output


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.asarray(values)[mask].mean())


def trajectory_metrics(pred: np.ndarray, data: dict[str, np.ndarray]) -> dict:
    gt = data["future_pos"]
    obj = data["object_valid"].astype(bool)
    future_vis = data["future_vis"].astype(bool)
    obs_vis = data["obs_vis"].astype(bool)
    mask = obj[:, :, None] & future_vis
    diff = pred - gt
    coord_count = max(float(mask.sum() * 2), 1.0)
    mse = float((np.square(diff).sum(-1) * mask).sum() / coord_count)
    mae_px = float((np.abs(diff).sum(-1) * mask).sum() / coord_count * PIXEL_SCALE)
    epe = np.linalg.norm(diff, axis=-1) * PIXEL_SCALE
    ade = safe_mean(epe, mask)
    fde_mask = mask[:, :, -1]
    fde = safe_mean(epe[:, :, -1], fde_mask)
    per_horizon_mse = []
    per_horizon_ade = []
    for horizon in range(8):
        count = max(float(mask[:, :, horizon].sum() * 2), 1.0)
        per_horizon_mse.append(float(
            (np.square(diff[:, :, horizon]).sum(-1) * mask[:, :, horizon]).sum() / count))
        per_horizon_ade.append(safe_mean(epe[:, :, horizon], mask[:, :, horizon]))

    obs_last = data["obs_pos"][:, :, -1]
    gt_seq = np.concatenate((obs_last[:, :, None], gt), axis=2)
    pred_seq = np.concatenate((obs_last[:, :, None], pred), axis=2)
    seq_vis = np.concatenate((obs_vis[:, :, -1:], future_vis), axis=2)
    gt_velocity = np.diff(gt_seq, axis=2)
    pred_velocity = np.diff(pred_seq, axis=2)
    velocity_valid = obj[:, :, None] & seq_vis[:, :, :-1] & seq_vis[:, :, 1:]
    velocity_error = np.linalg.norm(pred_velocity - gt_velocity, axis=-1) * PIXEL_SCALE
    velocity_epe = safe_mean(velocity_error, velocity_valid)
    speed_error = np.abs(np.linalg.norm(pred_velocity, axis=-1)
                         - np.linalg.norm(gt_velocity, axis=-1)) * PIXEL_SCALE
    speed_mae = safe_mean(speed_error, velocity_valid)
    gt_speed = np.linalg.norm(gt_velocity, axis=-1)
    pred_speed = np.linalg.norm(pred_velocity, axis=-1)
    moving = velocity_valid & (gt_speed * PIXEL_SCALE > 0.25)
    cosine = ((pred_velocity * gt_velocity).sum(-1)
              / np.maximum(pred_speed * gt_speed, 1e-8))
    direction_cosine = safe_mean(cosine, moving)

    last_obs_velocity = data["obs_pos"][:, :, 3] - data["obs_pos"][:, :, 2]
    gt_acceleration = np.empty_like(gt_velocity)
    pred_acceleration = np.empty_like(pred_velocity)
    gt_acceleration[:, :, 0] = gt_velocity[:, :, 0] - last_obs_velocity
    pred_acceleration[:, :, 0] = pred_velocity[:, :, 0] - last_obs_velocity
    gt_acceleration[:, :, 1:] = np.diff(gt_velocity, axis=2)
    pred_acceleration[:, :, 1:] = np.diff(pred_velocity, axis=2)
    acceleration_valid = np.zeros_like(velocity_valid)
    acceleration_valid[:, :, 0] = (obj & obs_vis[:, :, 2] & obs_vis[:, :, 3]
                                         & future_vis[:, :, 0])
    acceleration_valid[:, :, 1:] = velocity_valid[:, :, :-1] & velocity_valid[:, :, 1:]
    acceleration_error = np.linalg.norm(pred_acceleration - gt_acceleration, axis=-1) * PIXEL_SCALE
    acceleration_epe = safe_mean(acceleration_error, acceleration_valid)

    return {
        "position_mse": mse,
        "coordinate_mae_px": mae_px,
        "ade_px": ade,
        "fde_px": fde,
        "velocity_epe_px_per_frame": velocity_epe,
        "speed_mae_px_per_frame": speed_mae,
        "velocity_direction_cosine": direction_cosine,
        "acceleration_epe_px_per_frame2": acceleration_epe,
        "per_horizon_mse": per_horizon_mse,
        "per_horizon_ade_px": per_horizon_ade,
    }


def motion_quantiles(data: dict[str, np.ndarray]) -> dict:
    obj = data["object_valid"].astype(bool)
    vis = data["obs_vis"].astype(bool)
    fvis = data["future_vis"].astype(bool)
    velocity = data["obs_pos"][:, :, 3] - data["obs_pos"][:, :, 2]
    first_future_velocity = data["future_pos"][:, :, 0] - data["obs_pos"][:, :, 3]
    speed = np.linalg.norm(velocity, axis=-1) * PIXEL_SCALE
    acceleration = np.linalg.norm(first_future_velocity - velocity, axis=-1) * PIXEL_SCALE
    speed_valid = obj & vis[:, :, 2] & vis[:, :, 3]
    acceleration_valid = speed_valid & fvis[:, :, 0]
    return {
        "speed_px_per_frame": np.quantile(speed[speed_valid], [0.25, 0.5, 0.75]).tolist(),
        "acceleration_px_per_frame2": np.quantile(
            acceleration[acceleration_valid], [0.25, 0.5, 0.75]).tolist(),
    }


def binned_position_mse(pred: np.ndarray, data: dict[str, np.ndarray],
                        thresholds: dict) -> dict:
    obj = data["object_valid"].astype(bool)
    vis = data["obs_vis"].astype(bool)
    fvis = data["future_vis"].astype(bool)
    velocity = data["obs_pos"][:, :, 3] - data["obs_pos"][:, :, 2]
    first_future_velocity = data["future_pos"][:, :, 0] - data["obs_pos"][:, :, 3]
    quantities = {
        "speed": np.linalg.norm(velocity, axis=-1) * PIXEL_SCALE,
        "acceleration": np.linalg.norm(first_future_velocity - velocity, axis=-1) * PIXEL_SCALE,
    }
    base_valid = obj & vis[:, :, 2] & vis[:, :, 3]
    sq = np.square(pred - data["future_pos"]).sum(-1)
    result = {}
    for name, values in quantities.items():
        cuts = np.asarray(thresholds[f"{name}_px_per_frame" if name == "speed"
                                     else "acceleration_px_per_frame2"])
        bins = np.digitize(values, cuts, right=True)
        result[name] = []
        valid_obj = base_valid & (fvis[:, :, 0] if name == "acceleration" else True)
        for index in range(4):
            object_mask = valid_obj & (bins == index)
            horizon_mask = object_mask[:, :, None] & fvis
            count = max(float(horizon_mask.sum() * 2), 1.0)
            value = float((sq * horizon_mask).sum() / count)
            result[name].append({"bin": index, "object_count": int(object_mask.sum()),
                                 "position_mse": value})
    persistent = base_valid & fvis.all(axis=2)
    occluding = base_valid & (~fvis.all(axis=2)) & fvis.any(axis=2)
    result["visibility"] = {}
    for name, object_mask in (("always_visible", persistent), ("future_occlusion", occluding)):
        horizon_mask = object_mask[:, :, None] & fvis
        count = max(float(horizon_mask.sum() * 2), 1.0)
        result["visibility"][name] = {
            "object_count": int(object_mask.sum()),
            "position_mse": float((sq * horizon_mask).sum() / count),
        }
    return result


def probe_targets(data: dict[str, np.ndarray]):
    obj = data["object_valid"].astype(bool)
    ovis = data["obs_vis"].astype(bool)
    fvis = data["future_vis"].astype(bool)
    p2 = data["obs_pos"][:, :, 2]
    p3 = data["obs_pos"][:, :, 3]
    p4 = data["future_pos"][:, :, 0]
    p11 = data["future_pos"][:, :, 7]
    recent_velocity = p3 - p2
    targets = {
        "observed_velocity": (recent_velocity, obj & ovis[:, :, 2] & ovis[:, :, 3]),
        "future_first_displacement": (p4 - p3, obj & ovis[:, :, 3] & fvis[:, :, 0]),
        "future_mean_velocity": ((p11 - p3) / 8.0, obj & ovis[:, :, 3] & fvis[:, :, 7]),
        "first_acceleration": (p4 - 2.0 * p3 + p2,
                               obj & ovis[:, :, 2] & ovis[:, :, 3] & fvis[:, :, 0]),
        "cv_residual_h8": (p11 - (p3 + 8.0 * recent_velocity),
                           obj & ovis[:, :, 2] & ovis[:, :, 3] & fvis[:, :, 7]),
    }
    return targets


def vector_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    sse = float(np.square(error).sum())
    centered = y - y.mean(axis=0, keepdims=True)
    sst = float(np.square(centered).sum())
    r2 = 1.0 - sse / max(sst, 1e-12)
    mae_px = float(np.abs(error).mean() * PIXEL_SCALE)
    epe_px = float(np.linalg.norm(error, axis=-1).mean() * PIXEL_SCALE)
    y_norm = np.linalg.norm(y, axis=-1)
    p_norm = np.linalg.norm(pred, axis=-1)
    moving = y_norm * PIXEL_SCALE > 0.25
    cosine = ((y * pred).sum(-1) / np.maximum(y_norm * p_norm, 1e-8))
    direction_cosine = float(cosine[moving].mean()) if moving.any() else float("nan")
    if moving.any():
        true_bin = np.floor((np.arctan2(y[moving, 1], y[moving, 0]) + np.pi)
                            / (2 * np.pi) * 8).astype(int) % 8
        pred_bin = np.floor((np.arctan2(pred[moving, 1], pred[moving, 0]) + np.pi)
                            / (2 * np.pi) * 8).astype(int) % 8
        direction_8bin_accuracy = float((true_bin == pred_bin).mean())
    else:
        direction_8bin_accuracy = float("nan")
    return {"r2": r2, "coordinate_mae_px": mae_px, "epe_px": epe_px,
            "direction_cosine": direction_cosine,
            "direction_8bin_accuracy": direction_8bin_accuracy,
            "sample_count": int(len(y))}


def ridge_probe(train_x, train_y, val_x, val_y, test_x, test_y):
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = train_y.mean(axis=0, keepdims=True)
    tx = (train_x - x_mean) / x_std
    vx = (val_x - x_mean) / x_std
    qx = (test_x - x_mean) / x_std
    ty = train_y - y_mean
    xtx = tx.T @ tx
    xty = tx.T @ ty
    identity = np.eye(tx.shape[1], dtype=np.float64)
    candidates = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    best = None
    for regularization in candidates:
        weights = np.linalg.solve(xtx + regularization * identity, xty)
        val_pred = vx @ weights + y_mean
        val_mse = float(np.square(val_pred - val_y).mean())
        if best is None or val_mse < best[0]:
            best = (val_mse, regularization, weights)
    _, regularization, weights = best
    test_pred = qx @ weights + y_mean
    return regularization, vector_metrics(test_y, test_pred)


def run_probes(split_outputs: dict, mode: str, seed: int) -> list[dict]:
    split_targets = {split: probe_targets(output) for split, output in split_outputs.items()}
    records = []
    for layer_index, layer_name in enumerate(LAYERS):
        features = {split: output["features"][layer_index].reshape(-1, 128).astype(np.float64)
                    for split, output in split_outputs.items()}
        for target_name in split_targets["train"]:
            selected_x = {}
            selected_y = {}
            for split in ("train", "validation", "test"):
                target, valid = split_targets[split][target_name]
                selected_x[split] = features[split][valid.reshape(-1)]
                selected_y[split] = target.reshape(-1, 2)[valid.reshape(-1)].astype(np.float64)
            regularization, metrics = ridge_probe(
                selected_x["train"], selected_y["train"],
                selected_x["validation"], selected_y["validation"],
                selected_x["test"], selected_y["test"],
            )
            records.append({"mode": mode, "seed": seed, "layer": layer_name,
                            "target": target_name, "ridge_lambda": regularization,
                            "test": metrics})
    return records


def transform_flow(batch: dict[str, torch.Tensor], kind: str, value: float):
    modified = {key: tensor.clone() for key, tensor in batch.items()}
    flow = modified["obs_flow"]
    if kind == "scale":
        modified["obs_flow"] = flow * value
    elif kind == "rotate":
        angle = math.radians(value)
        c, s = math.cos(angle), math.sin(angle)
        x, y = flow[..., 0], flow[..., 1]
        modified["obs_flow"] = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
    elif kind == "time_zero":
        modified["obs_flow"][:, :, int(value)] = 0
    else:
        raise ValueError(kind)
    return modified


@torch.no_grad()
def evaluate_intervention(model: Predictor, loader: DataLoader, device: torch.device,
                          base_prediction: np.ndarray, kind: str, value: float) -> dict:
    pred_parts, data_parts = [], defaultdict(list)
    for cpu_batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
        modified = transform_flow(batch, kind, value)
        pred, _ = model(modified)
        pred_parts.append(pred.cpu().numpy())
        for key in ("obs_pos", "obs_vis", "obs_flow", "obs_flow_valid",
                    "future_pos", "future_vis", "object_valid"):
            data_parts[key].append(cpu_batch[key].numpy())
    pred = np.concatenate(pred_parts, axis=0)
    data = {key: np.concatenate(parts, axis=0) for key, parts in data_parts.items()}
    metrics = trajectory_metrics(pred, data)
    valid = data["future_vis"].astype(bool) & data["object_valid"][:, :, None].astype(bool)
    change = np.linalg.norm(pred - base_prediction, axis=-1) * PIXEL_SCALE
    metrics["output_change_epe_px"] = safe_mean(change, valid)
    weights = data["obs_flow_valid"]
    original_flow = ((data["obs_flow"] * weights[..., None]).sum(axis=2)
                     / np.maximum(weights.sum(axis=2, keepdims=True), 1.0))
    direction = original_flow / np.maximum(np.linalg.norm(original_flow, axis=-1, keepdims=True), 1e-8)
    final_disp = pred[:, :, -1] - data["obs_pos"][:, :, -1]
    projection = (final_disp * direction).sum(-1) * PIXEL_SCALE
    flow_valid = (data["object_valid"].astype(bool) & data["obs_vis"][:, :, -1].astype(bool)
                  & data["future_vis"][:, :, -1].astype(bool)
                  & (np.linalg.norm(original_flow, axis=-1) * PIXEL_SCALE > 0.25))
    metrics["final_displacement_projection_on_original_flow_px"] = safe_mean(projection, flow_valid)
    return metrics


@torch.no_grad()
def object_flow_ablation(model: Predictor, loader: DataLoader, device: torch.device,
                         base_prediction: np.ndarray) -> dict:
    matrices = []
    counts = []
    for input_object in range(10):
        sums = np.zeros(10, dtype=np.float64)
        nums = np.zeros(10, dtype=np.float64)
        offset = 0
        for cpu_batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
            batch["obs_flow"] = batch["obs_flow"].clone()
            batch["obs_flow"][:, input_object] = 0
            pred, _ = model(batch)
            pred = pred.cpu().numpy()
            base = base_prediction[offset:offset + len(pred)]
            offset += len(pred)
            change = np.linalg.norm(pred - base, axis=-1) * PIXEL_SCALE
            valid = (cpu_batch["future_vis"].numpy().astype(bool)
                     & cpu_batch["object_valid"].numpy()[:, :, None].astype(bool))
            input_valid = cpu_batch["object_valid"].numpy()[:, input_object].astype(bool)
            for output_object in range(10):
                mask = valid[:, output_object] & input_valid[:, None]
                sums[output_object] += float((change[:, output_object] * mask).sum())
                nums[output_object] += float(mask.sum())
        matrices.append((sums / np.maximum(nums, 1.0)).tolist())
        counts.append(nums.tolist())
    matrix = np.asarray(matrices)
    diagonal = float(np.diag(matrix).mean())
    off_diagonal = float((matrix.sum() - np.trace(matrix)) / (matrix.size - len(matrix)))
    return {"output_change_epe_px_matrix": matrix.tolist(), "counts": counts,
            "diagonal_mean": diagonal, "off_diagonal_mean": off_diagonal,
            "diagonal_to_off_diagonal_ratio": diagonal / max(off_diagonal, 1e-12)}


def summarize(records: list[dict], key_fields: tuple[str, ...], metric_path: tuple[str, ...]):
    groups = defaultdict(list)
    for record in records:
        group = tuple(record[key] for key in key_fields)
        value = record
        for key in metric_path:
            value = value[key]
        groups[group].append(float(value))
    output = []
    for group, values in sorted(groups.items()):
        row = {key: value for key, value in zip(key_fields, group)}
        row.update({"mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values)})
        output.append(row)
    return output


def make_plots(output_dir: Path, motion_records: list[dict], probe_records: list[dict],
               perturb_records: list[dict], thresholds: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    modes = [mode for mode in ("no_flow", "aligned_flow", "input_aux")
             if any(r["mode"] == mode for r in motion_records)]
    metrics = (("position_mse", "Position MSE"),
               ("velocity_epe_px_per_frame", "Velocity EPE (px/frame)"),
               ("acceleration_epe_px_per_frame2", "Acceleration EPE (px/frame²)"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for axis, (key, title) in zip(axes, metrics):
        means, stds = [], []
        for mode in modes:
            values = [r["test"][key] for r in motion_records if r["mode"] == mode]
            means.append(np.mean(values)); stds.append(np.std(values, ddof=1))
        axis.bar(np.arange(len(modes)), means, yerr=stds, capsize=4,
                 color=["#7f8c8d", "#2e7d32", "#1565c0"][:len(modes)])
        axis.set_xticks(np.arange(len(modes)), modes, rotation=20)
        axis.set_title(title); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(figures / "motion_metric_summary.png", dpi=180); plt.close(fig)

    scale = [r for r in perturb_records if r["kind"] == "scale"]
    if scale:
        values = sorted(set(r["value"] for r in scale))
        mse = [np.mean([r["test"]["position_mse"] for r in scale if r["value"] == v]) for v in values]
        projection = [np.mean([r["test"]["final_displacement_projection_on_original_flow_px"]
                               for r in scale if r["value"] == v]) for v in values]
        fig, left = plt.subplots(figsize=(6.5, 4.2)); right = left.twinx()
        left.plot(values, mse, "o-", color="#c62828", label="position MSE")
        right.plot(values, projection, "s-", color="#2e7d32", label="flow-axis projection")
        left.axvline(1.0, color="black", linestyle="--", alpha=0.4)
        left.set_xlabel("flow scale α"); left.set_ylabel("position MSE", color="#c62828")
        right.set_ylabel("final displacement projection (px)", color="#2e7d32")
        left.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(figures / "flow_scale_response.png", dpi=180); plt.close(fig)

    rotate = [r for r in perturb_records if r["kind"] == "rotate"]
    if rotate:
        values = sorted(set(r["value"] for r in rotate))
        mse = [np.mean([r["test"]["position_mse"] for r in rotate if r["value"] == v]) for v in values]
        fig, axis = plt.subplots(figsize=(6.5, 4.2))
        axis.plot(values, mse, "o-", color="#6a1b9a")
        axis.axvline(0.0, color="black", linestyle="--", alpha=0.4)
        axis.set_xlabel("flow rotation (degrees)"); axis.set_ylabel("position MSE")
        axis.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(figures / "flow_rotation_response.png", dpi=180); plt.close(fig)

    targets = ("observed_velocity", "future_first_displacement", "first_acceleration",
               "cv_residual_h8")
    fig, axes = plt.subplots(1, len(targets), figsize=(16, 3.8), sharey=False)
    for axis, target in zip(axes, targets):
        for mode, color in (("no_flow", "#7f8c8d"), ("aligned_flow", "#2e7d32"),
                            ("input_aux", "#1565c0")):
            if mode not in modes:
                continue
            means = []
            for layer in LAYERS:
                vals = [r["test"]["r2"] for r in probe_records
                        if r["mode"] == mode and r["target"] == target and r["layer"] == layer]
                means.append(np.mean(vals))
            axis.plot(range(len(LAYERS)), means, "o-", label=mode, color=color)
        axis.set_xticks(range(len(LAYERS)), ("input", "L1", "L2"))
        axis.set_title(target.replace("_", "\n")); axis.grid(alpha=0.25)
    axes[0].set_ylabel("linear-probe test R²"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "layerwise_probe_r2.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(4)
    for mode, color in (("no_flow", "#7f8c8d"), ("aligned_flow", "#2e7d32")):
        rows = [r for r in motion_records if r["mode"] == mode]
        means = [np.mean([r["binned"]["acceleration"][i]["position_mse"] for r in rows])
                 for i in range(4)]
        axis.plot(x, means, "o-", label=mode, color=color)
    axis.set_xticks(x, ("Q1 low", "Q2", "Q3", "Q4 high"))
    axis.set_xlabel("first-step acceleration bin (train thresholds)")
    axis.set_ylabel("position MSE"); axis.grid(alpha=0.25); axis.legend()
    axis.set_title("Error by non-uniform-motion strength")
    fig.tight_layout(); fig.savefig(figures / "acceleration_regime_mse.png", dpi=180); plt.close(fig)

    write_json(output_dir / "figure_metadata.json", {"motion_thresholds": thresholds})


def make_case_studies(output_dir: Path, data: dict[str, np.ndarray], ids: list[str],
                      no_flow_predictions: np.ndarray, flow_predictions: np.ndarray,
                      case_count: int) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = data["future_vis"].astype(bool) & data["object_valid"][:, :, None].astype(bool)
    no_sq = np.square(no_flow_predictions - data["future_pos"]).sum(-1)
    fl_sq = np.square(flow_predictions - data["future_pos"]).sum(-1)
    denominator = np.maximum(valid.sum(axis=(1, 2)) * 2, 1)
    no_scene = (no_sq * valid).sum(axis=(1, 2)) / denominator
    fl_scene = (fl_sq * valid).sum(axis=(1, 2)) / denominator
    gain = no_scene - fl_scene
    good_count = max(case_count - 2, 1)
    selected = list(np.argsort(gain)[-good_count:][::-1])
    if case_count > good_count:
        selected += list(np.argsort(gain)[:case_count - good_count])
    folder = output_dir / "figures" / "case_studies"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, scene in enumerate(selected):
        per_object_gain = ((no_sq[scene] - fl_sq[scene]) * valid[scene]).sum(axis=1)
        object_index = int(np.argmax(per_object_gain))
        fig, axis = plt.subplots(figsize=(5.2, 5.2))
        for other in range(10):
            if not data["object_valid"][scene, other]:
                continue
            obs_mask = data["obs_vis"][scene, other].astype(bool)
            obs = data["obs_pos"][scene, other, obs_mask]
            if len(obs):
                axis.plot(obs[:, 0], obs[:, 1], color="#bdbdbd", alpha=0.5, linewidth=1)
        obs_mask = data["obs_vis"][scene, object_index].astype(bool)
        future_mask = data["future_vis"][scene, object_index].astype(bool)
        obs = data["obs_pos"][scene, object_index, obs_mask]
        gt = data["future_pos"][scene, object_index, future_mask]
        nf = no_flow_predictions[scene, object_index, future_mask]
        af = flow_predictions[scene, object_index, future_mask]
        axis.plot(obs[:, 0], obs[:, 1], "o-", color="#424242", label="observed")
        axis.plot(gt[:, 0], gt[:, 1], "o-", color="#000000", label="GT future")
        axis.plot(nf[:, 0], nf[:, 1], "o--", color="#c62828", label="no-flow")
        axis.plot(af[:, 0], af[:, 1], "o--", color="#2e7d32", label="aligned-flow")
        mean_flow = data["obs_flow"][scene, object_index].mean(axis=0)
        if len(obs):
            axis.arrow(obs[-1, 0], obs[-1, 1], mean_flow[0] * 4, mean_flow[1] * 4,
                       color="#1565c0", width=0.002, length_includes_head=True)
        axis.set_xlim(0, 1); axis.set_ylim(1, 0); axis.set_aspect("equal")
        axis.grid(alpha=0.2); axis.legend(fontsize=8)
        axis.set_title(f"{ids[scene]} obj={object_index} gain={gain[scene]:.5f}")
        path = folder / f"case_{rank:02d}_{ids[scene]}_obj{object_index}.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        manifest.append({"rank": rank, "scene_index": int(scene), "video_id": ids[scene],
                         "object_index": object_index, "no_flow_mse": float(no_scene[scene]),
                         "aligned_flow_mse": float(fl_scene[scene]),
                         "mse_gain": float(gain[scene]), "figure": str(path)})
    return manifest


def main() -> None:
    args = parse_args()
    set_deterministic()
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device_resolved": str(device), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "pixel_scale": PIXEL_SCALE})
    write_json(args.output_dir / "config.json", config)
    print(f"device={device}", flush=True)

    datasets = load_datasets(args.data_root, args.flow_order, args.max_scenes)
    loaders = {split: loader_for(dataset, args.batch_size) for split, dataset in datasets.items()}
    threshold_data = {key: np.stack([record[key] for record in datasets["train"].records])
                      for key in ("obs_pos", "obs_vis", "future_pos", "future_vis", "object_valid")}
    thresholds = motion_quantiles(threshold_data)
    print(f"motion thresholds={thresholds}", flush=True)

    motion_records, probe_records, perturb_records = [], [], []
    ablation_records = []
    test_predictions = defaultdict(list)
    reference_test_data = None

    for mode in args.modes:
        for seed in args.seeds:
            print(f"model mode={mode} seed={seed}", flush=True)
            model = load_model(mode, seed, args.checkpoint_root, device)
            split_outputs = {}
            for split in ("train", "validation", "test"):
                split_outputs[split] = collect_split(model, loaders[split], device, keep_features=True)
                print(f"  collected {split}", flush=True)
            test = split_outputs["test"]
            metrics = trajectory_metrics(test["prediction"], test)
            binned = binned_position_mse(test["prediction"], test, thresholds)
            motion_records.append({"mode": mode, "seed": seed, "test": metrics, "binned": binned})
            probe_records.extend(run_probes(split_outputs, mode, seed))
            test_predictions[mode].append(test["prediction"])
            if reference_test_data is None:
                reference_test_data = {key: value for key, value in test.items()
                                       if key not in ("prediction", "features")}

            if mode == "aligned_flow":
                for alpha in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
                    result = evaluate_intervention(model, loaders["test"], device,
                                                   test["prediction"], "scale", alpha)
                    perturb_records.append({"seed": seed, "kind": "scale",
                                            "value": alpha, "test": result})
                    print(f"  scale={alpha:+.1f} mse={result['position_mse']:.6f}", flush=True)
                for angle in (-90.0, -45.0, 0.0, 45.0, 90.0, 180.0):
                    result = evaluate_intervention(model, loaders["test"], device,
                                                   test["prediction"], "rotate", angle)
                    perturb_records.append({"seed": seed, "kind": "rotate",
                                            "value": angle, "test": result})
                for time_index in range(3):
                    result = evaluate_intervention(model, loaders["test"], device,
                                                   test["prediction"], "time_zero", time_index)
                    perturb_records.append({"seed": seed, "kind": "time_zero",
                                            "value": time_index, "test": result})
                ablation = object_flow_ablation(model, loaders["test"], device,
                                                test["prediction"])
                ablation_records.append({"seed": seed, **ablation})
            del model, split_outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    motion_output = {
        "thresholds_from_train": thresholds,
        "per_seed": motion_records,
        "position_mse_summary": summarize(motion_records, ("mode",), ("test", "position_mse")),
        "velocity_epe_summary": summarize(
            motion_records, ("mode",), ("test", "velocity_epe_px_per_frame")),
        "acceleration_epe_summary": summarize(
            motion_records, ("mode",), ("test", "acceleration_epe_px_per_frame2")),
    }
    probe_output = {
        "protocol": "Frozen predictor; train-only ridge probe; lambda chosen on validation; test once.",
        "per_seed_layer_target": probe_records,
        "r2_summary": summarize(probe_records, ("mode", "layer", "target"), ("test", "r2")),
    }
    perturb_output = {
        "protocol": "Same aligned-flow checkpoint; only test observed flow is transformed.",
        "per_seed": perturb_records,
        "position_mse_summary": summarize(
            perturb_records, ("kind", "value"), ("test", "position_mse")),
        "output_change_summary": summarize(
            perturb_records, ("kind", "value"), ("test", "output_change_epe_px")),
    }
    write_json(args.output_dir / "motion_metrics.json", motion_output)
    write_json(args.output_dir / "probe_metrics.json", probe_output)
    write_json(args.output_dir / "flow_intervention_metrics.json", perturb_output)
    write_json(args.output_dir / "object_flow_ablation.json", {"per_seed": ablation_records})
    make_plots(args.output_dir, motion_records, probe_records, perturb_records, thresholds)

    if "no_flow" in test_predictions and "aligned_flow" in test_predictions:
        no_flow_mean = np.mean(np.stack(test_predictions["no_flow"]), axis=0)
        flow_mean = np.mean(np.stack(test_predictions["aligned_flow"]), axis=0)
        manifest = make_case_studies(args.output_dir, reference_test_data,
                                     datasets["test"].ids, no_flow_mean, flow_mean,
                                     args.case_count)
        write_json(args.output_dir / "case_manifest.json", manifest)

    combined = {
        "experiment_id": "WM-FLOW-004",
        "config": config,
        "motion": motion_output,
        "probes": probe_output,
        "interventions": perturb_output,
        "object_ablation": {"per_seed": ablation_records},
    }
    write_json(args.output_dir / "metrics.json", combined)
    print(f"COMPLETE: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
