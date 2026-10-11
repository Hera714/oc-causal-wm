#!/usr/bin/env python3
"""Fit a tiny train-only logistic proposal calibrator and tune selection on validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


FEATURES = [
    "sam_score", "stability", "sam_quality", "log_area_ratio", "bbox_fill",
    "abs_log_aspect", "touch_sides", "log_source_count", "track_fraction",
    "edge_distance_norm",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-metrics", type=Path, required=True)
    p.add_argument("--validation-metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=1e-3)
    return p.parse_args()


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scene in data["scenes"]:
        for row in scene["candidate_records"]:
            rows.append({"scene_id": scene["scene_id"], **row})
    return data, rows


def winner_labels(rows):
    winners = {}
    for index, row in enumerate(rows):
        if row["best_gt_id"] < 0 or row["best_iou"] < 0.5:
            continue
        key = (row["scene_id"], row["frame"], row["best_gt_id"])
        if key not in winners or row["best_iou"] > rows[winners[key]]["best_iou"]:
            winners[key] = index
    y = np.zeros(len(rows), dtype=np.float64)
    y[list(winners.values())] = 1.0
    return y


def matrix(rows):
    return np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64)


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(x, y, steps, lr, l2):
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    w = np.zeros(z.shape[1], dtype=np.float64)
    b = float(math.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4)))
    pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))
    weights = np.where(y > 0.5, pos_weight, 1.0)
    weights /= weights.mean()
    for step in range(steps):
        p = sigmoid(z @ w + b)
        residual = (p - y) * weights
        grad_w = z.T @ residual / len(y) + l2 * w
        grad_b = residual.mean()
        rate = lr / math.sqrt(1.0 + step / 1500.0)
        w -= rate * grad_w
        b -= rate * grad_b
    return mean, std, w, b


def select_metrics(data, rows, probabilities, threshold, topk):
    selected = []
    groups = {}
    for row, probability in zip(rows, probabilities):
        groups.setdefault((row["scene_id"], row["frame"]), []).append((float(probability), row))
    for items in groups.values():
        items.sort(key=lambda item: item[0], reverse=True)
        selected.extend((probability, row) for probability, row in items[:topk] if probability >= threshold)

    best = {}
    for probability, row in selected:
        if row["best_gt_id"] < 0 or row["best_iou"] < 0.5:
            continue
        key = (row["scene_id"], row["frame"], row["best_gt_id"])
        value = (row["best_iou"], probability, row)
        if key not in best or value[:2] > best[key][:2]:
            best[key] = value
    tp = len(best)
    predicted = len(selected)
    gt = int(data["summary"]["gt_object_frames"])
    precision = tp / max(predicted, 1)
    recall = tp / max(gt, 1)

    histories = {}
    for (scene_id, frame, gt_id), (_, _, row) in best.items():
        histories.setdefault((scene_id, gt_id), []).append((frame, row["track_id"]))
    switches = links = 0
    for history in histories.values():
        history.sort()
        for (_, previous), (_, current) in zip(history, history[1:]):
            links += 1
            switches += int(previous != current)
    return {
        "threshold": threshold, "topk": topk, "predicted_object_frames": predicted,
        "tp_iou50": tp, "mask_precision_iou50_micro": precision,
        "mask_recall_iou50_micro": recall, "track_id_switches": switches,
        "track_identity_links": links,
        "track_id_switch_rate_micro": switches / max(links, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def main():
    args = parse_args()
    train_data, train_rows = load(args.train_metrics)
    val_data, val_rows = load(args.validation_metrics)
    train_x = matrix(train_rows)
    train_y = winner_labels(train_rows)
    mean, std, weights, bias = fit_logistic(train_x, train_y, args.steps, args.lr, args.l2)
    val_p = sigmoid(((matrix(val_rows) - mean) / std) @ weights + bias)

    grid = []
    for topk in range(5, 17):
        for threshold in np.linspace(0.05, 0.95, 181):
            grid.append(select_metrics(val_data, val_rows, val_p, float(threshold), topk))
    feasible = [item for item in grid
                if item["mask_recall_iou50_micro"] >= 0.85
                and item["mask_precision_iou50_micro"] >= 0.50]
    if feasible:
        chosen = max(feasible, key=lambda item: (item["f1"], -item["predicted_object_frames"]))
        selection = "gate_feasible_best_f1"
    else:
        chosen = max(grid, key=lambda item: (
            min(item["mask_recall_iou50_micro"] / 0.85, 1.0)
            * min(item["mask_precision_iou50_micro"] / 0.50, 1.0),
            item["f1"],
        ))
        selection = "closest_to_gate"

    output = {
        "type": "standardized_logistic_candidate_calibrator",
        "training_split": "train only",
        "validation_role": "threshold/top-k selection only",
        "feature_names": FEATURES,
        "mean": mean.tolist(), "std": std.tolist(),
        "weights": weights.tolist(), "bias": float(bias),
        "threshold": chosen["threshold"], "topk": chosen["topk"],
        "selection_policy": selection,
        "train_rows": len(train_rows), "train_positive_winners": int(train_y.sum()),
        "validation_metrics_estimated_from_candidate_bank": chosen,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
