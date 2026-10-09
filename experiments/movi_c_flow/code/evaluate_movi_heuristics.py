#!/usr/bin/env python3
"""Evaluate copy-last and constant-velocity baselines on MOVi-C mini test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/autodl-fs/data/movi_c_flow_v1")
OUTPUT = Path(
    "/root/autodl-tmp/oc-causal-wm/out/movi_c_flow/mini/movi_c_flow_mini/"
    "heuristic_baselines_test.json"
)

OBS = 4
PRED = 8
FUTURE_TIMES = np.arange(OBS, OBS + PRED, dtype=np.float32)


def empty_stats():
    return {
        "sse": 0.0,
        "sae": 0.0,
        "count": 0,
        "h_sse": np.zeros(PRED, dtype=np.float64),
        "h_count": np.zeros(PRED, dtype=np.float64),
    }


def update(stats, prediction, target, valid):
    error = prediction - target
    scalar_count = int(valid.sum()) * 2

    stats["sse"] += float((error[valid] ** 2).sum())
    stats["sae"] += float(np.abs(error[valid]).sum())
    stats["count"] += scalar_count

    for horizon in range(PRED):
        h_valid = valid[:, horizon]
        h_error = error[:, horizon]
        stats["h_sse"][horizon] += float((h_error[h_valid] ** 2).sum())
        stats["h_count"][horizon] += int(h_valid.sum()) * 2


def summarize(stats):
    mse = stats["sse"] / max(stats["count"], 1)
    mae = stats["sae"] / max(stats["count"], 1)

    return {
        "center_mse_norm": mse,
        "center_mae_norm": mae,
        "center_rmse_px": (mse ** 0.5) * 127.0,
        "center_mae_px": mae * 127.0,
        "per_horizon_mse_norm": (
            stats["h_sse"] / np.maximum(stats["h_count"], 1)
        ).tolist(),
        "valid_scalar_count": stats["count"],
    }


def make_predictions(centers, visible, object_valid):
    """
    copy_last:
      Repeat each object's last visible observed centroid.

    constant_velocity:
      Fit velocity using all visible observed centroids, then extrapolate
      from the latest visible centroid.
    """
    num_objects = centers.shape[0]

    copy_last = np.zeros((num_objects, PRED, 2), dtype=np.float32)
    constant_velocity = np.zeros((num_objects, PRED, 2), dtype=np.float32)

    for obj in range(num_objects):
        if not object_valid[obj]:
            continue

        valid_times = [
            t for t in range(OBS)
            if visible[obj, t] and np.isfinite(centers[obj, t]).all()
        ]

        if not valid_times:
            continue

        last_t = valid_times[-1]
        last_center = centers[obj, last_t]
        copy_last[obj] = last_center

        if len(valid_times) < 2:
            constant_velocity[obj] = last_center
            continue

        times = np.asarray(valid_times, dtype=np.float32)
        positions = centers[obj, valid_times]

        centered_times = times - times.mean()
        denominator = float((centered_times ** 2).sum())

        velocity = (
            (centered_times[:, None] * (positions - positions.mean(axis=0)))
            .sum(axis=0)
            / max(denominator, 1e-8)
        )

        constant_velocity[obj] = (
            last_center[None]
            + (FUTURE_TIMES - float(last_t))[:, None] * velocity[None]
        )

    return copy_last, constant_velocity


def main():
    files = sorted((DATA_ROOT / "test").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No test files in {DATA_ROOT / 'test'}")

    stats = {
        "copy_last": empty_stats(),
        "constant_velocity_lsq": empty_stats(),
    }

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            centers = data["mask_center_norm"].astype(np.float32)
            visible = data["visible"].astype(bool)
            object_valid = data["object_valid"].astype(bool)

        target = centers[:, OBS : OBS + PRED]
        target_valid = (
            visible[:, OBS : OBS + PRED]
            & object_valid[:, None]
            & np.isfinite(target).all(axis=-1)
        )

        target = np.nan_to_num(target, nan=0.0)
        copy_last, constant_velocity = make_predictions(
            centers=centers,
            visible=visible,
            object_valid=object_valid,
        )

        update(stats["copy_last"], copy_last, target, target_valid)
        update(
            stats["constant_velocity_lsq"],
            constant_velocity,
            target,
            target_valid,
        )

    result = {
        "split": "test",
        "num_videos": len(files),
        "protocol": (
            "GT mask-center diagnostic; copy-last repeats the final visible "
            "observed center; constant_velocity_lsq fits velocity from all "
            "visible observed centers."
        ),
        "copy_last": summarize(stats["copy_last"]),
        "constant_velocity_lsq": summarize(stats["constant_velocity_lsq"]),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()