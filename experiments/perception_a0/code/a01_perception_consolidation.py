#!/usr/bin/env python3
"""WM-PERCEPT-005 A0.1/A0.2: consolidate a non-GT MOVi-C perception front-end.

Model inputs are observed RGB only. MOVi-C annotations are read exclusively for
evaluation (mask matching, centroid error, tracking identity, and flow EPE).
No predictor is trained in this script. A0.1 tunes annotation-free proposal
consolidation on validation. A0.2 freezes the chosen configuration on test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models.optical_flow import raft_large


@dataclass
class Candidate:
    mask: np.ndarray
    score: float
    stability: float
    area: int
    centroid: np.ndarray
    bbox: tuple[int, int, int, int]
    feature: np.ndarray | None = None
    track_id: int | None = None
    gt_id: int | None = None
    gt_iou: float = 0.0
    source_count: int = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    p.add_argument("--num-scenes", type=int, default=8)
    p.add_argument("--obs-frames", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sam2-repo", type=Path, required=True)
    p.add_argument("--sam2-checkpoint", type=Path, required=True)
    p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--raft-checkpoint", type=Path, required=True)
    p.add_argument("--dino-checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--flow-order", choices=("auto", "xy", "row_col"), default="auto")
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--points-per-batch", type=int, default=64)
    p.add_argument("--pred-iou-thresh", type=float, default=0.72)
    p.add_argument("--stability-thresh", type=float, default=0.82)
    p.add_argument("--min-mask-area", type=int, default=18)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.45)
    p.add_argument("--duplicate-iou", type=float, default=0.80)
    p.add_argument("--pre-max-masks", type=int, default=48)
    p.add_argument("--max-masks", type=int, default=12)
    p.add_argument("--containment-thresh", type=float, default=0.88)
    p.add_argument("--containment-max-area-ratio", type=float, default=8.0)
    p.add_argument("--containment-quality-slack", type=float, default=0.12)
    p.add_argument("--area-rank-weight", type=float, default=0.015)
    p.add_argument("--source-rank-weight", type=float, default=0.08)
    p.add_argument("--border-large-area-ratio", type=float, default=0.08)
    p.add_argument("--border-touch-sides", type=int, default=3)
    p.add_argument("--min-track-length", type=int, default=2)
    p.add_argument("--singleton-score", type=float, default=0.92)
    p.add_argument("--dump-candidates", action="store_true")
    p.add_argument("--calibrator-json", type=Path)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--track-score-thresh", type=float, default=0.22)
    p.add_argument("--visualize-scenes", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def infer_flow_order(paths: list[Path]) -> str:
    """Infer whether stored MOVi flow is (x,y) or (row,col) using train-only motion."""
    errors = [0.0, 0.0]
    count = 0
    for path in paths[:min(24, len(paths))]:
        with np.load(path, allow_pickle=False) as z:
            if "flow_component_order" in z:
                stored = str(z["flow_component_order"].item())
                return "xy" if stored == "xy" else "row_col"
            seg = np.asarray(z["seg"])
            flow = np.asarray(z["flow_fw"], dtype=np.float32)
            pos = np.asarray(z["mask_center_norm"], dtype=np.float32)
            visible = np.asarray(z["visible"], dtype=bool)
            for obj in range(min(10, pos.shape[0])):
                for t in range(min(3, flow.shape[0])):
                    mask = (seg[t] == obj + 1) & np.isfinite(flow[t]).all(-1)
                    if mask.sum() < 16 or not (visible[obj, t] and visible[obj, t + 1]):
                        continue
                    if not np.isfinite(pos[obj, t:t + 2]).all():
                        continue
                    pooled = flow[t][mask].mean(0) / 127.0
                    delta = pos[obj, t + 1] - pos[obj, t]
                    errors[0] += float(np.square(pooled - delta).sum())
                    errors[1] += float(np.square(pooled[::-1] - delta).sum())
                    count += 1
    if count < 5:
        raise RuntimeError("Cannot infer flow component order; pass --flow-order explicitly")
    chosen = "xy" if errors[0] <= errors[1] else "row_col"
    print(
        f"flow order train-only check: xy_error={errors[0] / count:.6g} "
        f"row_col_error={errors[1] / count:.6g}; chosen={chosen}",
        flush=True,
    )
    return chosen


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter / union) if union else 0.0


def mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.array([np.nan, np.nan], np.float32), (0, 0, 0, 0)
    center = np.array([xs.mean(), ys.mean()], np.float32)
    return center, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def filter_sam_masks(records: list[dict[str, Any]], height: int, width: int,
                     args: argparse.Namespace) -> list[Candidate]:
    max_area = int(height * width * args.max_mask_area_ratio)
    candidates: list[Candidate] = []
    for rec in records:
        mask = np.asarray(rec["segmentation"], dtype=bool)
        area = int(mask.sum())
        if area < args.min_mask_area or area > max_area:
            continue
        center, bbox = mask_geometry(mask)
        candidates.append(Candidate(
            mask=mask,
            score=float(rec["predicted_iou"]),
            stability=float(rec["stability_score"]),
            area=area,
            centroid=center,
            bbox=bbox,
        ))

    candidates.sort(key=lambda c: (c.score * c.stability, c.area), reverse=True)
    kept: list[Candidate] = []
    for candidate in candidates:
        if any(mask_iou(candidate.mask, previous.mask) >= args.duplicate_iou for previous in kept):
            continue
        kept.append(candidate)
        if len(kept) >= args.pre_max_masks:
            break

    # Reject only unmistakable large border/background regions. This rule does
    # not use dataset annotations and deliberately keeps small edge objects.
    foreground_like: list[Candidate] = []
    large_area = height * width * args.border_large_area_ratio
    for candidate in kept:
        x0, y0, x1, y1 = candidate.bbox
        touched = int(x0 == 0) + int(y0 == 0) + int(x1 == width) + int(y1 == height)
        if candidate.area >= large_area and touched >= args.border_touch_sides:
            continue
        foreground_like.append(candidate)

    # SAM frequently returns both a whole-object mask and several nested part
    # masks. Prefer the enclosing proposal when its confidence is comparable.
    by_area = sorted(foreground_like, key=lambda c: c.area, reverse=True)
    consolidated: list[Candidate] = []
    for candidate in by_area:
        quality = candidate.score * candidate.stability
        nested = False
        for larger in consolidated:
            if larger.area <= candidate.area:
                continue
            ratio = larger.area / max(candidate.area, 1)
            if ratio > args.containment_max_area_ratio:
                continue
            inter = np.logical_and(candidate.mask, larger.mask).sum(dtype=np.int64)
            containment = float(inter / max(candidate.area, 1))
            larger_quality = larger.score * larger.stability
            if (containment >= args.containment_thresh and
                    larger_quality >= quality - args.containment_quality_slack):
                larger.source_count += candidate.source_count
                nested = True
                break
        if not nested:
            consolidated.append(candidate)

    # Confidence first, with a small preference for complete (larger) objects.
    consolidated.sort(
        key=lambda c: (
            c.score * c.stability
            + args.area_rank_weight * math.log1p(c.area)
            + args.source_rank_weight * math.log1p(c.source_count),
            c.area,
        ),
        reverse=True,
    )
    return consolidated[:args.max_masks]


class DinoExtractor:
    def __init__(self, checkpoint: Path, device: torch.device):
        import timm

        self.device = device
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            checkpoint_path=str(checkpoint),
            num_classes=0,
            dynamic_img_size=True,
        ).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray, candidates: list[Candidate]) -> None:
        if not candidates:
            return
        crops = []
        for candidate in candidates:
            x0, y0, x1, y1 = candidate.bbox
            crop = rgb[y0:y1, x0:x1].copy()
            local_mask = candidate.mask[y0:y1, x0:x1]
            crop[~local_mask] = 127
            tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)
            tensor = F.interpolate(tensor[None], size=(112, 112), mode="bilinear", align_corners=False)[0]
            crops.append(tensor)
        batch = torch.stack(crops).to(self.device)
        batch = (batch - self.mean) / self.std
        features = self.model(batch)
        features = F.normalize(features.float(), dim=-1).cpu().numpy()
        for candidate, feature in zip(candidates, features):
            candidate.feature = feature.astype(np.float32, copy=False)


class RaftEstimator:
    def __init__(self, checkpoint: Path, device: torch.device):
        self.device = device
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = raft_large(weights=None).to(device).eval()
        self.model.load_state_dict(state)

    @torch.inference_mode()
    def estimate(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        def tensor(image: np.ndarray) -> torch.Tensor:
            x = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
            return x.mul_(2.0).sub_(1.0)

        pair_a = tensor(first)[None].to(self.device)
        pair_b = tensor(second)[None].to(self.device)
        prediction = self.model(pair_a, pair_b)[-1][0]
        return prediction.permute(1, 2, 0).float().cpu().numpy()


def shifted_mask(mask: np.ndarray, vector: np.ndarray) -> np.ndarray:
    transform = np.array([[1.0, 0.0, float(vector[0])],
                          [0.0, 1.0, float(vector[1])]], dtype=np.float32)
    return cv2.warpAffine(
        mask.astype(np.uint8), transform, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)


def candidate_similarity(previous: Candidate, current: Candidate, diagonal: float,
                         flow: np.ndarray | None = None) -> float:
    predicted_center = previous.centroid
    predicted_mask = previous.mask
    if flow is not None:
        vector = pool_flow(flow, previous.mask)
        if vector is not None and np.isfinite(vector).all():
            predicted_center = previous.centroid + vector
            predicted_mask = shifted_mask(previous.mask, vector)
    iou = mask_iou(predicted_mask, current.mask)
    distance = float(np.linalg.norm(predicted_center - current.centroid)) / max(diagonal, 1.0)
    proximity = math.exp(-0.5 * (distance / 0.18) ** 2)
    if previous.feature is None or current.feature is None:
        appearance = 0.5
    else:
        appearance = float(np.clip(np.dot(previous.feature, current.feature), -1.0, 1.0))
        appearance = (appearance + 1.0) / 2.0
    return 0.45 * iou + 0.25 * proximity + 0.30 * appearance


def assign_tracks(frames: list[list[Candidate]], height: int, width: int,
                  threshold: float, flows: list[np.ndarray] | None = None) -> None:
    next_id = 0
    if not frames:
        return
    for candidate in frames[0]:
        candidate.track_id = next_id
        next_id += 1
    diagonal = math.hypot(height, width)
    for t in range(1, len(frames)):
        previous = frames[t - 1]
        current = frames[t]
        if not previous:
            for candidate in current:
                candidate.track_id = next_id
                next_id += 1
            continue
        if not current:
            continue
        flow = flows[t - 1] if flows is not None and t - 1 < len(flows) else None
        similarity = np.array([
            [candidate_similarity(a, b, diagonal, flow) for b in current]
            for a in previous
        ], dtype=np.float32)
        rows, cols = linear_sum_assignment(1.0 - similarity)
        assigned_current: set[int] = set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if float(similarity[row, col]) >= threshold:
                current[col].track_id = previous[row].track_id
                assigned_current.add(col)
        for index, candidate in enumerate(current):
            if index not in assigned_current:
                candidate.track_id = next_id
                next_id += 1


def prune_short_tracks(frames: list[list[Candidate]], args: argparse.Namespace) -> None:
    counts: dict[int, int] = {}
    for frame in frames:
        for candidate in frame:
            counts[int(candidate.track_id)] = counts.get(int(candidate.track_id), 0) + 1
    for index, frame in enumerate(frames):
        frames[index] = [
            candidate for candidate in frame
            if counts.get(int(candidate.track_id), 0) >= args.min_track_length
            or candidate.score * candidate.stability >= args.singleton_score
        ]


def candidate_features(candidate: Candidate, track_counts: dict[int, int],
                       height: int, width: int) -> dict[str, float]:
    x0, y0, x1, y1 = candidate.bbox
    bbox_area = max((x1 - x0) * (y1 - y0), 1)
    touches = int(x0 == 0) + int(y0 == 0) + int(x1 == width) + int(y1 == height)
    cx, cy = candidate.centroid
    edge_distance = min(cx, cy, width - 1 - cx, height - 1 - cy)
    return {
        "sam_score": float(candidate.score),
        "stability": float(candidate.stability),
        "sam_quality": float(candidate.score * candidate.stability),
        "log_area_ratio": float(math.log(max(candidate.area / (height * width), 1e-8))),
        "bbox_fill": float(candidate.area / bbox_area),
        "abs_log_aspect": float(abs(math.log(max((x1 - x0) / max(y1 - y0, 1), 1e-6)))),
        "touch_sides": float(touches),
        "log_source_count": float(math.log1p(candidate.source_count)),
        "track_fraction": float(track_counts.get(int(candidate.track_id), 1) / 4.0),
        "edge_distance_norm": float(edge_distance / max(min(height, width), 1)),
    }


def apply_calibrator(frames: list[list[Candidate]], calibrator: dict[str, Any],
                     height: int, width: int) -> None:
    track_counts: dict[int, int] = {}
    for frame in frames:
        for candidate in frame:
            track_counts[int(candidate.track_id)] = track_counts.get(int(candidate.track_id), 0) + 1
    names = list(calibrator["feature_names"])
    mean = np.asarray(calibrator["mean"], dtype=np.float64)
    std = np.asarray(calibrator["std"], dtype=np.float64)
    weights = np.asarray(calibrator["weights"], dtype=np.float64)
    bias = float(calibrator["bias"])
    threshold = float(calibrator["threshold"])
    topk = int(calibrator["topk"])
    for frame_index, frame in enumerate(frames):
        scored = []
        for candidate in frame:
            values = candidate_features(candidate, track_counts, height, width)
            vector = np.asarray([values[name] for name in names], dtype=np.float64)
            logit = float(((vector - mean) / std) @ weights + bias)
            probability = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))
            scored.append((probability, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        frames[frame_index] = [candidate for probability, candidate in scored[:topk]
                               if probability >= threshold]


def gt_masks(segmentation: np.ndarray, object_valid: np.ndarray, visible: np.ndarray,
             frame: int) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for object_index in range(len(object_valid)):
        if not bool(object_valid[object_index]) or not bool(visible[object_index, frame]):
            continue
        mask = segmentation[frame] == (object_index + 1)
        if mask.any():
            masks[object_index] = mask
    return masks


def match_predictions_to_gt(candidates: list[Candidate], targets: dict[int, np.ndarray]) -> list[tuple[int, int, float]]:
    if not candidates or not targets:
        return []
    gt_ids = list(targets)
    matrix = np.array([
        [mask_iou(candidate.mask, targets[gt_id]) for gt_id in gt_ids]
        for candidate in candidates
    ], dtype=np.float32)
    rows, cols = linear_sum_assignment(1.0 - matrix)
    matches: list[tuple[int, int, float]] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        iou = float(matrix[row, col])
        candidates[row].gt_id = gt_ids[col]
        candidates[row].gt_iou = iou
        matches.append((row, gt_ids[col], iou))
    return matches


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def epe(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    values = np.linalg.norm(predicted.astype(np.float32) - target.astype(np.float32), axis=-1)
    if mask is not None:
        values = values[mask]
    return float(values.mean()) if values.size else None


def pool_flow(flow: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    values = flow[mask]
    if len(values) == 0:
        return None
    return np.median(values.astype(np.float32), axis=0)


def audit_scene(path: Path, mask_generator: Any, dino: DinoExtractor,
                raft: RaftEstimator, flow_order: str, calibrator: dict[str, Any] | None,
                args: argparse.Namespace) -> tuple[dict[str, Any], list[list[Candidate]], list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        rgb = np.asarray(z["rgb"][:args.obs_frames], dtype=np.uint8)
        seg = np.asarray(z["seg"][:args.obs_frames], dtype=np.uint8)
        gt_flow = np.asarray(z["flow_fw"][:args.obs_frames - 1], dtype=np.float32)
        object_valid = np.asarray(z["object_valid"], dtype=bool)
        visible = np.asarray(z["visible"][:, :args.obs_frames], dtype=bool)
    if flow_order == "row_col":
        gt_flow = gt_flow[..., ::-1].copy()

    height, width = rgb.shape[1:3]
    predicted_frames: list[list[Candidate]] = []
    sam_seconds = 0.0
    for frame in rgb:
        start = time.perf_counter()
        records = mask_generator.generate(frame)
        sam_seconds += time.perf_counter() - start
        candidates = filter_sam_masks(records, height, width, args)
        dino.encode(frame, candidates)
        predicted_frames.append(candidates)

    # Estimate observed-frame flow before association so the tracker compares
    # each candidate with its motion-propagated location in the next frame.
    raft_predictions: list[np.ndarray] = []
    raft_seconds = 0.0
    for t in range(args.obs_frames - 1):
        start = time.perf_counter()
        raft_predictions.append(raft.estimate(rgb[t], rgb[t + 1]))
        raft_seconds += time.perf_counter() - start
    assign_tracks(
        predicted_frames, height, width, args.track_score_thresh,
        flows=raft_predictions,
    )
    if calibrator is not None:
        apply_calibrator(predicted_frames, calibrator, height, width)
    prune_short_tracks(predicted_frames, args)

    gt_objects = 0
    predicted_objects = 0
    true_positive_50 = 0
    all_matched_ious: list[float] = []
    accepted_ious: list[float] = []
    centroid_errors: list[float] = []
    gt_track_history: dict[int, list[int]] = {}
    candidate_records: list[dict[str, Any]] = []
    track_counts: dict[int, int] = {}
    for frame_candidates in predicted_frames:
        for candidate in frame_candidates:
            track_counts[int(candidate.track_id)] = track_counts.get(int(candidate.track_id), 0) + 1

    for frame_index, candidates in enumerate(predicted_frames):
        targets = gt_masks(seg, object_valid, visible, frame_index)
        if args.dump_candidates:
            for candidate_index, candidate in enumerate(candidates):
                overlaps = [(gt_id, mask_iou(candidate.mask, target_mask))
                            for gt_id, target_mask in targets.items()]
                best_gt, best_iou = max(overlaps, key=lambda item: item[1]) if overlaps else (-1, 0.0)
                candidate_records.append({
                    "frame": frame_index,
                    "candidate_index": candidate_index,
                    "track_id": int(candidate.track_id),
                    "best_gt_id": int(best_gt),
                    "best_iou": float(best_iou),
                    "label_iou50": int(best_iou >= args.match_iou),
                    **candidate_features(candidate, track_counts, height, width),
                })
        gt_objects += len(targets)
        predicted_objects += len(candidates)
        matches = match_predictions_to_gt(candidates, targets)
        all_matched_ious.extend(iou for _, _, iou in matches)
        for candidate_index, gt_id, iou in matches:
            if iou < args.match_iou:
                continue
            true_positive_50 += 1
            accepted_ious.append(iou)
            gt_center, _ = mask_geometry(targets[gt_id])
            centroid_errors.append(float(np.linalg.norm(candidates[candidate_index].centroid - gt_center)))
            gt_track_history.setdefault(gt_id, []).append(int(candidates[candidate_index].track_id))

    id_switches = 0
    identity_links = 0
    for history in gt_track_history.values():
        for previous, current in zip(history, history[1:]):
            identity_links += 1
            id_switches += int(previous != current)

    dense_all: list[float] = []
    dense_fg: list[float] = []
    zero_all: list[float] = []
    zero_fg: list[float] = []
    object_gt_mask: list[float] = []
    object_pred_mask: list[float] = []
    for t, predicted in enumerate(raft_predictions):
        target = gt_flow[t]
        foreground = seg[t] > 0
        dense_all.append(float(epe(predicted, target)))
        zero_all.append(float(epe(np.zeros_like(target), target)))
        fg_value = epe(predicted, target, foreground)
        zero_fg_value = epe(np.zeros_like(target), target, foreground)
        if fg_value is not None:
            dense_fg.append(fg_value)
        if zero_fg_value is not None:
            zero_fg.append(zero_fg_value)

        targets = gt_masks(seg, object_valid, visible, t)
        for gt_id, target_mask in targets.items():
            predicted_vector = pool_flow(predicted, target_mask)
            target_vector = pool_flow(target, target_mask)
            if predicted_vector is not None and target_vector is not None:
                object_gt_mask.append(float(np.linalg.norm(predicted_vector - target_vector)))

        for candidate in predicted_frames[t]:
            if candidate.gt_id is None or candidate.gt_iou < args.match_iou:
                continue
            target_mask = targets.get(candidate.gt_id)
            if target_mask is None:
                continue
            predicted_vector = pool_flow(predicted, candidate.mask)
            target_vector = pool_flow(target, target_mask)
            if predicted_vector is not None and target_vector is not None:
                object_pred_mask.append(float(np.linalg.norm(predicted_vector - target_vector)))

    precision = true_positive_50 / max(predicted_objects, 1)
    recall = true_positive_50 / max(gt_objects, 1)
    scene = {
        "scene_id": path.stem,
        "file": str(path),
        "frames": args.obs_frames,
        "gt_object_frames": gt_objects,
        "predicted_object_frames": predicted_objects,
        "sam_source_masks_represented": sum(
            candidate.source_count for frame in predicted_frames for candidate in frame
        ),
        "tp_iou50": true_positive_50,
        "mask_precision_iou50": precision,
        "mask_recall_iou50": recall,
        "matched_iou_all": safe_mean(all_matched_ious),
        "matched_iou_iou50": safe_mean(accepted_ious),
        "centroid_mae_px_iou50": safe_mean(centroid_errors),
        "track_id_switches": id_switches,
        "track_identity_links": identity_links,
        "track_id_switch_rate": id_switches / max(identity_links, 1),
        "raft_dense_epe_all_px": safe_mean(dense_all),
        "raft_dense_epe_foreground_px": safe_mean(dense_fg),
        "zero_dense_epe_all_px": safe_mean(zero_all),
        "zero_dense_epe_foreground_px": safe_mean(zero_fg),
        "raft_object_epe_gt_mask_px": safe_mean(object_gt_mask),
        "raft_object_epe_pred_mask_px": safe_mean(object_pred_mask),
        "sam_seconds": sam_seconds,
        "raft_seconds": raft_seconds,
    }
    if args.dump_candidates:
        scene["candidate_records"] = candidate_records
    return scene, predicted_frames, raft_predictions


def color_for_id(index: int) -> tuple[float, float, float]:
    palette = plt.get_cmap("tab20")
    return tuple(float(x) for x in palette(index % 20)[:3])


def visualize(path: Path, output: Path, candidates: list[list[Candidate]],
              flows: list[np.ndarray], obs_frames: int) -> None:
    with np.load(path, allow_pickle=False) as z:
        rgb = np.asarray(z["rgb"][:obs_frames], dtype=np.uint8)
    fig, axes = plt.subplots(1, obs_frames, figsize=(4 * obs_frames, 4), dpi=140)
    if obs_frames == 1:
        axes = [axes]
    for t, axis in enumerate(axes):
        axis.imshow(rgb[t])
        for candidate in candidates[t]:
            track_id = int(candidate.track_id)
            color = color_for_id(track_id)
            contour, _ = cv2.findContours(candidate.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for item in contour:
                points = item[:, 0, :]
                axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.2)
            x, y = candidate.centroid
            label = f"P{track_id}"
            if candidate.gt_id is not None:
                label += f"/G{candidate.gt_id + 1}:{candidate.gt_iou:.2f}"
            axis.text(x, y, label, fontsize=6, color="white",
                      bbox={"facecolor": color, "alpha": 0.75, "pad": 1})
            if t < len(flows):
                vector = pool_flow(flows[t], candidate.mask)
                if vector is not None:
                    axis.arrow(x, y, float(vector[0]) * 3.0, float(vector[1]) * 3.0,
                               color="cyan", width=0.35, head_width=2.0, length_includes_head=True)
        axis.set_title(f"frame {t}: {len(candidates[t])} predicted masks")
        axis.set_xlim(0, rgb.shape[2])
        axis.set_ylim(rgb.shape[1], 0)
        axis.axis("off")
    fig.suptitle(f"A0 predicted masks/tracks + RAFT arrows | scene {path.stem}")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def aggregate(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "gt_object_frames": sum(int(x["gt_object_frames"]) for x in scenes),
        "predicted_object_frames": sum(int(x["predicted_object_frames"]) for x in scenes),
        "sam_source_masks_represented": sum(int(x["sam_source_masks_represented"]) for x in scenes),
        "tp_iou50": sum(int(x["tp_iou50"]) for x in scenes),
        "track_id_switches": sum(int(x["track_id_switches"]) for x in scenes),
        "track_identity_links": sum(int(x["track_identity_links"]) for x in scenes),
    }
    totals["mask_precision_iou50_micro"] = totals["tp_iou50"] / max(totals["predicted_object_frames"], 1)
    totals["mask_recall_iou50_micro"] = totals["tp_iou50"] / max(totals["gt_object_frames"], 1)
    totals["track_id_switch_rate_micro"] = totals["track_id_switches"] / max(totals["track_identity_links"], 1)
    mean_keys = [
        "matched_iou_all", "matched_iou_iou50", "centroid_mae_px_iou50",
        "raft_dense_epe_all_px", "raft_dense_epe_foreground_px",
        "zero_dense_epe_all_px", "zero_dense_epe_foreground_px",
        "raft_object_epe_gt_mask_px", "raft_object_epe_pred_mask_px",
        "sam_seconds", "raft_seconds",
    ]
    for key in mean_keys:
        values = [float(x[key]) for x in scenes if x.get(key) is not None]
        totals[f"{key}_scene_mean"] = safe_mean(values)
    return totals


def main() -> None:
    args = parse_args()
    if args.obs_frames < 2 or args.obs_frames > 12:
        raise ValueError("--obs-frames must be in [2, 12]")
    for path in (args.data_root, args.sam2_repo, args.sam2_checkpoint,
                 args.raft_checkpoint, args.dino_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.calibrator_json is not None and not args.calibrator_json.exists():
        raise FileNotFoundError(args.calibrator_json)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty: {args.output_dir}; pass --overwrite")
        shutil.rmtree(args.output_dir)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    split_dir = args.data_root / args.split
    files = sorted(split_dir.glob("*.npz"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No npz files in {split_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    selected = files[:min(args.num_scenes, len(files))]
    if args.flow_order == "auto":
        train_files = sorted((args.data_root / "train").glob("*.npz"), key=lambda p: p.name)
        if not train_files:
            raise FileNotFoundError(f"No train NPZ files for flow-order inference in {args.data_root / 'train'}")
        flow_order = infer_flow_order(train_files)
    else:
        flow_order = args.flow_order

    sys.path.insert(0, str(args.sam2_repo))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    print(f"device={device} split={args.split} scenes={len(selected)}")
    print("loading SAM2 ...", flush=True)
    sam = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=str(device), mode="eval")
    mask_generator = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )
    print("loading DINOv2 ...", flush=True)
    dino = DinoExtractor(args.dino_checkpoint, device)
    print("loading RAFT ...", flush=True)
    raft = RaftEstimator(args.raft_checkpoint, device)
    calibrator = None
    if args.calibrator_json is not None:
        calibrator = json.loads(args.calibrator_json.read_text(encoding="utf-8"))
        print(f"loading candidate calibrator: {args.calibrator_json}", flush=True)

    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "repository_commit": git_commit(Path(__file__).resolve().parents[3]),
        "sam2_repository_commit": git_commit(args.sam2_repo),
        "sam2_checkpoint_sha256": sha256(args.sam2_checkpoint),
        "raft_checkpoint_sha256": sha256(args.raft_checkpoint),
        "dino_checkpoint_sha256": sha256(args.dino_checkpoint),
        "calibrator_sha256": sha256(args.calibrator_json) if args.calibrator_json else None,
    }
    config = {key: json_ready(value) for key, value in vars(args).items()}
    config["resolved_flow_order"] = flow_order
    (args.output_dir / "config.json").write_text(
        json.dumps({"config": config, "environment": environment,
                    "selected_scenes": [p.stem for p in selected]}, indent=2), encoding="utf-8")

    scenes: list[dict[str, Any]] = []
    detail_path = args.output_dir / "per_scene.jsonl"
    with detail_path.open("w", encoding="utf-8") as detail:
        for index, path in enumerate(selected):
            start = time.perf_counter()
            scene, candidates, flows = audit_scene(
                path, mask_generator, dino, raft, flow_order, calibrator, args
            )
            scene["wall_seconds"] = time.perf_counter() - start
            scenes.append(scene)
            detail.write(json.dumps(json_ready(scene), ensure_ascii=False) + "\n")
            detail.flush()
            print(
                f"[{index + 1:02d}/{len(selected):02d}] {path.stem} "
                f"recall={scene['mask_recall_iou50']:.3f} "
                f"centroid={scene['centroid_mae_px_iou50']} "
                f"flow_fg={scene['raft_dense_epe_foreground_px']} "
                f"time={scene['wall_seconds']:.1f}s",
                flush=True,
            )
            if index < args.visualize_scenes:
                visualize(path, figures_dir / f"scene_{path.stem}.png", candidates, flows, args.obs_frames)

    result = {
        "experiment_id": "WM-PERCEPT-005",
        "stage": "A0.1 validation tuning" if args.split == "validation" else "A0.2 frozen test audit",
        "status": "completed",
        "annotation_policy": "GT used only for evaluation; model input is observed RGB",
        "config": config,
        "environment": environment,
        "summary": aggregate(scenes),
        "scenes": scenes,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_ready(result["summary"]), indent=2), flush=True)
    print(f"DONE: {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
