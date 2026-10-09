#!/usr/bin/env python3
"""Download MOVi-C TFRecord shards one at a time and build 1200/300/300 NPZ splits.

The output is a new, versioned dataset. It does not alter the 120/30/30 pilot.
Source scenes all come from the official MOVi-C train split; our three splits
are disjoint at the video level. The official validation split is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REMOTE_ROOT = "kubric-public/tfds/movi_c/128x128/1.0.0"
SPLITS = ("train", "validation", "test")
FIELDS = (
    "rgb", "seg", "flow_fw", "flow_bw", "mask_center_px", "mask_center_norm",
    "visible", "mask_area", "image_positions", "world_positions", "velocities",
    "visibility_raw", "object_valid", "frame_ids", "visual_grid", "video_id",
    "rgb_frame0_sha256", "source_shard", "flow_component_order",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--validation-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--max-shards", type=int, default=0,
                        help="0 means no limit; useful for a first-shard smoke test")
    return parser.parse_args()


def scalar_string(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_flow(encoded: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = np.asarray(bounds, dtype=np.float32)
    flow_row_col = encoded.astype(np.float32) * ((hi - lo) / 65535.0) + lo
    # Kubric stores (delta_row, delta_column); our positions use (x, y).
    return flow_row_col[..., [1, 0]].astype(np.float16)


def visual_grid(rgb: np.ndarray, seg: np.ndarray, n_objects: int,
                max_objects: int) -> np.ndarray:
    """Spatial RGB/mask evidence, with no explicit centroid or velocity input."""
    output = np.zeros((max_objects, 4, 8, 8, 4), dtype=np.float16)
    for obj in range(n_objects):
        for frame in range(4):
            mask = seg[frame] == (obj + 1)
            counts = mask.reshape(8, 16, 8, 16).sum(axis=(1, 3))
            masked = rgb[frame].astype(np.float32) * mask[..., None]
            sums = masked.reshape(8, 16, 8, 16, 3).sum(axis=(1, 3))
            output[obj, frame, :, :, :3] = (
                sums / np.maximum(counts[..., None], 1) / 255.0
            ).astype(np.float16)
            output[obj, frame, :, :, 3] = (counts / 256.0).astype(np.float16)
    return output


def convert_sample(sample: dict, shard_name: str, frames: int,
                   max_objects: int) -> dict[str, np.ndarray]:
    if frames != 12:
        raise ValueError("This protocol requires exactly 12 frames")
    rgb = np.asarray(sample["video"][:frames], dtype=np.uint8)
    seg = np.asarray(sample["segmentations"][:frames, :, :, 0], dtype=np.uint8)
    if rgb.shape != (12, 128, 128, 3) or seg.shape != (12, 128, 128):
        raise ValueError(f"Unexpected RGB/seg shapes: {rgb.shape}, {seg.shape}")
    n_objects = int(np.asarray(sample["metadata"]["num_instances"]).item())
    if n_objects < 1 or n_objects > max_objects:
        raise ValueError(f"Object count {n_objects} outside 1..{max_objects}")

    flow_fw = decode_flow(
        sample["forward_flow"][:frames - 1],
        sample["metadata"]["forward_flow_range"],
    )
    flow_bw = decode_flow(
        sample["backward_flow"][:frames - 1],
        sample["metadata"]["backward_flow_range"],
    )
    if flow_fw.shape != (11, 128, 128, 2):
        raise ValueError(f"Unexpected flow shape: {flow_fw.shape}")

    center_px = np.zeros((max_objects, frames, 2), dtype=np.float32)
    center_norm = np.zeros_like(center_px)
    areas = np.zeros((max_objects, frames), dtype=np.uint16)
    visible = np.zeros((max_objects, frames), dtype=np.bool_)
    for obj in range(n_objects):
        for frame in range(frames):
            ys, xs = np.nonzero(seg[frame] == (obj + 1))
            area = len(xs)
            if not area:
                continue
            areas[obj, frame] = area
            visible[obj, frame] = True
            center_px[obj, frame] = (xs.mean(), ys.mean())
            center_norm[obj, frame] = center_px[obj, frame] / 127.0

    instances = sample["instances"]
    image_positions = np.zeros((max_objects, frames, 2), dtype=np.float32)
    world_positions = np.zeros((max_objects, frames, 3), dtype=np.float32)
    velocities = np.zeros((max_objects, frames, 3), dtype=np.float32)
    visibility_raw = np.zeros((max_objects, frames), dtype=np.uint16)
    image_positions[:n_objects] = np.asarray(instances["image_positions"][:n_objects, :frames])
    world_positions[:n_objects] = np.asarray(instances["positions"][:n_objects, :frames])
    velocities[:n_objects] = np.asarray(instances["velocities"][:n_objects, :frames])
    visibility_raw[:n_objects] = np.asarray(instances["visibility"][:n_objects, :frames])

    video_id = scalar_string(sample["metadata"]["video_name"])
    frame_hash = hashlib.sha256(rgb[0].tobytes()).hexdigest()
    object_valid = np.zeros(max_objects, dtype=np.bool_)
    object_valid[:n_objects] = True
    result = {
        "rgb": rgb,
        "seg": seg,
        "flow_fw": flow_fw,
        "flow_bw": flow_bw,
        "mask_center_px": center_px,
        "mask_center_norm": center_norm,
        "visible": visible,
        "mask_area": areas,
        "image_positions": image_positions,
        "world_positions": world_positions,
        "velocities": velocities,
        "visibility_raw": visibility_raw,
        "object_valid": object_valid,
        "frame_ids": np.arange(frames, dtype=np.uint8),
        "visual_grid": visual_grid(rgb, seg, n_objects, max_objects),
        "video_id": np.asarray(video_id),
        "rgb_frame0_sha256": np.asarray(frame_hash),
        "source_shard": np.asarray(shard_name),
        "flow_component_order": np.asarray("xy"),
    }
    assert tuple(result) == FIELDS
    return result


def existing_records(root: Path) -> tuple[list[dict], set[str], set[str]]:
    records: list[dict] = []
    video_ids: set[str] = set()
    frame_hashes: set[str] = set()
    for split in SPLITS:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(split_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                video_id = scalar_string(data["video_id"])
                frame_hash = scalar_string(data["rgb_frame0_sha256"])
                shard = scalar_string(data["source_shard"])
            if video_id in video_ids or frame_hash in frame_hashes:
                raise RuntimeError(f"Duplicate video/frame detected in existing data: {path}")
            video_ids.add(video_id)
            frame_hashes.add(frame_hash)
            records.append({"split": split, "file": str(path.relative_to(root)),
                            "video_id": video_id, "rgb_frame0_sha256": frame_hash,
                            "source_shard": shard})
    return records, video_ids, frame_hashes


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_manifest(root: Path, records: list[dict], config: dict) -> None:
    records = sorted(records, key=lambda x: (x["split"], x["video_id"]))
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {split: sum(r["split"] == split for r in records) for split in SPLITS}
    (root / "dataset_config.json").write_text(
        json.dumps({**config, "actual_counts": counts,
                    "source_note": "All scenes sampled from official MOVi-C train TFRecords; "
                                   "internal split by unique video ID; official validation unused."},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    targets = {"train": args.train_count,
               "validation": args.validation_count, "test": args.test_count}
    if min(targets.values()) <= 0 or args.max_objects != 10:
        raise ValueError("Positive split sizes and max-objects=10 are required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.staging_root.mkdir(parents=True, exist_ok=True)
    config = {"dataset": "movi_c/128x128/1.0.0", "source": f"gs://{REMOTE_ROOT}",
              "target_counts": targets, "seed": args.seed, "frames": args.frames,
              "max_objects": args.max_objects, "flow_component_order": "xy",
              "flow_units": "pixels/frame", "schema_version": "movi_c_flow_v2"}
    config_path = args.output_root / "dataset_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("target_counts", "seed", "frames", "max_objects", "schema_version"):
            if old.get(key) != config.get(key):
                raise RuntimeError(f"Existing output has different {key}; choose a new output directory")
    records, known_ids, known_hashes = existing_records(args.output_root)
    counts = {s: sum(r["split"] == s for r in records) for s in SPLITS}
    for split in SPLITS:
        if counts[split] > targets[split]:
            raise RuntimeError(f"Existing {split} count exceeds target")

    import gcsfs
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    fs = gcsfs.GCSFileSystem(token="anon")
    metadata_dir = args.staging_root / "tfds_metadata"
    metadata_dir.mkdir(exist_ok=True)
    for name in ("dataset_info.json", "features.json"):
        local = metadata_dir / name
        if not local.exists():
            fs.get(f"{REMOTE_ROOT}/{name}", str(local))
    builder = tfds.builder_from_directory(str(metadata_dir))
    if builder.info.full_name != "movi_c/128x128/1.0.0":
        raise RuntimeError(f"Unexpected TFDS builder: {builder.info.full_name}")

    shards = [p for p in fs.ls(REMOTE_ROOT, detail=False)
              if re.search(r"movi_c-train\.tfrecord-\d{5}-of-\d{5}$", p)]
    random.Random(args.seed).shuffle(shards)
    remaining = [split for split in SPLITS for _ in range(targets[split] - counts[split])]
    random.Random(args.seed + len(records)).shuffle(remaining)
    print(f"Resume counts={counts}; need={len(remaining)}; available train shards={len(shards)}", flush=True)
    skipped_log = args.output_root / "skipped.jsonl"
    processed_shards = 0
    start = time.monotonic()

    for remote in shards:
        if not remaining:
            break
        if args.max_shards and processed_shards >= args.max_shards:
            break
        shard_name = Path(remote).name
        local = args.staging_root / shard_name
        if not local.exists():
            part = local.with_suffix(local.suffix + ".part")
            if part.exists():
                part.unlink()
            print(f"Downloading {shard_name}", flush=True)
            fs.get(remote, str(part))
            os.replace(part, local)
        try:
            raw_dataset = tf.data.TFRecordDataset([str(local)], num_parallel_reads=1)
            for raw in raw_dataset:
                if not remaining:
                    break
                try:
                    sample = builder.info.features.deserialize_example(raw)
                    sample = tf.nest.map_structure(
                        lambda x: x.numpy() if hasattr(x, "numpy") else x, sample)
                    video_id = scalar_string(sample["metadata"]["video_name"])
                    if video_id in known_ids:
                        continue
                    arrays = convert_sample(sample, shard_name, args.frames,
                                            args.max_objects)
                    frame_hash = scalar_string(arrays["rgb_frame0_sha256"])
                    if frame_hash in known_hashes:
                        raise ValueError("First RGB frame duplicates an existing sample")
                    split = remaining.pop()
                    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", video_id)
                    file_path = args.output_root / split / f"{safe_id}.npz"
                    if file_path.exists():
                        raise RuntimeError(f"Filename collision: {file_path}")
                    atomic_npz(file_path, arrays)
                    record = {"split": split, "file": str(file_path.relative_to(args.output_root)),
                              "video_id": video_id, "rgb_frame0_sha256": frame_hash,
                              "source_shard": shard_name}
                    records.append(record)
                    known_ids.add(video_id)
                    known_hashes.add(frame_hash)
                    counts[split] += 1
                    if len(records) % 25 == 0:
                        write_manifest(args.output_root, records, config)
                        print(f"Stored {len(records)}/{sum(targets.values())}: {counts}; "
                              f"elapsed {time.monotonic()-start:.0f}s", flush=True)
                except Exception as exc:
                    with skipped_log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"shard": shard_name, "error": repr(exc),
                                                 "traceback": traceback.format_exc()},
                                                ensure_ascii=False) + "\n")
                    print(f"Skipped one sample in {shard_name}: {exc}", file=sys.stderr,
                          flush=True)
        finally:
            # Only this dedicated staging copy is removed. Prepared NPZs are retained.
            local.unlink(missing_ok=True)
        processed_shards += 1
        write_manifest(args.output_root, records, config)
        print(f"Completed shard {processed_shards}: {counts}", flush=True)

    write_manifest(args.output_root, records, config)
    if remaining:
        raise RuntimeError(f"Incomplete dataset after {processed_shards} shards: {counts}; "
                           "rerun the same command to resume")
    final_records, ids, hashes = existing_records(args.output_root)
    actual = {s: sum(r["split"] == s for r in final_records) for s in SPLITS}
    if actual != targets or len(ids) != sum(targets.values()) or len(hashes) != len(ids):
        raise RuntimeError(f"Final integrity check failed: {actual}")
    print(f"COMPLETE: {actual}; unique videos={len(ids)}; "
          f"output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
