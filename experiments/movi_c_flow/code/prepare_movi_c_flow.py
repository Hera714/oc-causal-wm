#!/usr/bin/env python3
"""
Build a compact, object-centric MOVi-C flow dataset.

Data source:
  gs://kubric-public/tfds/movi_c/128x128/1.0.0

Output for each scene:
  rgb                uint8    [12, 128, 128, 3]
  seg                uint8    [12, 128, 128]
  flow_fw            float16  [11, 128, 128, 2], frame t -> t+1
  flow_bw            float16  [11, 128, 128, 2]
  mask_center_px     float32  [M, 12, 2], (x, y), NaN if invisible
  mask_center_norm   float32  [M, 12, 2], normalized to [0, 1]
  visible            bool     [M, 12]
  mask_area          uint16   [M, 12]
  image_positions    float32  [M, 12, 2], original MOVi annotation
  world_positions    float32  [M, 12, 3]
  velocities         float32  [M, 12, 3]
  visibility_raw     uint16   [M, 12], original MOVi annotation
  object_valid       bool     [M]

Important:
  - M=10, matching MOVi-C's maximum object count.
  - Observed frames later: 0..3
  - Predicted frames later: 4..11
  - Observed flow later: 0->1, 1->2, 2->3
  - Future flow target later: 3->4 ... 10->11
  - Raw uint16 flow is dequantized per scene using forward_flow_range /
    backward_flow_range and saved as float16 pixel displacement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import gcsfs
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from fsspec.callbacks import TqdmCallback

REMOTE_ROOT = "kubric-public/tfds/movi_c/128x128/1.0.0"
SPLIT_OFFSETS = {"train": 0, "validation": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/autodl-fs/data/movi_c_flow_v1"),
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path("/autodl-fs/data/movi_c_probe"),
        help="Directory containing dataset_info.json and features.json.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        required=True,
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--min-visible-pixels", type=int, default=16)
    parser.add_argument(
        "--max-output-gb",
        type=float,
        default=10.0,
        help="Hard limit over all generated .npz files.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
        help="Stop before downloading another TFRecord if less space remains.",
    )
    return parser.parse_args()


def as_numpy_tree(value):
    return tf.nest.map_structure(
        lambda x: x.numpy() if hasattr(x, "numpy") else x,
        value,
    )


def decode_text(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def dequantize_flow(encoded: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    low, high = np.asarray(value_range, dtype=np.float32).reshape(2)
    decoded = encoded.astype(np.float32) / 65535.0
    decoded = decoded * (high - low) + low
    return decoded.astype(np.float16)


def compute_mask_geometry(
    seg: np.ndarray,
    num_objects: int,
    max_objects: int,
    min_visible_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      mask_area       [M, T]
      visible         [M, T]
      center_px       [M, T, 2]
      center_norm     [M, T, 2]
    """
    num_frames, height, width = seg.shape

    mask_area = np.zeros((max_objects, num_frames), dtype=np.uint16)
    visible = np.zeros((max_objects, num_frames), dtype=bool)
    center_px = np.full((max_objects, num_frames, 2), np.nan, dtype=np.float32)
    center_norm = np.full(
        (max_objects, num_frames, 2),
        np.nan,
        dtype=np.float32,
    )

    for object_index in range(num_objects):
        object_id = object_index + 1

        for frame_index in range(num_frames):
            ys, xs = np.nonzero(seg[frame_index] == object_id)
            area = len(xs)
            mask_area[object_index, frame_index] = area

            if area < min_visible_pixels:
                continue

            x_center = xs.mean(dtype=np.float64)
            y_center = ys.mean(dtype=np.float64)

            visible[object_index, frame_index] = True
            center_px[object_index, frame_index] = (x_center, y_center)
            center_norm[object_index, frame_index] = (
                x_center / (width - 1),
                y_center / (height - 1),
            )

    return mask_area, visible, center_px, center_norm


def pad_object_state(
    state: np.ndarray,
    max_objects: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    padded = np.full(
        (max_objects,) + state.shape[1:],
        fill_value,
        dtype=np.float32,
    )
    padded[: state.shape[0]] = state.astype(np.float32)
    return padded


def build_scene(
    sample: dict,
    frames: int,
    max_objects: int,
    min_visible_pixels: int,
) -> tuple[dict[str, np.ndarray], dict] | tuple[None, str]:
    rgb = sample["video"][:frames].astype(np.uint8)
    seg = sample["segmentations"][:frames, ..., 0].astype(np.uint8)

    instances = sample["instances"]
    num_objects = int(instances["positions"].shape[0])

    if num_objects > max_objects:
        return None, f"num_objects={num_objects} exceeds max_objects={max_objects}"

    if rgb.shape[0] != frames or seg.shape[0] != frames:
        return None, "insufficient frames"

    max_seg_id = int(seg.max())
    if max_seg_id > num_objects:
        return None, (
            f"segmentation has id={max_seg_id}, "
            f"but annotations have only {num_objects} objects"
        )

    # Keep transitions 0->1 through 10->11 for the 12-frame window.
    flow_fw = dequantize_flow(
        sample["forward_flow"][: frames - 1],
        sample["metadata"]["forward_flow_range"],
    )
    flow_bw = dequantize_flow(
        sample["backward_flow"][: frames - 1],
        sample["metadata"]["backward_flow_range"],
    )

    mask_area, visible, center_px, center_norm = compute_mask_geometry(
        seg=seg,
        num_objects=num_objects,
        max_objects=max_objects,
        min_visible_pixels=min_visible_pixels,
    )

    image_positions = pad_object_state(
        instances["image_positions"][:num_objects, :frames],
        max_objects=max_objects,
    )
    world_positions = pad_object_state(
        instances["positions"][:num_objects, :frames],
        max_objects=max_objects,
    )
    velocities = pad_object_state(
        instances["velocities"][:num_objects, :frames],
        max_objects=max_objects,
    )

    visibility_raw = np.zeros((max_objects, frames), dtype=np.uint16)
    visibility_raw[:num_objects] = instances["visibility"][
        :num_objects, :frames
    ].astype(np.uint16)

    object_valid = np.zeros((max_objects,), dtype=bool)
    object_valid[:num_objects] = True

    video_id = decode_text(sample["metadata"]["video_name"])

    arrays = {
        "rgb": rgb,
        "seg": seg,
        "flow_fw": flow_fw,
        "flow_bw": flow_bw,
        "mask_center_px": center_px,
        "mask_center_norm": center_norm,
        "visible": visible,
        "mask_area": mask_area,
        "image_positions": image_positions,
        "world_positions": world_positions,
        "velocities": velocities,
        "visibility_raw": visibility_raw,
        "object_valid": object_valid,
        "frame_ids": np.arange(frames, dtype=np.uint8),
    }

    metadata = {
        "video_id": video_id,
        "num_objects": num_objects,
        "height": int(rgb.shape[1]),
        "width": int(rgb.shape[2]),
        "visible_object_frames": int(visible[:num_objects].sum()),
    }
    return arrays, metadata


def get_shards(fs: gcsfs.GCSFileSystem, split: str) -> list[str]:
    prefix = f"{REMOTE_ROOT}/movi_c-{split}.tfrecord-"
    paths = fs.find(REMOTE_ROOT, detail=False)
    shards = sorted(path for path in paths if path.startswith(prefix))

    if not shards:
        raise RuntimeError(f"No TFRecord shards found for split={split}")

    return shards


def total_output_bytes(output_root: Path) -> int:
    return sum(path.stat().st_size for path in output_root.rglob("*.npz"))


def write_manifest(output_root: Path, args: argparse.Namespace) -> None:
    manifest_path = output_root / "dataset_manifest.json"
    if manifest_path.exists():
        return

    manifest = {
        "dataset": "MOVi-C 128x128",
        "source": f"gs://{REMOTE_ROOT}",
        "version": "flow_v1",
        "frames_saved": args.frames,
        "max_objects": args.max_objects,
        "min_visible_pixels": args.min_visible_pixels,
        "flow_storage": "float16, dequantized pixel displacement",
        "flow_alignment": {
            "observed_flow": ["0->1", "1->2", "2->3"],
            "future_flow_target": ["3->4", "...", "10->11"],
        },
        "position_target_recommendation": (
            "Use mask_center_norm with visible as the primary first-round "
            "position target and loss mask. image_positions is retained "
            "for later annotation-based experiments."
        ),
        "collision_note": (
            "Collision fields are intentionally omitted in v1 because "
            "the decoded TFDS event tensors use fixed-length padding."
        ),
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temp_path = path.with_suffix(".npz.tmp")

    with temp_path.open("wb") as f:
        np.savez(f, **arrays)

    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    if args.frames != 12:
        raise ValueError(
            "This v1 protocol is defined for exactly 12 saved frames."
        )

    if args.max_objects != 10:
        raise ValueError(
            "MOVi-C v1 is fixed to max_objects=10 for Route B compatibility."
        )

    if not (args.schema_dir / "dataset_info.json").exists():
        raise FileNotFoundError(
            f"Missing {(args.schema_dir / 'dataset_info.json')}"
        )

    if not (args.schema_dir / "features.json").exists():
        raise FileNotFoundError(
            f"Missing {(args.schema_dir / 'features.json')}"
        )

    output_root = args.output_root
    split_dir = output_root / args.split
    temp_dir = output_root / "_tmp"

    split_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_root, args)

    # This builder is local: TensorFlow never contacts Google Cloud.
    builder = tfds.builder_from_directory(str(args.schema_dir))

    fs = gcsfs.GCSFileSystem(token="anon")
    shards = get_shards(fs, args.split)

    rng = np.random.default_rng(args.seed + SPLIT_OFFSETS[args.split])
    shard_order = rng.permutation(len(shards))

    existing_files = {path.stem for path in split_dir.glob("*.npz")}
    completed = len(existing_files)

    if completed >= args.num_samples:
        print(
            f"{args.split}: already has {completed} samples, "
            f"target is {args.num_samples}; nothing to do."
        )
        return

    budget_bytes = int(args.max_output_gb * 1024**3)
    min_free_bytes = int(args.min_free_gb * 1024**3)
    output_bytes = total_output_bytes(output_root)

    index_path = output_root / f"index_{args.split}.jsonl"
    skipped = 0

    print(f"split={args.split}")
    print(f"available shards={len(shards)}")
    print(f"existing samples={completed}")
    print(f"target samples={args.num_samples}")
    print(f"existing output={output_bytes / 1024**3:.2f} GB")
    print(f"hard output limit={args.max_output_gb:.2f} GB")

    progress = tqdm(
        total=args.num_samples,
        initial=completed,
        desc=f"build {args.split}",
        unit="scene",
    )

    try:
        for shard_index in shard_order:
            if completed >= args.num_samples:
                break

            free_bytes = shutil.disk_usage(output_root).free
            if free_bytes < min_free_bytes:
                raise RuntimeError(
                    f"Only {free_bytes / 1024**3:.2f} GB free; "
                    f"minimum allowed is {args.min_free_gb:.2f} GB."
                )

            remote_shard = shards[int(shard_index)]
            shard_name = Path(remote_shard).name
            local_shard = temp_dir / shard_name

            if local_shard.exists():
                local_shard.unlink()

            print(f"\nDownloading temporary shard: {shard_name}")
            download_progress = TqdmCallback(
                tqdm_kwargs={
                    "desc": f"download {shard_name}",
                    "unit": "B",
                    "unit_scale": True,
                    "unit_divisor": 1024,
                    "leave": False,
                }
            )
            
            fs.get(
                remote_shard,
                str(local_shard),
                callback=download_progress,
            )

            try:
                raw_records = list(
                    tf.data.TFRecordDataset([str(local_shard)]).as_numpy_iterator()
                )

                record_order = rng.permutation(len(raw_records))

                for record_index in record_order:
                    if completed >= args.num_samples:
                        break

                    raw = tf.convert_to_tensor(raw_records[int(record_index)])
                    sample = as_numpy_tree(
                        builder.info.features.deserialize_example(raw)
                    )

                    arrays, metadata_or_reason = build_scene(
                        sample=sample,
                        frames=args.frames,
                        max_objects=args.max_objects,
                        min_visible_pixels=args.min_visible_pixels,
                    )

                    if arrays is None:
                        skipped += 1
                        print(f"Skip sample: {metadata_or_reason}")
                        continue

                    metadata = metadata_or_reason
                    video_id = metadata["video_id"]
                    filename = f"{safe_name(video_id)}.npz"
                    output_path = split_dir / filename

                    if output_path.stem in existing_files:
                        continue

                    estimated_bytes = sum(
                        value.nbytes
                        for value in arrays.values()
                        if isinstance(value, np.ndarray)
                    )

                    if output_bytes + estimated_bytes > budget_bytes:
                        raise RuntimeError(
                            "The next sample would exceed --max-output-gb. "
                            "No additional scene was written."
                        )

                    atomic_save_npz(output_path, arrays)

                    actual_bytes = output_path.stat().st_size
                    output_bytes += actual_bytes
                    completed += 1
                    existing_files.add(output_path.stem)

                    index_record = {
                        "split": args.split,
                        "file": str(output_path.relative_to(output_root)),
                        "source_shard": shard_name,
                        **metadata,
                        "bytes": actual_bytes,
                    }

                    with index_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(index_record, ensure_ascii=False) + "\n")

                    progress.update(1)
                    progress.set_postfix(
                        output_gb=f"{output_bytes / 1024**3:.2f}",
                        skipped=skipped,
                    )
            finally:
                local_shard.unlink(missing_ok=True)
    finally:
        progress.close()

    if completed < args.num_samples:
        raise RuntimeError(
            f"Only built {completed}/{args.num_samples} samples. "
            "Check the messages above."
        )

    print("\nFinished.")
    print(f"split={args.split}")
    print(f"samples={completed}")
    print(f"skipped={skipped}")
    print(f"total .npz size={output_bytes / 1024**3:.2f} GB")
    print(f"index={index_path}")


if __name__ == "__main__":
    main()