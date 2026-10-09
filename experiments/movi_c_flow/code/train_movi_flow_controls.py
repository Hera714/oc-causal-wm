#!/usr/bin/env python3
"""
Controls for WM-FLOW-001-mini.

Requires train_movi_flow_mini.py in the same directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_movi_flow_mini import (
    MoviMiniDataset,
    ObjectFuturePredictor,
    evaluate,
    masked_sums,
    parameter_count,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/autodl-fs/data/movi_c_flow_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/oc-causal-wm/out/movi_c_flow/controls/movi_c_flow_controls"),
    )
    parser.add_argument(
        "--flow-mode",
        choices=["zero", "shuffled"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation with no sample mapped to itself."""
    if size < 2:
        raise ValueError("Shuffled-flow control needs at least two samples.")

    identity = np.arange(size)

    for _ in range(100):
        permutation = rng.permutation(size)
        if not np.any(permutation == identity):
            return permutation

    # Guaranteed fallback.
    return np.roll(identity, 1)


def apply_flow_control(
    dataset: MoviMiniDataset,
    mode: str,
    seed: int,
) -> None:
    """
    zero:
      Keep obs_flow_valid unchanged; erase only flow vector values.
      This preserves architecture and validity-mask information.

    shuffled:
      Move flow values and corresponding flow-valid masks to another video.
      Distribution is preserved but video-flow alignment is destroyed.
    """
    if mode == "zero":
        for sample in dataset.samples:
            sample["obs_flow"].fill(0.0)
        return

    rng = np.random.default_rng(seed)
    permutation = derangement(len(dataset.samples), rng)

    source_flow = [
        sample["obs_flow"].copy()
        for sample in dataset.samples
    ]
    source_valid = [
        sample["obs_flow_valid"].copy()
        for sample in dataset.samples
    ]

    for destination_index, source_index in enumerate(permutation):
        dataset.samples[destination_index]["obs_flow"] = source_flow[source_index]
        dataset.samples[destination_index]["obs_flow_valid"] = source_valid[
            source_index
        ]


def make_loader(
    dataset: MoviMiniDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    train_data = MoviMiniDataset(args.data_root, "train")
    validation_data = MoviMiniDataset(args.data_root, "validation")
    test_data = MoviMiniDataset(args.data_root, "test")

    # Different deterministic permutations for each split.
    apply_flow_control(train_data, args.flow_mode, args.seed + 100)
    apply_flow_control(validation_data, args.flow_mode, args.seed + 200)
    apply_flow_control(test_data, args.flow_mode, args.seed + 300)

    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    validation_loader = make_loader(
        validation_data,
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(test_data, args.batch_size, shuffle=False)

    model = ObjectFuturePredictor(
        use_flow=True,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = args.output_dir / f"{args.flow_mode}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.jsonl"

    best_validation_mse = float("inf")
    patience_counter = 0

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"flow control: {args.flow_mode}")
    print(f"parameters: {parameter_count(model):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_sse = 0.0
        train_count = 0.0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)

            sse, _, count = masked_sums(
                prediction,
                batch["target_centers"],
                batch["target_valid"],
            )

            loss = sse / count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sse += sse.item()
            train_count += count.item()

        validation = evaluate(model, validation_loader, device)
        train_mse = train_sse / train_count

        record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation": validation,
        }

        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"epoch={epoch:03d} "
            f"train_mse={train_mse:.6f} "
            f"val_mse={validation['center_mse_norm']:.6f} "
            f"val_mae_px={validation['center_mae_px']:.3f}"
        )

        if validation["center_mse_norm"] < best_validation_mse:
            best_validation_mse = validation["center_mse_norm"]
            patience_counter = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                },
                best_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    metrics = {
        "variant": "flow_control",
        "flow_mode": args.flow_mode,
        "seed": args.seed,
        "best_epoch": checkpoint["epoch"],
        "best_validation": evaluate(model, validation_loader, device),
        "test": evaluate(model, test_loader, device),
        "control_definition": (
            "zero: flow vectors zeroed while flow-valid masks remain; "
            "shuffled: flow vectors and flow-valid masks are moved across "
            "videos using a deterministic derangement."
        ),
        "protocol": (
            "GT mask-center / GT pooled-flow diagnostic; "
            "not an RGB-only experiment."
        ),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== final result ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()