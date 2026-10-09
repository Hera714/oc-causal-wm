#!/usr/bin/env python3
"""
WM-FLOW-001-mini

baseline:
    observed object mask centers + visibility -> future centers

flow:
    observed object mask centers + visibility + pooled observed optical flow
    -> future centers

This is a GT-geometry / GT-flow diagnostic pilot, not an RGB-only experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


OBS_FRAMES = 4
PRED_FRAMES = 8
MAX_OBJECTS = 10


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
        default=Path("/root/autodl-tmp/oc-causal-wm/out/movi_c_flow/mini/movi_c_flow_mini"),
    )
    parser.add_argument("--variant", choices=["baseline", "flow"], required=True)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pooled_object_flow(
    seg: np.ndarray,
    flow_fw: np.ndarray,
    object_valid: np.ndarray,
    min_pixels: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool dense flow inside each source-frame object mask.

    flow[t] maps frame t -> t+1.
    We only use t=0,1,2 because those pairs lie fully inside observations.
    """
    flow = np.zeros((MAX_OBJECTS, OBS_FRAMES - 1, 2), dtype=np.float32)
    valid = np.zeros((MAX_OBJECTS, OBS_FRAMES - 1), dtype=np.float32)

    num_objects = int(object_valid.sum())

    for t in range(OBS_FRAMES - 1):
        for object_index in range(num_objects):
            mask = seg[t] == (object_index + 1)

            if int(mask.sum()) < min_pixels:
                continue

            flow[object_index, t] = flow_fw[t][mask].mean(axis=0) / 127.0
            valid[object_index, t] = 1.0

    return flow, valid


class MoviMiniDataset(Dataset):
    def __init__(self, data_root: Path, split: str):
        self.files = sorted((data_root / split).glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_root / split}")

        self.samples = [self._load_sample(path) for path in self.files]
        print(f"Loaded {len(self.samples)} {split} samples.")

    @staticmethod
    def _load_sample(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            centers_raw = data["mask_center_norm"].astype(np.float32)
            visible = data["visible"].astype(bool)
            object_valid = data["object_valid"].astype(bool)
            seg = data["seg"].astype(np.uint8)
            flow_fw = data["flow_fw"].astype(np.float32)

        obs_centers = centers_raw[:, :OBS_FRAMES]
        future_centers = centers_raw[
            :, OBS_FRAMES : OBS_FRAMES + PRED_FRAMES
        ]

        obs_visible = visible[:, :OBS_FRAMES]
        future_visible = visible[
            :, OBS_FRAMES : OBS_FRAMES + PRED_FRAMES
        ]

        future_valid = (
            future_visible
            & object_valid[:, None]
            & np.isfinite(future_centers).all(axis=-1)
        )

        object_flow, flow_valid = pooled_object_flow(
            seg=seg,
            flow_fw=flow_fw,
            object_valid=object_valid,
        )

        return {
            "obs_centers": np.nan_to_num(
                obs_centers,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32),
            "obs_visible": obs_visible.astype(np.float32),
            "obs_flow": object_flow.astype(np.float32),
            "obs_flow_valid": flow_valid.astype(np.float32),
            "target_centers": np.nan_to_num(
                future_centers,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32),
            "target_valid": future_valid.astype(bool),
            "object_valid": object_valid.astype(bool),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {key: torch.from_numpy(value) for key, value in sample.items()}


class ObjectFuturePredictor(nn.Module):
    def __init__(
        self,
        use_flow: bool,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.use_flow = use_flow

        # centers: 4 x 2, observed visibility: 4
        input_dim = OBS_FRAMES * 2 + OBS_FRAMES

        if use_flow:
            # three 2D pooled-flow vectors + three validity flags
            input_dim += (OBS_FRAMES - 1) * 2 + (OBS_FRAMES - 1)

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.object_embedding = nn.Embedding(MAX_OBJECTS, d_model)
        self.horizon_embedding = nn.Embedding(PRED_FRAMES, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, num_objects, _, _ = batch["obs_centers"].shape

        pieces = [
            batch["obs_centers"].reshape(batch_size, num_objects, -1),
            batch["obs_visible"].reshape(batch_size, num_objects, -1),
        ]

        if self.use_flow:
            pieces.extend(
                [
                    batch["obs_flow"].reshape(batch_size, num_objects, -1),
                    batch["obs_flow_valid"].reshape(batch_size, num_objects, -1),
                ]
            )

        features = torch.cat(pieces, dim=-1)
        tokens = self.input_projection(features)

        object_ids = torch.arange(
            num_objects,
            device=tokens.device,
        )
        tokens = tokens + self.object_embedding(object_ids)[None]

        encoded = self.encoder(
            tokens,
            src_key_padding_mask=~batch["object_valid"],
        )

        horizon_ids = torch.arange(PRED_FRAMES, device=tokens.device)
        horizon = self.horizon_embedding(horizon_ids)[None, None]

        decoded = encoded[:, :, None, :] + horizon
        return self.output_head(decoded)


def masked_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns sum squared error, sum absolute error, and valid scalar count.
    """
    valid = valid.unsqueeze(-1).float()
    difference = prediction - target

    squared_error = (difference.square() * valid).sum()
    absolute_error = (difference.abs() * valid).sum()
    count = valid.sum() * prediction.shape[-1]

    return squared_error, absolute_error, count.clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    total_sse = 0.0
    total_sae = 0.0
    total_count = 0.0

    horizon_sse = np.zeros(PRED_FRAMES, dtype=np.float64)
    horizon_count = np.zeros(PRED_FRAMES, dtype=np.float64)

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(batch)

        sse, sae, count = masked_sums(
            prediction,
            batch["target_centers"],
            batch["target_valid"],
        )

        total_sse += sse.item()
        total_sae += sae.item()
        total_count += count.item()

        for horizon in range(PRED_FRAMES):
            horizon_valid = batch["target_valid"][:, :, horizon]
            horizon_prediction = prediction[:, :, horizon]
            horizon_target = batch["target_centers"][:, :, horizon]

            h_sse, _, h_count = masked_sums(
                horizon_prediction[:, :, None, :],
                horizon_target[:, :, None, :],
                horizon_valid[:, :, None],
            )

            horizon_sse[horizon] += h_sse.item()
            horizon_count[horizon] += h_count.item()

    mse = total_sse / total_count
    mae = total_sae / total_count

    per_horizon_mse = (
        horizon_sse / np.maximum(horizon_count, 1.0)
    ).tolist()

    return {
        "center_mse_norm": mse,
        "center_mae_norm": mae,
        "center_rmse_px": (mse**0.5) * 127.0,
        "center_mae_px": mae * 127.0,
        "per_horizon_mse_norm": per_horizon_mse,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ObjectFuturePredictor(
        use_flow=args.variant == "flow",
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

    run_dir = args.output_dir / f"{args.variant}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.jsonl"

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["output_dir"] = str(args.output_dir)
    config["parameter_count"] = parameter_count(model)
    config["torch_version"] = torch.__version__
    config["device_name"] = (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else "cpu"
    )

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"device: {config['device_name']}")
    print(f"parameters: {config['parameter_count']:,}")
    print(f"variant: {args.variant}")

    best_validation_mse = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_sse = 0.0
        train_sae = 0.0
        train_count = 0.0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            prediction = model(batch)
            sse, sae, count = masked_sums(
                prediction,
                batch["target_centers"],
                batch["target_valid"],
            )

            loss = sse / count
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sse += sse.item()
            train_sae += sae.item()
            train_count += count.item()

        train_metrics = {
            "center_mse_norm": train_sse / train_count,
            "center_mae_norm": train_sae / train_count,
        }
        validation_metrics = evaluate(model, validation_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }

        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record) + "\n")

        print(
            f"epoch={epoch:03d} "
            f"train_mse={train_metrics['center_mse_norm']:.6f} "
            f"val_mse={validation_metrics['center_mse_norm']:.6f} "
            f"val_mae_px={validation_metrics['center_mae_px']:.3f}"
        )

        if validation_metrics["center_mse_norm"] < best_validation_mse:
            best_validation_mse = validation_metrics["center_mse_norm"]
            patience_counter = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation_metrics,
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

    best_validation = evaluate(model, validation_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    final_metrics = {
        "variant": args.variant,
        "seed": args.seed,
        "best_epoch": checkpoint["epoch"],
        "best_validation": best_validation,
        "test": test_metrics,
        "protocol": (
            "GT mask-center / GT pooled-flow diagnostic; "
            "not an RGB-only experiment."
        ),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("\n=== final result ===")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(f"saved to: {run_dir}")


if __name__ == "__main__":
    main()