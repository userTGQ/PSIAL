#!/usr/bin/env python3
"""Evaluate a trained PSIAL checkpoint on a selected PEMS dataset.

The pattern library is selected automatically from --data:
    PEMS03 / PEMS03_24 / PEMS03_36 / PEMS03_48
        -> data/PEMS03/K32_L12_centers.npy
    PEMS04 / PEMS04_24 / PEMS04_36 / PEMS04_48
        -> data/PEMS04/K32_L12_centers.npy
    PEMS07 / PEMS07_24 / PEMS07_36 / PEMS07_48
        -> data/PEMS07/K32_L12_centers.npy
    PEMS08 / PEMS08_24 / PEMS08_36 / PEMS08_48
        -> data/PEMS08/K32_L12_centers.npy

The checkpoint must have been trained with the same pattern library and
compatible model configuration. The script checks the pattern buffers stored
in the checkpoint against the selected .npy file when those buffers exist.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import util
from model import PSIAL


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_CONFIG = {
    "PEMS03": {
        "num_nodes": 358,
        "pattern_file": "data/PEMS03/K32_L12_centers.npy",
    },
    "PEMS04": {
        "num_nodes": 307,
        "pattern_file": "data/PEMS04/K32_L12_centers.npy",
    },
    "PEMS07": {
        "num_nodes": 883,
        "pattern_file": "data/PEMS07/K32_L12_centers.npy",
    },
    "PEMS08": {
        "num_nodes": 170,
        "pattern_file": "data/PEMS08/K32_L12_centers.npy",
    },
}

OTHER_DATASET_NODES = {
    "bike_drop": 250,
    "bike_pick": 250,
    "taxi_drop": 266,
    "taxi_pick": 266,
}

SUPPORTED_SUFFIX_LENGTHS = {12, 24, 36, 48, 60}


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--data", type=str, default="PEMS08", help="dataset name")
parser.add_argument("--input_dim", type=int, default=3, help="input dimension")
parser.add_argument(
    "--channels", type=int, default=None,
    help="model channels; inferred from checkpoint when omitted",
)
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input length")
parser.add_argument("--output_len", type=int, default=12, help="prediction length")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--pattern_file",
    type=str,
    default=None,
    help=(
        "Optional pattern .npy path. If omitted, the path is selected "
        "automatically according to --data."
    ),
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./logs/2026-11-09-21-45-32-PEMS08/best_model.pth",
    help="model checkpoint path; must match --data and its training configuration",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./results",
    help="directory to save CSV files",
)
parser.add_argument(
    "--continuous_horizon",
    type=int,
    default=6,
    help=(
        "1-based forecast horizon used to build continuous CSV files "
        "(default: 6)"
    ),
)
args = parser.parse_args()


def resolve_dataset_config(data_name: str):
    """Resolve dataset folder, node count, sequence length suffix, and pattern file."""
    data_name = str(data_name).strip()
    if not data_name:
        raise ValueError("--data cannot be empty.")

    # Dataset names such as PEMS08_24 are split into base name and length suffix.
    parts = data_name.split("_")
    base_name = parts[0].upper()
    suffix = parts[-1] if len(parts) > 1 else None

    input_len = args.input_len
    output_len = args.output_len

    if base_name in DATASET_CONFIG:
        num_nodes = DATASET_CONFIG[base_name]["num_nodes"]

        if suffix is not None and suffix.isdigit():
            suffix_length = int(suffix)
            if suffix_length in SUPPORTED_SUFFIX_LENGTHS:
                input_len = suffix_length
                output_len = suffix_length

        if args.pattern_file:
            pattern_path = Path(args.pattern_file).expanduser()
            if not pattern_path.is_absolute():
                pattern_path = PROJECT_ROOT / pattern_path
        else:
            pattern_path = PROJECT_ROOT / DATASET_CONFIG[base_name]["pattern_file"]

    elif data_name.lower() in OTHER_DATASET_NODES:
        num_nodes = OTHER_DATASET_NODES[data_name.lower()]
        if not args.pattern_file:
            raise ValueError(
                f"{data_name} has no automatic pattern-file mapping. "
                "Please provide --pattern_file explicitly."
            )
        pattern_path = Path(args.pattern_file).expanduser()
        if not pattern_path.is_absolute():
            pattern_path = PROJECT_ROOT / pattern_path

    else:
        raise ValueError(
            f"Unsupported dataset: {data_name}. "
            f"Supported PEMS datasets: {list(DATASET_CONFIG)}."
        )

    pattern_path = pattern_path.resolve()
    if not pattern_path.is_file():
        raise FileNotFoundError(
            f"Pattern file not found: {pattern_path}\n"
            "Check the data/<DATASET>/K32_L12_centers.npy directory."
        )

    data_path = PROJECT_ROOT / "data" / data_name

    return {
        "data_name": data_name,
        "data_path": str(data_path),
        "base_name": base_name,
        "num_nodes": num_nodes,
        "input_len": input_len,
        "output_len": output_len,
        "pattern_file": str(pattern_path),
    }


def save_continuous_csv(values, csv_path, horizon_index=5):
    """Save one forecast horizon from every sliding window as a time series.

    values has shape [samples, nodes, horizons]. The default horizon_index=5
    selects the 6th predicted time step from every sample/window.
    """
    array = (
        values.detach().cpu().numpy()
        if torch.is_tensor(values)
        else np.asarray(values)
    )

    if array.ndim != 3:
        raise ValueError(
            f"Expected [samples, nodes, horizons], got shape {array.shape}"
        )
    if not 0 <= horizon_index < array.shape[2]:
        raise ValueError(
            f"horizon_index={horizon_index} is invalid for output length "
            f"{array.shape[2]}; expected 0 <= horizon_index < {array.shape[2]}."
        )

    # Select the same forecast horizon from EVERY sliding window.
    continuous = array[:, :, horizon_index]

    df = pd.DataFrame(
        continuous,
        columns=[f"node_{i}" for i in range(continuous.shape[1])],
    )
    df.insert(0, "time_idx", np.arange(len(df)))
    df.to_csv(csv_path, index=False, float_format="%.6f", encoding="utf-8-sig")
    return continuous.shape


def extract_state_dict(checkpoint):
    """Support a raw state_dict or a checkpoint containing 'state_dict'."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def remove_thop_statistics(state_dict):
    """Remove THOP profiling counters accidentally saved in older checkpoints."""
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected checkpoint state_dict to be a dict, got {type(state_dict)}."
        )

    cleaned = {}
    removed = []
    for key, value in state_dict.items():
        leaf_name = key.rsplit(".", 1)[-1]
        if leaf_name in {"total_ops", "total_params"}:
            removed.append(key)
        else:
            cleaned[key] = value

    if removed:
        print(
            f"Removed {len(removed)} THOP profiling entries "
            "(total_ops / total_params) from checkpoint."
        )
    return cleaned


def infer_channels_from_checkpoint(state_dict):
    """Infer the full model channel dimension from the shared input projection."""
    key = "shared_input_projection.weight"
    if key not in state_dict:
        raise KeyError(
            f"Cannot infer channels: checkpoint is missing '{key}'. "
            "Please pass --channels explicitly."
        )

    weight = state_dict[key]
    if not torch.is_tensor(weight) or weight.ndim != 4:
        raise ValueError(
            f"Unexpected shape for {key}: {getattr(weight, 'shape', None)}"
        )

    # PSIAL splits the full channel dimension equally between shared/private branches.
    return int(weight.shape[0]) * 2


def check_checkpoint_pattern(state_dict, model, pattern_file):
    """Check that checkpoint-embedded pattern buffers match the selected .npy."""
    key = "pattern_decomposition.pattern_features"

    # Older checkpoints may not contain the pattern buffer; load_state_dict
    # will report any other incompatibilities.
    if not isinstance(state_dict, dict) or key not in state_dict:
        print(
            "Warning: checkpoint does not contain the pattern buffer; "
            "the selected pattern file will initialize the model."
        )
        return

    checkpoint_patterns = state_dict[key].detach().cpu()
    selected_patterns = model.pattern_decomposition.pattern_features.detach().cpu()

    if checkpoint_patterns.shape != selected_patterns.shape:
        raise ValueError(
            "Pattern-library shape mismatch between checkpoint and --pattern_file.\n"
            f"Checkpoint shape: {tuple(checkpoint_patterns.shape)}\n"
            f"Selected file ({pattern_file}) shape: {tuple(selected_patterns.shape)}\n"
            "Use the pattern file that was used when training this checkpoint."
        )

    if not torch.allclose(
        checkpoint_patterns, selected_patterns, rtol=1e-5, atol=1e-6
    ):
        raise ValueError(
            "The pattern library stored in the checkpoint differs from the "
            "selected --pattern_file:\n"
            f"{pattern_file}\n"
            "Use the exact pattern file used for training this checkpoint. "
            "Do not mix checkpoints and pattern libraries from different datasets."
        )


def main():
    config = resolve_dataset_config(args.data)

    data_name = config["data_name"]
    data_path = config["data_path"]
    args.num_nodes = config["num_nodes"]
    args.input_len = config["input_len"]
    args.output_len = config["output_len"]
    pattern_file = config["pattern_file"]

    if not 1 <= args.continuous_horizon <= args.output_len:
        raise ValueError(
            f"--continuous_horizon must be between 1 and output_len "
            f"({args.output_len}), got {args.continuous_horizon}."
        )
    horizon_index = args.continuous_horizon - 1

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please pass the checkpoint trained on the selected dataset "
            "using --checkpoint."
        )

    device = torch.device(args.device)

    print(f"Dataset: {data_name}")
    print(f"Data path: {data_path}")
    print(f"Number of nodes: {args.num_nodes}")
    print(f"Input length: {args.input_len}")
    print(f"Output length: {args.output_len}")
    print(f"Pattern file: {pattern_file}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load the checkpoint first, then infer the trained channel dimension.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_thop_statistics(state_dict)

    checkpoint_channels = infer_channels_from_checkpoint(state_dict)

    # Validate key dimensions against the selected dataset and requested horizon.
    node_scale_key = "spatial_encoder2.node_channel_scale"
    if node_scale_key in state_dict:
        checkpoint_nodes = int(state_dict[node_scale_key].shape[2])
        if checkpoint_nodes != args.num_nodes:
            raise ValueError(
                f"Dataset {data_name} has {args.num_nodes} nodes, but the "
                f"checkpoint was trained with {checkpoint_nodes} nodes. "
                "Please use the checkpoint belonging to this dataset."
            )

    output_weight_key = "output_projection.weight"
    if output_weight_key in state_dict:
        checkpoint_output_len = int(state_dict[output_weight_key].shape[0])
        if checkpoint_output_len != args.output_len:
            raise ValueError(
                f"Requested output_len={args.output_len}, but the checkpoint "
                f"was trained with output_len={checkpoint_output_len}. "
                "Use the matching dataset/horizon checkpoint."
            )

    if args.channels is None:
        args.channels = checkpoint_channels
        print(f"Inferred channels from checkpoint: {args.channels}")
    elif args.channels != checkpoint_channels:
        raise ValueError(
            f"--channels={args.channels} does not match the checkpoint "
            f"configuration (channels={checkpoint_channels}). "
            f"Please omit --channels or set --channels {checkpoint_channels}."
        )

    model = PSIAL(
        device=device,
        input_dim=args.input_dim,
        channels=args.channels,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        dropout=args.dropout,
        pattern_file=pattern_file,
    ).to(device)

    check_checkpoint_pattern(state_dict, model, pattern_file)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully.")

    dataloader = util.load_dataset(
        data_path, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    outputs = []
    realy = torch.as_tensor(
        dataloader["y_test"], dtype=torch.float32, device=device
    )
    realy = realy.transpose(1, 3)[:, 0, :, :]  # [samples, nodes, output_len]

    with torch.no_grad():
        for x, _ in dataloader["test_loader"].get_iterator():
            testx = torch.as_tensor(x, dtype=torch.float32, device=device)
            testx = testx.transpose(1, 3)
            preds = model(testx).transpose(1, 3)  # [batch, 1, nodes, output_len]
            outputs.append(preds.squeeze(1))  # [batch, nodes, output_len]

    if not outputs:
        raise RuntimeError("The test loader is empty; no predictions were produced.")

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]
    yhat = scaler.inverse_transform(yhat)

    # Evaluate every forecast horizon.
    amae, amape, armse, awmape = [], [], [], []
    for horizon in range(args.output_len):
        pred = yhat[:, :, horizon]
        real = realy[:, :, horizon]
        metrics = util.metric(pred, real)

        print(
            "Evaluate test data for horizon {:d}, Test MAE: {:.4f}, "
            "Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}".format(
                horizon + 1, metrics[0], metrics[1], metrics[2], metrics[3]
            ),
            flush=True,
        )

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    print(
        "Average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, "
        "Test RMSE: {:.4f}, Test WMAPE: {:.4f}".format(
            args.output_len,
            np.mean(amae),
            np.mean(amape),
            np.mean(armse),
            np.mean(awmape),
        ),
        flush=True,
    )

    # Export the selected horizon from every sliding window.
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    real_csv = output_dir / f"{data_name}_real.csv"
    pred_csv = output_dir / f"{data_name}_pred.csv"

    real_shape = save_continuous_csv(
        realy, real_csv, horizon_index=horizon_index
    )
    pred_shape = save_continuous_csv(
        yhat, pred_csv, horizon_index=horizon_index
    )

    print(
        f"Continuous series uses forecast horizon {args.continuous_horizon} "
        "from every sliding window."
    )
    print(f"Continuous real series shape [time_steps, nodes]: {real_shape}")
    print(f"Continuous prediction series shape [time_steps, nodes]: {pred_shape}")
    print(f"Real values CSV saved to: {real_csv.resolve()}")
    print(f"Predictions CSV saved to: {pred_csv.resolve()}")


if __name__ == "__main__":
    main()
