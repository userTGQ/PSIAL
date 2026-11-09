#!/usr/bin/env python3
"""Cluster 12-step traffic-flow windows into K-shape traffic patterns.

Expected train.npz array shape:
    [num_samples, history_len, num_nodes, num_features]
Example:
    [10699, 12, 170, 3]

Feature index 0 is treated as traffic flow.

Outputs:
    <prefix>_centers.npy      normalized K-shape centroids, shape [K, 12]
    <prefix>_result.npz       centers + labels + sampled indices + counts
    <prefix>_patterns.png     4x4 plot for K=16

Example:
    python kshape_cluster_traffic.py \
        --input train.npz \
        --clusters 16 \
        --samples-per-node 1000 \
        --output-prefix pems08_kshape_k16_l12

Use all 1,818,830 candidate sequences (potentially very slow / memory heavy):
    python kshape_cluster_traffic.py --input train.npz --clusters 16 --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tslearn.clustering import KShape
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
except ImportError as exc:
    raise SystemExit(
        "tslearn is required. Install it first with:\n"
        "    pip install tslearn\n"
    ) from exc


def load_npz_array(path: Path, key: str | None) -> tuple[np.ndarray, str]:
    """Load a 4-D training array from an NPZ file."""
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if not keys:
            raise ValueError(f"No arrays found in {path}.")

        if key is not None:
            if key not in archive:
                raise KeyError(
                    f"Key {key!r} not found. Available keys: {keys}"
                )
            array = archive[key]
            return array, key

        # Common key names first.
        for candidate in ("data", "x_train", "x", "train"):
            if candidate in archive:
                return archive[candidate], candidate

        # Otherwise use the only key, or fail clearly if ambiguous.
        if len(keys) == 1:
            return archive[keys[0]], keys[0]

        raise ValueError(
            "The NPZ contains multiple arrays. Pass --key explicitly. "
            f"Available keys: {keys}"
        )


def build_candidates(
    x_train: np.ndarray,
    flow_feature: int,
    samples_per_node: int | None,
    random_state: int,
    std_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build [num_series, time, 1] K-shape input.

    Sampling is performed *per node* so every road contributes the same
    number of candidate windows.

    Returns:
        candidates: [M, T, 1]
        sample_indices: [M] original training-window index for each series
        node_indices: [M] original node index for each series
    """
    if x_train.ndim != 4:
        raise ValueError(
            "Expected train array shape [samples, time, nodes, features], "
            f"but got {x_train.shape}."
        )

    num_samples, history_len, num_nodes, num_features = x_train.shape
    if not (0 <= flow_feature < num_features):
        raise ValueError(
            f"flow_feature={flow_feature} is invalid for {num_features} features."
        )

    # [samples, time, nodes]
    flow = np.asarray(x_train[..., flow_feature], dtype=np.float32)

    rng = np.random.default_rng(random_state)

    if samples_per_node is None or samples_per_node >= num_samples:
        selected_per_node = [np.arange(num_samples) for _ in range(num_nodes)]
    else:
        selected_per_node = [
            np.sort(
                rng.choice(
                    num_samples,
                    size=samples_per_node,
                    replace=False,
                )
            )
            for _ in range(num_nodes)
        ]

    series_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    node_parts: list[np.ndarray] = []

    for node_id, sample_ids in enumerate(selected_per_node):
        # [m, T]
        node_series = flow[sample_ids, :, node_id]

        # Remove NaN/Inf and essentially constant sequences. K-shape is based
        # on normalized shape, so zero-variance sequences are uninformative.
        finite_mask = np.isfinite(node_series).all(axis=1)
        std_mask = node_series.std(axis=1) > std_eps
        keep = finite_mask & std_mask

        if not np.any(keep):
            continue

        node_series = node_series[keep]
        kept_sample_ids = sample_ids[keep]

        series_parts.append(node_series)
        sample_parts.append(kept_sample_ids.astype(np.int32, copy=False))
        node_parts.append(
            np.full(node_series.shape[0], node_id, dtype=np.int16)
        )

    if not series_parts:
        raise ValueError("No valid non-constant traffic sequences remain.")

    # [M, T] -> [M, T, 1]
    candidates_2d = np.concatenate(series_parts, axis=0)
    candidates = candidates_2d[..., None]
    sample_indices = np.concatenate(sample_parts, axis=0)
    node_indices = np.concatenate(node_parts, axis=0)

    return candidates, sample_indices, node_indices


def plot_centers(centers: np.ndarray, output_path: Path) -> None:
    """Save a compact plot of all learned K-shape patterns."""
    k, length = centers.shape
    cols = 4
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), squeeze=False)
    x = np.arange(length)

    for pattern_id in range(rows * cols):
        ax = axes.flat[pattern_id]
        if pattern_id < k:
            ax.plot(x, centers[pattern_id], linewidth=1.8)
            ax.set_title(f"Pattern {pattern_id}")
            ax.set_xlim(0, length - 1)
            ax.grid(alpha=0.25)
        else:
            ax.axis("off")

    fig.suptitle(f"K-shape Traffic Patterns (K={k}, Length={length})")
    fig.supxlabel("History step")
    fig.supylabel("Z-normalized traffic flow")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("PEMS08/train.npz"))
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="NPZ array key. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--flow-feature",
        type=int,
        default=0,
        help="Feature index containing traffic flow (default: 0).",
    )
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--samples-per-node",
        type=int,
        default=1000,
        help=(
            "Number of windows sampled independently from each node. "
            "Default 1000 => about 170k sequences for 170 nodes."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use every valid sample from every node; overrides --samples-per-node.",
    )
    parser.add_argument("--std-eps", type=float, default=1e-6)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("K32_L12"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input.resolve()}")

    x_train, used_key = load_npz_array(args.input, args.key)
    print(f"Loaded: {args.input}")
    print(f"NPZ key: {used_key}")
    print(f"train shape: {x_train.shape}")

    if x_train.ndim != 4:
        raise ValueError(
            "Expected train.npz data shape [samples, time, nodes, features]."
        )

    expected_candidates = x_train.shape[0] * x_train.shape[2]
    print(f"All possible node-window sequences: {expected_candidates:,}")

    samples_per_node = None if args.all else args.samples_per_node
    candidates, sample_indices, node_indices = build_candidates(
        x_train=x_train,
        flow_feature=args.flow_feature,
        samples_per_node=samples_per_node,
        random_state=args.random_state,
        std_eps=args.std_eps,
    )

    print(f"K-shape candidate shape before normalization: {candidates.shape}")
    print(f"Candidate sequences used: {candidates.shape[0]:,}")

    # K-shape is shape-based: normalize each individual time series to zero
    # mean and unit standard deviation before clustering.
    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
    candidates_z = scaler.fit_transform(candidates)

    print("Clustering with K-shape...")
    model = KShape(
        n_clusters=args.clusters,
        max_iter=args.max_iter,
        n_init=args.n_init,
        random_state=args.random_state,
        verbose=True,
    )
    labels = model.fit_predict(candidates_z)

    # tslearn: [K, T, 1] -> save convenient [K, T]
    centers = np.asarray(model.cluster_centers_, dtype=np.float32).squeeze(-1)
    if centers.ndim != 2:
        raise RuntimeError(
            f"Unexpected cluster-center shape: {model.cluster_centers_.shape}"
        )

    counts = np.bincount(labels, minlength=args.clusters)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    centers_path = Path(f"{prefix}_centers.npy")
    result_path = Path(f"{prefix}_result.npz")
    figure_path = Path(f"{prefix}_patterns.png")

    np.save(centers_path, centers)
    np.savez_compressed(
        result_path,
        centers=centers,
        labels=labels.astype(np.int16),
        cluster_counts=counts.astype(np.int64),
        sample_indices=sample_indices,
        node_indices=node_indices,
        source_shape=np.asarray(x_train.shape, dtype=np.int64),
        flow_feature=np.asarray(args.flow_feature, dtype=np.int64),
        random_state=np.asarray(args.random_state, dtype=np.int64),
    )
    plot_centers(centers, figure_path)

    print("\nDone.")
    print(f"Cluster centers shape: {centers.shape}")
    print(f"Cluster counts: {counts.tolist()}")
    print(f"Saved centers: {centers_path}")
    print(f"Saved full result: {result_path}")
    print(f"Saved plot: {figure_path}")


if __name__ == "__main__":
    main()
