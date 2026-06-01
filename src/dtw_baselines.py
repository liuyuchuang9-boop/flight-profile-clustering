"""DTW / Soft-DTW time-series clustering baselines.

Input array format:
    series.npy with shape = (n_missions, sequence_length, n_variables)

Typical variables:
    altitude, Mach number, low-pressure rotor speed, high-pressure rotor speed,
    gas temperature, load parameter, damage proxy, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans


def find_nearest_to_barycenter(series: np.ndarray, labels: np.ndarray, barycenters: np.ndarray) -> dict[int, int]:
    """Find the nearest real mission to each barycenter using Euclidean distance.

    This is a simple representative-task constraint. For a stricter paper version,
    replace this part with DTW distance or a damage-weighted DTW distance.
    """
    representatives: dict[int, int] = {}
    for cluster_id in sorted(set(labels.tolist())):
        cluster_indices = np.where(labels == cluster_id)[0]
        center = barycenters[int(cluster_id)]
        distances = np.linalg.norm((series[cluster_indices] - center).reshape(len(cluster_indices), -1), axis=1)
        representatives[int(cluster_id)] = int(cluster_indices[int(np.argmin(distances))])
    return representatives


def run_dtw_baseline(input_path: str, output_dir: str, n_clusters: int = 4, metric: str = "dtw") -> None:
    series = np.load(input_path)
    if series.ndim != 3:
        raise ValueError("Input series must have shape (n_missions, sequence_length, n_variables).")

    if metric not in {"euclidean", "dtw", "softdtw"}:
        raise ValueError("metric must be one of: euclidean, dtw, softdtw")

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=metric,
        random_state=42,
        n_init=2,
        max_iter=20,
        verbose=True,
    )
    labels = model.fit_predict(series)
    barycenters = model.cluster_centers_
    representatives = find_nearest_to_barycenter(series, labels, barycenters)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_df = pd.DataFrame({
        "sample_index": np.arange(len(series)),
        f"timeseries_kmeans_{metric}_label": labels,
        "is_representative": False,
    })
    for sample_index in representatives.values():
        label_df.loc[sample_index, "is_representative"] = True

    label_df.to_csv(out_dir / f"labels_{metric}.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / f"barycenters_{metric}.npy", barycenters)

    print(f"Saved labels to: {out_dir / f'labels_{metric}.csv'}")
    print(f"Saved barycenters to: {out_dir / f'barycenters_{metric}.npy'}")
    print("Representative sample indices:", representatives)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DTW / Soft-DTW time-series clustering baselines.")
    parser.add_argument("--input", required=True, help="Input .npy array with shape (n_missions, sequence_length, n_variables).")
    parser.add_argument("--output-dir", default="outputs/dtw", help="Output directory.")
    parser.add_argument("--n-clusters", type=int, default=4, help="Cluster number.")
    parser.add_argument("--metric", default="dtw", choices=["euclidean", "dtw", "softdtw"], help="Distance metric.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dtw_baseline(args.input, args.output_dir, args.n_clusters, args.metric)
