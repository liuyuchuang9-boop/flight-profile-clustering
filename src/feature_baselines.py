"""Feature-based clustering baselines for flight mission profiles.

This script is intended for task-level physical / damage features, such as:
- mission duration
- phase ratios
- speed / temperature / altitude / Mach statistics
- exceedance duration
- rainflow cycle statistics
- Miner damage
- equivalent load
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hdbscan
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from utils import ensure_parent_dir, nearest_real_samples, read_feature_table, standardize_features


def run_feature_baselines(input_path: str, output_path: str, n_clusters: int = 4) -> None:
    ids, features = read_feature_table(input_path)
    x, _ = standardize_features(features)

    result = ids.copy()
    if result.empty:
        result["sample_index"] = range(len(features))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    result["kmeans_label"] = kmeans.fit_predict(x)

    minibatch = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init="auto", batch_size=256)
    result["minibatch_kmeans_label"] = minibatch.fit_predict(x)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    result["gmm_label"] = gmm.fit_predict(x)
    result["gmm_log_likelihood"] = gmm.score_samples(x)

    hdb = hdbscan.HDBSCAN(min_cluster_size=max(5, len(features) // 50), prediction_data=True)
    result["hdbscan_label"] = hdb.fit_predict(x)
    result["hdbscan_outlier_score"] = hdb.outlier_scores_

    representatives = nearest_real_samples(x, result["kmeans_label"].to_numpy(), kmeans.cluster_centers_)
    result["is_kmeans_representative"] = False
    for sample_index in representatives.values():
        result.loc[sample_index, "is_kmeans_representative"] = True

    ensure_parent_dir(output_path)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved feature clustering labels to: {output_path}")
    print("KMeans representative sample indices:", representatives)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-based clustering baselines.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX feature table.")
    parser.add_argument("--output", default="outputs/feature_labels.csv", help="Output label CSV file.")
    parser.add_argument("--n-clusters", type=int, default=4, help="Cluster number for KMeans/GMM.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_baselines(args.input, args.output, args.n_clusters)
