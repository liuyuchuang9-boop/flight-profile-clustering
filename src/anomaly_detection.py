"""Abnormal mission detection using HDBSCAN or OPTICS.

This script is designed for damage-weighted physical feature tables.
Noise / abnormal missions are usually labeled as -1.
"""

from __future__ import annotations

import argparse

import hdbscan
import pandas as pd
from sklearn.cluster import OPTICS, DBSCAN

from utils import ensure_parent_dir, read_feature_table, standardize_features


def run_anomaly_detection(input_path: str, output_path: str, method: str = "hdbscan") -> None:
    ids, features = read_feature_table(input_path)
    x, _ = standardize_features(features)

    result = ids.copy()
    if result.empty:
        result["sample_index"] = range(len(features))

    if method == "hdbscan":
        model = hdbscan.HDBSCAN(min_cluster_size=max(5, len(features) // 50), prediction_data=True)
        labels = model.fit_predict(x)
        result["outlier_score"] = model.outlier_scores_
    elif method == "optics":
        model = OPTICS(min_samples=max(5, len(features) // 50), xi=0.05, min_cluster_size=0.05)
        labels = model.fit_predict(x)
        result["reachability"] = model.reachability_
    elif method == "dbscan":
        model = DBSCAN(eps=1.5, min_samples=max(5, len(features) // 50))
        labels = model.fit_predict(x)
    else:
        raise ValueError("method must be one of: hdbscan, optics, dbscan")

    result[f"{method}_label"] = labels
    result["is_noise_or_abnormal"] = labels == -1

    ensure_parent_dir(output_path)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved anomaly detection labels to: {output_path}")
    print("Number of abnormal/noise samples:", int((labels == -1).sum()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run abnormal mission detection.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX feature table.")
    parser.add_argument("--output", default="outputs/anomaly_labels.csv", help="Output CSV file.")
    parser.add_argument("--method", default="hdbscan", choices=["hdbscan", "optics", "dbscan"], help="Anomaly detection method.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_anomaly_detection(args.input, args.output, args.method)
