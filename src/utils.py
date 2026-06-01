"""Utility functions for flight mission profile clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory of a file path if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_feature_table(path: str | Path, id_columns: Iterable[str] = ("mission_id",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a feature table and split identifier columns from numeric feature columns.

    Parameters
    ----------
    path:
        CSV or Excel file path.
    id_columns:
        Columns preserved as identifiers rather than clustering features.

    Returns
    -------
    ids:
        Identifier columns available in the input table.
    features:
        Numeric feature matrix as a pandas DataFrame.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    existing_id_columns = [c for c in id_columns if c in df.columns]
    ids = df[existing_id_columns].copy() if existing_id_columns else pd.DataFrame(index=df.index)

    features = df.drop(columns=existing_id_columns, errors="ignore")
    features = features.select_dtypes(include=[np.number]).copy()

    if features.empty:
        raise ValueError("No numeric feature columns were found in the input table.")

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    return ids, features


def standardize_features(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize feature columns before distance-based clustering."""
    scaler = StandardScaler()
    x = scaler.fit_transform(features.to_numpy(dtype=float))
    return x, scaler


def nearest_real_samples(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> dict[int, int]:
    """Find the nearest real sample index for each cluster center.

    Noise label -1 is ignored.
    """
    representatives: dict[int, int] = {}
    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == -1:
            continue
        cluster_indices = np.where(labels == cluster_id)[0]
        center = centers[int(cluster_id)]
        distances = np.linalg.norm(x[cluster_indices] - center, axis=1)
        representatives[int(cluster_id)] = int(cluster_indices[int(np.argmin(distances))])
    return representatives
