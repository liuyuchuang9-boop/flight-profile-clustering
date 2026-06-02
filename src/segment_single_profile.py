from __future__ import annotations

import argparse
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HEADER_MARKERS = {"PKWZ_L", "PKWZ_R", "T4_L", "T4_R", "Ny", "Hp", "M", "n2_L", "n1_L", "H2", "H4"}
FEATURE_COLUMNS = ["Hp", "M", "T4_L", "T4_R", "n2_L", "n1_L", "n2_R", "n1_R", "H2", "H4", "Ny", "Nz", "Nx", "YMG_L", "YMG_R"]
STATUS_COLUMNS = ["KG3", "KG4", "KG5", "KG16", "KG17", "KG48"]


def read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=None)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=None, encoding="utf-8", engine="python")
    raise ValueError(f"Unsupported file type: {path}")


def find_header_row(raw: pd.DataFrame) -> int:
    rows = []
    for idx in range(len(raw)):
        values = {str(v).strip() for v in raw.iloc[idx].tolist() if pd.notna(v)}
        if len(values.intersection(HEADER_MARKERS)) >= 5:
            rows.append(idx)
    if not rows:
        raise ValueError("Header row was not found.")
    return rows[-1]


def load_profile(path: str) -> pd.DataFrame:
    raw = read_raw(Path(path))
    header_row = find_header_row(raw)
    columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = columns
    df = df.loc[:, [c for c in df.columns if c and c != "nan"]]
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(how="all").reset_index(drop=True)


def clean(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(limit_direction="both")
    s = s.fillna(s.median())
    return s


def extract_segment_features(seg: pd.DataFrame, segment_id: str, start: int, end: int, sample_interval: float) -> dict:
    row = {
        "segment_id": segment_id,
        "start_index": int(start),
        "end_index": int(end - 1),
        "start_time_s": float(start * sample_interval),
        "end_time_s": float((end - 1) * sample_interval),
        "duration_s": float((end - start) * sample_interval),
        "n_samples": int(len(seg)),
    }
    for col in FEATURE_COLUMNS:
        if col not in seg.columns:
            continue
        s = clean(seg[col])
        row[f"{col}_mean"] = float(s.mean())
        row[f"{col}_std"] = float(s.std(ddof=0))
        row[f"{col}_min"] = float(s.min())
        row[f"{col}_max"] = float(s.max())
        row[f"{col}_range"] = float(s.max() - s.min())
        row[f"{col}_slope"] = float((s.iloc[-1] - s.iloc[0]) / max(len(s) - 1, 1))
    for col in STATUS_COLUMNS:
        if col in seg.columns:
            active = clean(seg[col]) > 0.5
            row[f"{col}_active_ratio"] = float(active.mean())
    if {"Ny", "Nz", "Nx"}.issubset(seg.columns):
        ny, nz, nx = clean(seg["Ny"]), clean(seg["Nz"]), clean(seg["Nx"])
        resultant = np.sqrt(ny**2 + nz**2 + nx**2)
        row["resultant_overload_mean"] = float(resultant.mean())
        row["resultant_overload_max"] = float(resultant.max())
        row["resultant_overload_range"] = float(resultant.max() - resultant.min())
    if "T4_L" in seg.columns and "n2_L" in seg.columns:
        row["left_thermal_speed_proxy"] = float((clean(seg["T4_L"]) * clean(seg["n2_L"])).mean())
    if "T4_R" in seg.columns and "n2_R" in seg.columns:
        row["right_thermal_speed_proxy"] = float((clean(seg["T4_R"]) * clean(seg["n2_R"])).mean())
    return row


def make_segments(df: pd.DataFrame, window: int, step: int, sample_interval: float) -> pd.DataFrame:
    rows = []
    sid = 0
    for start in range(0, len(df) - window + 1, step):
        end = start + window
        sid += 1
        rows.append(extract_segment_features(df.iloc[start:end], f"S{sid:03d}", start, end, sample_interval))
    if not rows:
        raise ValueError("No segment was created. Reduce --window-size.")
    return pd.DataFrame(rows)


def cluster_segments(feature_df: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_cols = ["segment_id", "start_index", "end_index", "start_time_s", "end_time_s", "duration_s", "n_samples"]
    xdf = feature_df.drop(columns=[c for c in id_cols if c in feature_df.columns], errors="ignore")
    xdf = xdf.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    xdf = xdf.fillna(xdf.median(numeric_only=True))
    x = StandardScaler().fit_transform(xdf.to_numpy(dtype=float))

    if len(feature_df) < n_clusters:
        raise ValueError("The number of segments must be no smaller than n_clusters.")

    labels = feature_df[id_cols].copy()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels["kmeans_label"] = km.fit_predict(x)
    gm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    labels["gmm_label"] = gm.fit_predict(x)
    labels["gmm_log_likelihood"] = gm.score_samples(x)

    min_cluster_size = max(3, min(8, len(feature_df) // 5))
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels["hdbscan_label"] = hdb.fit_predict(x)

    rep_rows = []
    for label in sorted(labels["kmeans_label"].unique()):
        idx = labels.index[labels["kmeans_label"] == label].to_numpy()
        center = km.cluster_centers_[int(label)]
        dist = np.linalg.norm(x[idx] - center, axis=1)
        rep_idx = int(idx[int(np.argmin(dist))])
        rep_rows.append(labels.loc[rep_idx].to_dict() | {"cluster_label": int(label)})
    representatives = pd.DataFrame(rep_rows)

    merged = labels.merge(feature_df, on=id_cols, how="left")
    summary = merged.groupby("kmeans_label").agg(
        segment_count=("segment_id", "count"),
        start_time_s_min=("start_time_s", "min"),
        end_time_s_max=("end_time_s", "max"),
        Hp_mean=("Hp_mean", "mean"),
        M_mean=("M_mean", "mean"),
        T4_L_mean=("T4_L_mean", "mean"),
        n2_L_mean=("n2_L_mean", "mean"),
        resultant_overload_mean=("resultant_overload_mean", "mean"),
    ).reset_index()
    return labels, summary, representatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment one flight profile and cluster the segments.")
    parser.add_argument("--input", required=True, help="Raw profile file, Excel or CSV.")
    parser.add_argument("--output-dir", default="outputs/segments")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--step-size", type=int, default=150)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--n-clusters", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_profile(args.input)
    features = make_segments(df, args.window_size, args.step_size, args.sample_interval)
    features.to_csv(out / "segment_features.csv", index=False, encoding="utf-8-sig")
    labels, summary, representatives = cluster_segments(features, args.n_clusters)
    labels.to_csv(out / "segment_labels.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    representatives.to_csv(out / "representative_segments.csv", index=False, encoding="utf-8-sig")
    print("Segment clustering finished.")
    print(f"Raw samples: {len(df)}")
    print(f"Segments: {len(features)}")
    print(f"Output directory: {out}")


if __name__ == "__main__":
    main()
