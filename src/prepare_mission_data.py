from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEADER_MARKERS = {"PKWZ_L", "PKWZ_R", "T4_L", "T4_R", "Ny", "Hp", "M", "n2_L", "n1_L", "H2", "H4"}
FEATURE_COLUMNS = ["PKWZ_L", "PKWZ_R", "T4_L", "T4_R", "Ny", "Nz", "Nx", "B_L", "B_R", "T1", "YMG_L", "YMG_R", "Hp", "M", "n2_L", "n1_L", "n2_R", "n1_R", "H1", "H2", "H3", "H4"]
STATUS_COLUMNS = ["KG3", "KG4", "KG5", "KG16", "KG17", "KG48"]
SERIES_COLUMNS = ["Hp", "M", "n2_L", "n1_L", "n2_R", "n1_R", "T4_L", "T4_R", "Ny", "Nz", "Nx", "H2", "H4"]


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
        raise ValueError("Header row was not found. Check symbols such as PKWZ_L, T4_L, Ny, Hp, M.")
    return rows[-1]


def load_mission_file(path: Path) -> pd.DataFrame:
    raw = read_raw(path)
    header_row = find_header_row(raw)
    columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = columns
    data = data.loc[:, [c for c in data.columns if c and c != "nan"]]
    data = data.replace([np.inf, -np.inf], np.nan)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(how="all")
    return data


def clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(limit_direction="both")
    s = s.fillna(s.median())
    return s


def extract_features(df: pd.DataFrame, mission_id: str) -> dict:
    row = {"mission_id": mission_id, "n_samples": int(len(df))}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        s = clean_series(df[col])
        row[f"{col}_mean"] = float(s.mean())
        row[f"{col}_std"] = float(s.std(ddof=0))
        row[f"{col}_min"] = float(s.min())
        row[f"{col}_max"] = float(s.max())
        row[f"{col}_range"] = float(s.max() - s.min())
    for col in STATUS_COLUMNS:
        if col not in df.columns:
            continue
        s = clean_series(df[col])
        active = s > 0.5
        row[f"{col}_active_count"] = int(active.sum())
        row[f"{col}_active_ratio"] = float(active.mean())
    if "T4_L" in df.columns and "n2_L" in df.columns:
        row["left_engine_thermal_speed_proxy"] = float((clean_series(df["T4_L"]) * clean_series(df["n2_L"])).mean())
    if "T4_R" in df.columns and "n2_R" in df.columns:
        row["right_engine_thermal_speed_proxy"] = float((clean_series(df["T4_R"]) * clean_series(df["n2_R"])).mean())
    if {"Ny", "Nz", "Nx"}.issubset(df.columns):
        ny = clean_series(df["Ny"])
        nz = clean_series(df["Nz"])
        nx = clean_series(df["Nx"])
        resultant = np.sqrt(ny**2 + nz**2 + nx**2)
        row["resultant_overload_mean"] = float(resultant.mean())
        row["resultant_overload_max"] = float(resultant.max())
    return row


def resample_series(df: pd.DataFrame, length: int) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in SERIES_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError("No valid series columns were found.")
    x_old = np.linspace(0.0, 1.0, len(df))
    x_new = np.linspace(0.0, 1.0, length)
    arr = []
    for col in cols:
        s = clean_series(df[col]).to_numpy(dtype=float)
        arr.append(np.interp(x_new, x_old, s))
    return np.stack(arr, axis=1), cols


def discover_files(input_dir: Path) -> list[Path]:
    files = []
    for suffix in ("*.xlsx", "*.xls", "*.csv"):
        files.extend(input_dir.glob(suffix))
    return [p for p in sorted(files) if not p.name.startswith("~$")]


def prepare(input_dir: str, output_dir: str, resample_length: int) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = discover_files(input_path)
    if not files:
        raise FileNotFoundError(f"No Excel or CSV files were found in: {input_path}")

    feature_rows = []
    series_list = []
    mission_ids = []
    expected_cols = None
    for file in files:
        print(f"Reading: {file}")
        mission_id = file.stem
        df = load_mission_file(file)
        feature_rows.append(extract_features(df, mission_id))
        series, cols = resample_series(df, resample_length)
        if expected_cols is None:
            expected_cols = cols
        elif cols != expected_cols:
            raise ValueError(f"Series columns are inconsistent in {file.name}.")
        series_list.append(series)
        mission_ids.append(mission_id)

    pd.DataFrame(feature_rows).to_csv(output_path / "features.csv", index=False, encoding="utf-8-sig")
    series_array = np.stack(series_list, axis=0)
    np.save(output_path / "series.npy", series_array)
    pd.DataFrame({"sample_index": range(len(mission_ids)), "mission_id": mission_ids}).to_csv(output_path / "mission_ids.csv", index=False, encoding="utf-8-sig")
    with open(output_path / "series_variables.txt", "w", encoding="utf-8") as f:
        for name in expected_cols or []:
            f.write(f"{name}\n")
    print("Prepared dataset successfully.")
    print(f"Mission files: {len(files)}")
    print(f"Feature table: {output_path / 'features.csv'}")
    print(f"Series array shape: {series_array.shape}")
    if len(files) < 2:
        print("Warning: clustering needs multiple missions. One file can only test preprocessing.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--resample-length", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.input_dir, args.output_dir, args.resample_length)
