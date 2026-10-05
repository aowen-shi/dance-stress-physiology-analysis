import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, find_peaks


EX_LIST = ["Ex1", "Ex2", "Ex3"]
SIGNALS = ["ACC", "BVP", "EDA", "TEMP"]
SKIP_SET = {("S77", "Ex3")}

REPO_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("DANCE_STRESS_DATA_DIR", REPO_DIR / "Phyiso_data"))

BVP_BAND = (0.7, 3.0)
BVP_TOTAL_BAND = (0.1, 10.0)
TARGET_FS = {
    "ACC": 32.0,
    "BVP": 64.0,
    "EDA": 4.0,
    "TEMP": 1.0,
}

def subject_num(subj_name: str) -> int:
    match = re.match(r"S(\d+)$", subj_name)
    return int(match.group(1)) if match else -1


def detect_device_group(subject_id: int) -> str:
    return "E4" if subject_id <= 47 else "Embrace"


def target_fs_for(signal: str) -> float:
    return TARGET_FS[signal]


def read_csv_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python", decimal=",")
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    return df


def parse_time_column(df: pd.DataFrame) -> np.ndarray | None:
    time_cols = [c for c in df.columns if c.lower() in ["temps", "time", "timestamp", "t"]]
    if not time_cols:
        return None

    s = df[time_cols[0]].astype(str).str.replace(",", ".", regex=False)
    try:
        t = s.astype(float).to_numpy()
    except Exception:
        return None

    med = np.nanmedian(t)
    if med > 1e12:
        dt_raw = np.diff(t[: min(len(t), 5000)])
        dt_raw = dt_raw[dt_raw > 0]
        if len(dt_raw) == 0:
            return None
        md = np.median(dt_raw)
        if md > 1e7:
            return t * 1e-9
        return t * 1e-6

    return t


def estimate_fs(t_sec: np.ndarray) -> float | None:
    if len(t_sec) < 3:
        return None
    t = np.asarray(t_sec, dtype=float)
    t = np.sort(t)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return float(1.0 / np.median(dt))


def resolve_signal_path(ex_dir: Path, subj: str, signal: str) -> Path | None:
    subj_root = ex_dir / subj
    if not subj_root.exists():
        return None

    subdirs = [p for p in subj_root.iterdir() if p.is_dir() and p.name.lower() == "donnees_coupees"]
    if not subdirs:
        return None
    subj_dir = subdirs[0]

    sid = subject_num(subj)
    if sid == -1:
        return None

    if sid <= 47:
        name_map = {"ACC": "ACC.csv", "BVP": "BVP.csv", "EDA": "EDA.csv", "TEMP": "TEMP.csv"}
        path = subj_dir / name_map[signal]
        return path if path.exists() else None

    name_map = {
        "ACC": f"{subj}_accelerometer_merged_clean.csv",
        "BVP": f"{subj}_bvp_merged_clean.csv",
        "EDA": f"{subj}_eda_merged_clean.csv",
        "TEMP": f"{subj}_temperature_merged_clean.csv",
    }
    path = subj_dir / name_map[signal]
    if path.exists():
        return path

    fuzzy_key = {"ACC": "accelerometer", "BVP": "bvp", "EDA": "eda", "TEMP": "temperature"}[signal]
    hits = list(subj_dir.glob(f"*{fuzzy_key}*clean*.csv"))
    return hits[0] if hits else None


def load_signal_series(path: Path, signal: str) -> tuple[np.ndarray, np.ndarray] | None:
    df = read_csv_robust(path)
    df = df.drop_duplicates()
    t = parse_time_column(df)
    if t is None:
        return None

    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    df = df.iloc[order].reset_index(drop=True)
    t = t[order]

    _, idx = np.unique(t, return_index=True)
    idx = np.sort(idx)
    df = df.iloc[idx].reset_index(drop=True)
    t = t[idx]

    if signal == "ACC":
        cols = df.columns
        if all(c in cols for c in ["X", "Y", "Z"]):
            ax = df["X"].to_numpy(dtype=float)
            ay = df["Y"].to_numpy(dtype=float)
            az = df["Z"].to_numpy(dtype=float)
        elif all(c in cols for c in ["x", "y", "z"]):
            ax = df["x"].to_numpy(dtype=float)
            ay = df["y"].to_numpy(dtype=float)
            az = df["z"].to_numpy(dtype=float)
        else:
            return None
        return t, np.vstack([ax, ay, az]).T

    non_time_cols = [c for c in df.columns if c.lower() not in ["temps", "time", "timestamp", "t"]]
    if not non_time_cols:
        return None
    v = df[non_time_cols[0]].astype(str).str.replace(",", ".", regex=False).astype(float).to_numpy()
    return t, v


def lin_slope(t: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(t) & np.isfinite(x)
    if np.sum(mask) < 3:
        return np.nan
    t = t[mask]
    x = x[mask]
    t0 = t - t[0]
    design = np.vstack([t0, np.ones_like(t0)]).T
    m, _ = np.linalg.lstsq(design, x, rcond=None)[0]
    return float(m)


def safe_nanstd(x: np.ndarray, min_count: int = 2) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_count:
        return np.nan
    return float(np.std(arr))


def basic_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {k: np.nan for k in ["mean", "std", "min", "max", "energy", "diff_std"]}
    dx = np.diff(x) if len(x) > 1 else np.array([], dtype=float)
    return {
        "mean": float(np.nanmean(x)),
        "std": safe_nanstd(x),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "energy": float(np.nanmean(x * x)),
        "diff_std": safe_nanstd(dx),
    }


def band_energy_ratio(x: np.ndarray, fs: float | None, band: tuple[float, float], total_band: tuple[float, float]) -> float:
    if fs is None or not np.isfinite(fs) or fs <= 0 or len(x) < 8:
        return np.nan
    x = np.asarray(x, dtype=float)
    if np.sum(np.isfinite(x)) < 8:
        return np.nan
    x = interpolate_1d(x)
    if np.sum(np.isfinite(x)) < 8:
        return np.nan
    x = x - np.nanmean(x)
    nfft = 1 << (len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    psd = (np.abs(spectrum) ** 2) / nfft

    b0, b1 = band
    t0, t1 = total_band
    band_mask = (freqs >= b0) & (freqs <= b1)
    total_mask = (freqs >= t0) & (freqs <= t1)

    band_energy = float(np.sum(psd[band_mask]))
    total_energy = float(np.sum(psd[total_mask]))
    if total_energy <= 0:
        return np.nan
    return band_energy / total_energy


def median_abs_deviation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(mad)


def robust_clip(x: np.ndarray, z: float = 6.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if x.size == 0 or not np.any(np.isfinite(x)):
        return x
    med = np.nanmedian(x)
    mad = median_abs_deviation(x)
    if not np.isfinite(mad) or mad == 0:
        return x
    scale = 1.4826 * mad
    lo = med - z * scale
    hi = med + z * scale
    return np.clip(x, lo, hi)


def estimate_outlier_ratio(x: np.ndarray, z: float = 6.0) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return np.nan
    med = np.nanmedian(x)
    mad = median_abs_deviation(x)
    if not np.isfinite(mad) or mad == 0:
        return 0.0
    scale = 1.4826 * mad
    mask = np.abs(x - med) > (z * scale)
    return float(np.nanmean(mask))


def interpolate_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not np.any(bad):
        return x
    idx = np.arange(len(x))
    good = ~bad
    if np.sum(good) < 2:
        return np.full_like(x, np.nan, dtype=float)
    x[bad] = np.interp(idx[bad], idx[good], x[good])
    return x


def build_regular_time_grid(t: np.ndarray, target_fs: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2 or target_fs <= 0:
        return t.copy()
    start = float(np.min(t))
    end = float(np.max(t))
    duration = end - start
    if duration <= 0:
        return np.array([start], dtype=float)
    n_steps = int(np.floor(duration * target_fs)) + 1
    grid = start + np.arange(n_steps, dtype=float) / target_fs
    if len(grid) == 0 or grid[-1] < end:
        grid = np.append(grid, end)
    return grid


def resample_1d_from_time(t: np.ndarray, x: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(t) & np.isfinite(x)
    if np.sum(mask) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    t_good = t[mask]
    x_good = x[mask]
    order = np.argsort(t_good)
    t_good = t_good[order]
    x_good = x_good[order]
    t_unique, idx = np.unique(t_good, return_index=True)
    x_unique = x_good[idx]
    if len(t_unique) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    grid = build_regular_time_grid(t_unique, target_fs)
    return grid, np.interp(grid, t_unique, x_unique)


def resample_nd_from_time(t: np.ndarray, x: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected a 2D array for multi-axis resampling.")
    cols = []
    grid_ref = None
    for i in range(x.shape[1]):
        grid, col = resample_1d_from_time(t, x[:, i], target_fs)
        if len(grid) == 0:
            return np.array([], dtype=float), np.empty((0, x.shape[1]), dtype=float)
        if grid_ref is None:
            grid_ref = grid
        cols.append(col)
    return grid_ref, np.column_stack(cols)


def butter_filter(x: np.ndarray, fs: float | None, low: float | None = None, high: float | None = None, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if fs is None or not np.isfinite(fs) or fs <= 0 or len(x) < 12:
        return x.copy()

    nyquist = 0.5 * fs
    if high is not None:
        high = min(high, 0.95 * nyquist)
    if low is not None and high is not None and low >= high:
        return x.copy()

    if low is None and high is None:
        return x.copy()
    if low is None:
        b, a = butter(order, high / nyquist, btype="low")
    elif high is None:
        b, a = butter(order, low / nyquist, btype="high")
    else:
        b, a = butter(order, [low / nyquist, high / nyquist], btype="band")

    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x.copy()
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x.copy()


def preprocess_acc(v: np.ndarray, fs: float | None) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(v, dtype=float).copy()
    finite_ratio_before = float(np.mean(np.isfinite(arr)))
    all_nan_input = int(not np.any(np.isfinite(arr)))
    outlier_vals = [estimate_outlier_ratio(arr[:, i]) for i in range(arr.shape[1])]
    outlier_vals = [v for v in outlier_vals if np.isfinite(v)]
    outlier_ratio = float(np.mean(outlier_vals)) if outlier_vals else np.nan
    for i in range(arr.shape[1]):
        arr[:, i] = interpolate_1d(robust_clip(arr[:, i], z=6.0))
        arr[:, i] = butter_filter(arr[:, i], fs, low=None, high=10.0, order=3)
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(arr))),
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "all_nan_input": all_nan_input,
    }
    return arr, qc


def harmonize_acc(t: np.ndarray, v: np.ndarray, fs: float | None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target_fs = target_fs_for("ACC")
    arr = np.asarray(v, dtype=float).copy()
    finite_ratio_before = float(np.mean(np.isfinite(arr)))
    all_nan_input = int(not np.any(np.isfinite(arr)))
    outlier_vals = [estimate_outlier_ratio(arr[:, i]) for i in range(arr.shape[1])]
    outlier_vals = [val for val in outlier_vals if np.isfinite(val)]
    outlier_ratio = float(np.mean(outlier_vals)) if outlier_vals else np.nan
    for i in range(arr.shape[1]):
        arr[:, i] = interpolate_1d(robust_clip(arr[:, i], z=6.0))
        arr[:, i] = butter_filter(arr[:, i], fs, low=None, high=10.0, order=3)
    t_h, arr_h = resample_nd_from_time(t, arr, target_fs)
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(arr_h))) if arr_h.size else np.nan,
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "all_nan_input": all_nan_input,
    }
    return t_h, arr_h, qc


def preprocess_bvp(v: np.ndarray, fs: float | None) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(v, dtype=float)
    finite_ratio_before = float(np.mean(np.isfinite(raw)))
    all_nan_input = int(not np.any(np.isfinite(raw)))
    arr = interpolate_1d(robust_clip(raw, z=6.0))
    outlier_ratio = estimate_outlier_ratio(v)
    if np.any(np.isfinite(arr)):
        arr = detrend(arr, type="linear")
        arr = butter_filter(arr, fs, low=0.5, high=8.0, order=3)
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(arr))),
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "all_nan_input": all_nan_input,
    }
    return arr, qc


def harmonize_bvp(t: np.ndarray, v: np.ndarray, fs: float | None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target_fs = target_fs_for("BVP")
    raw = np.asarray(v, dtype=float)
    finite_ratio_before = float(np.mean(np.isfinite(raw)))
    all_nan_input = int(not np.any(np.isfinite(raw)))
    arr = interpolate_1d(robust_clip(raw, z=6.0))
    outlier_ratio = estimate_outlier_ratio(v)
    if np.any(np.isfinite(arr)):
        arr = detrend(arr, type="linear")
        arr = butter_filter(arr, fs, low=0.5, high=8.0, order=3)
    t_h, arr_h = resample_1d_from_time(t, arr, target_fs)
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(arr_h))) if arr_h.size else np.nan,
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "all_nan_input": all_nan_input,
    }
    return t_h, arr_h, qc


def preprocess_eda(v: np.ndarray, fs: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    arr = np.asarray(v, dtype=float).copy()
    finite_ratio_before = float(np.mean(np.isfinite(arr)))
    arr[arr < 0] = np.nan
    finite_ratio_after_neg_clip = float(np.mean(np.isfinite(arr)))
    all_nan_input = int(not np.any(np.isfinite(arr)))
    outlier_ratio = estimate_outlier_ratio(arr)
    arr = interpolate_1d(robust_clip(arr, z=6.0))
    clean = butter_filter(arr, fs, low=None, high=1.0, order=3) if np.any(np.isfinite(arr)) else arr.copy()
    tonic = butter_filter(clean, fs, low=None, high=0.05, order=3) if np.any(np.isfinite(clean)) else clean.copy()
    phasic = clean - tonic
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(clean))),
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "finite_ratio_after_neg_clip": finite_ratio_after_neg_clip,
        "all_nan_input": all_nan_input,
    }
    return clean, tonic, phasic, qc


def harmonize_eda(t: np.ndarray, v: np.ndarray, fs: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    target_fs = target_fs_for("EDA")
    arr = np.asarray(v, dtype=float).copy()
    finite_ratio_before = float(np.mean(np.isfinite(arr)))
    arr[arr < 0] = np.nan
    finite_ratio_after_neg_clip = float(np.mean(np.isfinite(arr)))
    all_nan_input = int(not np.any(np.isfinite(arr)))
    outlier_ratio = estimate_outlier_ratio(arr)
    arr = interpolate_1d(robust_clip(arr, z=6.0))
    clean = butter_filter(arr, fs, low=None, high=1.0, order=3) if np.any(np.isfinite(arr)) else arr.copy()
    t_h, clean_h = resample_1d_from_time(t, clean, target_fs)
    tonic_h = butter_filter(clean_h, target_fs, low=None, high=0.05, order=3) if len(clean_h) else clean_h.copy()
    phasic_h = clean_h - tonic_h
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(clean_h))) if len(clean_h) else np.nan,
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "finite_ratio_after_neg_clip": finite_ratio_after_neg_clip,
        "all_nan_input": all_nan_input,
    }
    return t_h, clean_h, tonic_h, phasic_h, qc


def preprocess_temp(v: np.ndarray, fs: float | None) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(v, dtype=float)
    finite_ratio_before = float(np.mean(np.isfinite(raw)))
    all_nan_input = int(not np.any(np.isfinite(raw)))
    arr = interpolate_1d(robust_clip(raw, z=6.0))
    outlier_ratio = estimate_outlier_ratio(v)
    arr = butter_filter(arr, fs, low=None, high=0.3, order=3) if np.any(np.isfinite(arr)) else arr.copy()
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(arr))),
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "all_nan_input": all_nan_input,
    }
    return arr, qc


def harmonize_temp(t: np.ndarray, v: np.ndarray, fs: float | None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target_fs = target_fs_for("TEMP")
    raw = np.asarray(v, dtype=float)
    finite_ratio_before = float(np.mean(np.isfinite(raw)))
    all_nan_input = int(not np.any(np.isfinite(raw)))
    arr = interpolate_1d(robust_clip(raw, z=6.0))
    outlier_ratio = estimate_outlier_ratio(v)
    arr = butter_filter(arr, fs, low=None, high=0.3, order=3) if np.any(np.isfinite(arr)) else arr.copy()
    t_h, arr_h = resample_1d_from_time(t, arr, target_fs)
    qc = {
        "nan_ratio": float(np.mean(~np.isfinite(arr_h))) if len(arr_h) else np.nan,
        "outlier_ratio": outlier_ratio,
        "finite_ratio_before": finite_ratio_before,
        "all_nan_input": all_nan_input,
    }
    return t_h, arr_h, qc


def peak_rate(x: np.ndarray, fs: float | None, min_distance_s: float, prominence_scale: float = 0.5) -> tuple[float, float]:
    if fs is None or not np.isfinite(fs) or fs <= 0 or len(x) < 8:
        return np.nan, np.nan
    x = np.asarray(x, dtype=float)
    if np.sum(np.isfinite(x)) < 8:
        return np.nan, np.nan
    x = interpolate_1d(x)
    if np.sum(np.isfinite(x)) < 8:
        return np.nan, np.nan
    distance = max(int(min_distance_s * fs), 1)
    sigma = safe_nanstd(x)
    prom = max(float((sigma if np.isfinite(sigma) else 0.0) * prominence_scale), 1e-8)
    peaks, props = find_peaks(x, distance=distance, prominence=prom)
    duration_s = len(x) / fs
    rate = float(len(peaks) / duration_s) if duration_s > 0 else np.nan
    prom_median = float(np.nanmedian(props["prominences"])) if len(peaks) else np.nan
    return rate, prom_median
