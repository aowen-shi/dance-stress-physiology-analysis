from pathlib import Path

import numpy as np
import pandas as pd

from analysis.preprocessing.signal_preprocessing import (
    BASE_DIR,
    BVP_BAND,
    BVP_TOTAL_BAND,
    EX_LIST,
    SIGNALS,
    SKIP_SET,
    band_energy_ratio,
    basic_stats,
    detect_device_group,
    estimate_fs,
    harmonize_acc,
    harmonize_bvp,
    harmonize_eda,
    harmonize_temp,
    lin_slope,
    load_signal_series,
    peak_rate,
    resolve_signal_path,
    subject_num,
)


REPO_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_DIR / "generated_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_stats(rec, prefix, x):
    for key, value in basic_stats(x).items():
        rec[f"{prefix}_{key}"] = value


def safe_corr(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan
    val = np.corrcoef(x[mask], y[mask])[0, 1]
    return float(val) if np.isfinite(val) else np.nan


def extract_one(exercise, subject):
    sid = subject_num(subject)
    device = detect_device_group(sid)
    rec = {
        "exercise": exercise,
        "subject": subject,
        "subject_id": sid,
        "device": device,
        "valid": True,
    }

    for signal in SIGNALS:
        path = resolve_signal_path(BASE_DIR / exercise, subject, signal)
        if path is None or not path.exists():
            rec[f"{signal}_missing"] = 1
            rec["valid"] = False
            continue

        rec[f"{signal}_missing"] = 0
        loaded = load_signal_series(path, signal)
        if loaded is None:
            rec[f"{signal}_parse_fail"] = 1
            rec["valid"] = False
            continue

        rec[f"{signal}_parse_fail"] = 0
        t, v = loaded
        fs = estimate_fs(t)
        rec[f"{signal}_fs_hat"] = fs
        rec[f"{signal}_duration_s"] = float(t[-1] - t[0]) if len(t) > 1 else np.nan

        if signal == "ACC":
            t_h, clean_xyz, qc = harmonize_acc(t, v, fs)
            rec["ACC_nan_ratio"] = qc["nan_ratio"]
            rec["ACC_outlier_ratio"] = qc["outlier_ratio"]
            rec["ACC_finite_ratio_before"] = qc["finite_ratio_before"]
            rec["ACC_all_nan_input"] = qc["all_nan_input"]
            if clean_xyz.size == 0:
                rec["valid"] = False
                continue

            ax, ay, az = clean_xyz[:, 0], clean_xyz[:, 1], clean_xyz[:, 2]
            mag = np.sqrt(ax * ax + ay * ay + az * az)
            add_stats(rec, "ACC_mag", mag)
            rec["ACC_mag_slope"] = lin_slope(t_h, mag)
            rec["ACC_axis_corr_xy"] = safe_corr(ax, ay)
            rec["ACC_axis_corr_xz"] = safe_corr(ax, az)
            rec["ACC_axis_corr_yz"] = safe_corr(ay, az)

            if len(mag) > 2:
                dt = np.diff(t_h)
                dt = np.where(dt <= 0, np.nan, dt)
                jerk = np.diff(mag) / dt
                jerk = jerk[np.isfinite(jerk)]
                add_stats(rec, "ACC_jerk", jerk)
            else:
                add_stats(rec, "ACC_jerk", np.array([], dtype=float))

        elif signal == "BVP":
            t_h, clean_bvp, qc = harmonize_bvp(t, v, fs)
            rec["BVP_nan_ratio"] = qc["nan_ratio"]
            rec["BVP_outlier_ratio"] = qc["outlier_ratio"]
            rec["BVP_finite_ratio_before"] = qc["finite_ratio_before"]
            rec["BVP_all_nan_input"] = qc["all_nan_input"]
            if len(clean_bvp) == 0:
                rec["valid"] = False
                continue
            add_stats(rec, "BVP", clean_bvp)
            rec["BVP_slope"] = lin_slope(t_h, clean_bvp)
            rec["BVP_band_energy_ratio"] = band_energy_ratio(clean_bvp, 64.0, BVP_BAND, BVP_TOTAL_BAND)
            peak_rate_hz, peak_prom = peak_rate(clean_bvp, 64.0, min_distance_s=0.35, prominence_scale=0.5)
            rec["BVP_peak_rate_hz"] = peak_rate_hz
            rec["BVP_peak_prominence_median"] = peak_prom

        elif signal == "EDA":
            t_h, clean_eda, tonic, phasic, qc = harmonize_eda(t, v, fs)
            rec["EDA_nan_ratio"] = qc["nan_ratio"]
            rec["EDA_outlier_ratio"] = qc["outlier_ratio"]
            rec["EDA_finite_ratio_before"] = qc["finite_ratio_before"]
            rec["EDA_finite_ratio_after_neg_clip"] = qc["finite_ratio_after_neg_clip"]
            rec["EDA_all_nan_input"] = qc["all_nan_input"]
            if len(clean_eda) == 0:
                rec["valid"] = False
                continue

            add_stats(rec, "EDA_clean", clean_eda)
            add_stats(rec, "EDA_tonic", tonic)
            add_stats(rec, "EDA_phasic", phasic)
            rec["EDA_tonic_slope"] = lin_slope(t_h, tonic)
            rec["EDA_phasic_slope"] = lin_slope(t_h, phasic)
            scr_rate, scr_prom = peak_rate(np.maximum(phasic, 0.0), 4.0, min_distance_s=1.0, prominence_scale=0.3)
            rec["EDA_scr_rate_hz"] = scr_rate
            rec["EDA_scr_prominence_median"] = scr_prom
            if np.sum(np.isfinite(clean_eda)) > 1:
                clean_eda_diff = np.diff(clean_eda)
                valid_diff = clean_eda_diff[np.isfinite(clean_eda_diff)]
                rec["EDA_flat_ratio"] = float(np.mean(np.abs(valid_diff) < 1e-4)) if len(valid_diff) else np.nan
            else:
                rec["EDA_flat_ratio"] = np.nan

        elif signal == "TEMP":
            t_h, clean_temp, qc = harmonize_temp(t, v, fs)
            rec["TEMP_nan_ratio"] = qc["nan_ratio"]
            rec["TEMP_outlier_ratio"] = qc["outlier_ratio"]
            rec["TEMP_finite_ratio_before"] = qc["finite_ratio_before"]
            rec["TEMP_all_nan_input"] = qc["all_nan_input"]
            if len(clean_temp) == 0:
                rec["valid"] = False
                continue
            add_stats(rec, "TEMP", clean_temp)
            rec["TEMP_slope"] = lin_slope(t_h, clean_temp)

    return rec


def main():
    rows = []
    for exercise in EX_LIST:
        ex_dir = BASE_DIR / exercise
        if not ex_dir.exists():
            print(f"[WARN] Missing exercise dir: {ex_dir}")
            continue
        subjects = sorted([p.name for p in ex_dir.iterdir() if p.is_dir() and p.name.startswith("S")])
        print(f"[INFO] {exercise}: {len(subjects)} subjects")
        for subject in subjects:
            if (subject, exercise) in SKIP_SET:
                continue
            rows.append(extract_one(exercise, subject))

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "exercise_features_harmonized.csv"
    df.to_csv(out_path, index=False)
    print("[OK] Saved:", out_path)
    print("[INFO] rows:", len(df), "cols:", len(df.columns))


if __name__ == "__main__":
    main()
