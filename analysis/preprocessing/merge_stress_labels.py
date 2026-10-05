import os
import re
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_DIR / "generated_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CSV = Path(
    os.environ.get(
        "DANCE_STRESS_FEATURE_CSV",
        OUTPUT_DIR / "exercise_features_harmonized.csv",
    )
)
DEFAULT_LABEL_XLSX = Path(os.environ.get("DANCE_STRESS_DATA_DIR", REPO_DIR / "Phyiso_data")) / "stress_label_84.xlsx"
LABEL_XLSX = Path(os.environ.get("DANCE_STRESS_LABEL_XLSX", str(DEFAULT_LABEL_XLSX)))
SHEET_NAME = 0
OUT_CSV = OUTPUT_DIR / "train_table_harmonized.csv"

LABEL_MAPPING = {
    "Ex1": "Stress ressenti plié (EX1)",
    "Ex2": "Stress ressenti frappé (EX2)",
    "Ex3": "Stress ressenti battement (EX3)",
}


def subject_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace(",", ".")
    match = re.search(r"(\d+)", s)
    if not match:
        return None
    return f"S{int(match.group(1))}"


def as_number(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def main():
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_CSV}")
    if not LABEL_XLSX.exists():
        raise FileNotFoundError(f"Missing label file: {LABEL_XLSX}")

    feat = pd.read_csv(FEATURE_CSV)
    lab = pd.read_excel(LABEL_XLSX, sheet_name=SHEET_NAME)
    lab.columns = [str(c).strip() for c in lab.columns]
    lab["subject"] = lab["Numéro"].apply(subject_id)

    long_rows = []
    for ex, col in LABEL_MAPPING.items():
        tmp = lab[["subject", col]].copy()
        tmp["exercise"] = ex
        tmp["stress"] = as_number(tmp[col])
        tmp = tmp.drop(columns=[col])
        long_rows.append(tmp)

    lab_long = pd.concat(long_rows, ignore_index=True).dropna(subset=["subject"])
    merged = feat.merge(lab_long, on=["subject", "exercise"], how="outer", indicator=True)

    print("[INFO] merge indicator counts:")
    print(merged["_merge"].value_counts().to_string())

    if (merged["_merge"] != "both").any():
        mismatch_path = OUT_CSV.parent / "train_table_harmonized_merge_mismatch.csv"
        merged.loc[merged["_merge"] != "both"].to_csv(mismatch_path, index=False)
        print("[WARN] Wrote mismatches to:", mismatch_path)

    df = merged.loc[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()
    df = df[~((df["subject"] == "S77") & (df["exercise"] == "Ex3"))].copy()
    if "valid" in df.columns:
        df = df[df["valid"] == True].copy()

    df.to_csv(OUT_CSV, index=False)
    print("[OK] Saved:", OUT_CSV)
    print("[INFO] rows:", len(df), "subjects:", df["subject"].nunique())


if __name__ == "__main__":
    main()
