import os
import re
from pathlib import Path

import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_DIR / "generated_results" / "trait_models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LABEL_XLSX = Path(os.environ.get("DANCE_STRESS_DATA_DIR", REPO_DIR / "Phyiso_data")) / "stress_label_84.xlsx"
LABEL_XLSX = Path(os.environ.get("DANCE_STRESS_LABEL_XLSX", str(DEFAULT_LABEL_XLSX)))
OUT_CSV = REPORT_DIR / "subject_level_labels.csv"


def subject_id(value):
    if pd.isna(value):
        return None
    match = re.search(r"(\d+)", str(value).strip())
    if not match:
        return None
    return f"S{int(match.group(1))}"


def find_col(columns, *candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Could not find any of the expected columns: {candidates}")


def main():
    if not LABEL_XLSX.exists():
        raise FileNotFoundError(f"Missing label workbook: {LABEL_XLSX}")

    df = pd.read_excel(LABEL_XLSX, sheet_name="annotations")
    df.columns = [str(c).strip() for c in df.columns]

    out = pd.DataFrame()
    out["subject"] = df[find_col(df.columns.tolist(), "Numéro")].apply(subject_id)
    out["stai_trait"] = pd.to_numeric(df[find_col(df.columns.tolist(), "STAI-Y trait")], errors="coerce")
    out["stress_cohen"] = pd.to_numeric(df[find_col(df.columns.tolist(), "Stress Cohen")], errors="coerce")
    out["tas20"] = pd.to_numeric(df[find_col(df.columns.tolist(), "TAS-20")], errors="coerce")
    out["age"] = pd.to_numeric(df[find_col(df.columns.tolist(), "Age")], errors="coerce")
    out["genre"] = df[find_col(df.columns.tolist(), "Genre")].astype(str).replace({"nan": pd.NA})
    out["dance_level"] = df[find_col(df.columns.tolist(), "Amateur/Pro/Pré")].astype(str).replace({"nan": pd.NA})
    out["dance_years_total"] = df[
        find_col(
            df.columns.tolist(),
            "nombre d'années de de TOTAL",
            "Nombre d'années de de TOTAL",
            "nombre d'années de danse TOTAL",
            "Nombre d'années de danse TOTAL",
        )
    ].astype(str).replace({"nan": pd.NA})
    out["alexithymie"] = df[find_col(df.columns.tolist(), "Alexithymie")].astype(str).replace({"nan": pd.NA})
    out["anxiete"] = df[find_col(df.columns.tolist(), "anxiété")].astype(str).replace({"nan": pd.NA})

    out = out.dropna(subset=["subject"]).drop_duplicates(subset=["subject"]).sort_values("subject").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print("[OK] Saved:", OUT_CSV)
    print({"rows": len(out), "subjects": int(out["subject"].nunique()), "label_source": str(LABEL_XLSX)})


if __name__ == "__main__":
    main()
