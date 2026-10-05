import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


REPO_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_DIR / "generated_results" / "trait_models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path(
    os.environ.get(
        "DANCE_STRESS_TRAIN_CSV",
        REPO_DIR / "generated_results" / "train_table_harmonized.csv",
    )
)
LABEL_CSV = Path(os.environ.get("DANCE_STRESS_SUBJECT_LABEL_CSV", REPORT_DIR / "subject_level_labels.csv"))

TARGET_COL = "stress"
GROUP_COL = "subject"
EXERCISE_COL = "exercise"

TRAIT_COLS = ["stai_trait", "stress_cohen", "tas20"]
PHYSIO_COLS = ["ACC_mag_mean", "EDA_tonic_mean", "EDA_scr_rate_hz"]


def zscore(series):
    std = float(series.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def load_table():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing harmonized train table: {TRAIN_CSV}")
    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"Missing subject-level label table: {LABEL_CSV}")

    stress_df = pd.read_csv(TRAIN_CSV)
    label_df = pd.read_csv(LABEL_CSV)[["subject"] + TRAIT_COLS].copy()
    df = stress_df.merge(label_df, on="subject", how="left")

    keep_cols = [GROUP_COL, EXERCISE_COL, TARGET_COL] + TRAIT_COLS + PHYSIO_COLS
    df = df[keep_cols].dropna().reset_index(drop=True)

    for col in TRAIT_COLS + PHYSIO_COLS:
        df[f"z_{col}"] = zscore(df[col].astype(float))

    return df


def fit_model(df: pd.DataFrame, formula: str):
    model = smf.mixedlm(formula=formula, data=df, groups=df[GROUP_COL])
    fit_kwargs = dict(reml=False, maxiter=2000, disp=False)
    methods = ["lbfgs", "powell", "cg"]
    last_error = None

    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(method=method, **fit_kwargs)
            return result, method
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    raise RuntimeError(f"MixedLM failed for formula={formula!r}: {last_error}")


def coef_row(result, model_id, term):
    if term not in result.params.index:
        return {
            "model_id": model_id,
            "term": term,
            "coef": np.nan,
            "std_err": np.nan,
            "z_value": np.nan,
            "p_value": np.nan,
        }
    return {
        "model_id": model_id,
        "term": term,
        "coef": float(result.params[term]),
        "std_err": float(result.bse[term]),
        "z_value": float(result.tvalues[term]),
        "p_value": float(result.pvalues[term]),
    }


def main():
    df = load_table()

    model_specs = [
        (
            "M0_exercise_only",
            f"{TARGET_COL} ~ C({EXERCISE_COL})",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
            ],
        ),
        (
            "M1_exercise_plus_stai",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) + z_stai_trait",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_stai_trait",
            ],
        ),
        (
            "M2_exercise_plus_cohen",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) + z_stress_cohen",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_stress_cohen",
            ],
        ),
        (
            "M3_exercise_plus_tas20",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) + z_tas20",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_tas20",
            ],
        ),
        (
            "M4_exercise_plus_all_traits",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) + z_stai_trait + z_stress_cohen + z_tas20",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_stai_trait",
                "z_stress_cohen",
                "z_tas20",
            ],
        ),
        (
            "M4b_exercise_plus_stai_tas20",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) + z_stai_trait + z_tas20",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_stai_trait",
                "z_tas20",
            ],
        ),
        (
            "M5_exercise_plus_acc_eda",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) + z_ACC_mag_mean + z_EDA_tonic_mean + z_EDA_scr_rate_hz",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_ACC_mag_mean",
                "z_EDA_tonic_mean",
                "z_EDA_scr_rate_hz",
            ],
        ),
        (
            "M6_exercise_plus_acc_eda_stai",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) + z_stai_trait + z_ACC_mag_mean + z_EDA_tonic_mean + z_EDA_scr_rate_hz",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_stai_trait",
                "z_ACC_mag_mean",
                "z_EDA_tonic_mean",
                "z_EDA_scr_rate_hz",
            ],
        ),
        (
            "M7_exercise_stai_interaction",
            f"{TARGET_COL} ~ C({EXERCISE_COL}) * z_stai_trait",
            [
                "Intercept",
                f"C({EXERCISE_COL})[T.Ex2]",
                f"C({EXERCISE_COL})[T.Ex3]",
                "z_stai_trait",
                f"C({EXERCISE_COL})[T.Ex2]:z_stai_trait",
                f"C({EXERCISE_COL})[T.Ex3]:z_stai_trait",
            ],
        ),
    ]

    comparison_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    summary_chunks: list[str] = []

    for model_id, formula, key_terms in model_specs:
        result, method = fit_model(df, formula)
        random_var = np.nan
        if hasattr(result, "cov_re") and result.cov_re is not None:
            cov_re = np.asarray(result.cov_re)
            if cov_re.size:
                random_var = float(cov_re.flat[0])

        comparison_rows.append(
            {
                "model_id": model_id,
                "formula": formula,
                "n_rows": int(result.nobs),
                "fit_method": method,
                "converged": bool(result.converged),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "loglik": float(result.llf),
                "random_intercept_var": random_var,
                "residual_var": float(result.scale),
            }
        )

        for term in key_terms:
            effect_rows.append(coef_row(result, model_id, term))

        summary_chunks.append(f"# {model_id}\n")
        summary_chunks.append(f"Formula: {formula}\n")
        summary_chunks.append(result.summary().as_text())
        summary_chunks.append("\n\n")

    comparison_df = pd.DataFrame(comparison_rows)
    effect_df = pd.DataFrame(effect_rows)

    numeric_cols_cmp = ["aic", "bic", "loglik", "random_intercept_var", "residual_var"]
    for col in numeric_cols_cmp:
        comparison_df[col] = comparison_df[col].round(4)

    numeric_cols_eff = ["coef", "std_err", "z_value", "p_value"]
    for col in numeric_cols_eff:
        effect_df[col] = effect_df[col].round(4)

    comparison_path = REPORT_DIR / "trait_lmm_model_comparison.csv"
    effects_path = REPORT_DIR / "trait_lmm_key_effects.csv"
    summary_path = REPORT_DIR / "trait_lmm_full_summary.txt"
    markdown_path = REPORT_DIR / "trait_lmm_summary.md"

    comparison_df.to_csv(comparison_path, index=False)
    effect_df.to_csv(effects_path, index=False)
    summary_path.write_text("".join(summary_chunks), encoding="utf-8")

    best_aic = comparison_df.sort_values("aic").iloc[0]
    md = [
        "# Trait-Aware Mixed-Effects Summary",
        "",
        f"- Rows analyzed: `{len(df)}`",
        f"- Subjects: `{df[GROUP_COL].nunique()}`",
        f"- Best AIC model: `{best_aic['model_id']}` (`AIC={best_aic['aic']}`)",
        "",
        "Generated files:",
        f"- `{comparison_path.name}`",
        f"- `{effects_path.name}`",
        f"- `{summary_path.name}`",
        "",
    ]
    markdown_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Saved model comparison to: {comparison_path}")
    print(f"[OK] Saved key effects to: {effects_path}")
    print(f"[OK] Saved full summaries to: {summary_path}")


if __name__ == "__main__":
    main()
