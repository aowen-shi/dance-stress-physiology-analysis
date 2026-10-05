import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

REPO_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_DIR / "generated_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV = Path(
    os.environ.get(
        "DANCE_STRESS_TRAIN_CSV",
        OUT_DIR / "train_table_harmonized.csv",
    )
)

GROUP_COL = "subject"
TARGET_COL = "stress"
EXERCISE_COL = "exercise"
DEVICE_COL = "device"
N_SPLITS = 5

NON_FEATURE_COLS = {"stress", "subject", "subject_id", "valid", "device", "exercise"}
META_SUFFIXES = (
    "_fs_hat",
    "_duration_s",
    "_missing",
    "_parse_fail",
    "_nan_ratio",
    "_outlier_ratio",
    "_finite_ratio_before",
    "_finite_ratio_after_neg_clip",
    "_all_nan_input",
)
SIGNALS = ("ACC", "BVP", "EDA", "TEMP")


def rmse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def corr_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    yt = pd.Series(y_true, dtype=float)
    yp = pd.Series(y_pred, dtype=float)
    pearson = yt.corr(yp, method="pearson")
    spearman = yt.corr(yp, method="spearman")
    return float(pearson) if pd.notna(pearson) else np.nan, float(spearman) if pd.notna(spearman) else np.nan


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = df[~((df["subject"] == "S77") & (df["exercise"] == "Ex3"))].copy()
    df = df.dropna(subset=[TARGET_COL]).copy()
    return df.reset_index(drop=True)


def is_meta_feature(col: str) -> bool:
    return col.endswith("_path") or any(col.endswith(suffix) for suffix in META_SUFFIXES)


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    usable = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and not is_meta_feature(c)
    ]

    feature_sets: dict[str, list[str]] = {}
    for signal in SIGNALS:
        feature_sets[f"{signal}_only"] = sorted([c for c in usable if c.startswith(f"{signal}_") or c.startswith(f"{signal.lower()}_")])

    feature_sets["ALL"] = sorted(set().union(*feature_sets.values()))
    return feature_sets


def build_feature_catalog(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    excluded = [c for c in df.columns if c not in NON_FEATURE_COLS and is_meta_feature(c)]
    for feature_set, cols in feature_sets.items():
        for col in cols:
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_name": col,
                    "signal": col.split("_", 1)[0],
                    "included_in_analysis": True,
                }
            )
    for col in sorted(excluded):
        rows.append(
            {
                "feature_set": "EXCLUDED_META",
                "feature_name": col,
                "signal": col.split("_", 1)[0],
                "included_in_analysis": False,
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "n_subjects": int(df[GROUP_COL].nunique()),
                "n_exercises": int(df[EXERCISE_COL].nunique()),
                "n_devices": int(df[DEVICE_COL].nunique()),
                "stress_mean": float(df[TARGET_COL].mean()),
                "stress_std": float(df[TARGET_COL].std()),
                "stress_min": float(df[TARGET_COL].min()),
                "stress_max": float(df[TARGET_COL].max()),
            }
        ]
    )

    by_exercise = (
        df.groupby(EXERCISE_COL)[TARGET_COL]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_rows"})
    )

    by_device = (
        df.groupby(DEVICE_COL)[TARGET_COL]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_rows"})
    )

    feature_count_rows = [{"feature_set": key, "n_features": len(val)} for key, val in feature_sets.items()]
    feature_counts = pd.DataFrame(feature_count_rows)
    return overall, by_exercise, by_device, feature_counts


def build_model(model_key: str):
    if model_key == "Ridge":
        return Ridge(alpha=300.0)
    if model_key == "RandomForest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1,
        )
    if model_key == "HistGB":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=3,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=SEED,
        )
    raise ValueError(f"Unsupported model_key: {model_key}")


def build_pipeline(model_key: str, feature_cols: list[str], include_exercise: bool) -> Pipeline:
    cat_cols = [EXERCISE_COL] if include_exercise else []
    num_cols = list(feature_cols)

    transformers = []
    if cat_cols:
        transformers.append(("cat", make_one_hot_encoder(), cat_cols))
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )

    if not transformers:
        raise ValueError("No features selected for pipeline.")

    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("prep", prep), ("model", build_model(model_key))])


def make_experiment(
    name: str,
    suite: str,
    family: str,
    variant: str,
    feature_set: str,
    feature_cols: list[str] | None = None,
    include_exercise: bool = False,
    baseline_key: str | None = None,
) -> dict:
    return {
        "name": name,
        "suite": suite,
        "family": family,
        "variant": variant,
        "feature_set": feature_set,
        "feature_cols": list(feature_cols or []),
        "include_exercise": include_exercise,
        "baseline_key": baseline_key,
    }


def predict_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, baseline_key: str) -> np.ndarray:
    if baseline_key == "global_mean":
        return np.full(len(test_df), float(train_df[TARGET_COL].mean()))
    if baseline_key == "exercise_mean":
        means = train_df.groupby(EXERCISE_COL)[TARGET_COL].mean()
        return test_df[EXERCISE_COL].map(means).astype(float).to_numpy()
    if baseline_key == "device_mean":
        means = train_df.groupby(DEVICE_COL)[TARGET_COL].mean()
        return test_df[DEVICE_COL].map(means).astype(float).to_numpy()
    if baseline_key == "exercise_device_mean":
        means = train_df.groupby([EXERCISE_COL, DEVICE_COL])[TARGET_COL].mean()
        idx = test_df.set_index([EXERCISE_COL, DEVICE_COL]).index
        return idx.map(means).astype(float)
    raise ValueError(f"Unsupported baseline_key: {baseline_key}")


def fit_predict_experiment(spec: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    if spec["family"] == "Baseline":
        return np.asarray(predict_baseline(train_df, test_df, spec["baseline_key"]), dtype=float)

    feature_cols = list(spec["feature_cols"])
    include_exercise = bool(spec["include_exercise"])

    if spec["variant"] == "exercise_only":
        pipe = build_pipeline(spec["family"], feature_cols=[], include_exercise=True)
        pipe.fit(train_df[[EXERCISE_COL]], train_df[TARGET_COL].to_numpy(dtype=float))
        pred = pipe.predict(test_df[[EXERCISE_COL]])
        return np.asarray(pred, dtype=float)

    if spec["variant"] == "residual_over_exercise":
        exercise_mean = train_df.groupby(EXERCISE_COL)[TARGET_COL].mean()
        y_train = train_df[TARGET_COL].to_numpy(dtype=float) - train_df[EXERCISE_COL].map(exercise_mean).to_numpy(dtype=float)
        baseline = test_df[EXERCISE_COL].map(exercise_mean).to_numpy(dtype=float)
        cols = feature_cols
        pipe = build_pipeline(spec["family"], feature_cols=cols, include_exercise=False)
        pipe.fit(train_df[cols], y_train)
        residual_pred = pipe.predict(test_df[cols])
        return np.asarray(baseline + residual_pred, dtype=float)

    input_cols = list(feature_cols)
    if include_exercise:
        input_cols = [EXERCISE_COL] + input_cols
    pipe = build_pipeline(spec["family"], feature_cols=feature_cols, include_exercise=include_exercise)
    pipe.fit(train_df[input_cols], train_df[TARGET_COL].to_numpy(dtype=float))
    pred = pipe.predict(test_df[input_cols])
    return np.asarray(pred, dtype=float)


def evaluate_experiment(df: pd.DataFrame, spec: dict, strat_type: str = "overall", strat_value: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[GROUP_COL].astype(str).to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=float)
    n_subjects = int(df[GROUP_COL].nunique())
    n_splits = min(N_SPLITS, n_subjects)
    if n_splits < 2:
        raise ValueError(f"Not enough subjects for CV: {n_subjects}")

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(df), np.nan, dtype=float)
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, y, groups=groups)):
        train_df = df.iloc[tr_idx].copy()
        test_df = df.iloc[te_idx].copy()
        y_test = y[te_idx]

        pred = fit_predict_experiment(spec, train_df, test_df)
        oof_pred[te_idx] = pred

        fold_rows.append(
            {
                "suite": spec["suite"],
                "name": spec["name"],
                "family": spec["family"],
                "variant": spec["variant"],
                "feature_set": spec["feature_set"],
                "include_exercise": bool(spec["include_exercise"]),
                "strat_type": strat_type,
                "strat_value": strat_value,
                "fold": fold,
                "n_test": int(len(te_idx)),
                "y_test_mean": float(np.mean(y_test)),
                "y_test_min": float(np.min(y_test)),
                "y_test_max": float(np.max(y_test)),
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse_metric(y_test, pred),
            }
        )

    pearson, spearman = corr_pair(y, oof_pred)
    fold_df = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(
        [
            {
                "suite": spec["suite"],
                "name": spec["name"],
                "family": spec["family"],
                "variant": spec["variant"],
                "feature_set": spec["feature_set"],
                "include_exercise": bool(spec["include_exercise"]),
                "strat_type": strat_type,
                "strat_value": strat_value,
                "n_rows": int(len(df)),
                "n_subjects": n_subjects,
                "n_splits": n_splits,
                "mae_mean": float(fold_df["mae"].mean()),
                "mae_std": float(fold_df["mae"].std(ddof=0)),
                "rmse_mean": float(fold_df["rmse"].mean()),
                "rmse_std": float(fold_df["rmse"].std(ddof=0)),
                "oof_mae": float(mean_absolute_error(y, oof_pred)),
                "oof_rmse": rmse_metric(y, oof_pred),
                "oof_pearson": pearson,
                "oof_spearman": spearman,
            }
        ]
    )
    return fold_df, summary.assign(oof_pred_json=json.dumps(oof_pred.tolist()))


def evaluate_suite(df: pd.DataFrame, specs: list[dict], strat_type: str = "overall", strat_value: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_frames = []
    summary_frames = []
    for spec in specs:
        fold_df, summary_df = evaluate_experiment(df, spec, strat_type=strat_type, strat_value=strat_value)
        fold_frames.append(fold_df)
        summary_frames.append(summary_df)
    return pd.concat(fold_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def build_core_specs(feature_sets: dict[str, list[str]]) -> list[dict]:
    specs = [
        make_experiment("baseline_global_mean", "core_overall", "Baseline", "baseline", "none", baseline_key="global_mean"),
        make_experiment("baseline_exercise_mean", "core_overall", "Baseline", "baseline", "none", baseline_key="exercise_mean"),
        make_experiment("baseline_device_mean", "core_overall", "Baseline", "baseline", "none", baseline_key="device_mean"),
        make_experiment("baseline_exercise_device_mean", "core_overall", "Baseline", "baseline", "none", baseline_key="exercise_device_mean"),
        make_experiment("ridge_exercise_only", "core_overall", "Ridge", "exercise_only", "none", include_exercise=True),
    ]

    for family in ("Ridge", "RandomForest", "HistGB"):
        specs.extend(
            [
                make_experiment(f"{family.lower()}_all_physio_only", "core_overall", family, "physio_only", "ALL", feature_cols=feature_sets["ALL"]),
                make_experiment(
                    f"{family.lower()}_all_plus_exercise",
                    "core_overall",
                    family,
                    "physio_plus_exercise",
                    "ALL",
                    feature_cols=feature_sets["ALL"],
                    include_exercise=True,
                ),
                make_experiment(
                    f"{family.lower()}_all_residual_over_exercise",
                    "core_overall",
                    family,
                    "residual_over_exercise",
                    "ALL",
                    feature_cols=feature_sets["ALL"],
                ),
            ]
        )
    return specs


def build_overall_modality_specs(feature_sets: dict[str, list[str]]) -> list[dict]:
    specs = []
    for feature_set, cols in feature_sets.items():
        for family in ("Ridge", "HistGB"):
            specs.extend(
                [
                    make_experiment(
                        f"{family.lower()}_{feature_set.lower()}_physio_only",
                        "overall_modality",
                        family,
                        "physio_only",
                        feature_set,
                        feature_cols=cols,
                    ),
                    make_experiment(
                        f"{family.lower()}_{feature_set.lower()}_plus_exercise",
                        "overall_modality",
                        family,
                        "physio_plus_exercise",
                        feature_set,
                        feature_cols=cols,
                        include_exercise=True,
                    ),
                    make_experiment(
                        f"{family.lower()}_{feature_set.lower()}_residual_over_exercise",
                        "overall_modality",
                        family,
                        "residual_over_exercise",
                        feature_set,
                        feature_cols=cols,
                    ),
                ]
            )
    return specs


def build_subset_specs(feature_sets: dict[str, list[str]]) -> list[dict]:
    specs = [make_experiment("subset_global_mean", "subset_scan", "Baseline", "baseline", "none", baseline_key="global_mean")]
    for feature_set, cols in feature_sets.items():
        for family in ("Ridge", "HistGB"):
            specs.append(
                make_experiment(
                    f"{family.lower()}_{feature_set.lower()}_physio_only",
                    "subset_scan",
                    family,
                    "physio_only",
                    feature_set,
                    feature_cols=cols,
                )
            )
    return specs


def build_oof_table(df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    out = df[[GROUP_COL, EXERCISE_COL, DEVICE_COL, TARGET_COL]].copy()
    for _, row in summary_df.iterrows():
        preds = np.asarray(json.loads(row["oof_pred_json"]), dtype=float)
        out[row["name"]] = preds
    return out


def attach_delta(summary_df: pd.DataFrame, baseline_df: pd.DataFrame, baseline_name: str, key_cols: list[str]) -> pd.DataFrame:
    baseline_cols = key_cols + ["mae_mean", "rmse_mean", "oof_mae", "oof_rmse"]
    base = (
        baseline_df[baseline_df["name"] == baseline_name][baseline_cols]
        .rename(
            columns={
                "mae_mean": "baseline_mae_mean",
                "rmse_mean": "baseline_rmse_mean",
                "oof_mae": "baseline_oof_mae",
                "oof_rmse": "baseline_oof_rmse",
            }
        )
        .copy()
    )
    merged = summary_df.merge(base, on=key_cols, how="left")
    merged["delta_mae_mean_vs_baseline"] = merged["mae_mean"] - merged["baseline_mae_mean"]
    merged["delta_rmse_mean_vs_baseline"] = merged["rmse_mean"] - merged["baseline_rmse_mean"]
    merged["delta_oof_mae_vs_baseline"] = merged["oof_mae"] - merged["baseline_oof_mae"]
    merged["delta_oof_rmse_vs_baseline"] = merged["oof_rmse"] - merged["baseline_oof_rmse"]
    return merged


def build_report(
    overall_df: pd.DataFrame,
    by_exercise_df: pd.DataFrame,
    by_device_df: pd.DataFrame,
    feature_counts_df: pd.DataFrame,
    core_summary: pd.DataFrame,
    modality_summary: pd.DataFrame,
    exercise_subset_summary: pd.DataFrame,
    device_subset_summary: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# Harmonized physiology analysis")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(overall_df.to_string(index=False))
    lines.append("")
    lines.append("### Stress by exercise")
    lines.append("")
    lines.append(by_exercise_df.round(4).to_string(index=False))
    lines.append("")
    lines.append("### Stress by device")
    lines.append("")
    lines.append(by_device_df.round(4).to_string(index=False))
    lines.append("")
    lines.append("### Feature counts kept for round 2")
    lines.append("")
    lines.append(feature_counts_df.to_string(index=False))
    lines.append("")

    core_view = core_summary.sort_values(["mae_mean", "rmse_mean"]).copy()
    lines.append("## Core overall comparison")
    lines.append("")
    lines.append(
        core_view[
            [
                "name",
                "family",
                "variant",
                "feature_set",
                "mae_mean",
                "rmse_mean",
                "oof_mae",
                "oof_rmse",
                "oof_pearson",
                "delta_mae_mean_vs_baseline",
            ]
        ].round(4).to_string(index=False)
    )
    lines.append("")

    best_exercise = core_view[core_view["name"] == "baseline_exercise_mean"].iloc[0]
    best_core = core_view.iloc[0]
    residual_best = core_view[core_view["variant"] == "residual_over_exercise"].iloc[0]

    lines.append("### Quick takeaways")
    lines.append("")
    lines.append(
        f"- Best overall MAE: `{best_core['name']}` at `{best_core['mae_mean']:.4f}`."
    )
    lines.append(
        f"- Exercise-mean baseline MAE: `{best_exercise['mae_mean']:.4f}`."
    )
    lines.append(
        f"- Best residual-over-exercise model: `{residual_best['name']}` at `{residual_best['mae_mean']:.4f}` "
        f"(delta vs exercise baseline `{residual_best['delta_mae_mean_vs_baseline']:.4f}`)."
    )
    lines.append("")

    modality_view = modality_summary.sort_values(["variant", "mae_mean"]).copy()
    lines.append("## Overall modality scan")
    lines.append("")
    lines.append(
        modality_view[
            [
                "name",
                "family",
                "variant",
                "feature_set",
                "mae_mean",
                "rmse_mean",
                "delta_mae_mean_vs_baseline",
            ]
        ].round(4).to_string(index=False)
    )
    lines.append("")

    ex_view = exercise_subset_summary.sort_values(["strat_value", "mae_mean"]).copy()
    lines.append("## Exercise-specific subset scan")
    lines.append("")
    lines.append(
        ex_view[
            [
                "strat_value",
                "name",
                "family",
                "feature_set",
                "mae_mean",
                "rmse_mean",
                "delta_mae_mean_vs_baseline",
            ]
        ].round(4).to_string(index=False)
    )
    lines.append("")

    dev_view = device_subset_summary.sort_values(["strat_value", "mae_mean"]).copy()
    lines.append("## Device-specific subset scan")
    lines.append("")
    lines.append(
        dev_view[
            [
                "strat_value",
                "name",
                "family",
                "feature_set",
                "mae_mean",
                "rmse_mean",
                "delta_mae_mean_vs_baseline",
            ]
        ].round(4).to_string(index=False)
    )
    lines.append("")
    return "\n".join(lines)


def main():
    print("[INFO] Loading data from:", INPUT_CSV)
    df = load_data()
    feature_sets = build_feature_sets(df)

    overall_df, by_exercise_df, by_device_df, feature_counts_df = summarize_dataset(df, feature_sets)
    feature_catalog_df = build_feature_catalog(df, feature_sets)

    overall_df.to_csv(OUT_DIR / "dataset_overall_summary.csv", index=False)
    by_exercise_df.to_csv(OUT_DIR / "dataset_by_exercise_summary.csv", index=False)
    by_device_df.to_csv(OUT_DIR / "dataset_by_device_summary.csv", index=False)
    feature_counts_df.to_csv(OUT_DIR / "feature_counts_harmonized.csv", index=False)
    feature_catalog_df.to_csv(OUT_DIR / "feature_catalog_harmonized.csv", index=False)

    print("[INFO] Running core overall suite...")
    core_specs = build_core_specs(feature_sets)
    core_folds, core_summary = evaluate_suite(df, core_specs)
    core_summary = attach_delta(
        core_summary,
        baseline_df=core_summary,
        baseline_name="baseline_exercise_mean",
        key_cols=["suite", "strat_type", "strat_value"],
    )
    core_oof = build_oof_table(df, core_summary)
    core_folds.to_csv(OUT_DIR / "core_overall_folds.csv", index=False)
    core_summary.drop(columns=["oof_pred_json"]).sort_values(["mae_mean", "rmse_mean"]).to_csv(
        OUT_DIR / "core_overall_summary.csv",
        index=False,
    )
    core_oof.to_csv(OUT_DIR / "core_overall_oof_predictions.csv", index=False)

    print("[INFO] Running overall modality suite...")
    modality_specs = build_overall_modality_specs(feature_sets)
    modality_folds, modality_summary = evaluate_suite(df, modality_specs)
    modality_summary = attach_delta(
        modality_summary,
        baseline_df=core_summary,
        baseline_name="baseline_exercise_mean",
        key_cols=["strat_type", "strat_value"],
    )
    modality_folds.to_csv(OUT_DIR / "overall_modality_folds.csv", index=False)
    modality_summary.drop(columns=["oof_pred_json"]).sort_values(["variant", "mae_mean", "rmse_mean"]).to_csv(
        OUT_DIR / "overall_modality_summary.csv",
        index=False,
    )

    print("[INFO] Running exercise-specific subset scans...")
    subset_specs = build_subset_specs(feature_sets)
    ex_folds_all = []
    ex_summary_all = []
    for exercise in sorted(df[EXERCISE_COL].unique()):
        df_ex = df[df[EXERCISE_COL] == exercise].copy().reset_index(drop=True)
        folds_df, summary_df = evaluate_suite(df_ex, subset_specs, strat_type="exercise", strat_value=exercise)
        summary_df = attach_delta(
            summary_df,
            baseline_df=summary_df,
            baseline_name="subset_global_mean",
            key_cols=["suite", "strat_type", "strat_value"],
        )
        ex_folds_all.append(folds_df)
        ex_summary_all.append(summary_df)

    exercise_folds = pd.concat(ex_folds_all, ignore_index=True)
    exercise_summary = pd.concat(ex_summary_all, ignore_index=True)
    exercise_folds.to_csv(OUT_DIR / "exercise_subset_folds.csv", index=False)
    exercise_summary.drop(columns=["oof_pred_json"]).sort_values(
        ["strat_value", "mae_mean", "rmse_mean"]
    ).to_csv(OUT_DIR / "exercise_subset_summary.csv", index=False)

    print("[INFO] Running device-specific subset scans...")
    dev_folds_all = []
    dev_summary_all = []
    for device in sorted(df[DEVICE_COL].unique()):
        df_dev = df[df[DEVICE_COL] == device].copy().reset_index(drop=True)
        folds_df, summary_df = evaluate_suite(df_dev, subset_specs, strat_type="device", strat_value=device)
        summary_df = attach_delta(
            summary_df,
            baseline_df=summary_df,
            baseline_name="subset_global_mean",
            key_cols=["suite", "strat_type", "strat_value"],
        )
        dev_folds_all.append(folds_df)
        dev_summary_all.append(summary_df)

    device_folds = pd.concat(dev_folds_all, ignore_index=True)
    device_summary = pd.concat(dev_summary_all, ignore_index=True)
    device_folds.to_csv(OUT_DIR / "device_subset_folds.csv", index=False)
    device_summary.drop(columns=["oof_pred_json"]).sort_values(
        ["strat_value", "mae_mean", "rmse_mean"]
    ).to_csv(OUT_DIR / "device_subset_summary.csv", index=False)

    report = build_report(
        overall_df=overall_df,
        by_exercise_df=by_exercise_df,
        by_device_df=by_device_df,
        feature_counts_df=feature_counts_df,
        core_summary=core_summary.drop(columns=["oof_pred_json"]).sort_values(["mae_mean", "rmse_mean"]),
        modality_summary=modality_summary.drop(columns=["oof_pred_json"]).sort_values(["variant", "mae_mean", "rmse_mean"]),
        exercise_subset_summary=exercise_summary.drop(columns=["oof_pred_json"]).copy(),
        device_subset_summary=device_summary.drop(columns=["oof_pred_json"]).copy(),
    )
    (OUT_DIR / "harmonized_analysis_report.md").write_text(report, encoding="utf-8")

    print("[OK] Analysis outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
