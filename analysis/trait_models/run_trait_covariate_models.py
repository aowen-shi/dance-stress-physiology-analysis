import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from analysis.shared.harmonized_eval import (
    EXERCISE_COL,
    GROUP_COL,
    SEED,
    TARGET_COL,
    build_feature_sets,
    make_one_hot_encoder,
)


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

OUTER_SPLITS = 5
INNER_SPLITS = 4
SEARCH_N_JOBS = 1
N_ITER_RF = 4
N_ITER_HGB = 4
FAMILIES = ("Ridge", "RandomForest", "HistGB")
TRAIT_COLS = ["stai_trait", "stress_cohen", "tas20"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_table() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing harmonized train table: {TRAIN_CSV}")
    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"Missing subject label table: {LABEL_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    lab = pd.read_csv(LABEL_CSV)[["subject"] + TRAIT_COLS].copy()
    merged = df.merge(lab, on="subject", how="left")
    merged = merged.dropna(subset=[TARGET_COL] + TRAIT_COLS).reset_index(drop=True)
    return merged


def subject_folds(df: pd.DataFrame) -> dict[str, int]:
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = df[GROUP_COL].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=OUTER_SPLITS)
    mapping: dict[str, int] = {}
    for fold, (_, te_idx) in enumerate(gkf.split(df, y, groups=groups)):
        for subject in df.iloc[te_idx][GROUP_COL].astype(str).unique():
            mapping[subject] = fold
    return mapping


def fold_ids(df: pd.DataFrame, fold_map: dict[str, int]) -> list[int]:
    folds = sorted(df[GROUP_COL].astype(str).map(fold_map).dropna().astype(int).unique().tolist())
    return [fold for fold in folds if fold in range(OUTER_SPLITS)]


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby(EXERCISE_COL)[TARGET_COL].mean()
    global_mean = float(train[TARGET_COL].mean())
    return test[EXERCISE_COL].map(means).fillna(global_mean).to_numpy(dtype=float)


def get_model(name: str):
    if name == "Ridge":
        return Ridge(random_state=SEED)
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=600,
            random_state=SEED,
            n_jobs=1,
        )
    if name == "HistGB":
        return HistGradientBoostingRegressor(
            random_state=SEED,
            early_stopping=True,
        )
    raise ValueError("Unknown model: " + name)


def preprocessor(cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", make_one_hot_encoder(), [EXERCISE_COL]),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                cols,
            ),
        ],
        remainder="drop",
    )


def tuner(name: str, cols: list[str], n_inner: int):
    pipe = Pipeline(
        [
            ("prep", preprocessor(cols)),
            ("model", get_model(name)),
        ]
    )
    inner_cv = GroupKFold(n_splits=n_inner)

    if name == "Ridge":
        return GridSearchCV(
            pipe,
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]},
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=SEARCH_N_JOBS,
            refit=True,
            verbose=0,
        )

    if name == "RandomForest":
        return RandomizedSearchCV(
            pipe,
            param_distributions={
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_leaf": [1, 2, 5],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", 0.8, 1.0],
            },
            n_iter=N_ITER_RF,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=SEARCH_N_JOBS,
            random_state=SEED,
            refit=True,
            verbose=0,
        )

    return RandomizedSearchCV(
        pipe,
        param_distributions={
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_depth": [3, 5, 7, None],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [5, 10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        },
        n_iter=N_ITER_HGB,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=SEARCH_N_JOBS,
        random_state=SEED,
        refit=True,
        verbose=0,
    )


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    fold_df = pd.DataFrame(rows)
    return {
        "mae_mean": float(fold_df["mae"].mean()),
        "mae_std": float(fold_df["mae"].std(ddof=0)),
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=0)),
    }


def eval_fixed(df: pd.DataFrame, cols: list[str], family: str | None, baseline: bool, fold_map: dict[str, int]) -> dict[str, float]:
    fold_rows = []
    for fold in fold_ids(df, fold_map):
        test_mask = df[GROUP_COL].astype(str).map(fold_map).eq(fold)
        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()
        y_test = test[TARGET_COL].to_numpy(dtype=float)
        if baseline:
            pred = baseline(train, test)
        else:
            pipe = Pipeline([
                ("prep", preprocessor(cols)),
                ("model", get_model(family)),
            ])
            input_cols = [EXERCISE_COL] + list(cols)
            pipe.fit(train[input_cols], train[TARGET_COL].to_numpy(dtype=float))
            pred = pipe.predict(test[input_cols])
        fold_rows.append(
            {
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse(y_test, pred),
            }
        )
    return summarize(fold_rows)


def eval_nested(df: pd.DataFrame, cols: list[str], family: str | None, baseline: bool, fold_map: dict[str, int]) -> dict[str, float]:
    fold_rows = []
    for fold in fold_ids(df, fold_map):
        test_mask = df[GROUP_COL].astype(str).map(fold_map).eq(fold)
        train = df.loc[~test_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        y_test = test[TARGET_COL].to_numpy(dtype=float)

        if baseline:
            pred = baseline(train, test)
        else:
            train_groups = train[GROUP_COL].astype(str).to_numpy()
            n_inner = min(INNER_SPLITS, int(pd.Series(train_groups).nunique()))
            if n_inner < 2:
                raise ValueError("Not enough training subjects for inner GroupKFold.")

            search = tuner(family, cols, n_inner)
            input_cols = [EXERCISE_COL] + list(cols)
            search.fit(train[input_cols], train[TARGET_COL].to_numpy(dtype=float), groups=train_groups)
            pred = search.best_estimator_.predict(test[input_cols])

        fold_rows.append(
            {
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse(y_test, pred),
            }
        )
    return summarize(fold_rows)


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["mae_mean", "mae_std", "rmse_mean", "rmse_std", "delta_mae_vs_baseline", "delta_rmse_vs_baseline"]:
        if col in out.columns:
            out[col] = out[col].round(4)
    return out


def variants(df: pd.DataFrame) -> list[dict]:
    feature_sets = build_feature_sets(df)
    acc_eda_cols = sorted(set(feature_sets["ACC_only"] + feature_sets["EDA_only"]))
    return [
        {"variant": "exercise_plus_stai", "feature_set": "STAI_only", "feature_cols": ["stai_trait"]},
        {"variant": "exercise_plus_tas20", "feature_set": "TAS20_only", "feature_cols": ["tas20"]},
        {"variant": "exercise_plus_stai_tas20", "feature_set": "STAI+TAS20", "feature_cols": ["stai_trait", "tas20"]},
        {"variant": "acc_eda_plus_exercise", "feature_set": "ACC+EDA", "feature_cols": acc_eda_cols},
        {"variant": "acc_eda_plus_exercise_plus_stai_tas20", "feature_set": "ACC+EDA+STAI+TAS20", "feature_cols": acc_eda_cols + ["stai_trait", "tas20"]},
    ]


def results_table(df: pd.DataFrame, evaluator, fold_map: dict[str, int]) -> pd.DataFrame:
    baseline = evaluator(df, [], None, True, fold_map)
    rows = [
        {
            "family": "Baseline",
            "model_name": "baseline_exercise_mean",
            "variant": "baseline",
            "feature_set": "exercise_only",
            **baseline,
            "delta_mae_vs_baseline": 0.0,
            "delta_rmse_vs_baseline": 0.0,
        }
    ]
    for spec in variants(df):
        for family in FAMILIES:
            metrics = evaluator(df, spec["feature_cols"], family, False, fold_map)
            rows.append(
                {
                    "family": family,
                    "model_name": family,
                    "variant": spec["variant"],
                    "feature_set": spec["feature_set"],
                    **metrics,
                    "delta_mae_vs_baseline": metrics["mae_mean"] - baseline["mae_mean"],
                    "delta_rmse_vs_baseline": metrics["rmse_mean"] - baseline["rmse_mean"],
                }
            )
    return rounded(pd.DataFrame(rows))


def best_table(results: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for variant, sub in results.groupby("variant"):
        row = sub.sort_values(["mae_mean", "rmse_mean", "family"], ascending=[True, True, True]).iloc[0]
        best_rows.append(row)
    return rounded(pd.DataFrame(best_rows).reset_index(drop=True))


def nested_checks(df: pd.DataFrame, fold_map: dict[str, int]) -> pd.DataFrame:
    feature_sets = build_feature_sets(df)
    acc_eda_cols = sorted(set(feature_sets["ACC_only"] + feature_sets["EDA_only"]))
    rows = []
    baseline = eval_nested(df, [], None, True, fold_map)
    rows.append({"setting": "baseline_exercise_mean", "family": "Baseline", "feature_set": "exercise_only", **baseline})

    checks = [
        ("exercise_plus_stai", "Ridge", ["stai_trait"]),
        ("exercise_plus_tas20", "Ridge", ["tas20"]),
        ("exercise_plus_stai_tas20", "Ridge", ["stai_trait", "tas20"]),
        ("acc_eda_plus_exercise", "HistGB", acc_eda_cols),
        ("acc_eda_plus_exercise_plus_stai_tas20", "HistGB", acc_eda_cols + ["stai_trait", "tas20"]),
    ]
    for setting, family, cols in checks:
        rows.append({"setting": setting, "family": family, "feature_set": "+".join(cols) if cols else "none", **eval_nested(df, cols, family, False, fold_map)})
    return rounded(pd.DataFrame(rows))


def main() -> None:
    print("[INFO] Loading harmonized stress table and trait covariates.")
    df = load_table()
    fold_map = subject_folds(df)

    fixed_results = results_table(df, eval_fixed, fold_map)
    fixed_best = best_table(fixed_results)
    fixed_results.to_csv(REPORT_DIR / "trait_cov_fixed_results_table.csv", index=False)
    fixed_best.to_csv(REPORT_DIR / "trait_cov_fixed_best_table.csv", index=False)
    print("[OK] Saved fixed trait-covariate tables.")

    nested_sanity = nested_checks(df, fold_map)
    nested_sanity.to_csv(REPORT_DIR / "trait_cov_nested_sanity_table.csv", index=False)
    print("[OK] Saved nested sanity table to:", REPORT_DIR)


if __name__ == "__main__":
    main()
