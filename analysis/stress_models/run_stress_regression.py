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
    DEVICE_COL,
    EXERCISE_COL,
    GROUP_COL,
    INPUT_CSV,
    OUT_DIR,
    SEED,
    TARGET_COL,
    build_feature_sets,
    build_pipeline,
    load_data,
    make_one_hot_encoder,
)


PRES_DIR = OUT_DIR / "stress_models"
PRES_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = ("Ridge", "RandomForest", "HistGB")
OUTER_SPLITS = 5
INNER_SPLITS = 4
N_ITER_RF = 6
N_ITER_HGB = 6
SEARCH_N_JOBS = 1
ALL_FEATURE_SET = "ALL"
MODALITY_ORDER = ["ACC_only", "BVP_only", "EDA_only", "TEMP_only", "ALL"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


def baseline(train: pd.DataFrame, test: pd.DataFrame, baseline_name: str) -> np.ndarray:
    if baseline_name == "exercise_mean":
        means = train.groupby(EXERCISE_COL)[TARGET_COL].mean()
        global_mean = float(train[TARGET_COL].mean())
        return test[EXERCISE_COL].map(means).fillna(global_mean).to_numpy(dtype=float)
    if baseline_name == "global_mean":
        return np.full(len(test), float(train[TARGET_COL].mean()))
    raise ValueError(f"Unsupported baseline_name: {baseline_name}")


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


def nested_preprocessor(cols: list[str], with_exercise: bool) -> ColumnTransformer:
    transformers = []
    if with_exercise:
        transformers.append(("cat", make_one_hot_encoder(), [EXERCISE_COL]))
    transformers.append(
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            cols,
        )
    )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def tuner(name: str, cols: list[str], with_exercise: bool):
    pipe = Pipeline(
        [
            ("prep", nested_preprocessor(cols, with_exercise=with_exercise)),
            ("model", get_model(name)),
        ]
    )

    if name == "Ridge":
        param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]}
        return GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=INNER_SPLITS),
            n_jobs=SEARCH_N_JOBS,
            refit=True,
            verbose=0,
        )

    if name == "RandomForest":
        param_dist = {
            "model__n_estimators": [300, 600, 1000],
            "model__max_depth": [None, 8, 12, 16, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", 0.5, 0.8, 1.0],
        }
        return RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=N_ITER_RF,
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=INNER_SPLITS),
            n_jobs=SEARCH_N_JOBS,
            random_state=SEED,
            refit=True,
            verbose=0,
        )

    param_dist = {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [3, 5, 7, None],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [5, 10, 20, 50, 80],
        "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    }
    return RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_HGB,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=INNER_SPLITS),
        n_jobs=SEARCH_N_JOBS,
        random_state=SEED,
        refit=True,
        verbose=0,
    )


def summarize(fold_rows: list[dict[str, float]]) -> dict[str, float]:
    fold_df = pd.DataFrame(fold_rows)
    return {
        "mae_mean": float(fold_df["mae"].mean()),
        "mae_std": float(fold_df["mae"].std(ddof=0)),
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=0)),
    }


def eval_fixed(
    df: pd.DataFrame,
    cols: list[str],
    family: str | None,
    with_exercise: bool,
    baseline_name: str | None,
    fold_map: dict[str, int],
) -> dict[str, float]:
    fold_rows = []
    for fold in fold_ids(df, fold_map):
        test_mask = df[GROUP_COL].astype(str).map(fold_map).eq(fold)
        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()
        if test.empty or train.empty:
            continue

        y_test = test[TARGET_COL].to_numpy(dtype=float)
        if baseline_name is not None:
            pred = baseline(train, test, baseline_name)
        else:
            pipe = build_pipeline(
                family, feature_cols=cols, include_exercise=with_exercise
            )
            input_cols = list(cols)
            if with_exercise:
                input_cols = [EXERCISE_COL] + input_cols
            pipe.fit(train[input_cols], train[TARGET_COL].to_numpy(dtype=float))
            pred = pipe.predict(test[input_cols])

        fold_rows.append(
            {
                "fold": fold,
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse(y_test, pred),
            }
        )
    return summarize(fold_rows)


def eval_nested(
    df: pd.DataFrame,
    cols: list[str],
    family: str | None,
    with_exercise: bool,
    baseline_name: str | None,
    fold_map: dict[str, int],
) -> dict[str, float]:
    fold_rows = []
    for fold in fold_ids(df, fold_map):
        test_mask = df[GROUP_COL].astype(str).map(fold_map).eq(fold)
        train = df.loc[~test_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        if test.empty or train.empty:
            continue

        y_tr = train[TARGET_COL].to_numpy(dtype=float)
        y_te = test[TARGET_COL].to_numpy(dtype=float)

        if baseline_name is not None:
            pred = baseline(train, test, baseline_name)
        else:
            train_groups = train[GROUP_COL].astype(str).to_numpy()
            train_subjects = pd.Series(train_groups).nunique()
            inner_splits = min(INNER_SPLITS, int(train_subjects))
            if inner_splits < 2:
                raise ValueError("Not enough training subjects for inner GroupKFold.")

            if with_exercise:
                X_tr = train[[EXERCISE_COL] + cols].copy()
                X_te = test[[EXERCISE_COL] + cols].copy()
            else:
                X_tr = train[cols].copy()
                X_te = test[cols].copy()

            search = tuner(family, cols, with_exercise=with_exercise)
            # Override the CV object when a subset has fewer training subjects than INNER_SPLITS.
            search.cv = GroupKFold(n_splits=inner_splits)
            search.fit(X_tr, y_tr, groups=train_groups)
            pred = search.best_estimator_.predict(X_te)

        fold_rows.append(
            {
                "fold": fold,
                "mae": float(mean_absolute_error(y_te, pred)),
                "rmse": rmse(y_te, pred),
            }
        )
    return summarize(fold_rows)


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["mae_mean", "mae_std", "rmse_mean", "rmse_std"]:
        out[col] = out[col].round(4)
    return out


def best_row(rows: list[dict]) -> dict:
    return pd.DataFrame(rows).sort_values(["mae_mean", "rmse_mean", "family"]).iloc[0].to_dict()


def overall_table(
    df: pd.DataFrame,
    cols: list[str],
    with_exercise: bool,
    evaluator,
    fold_map: dict[str, int],
) -> pd.DataFrame:
    rows = [
        {
            "family": "Baseline",
            "model_name": "baseline_exercise_mean",
            **evaluator(
                df,
                cols=[],
                family=None,
                with_exercise=False,
                baseline_name="exercise_mean",
                fold_map=fold_map,
            ),
        }
    ]
    for family in FAMILIES:
        rows.append(
            {
                "family": family,
                "model_name": family,
                **evaluator(
                    df,
                    cols=cols,
                    family=family,
                    with_exercise=with_exercise,
                    baseline_name=None,
                    fold_map=fold_map,
                ),
            }
        )
    return rounded(pd.DataFrame(rows))


def modality_table(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    evaluator,
    fold_map: dict[str, int],
) -> pd.DataFrame:
    baseline_metrics = evaluator(
        df,
        cols=[],
        family=None,
        with_exercise=False,
        baseline_name="exercise_mean",
        fold_map=fold_map,
    )
    display_rows = [
        {
            "Modality": "Baseline",
            "Best model": "exercise_mean",
            "MAE": round(baseline_metrics["mae_mean"], 4),
            "MAE std": round(baseline_metrics["mae_std"], 4),
            "RMSE": round(baseline_metrics["rmse_mean"], 4),
            "RMSE std": round(baseline_metrics["rmse_std"], 4),
        }
    ]

    for feature_set in MODALITY_ORDER:
        rows = []
        for family in FAMILIES:
            metrics = evaluator(
                df,
                cols=feature_sets[feature_set],
                family=family,
                with_exercise=False,
                baseline_name=None,
                fold_map=fold_map,
            )
            rows.append({"family": family, **metrics})
        best = best_row(rows)
        display_rows.append(
            {
                "Modality": feature_set,
                "Best model": best["family"],
                "MAE": round(best["mae_mean"], 4),
                "MAE std": round(best["mae_std"], 4),
                "RMSE": round(best["rmse_mean"], 4),
                "RMSE std": round(best["rmse_std"], 4),
            }
        )
    return pd.DataFrame(display_rows)


def exercise_table(
    df: pd.DataFrame,
    cols: list[str],
    evaluator,
    fold_map: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for exercise in sorted(df[EXERCISE_COL].unique()):
        sub = df[df[EXERCISE_COL] == exercise].copy()
        rows.append(
            {
                "group": exercise,
                "family": "Baseline",
                "model_name": "subset_global_mean",
                **evaluator(
                    sub,
                    cols=[],
                    family=None,
                    with_exercise=False,
                    baseline_name="global_mean",
                    fold_map=fold_map,
                ),
            }
        )
        for family in FAMILIES:
            rows.append(
                {
                    "group": exercise,
                    "family": family,
                    "model_name": family,
                    **evaluator(
                        sub,
                        cols=cols,
                        family=family,
                        with_exercise=False,
                        baseline_name=None,
                        fold_map=fold_map,
                    ),
                }
            )
    return rounded(pd.DataFrame(rows))


def device_table(
    df: pd.DataFrame,
    cols: list[str],
    evaluator,
    fold_map: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for device in sorted(df[DEVICE_COL].unique()):
        sub = df[df[DEVICE_COL] == device].copy()
        rows.append(
            {
                "group": device,
                "family": "Baseline",
                "model_name": "device_exercise_mean",
                **evaluator(
                    sub,
                    cols=[],
                    family=None,
                    with_exercise=False,
                    baseline_name="exercise_mean",
                    fold_map=fold_map,
                ),
            }
        )
        for family in FAMILIES:
            rows.append(
                {
                    "group": device,
                    "family": family,
                    "model_name": family,
                    **evaluator(
                        sub,
                        cols=cols,
                        family=family,
                        with_exercise=False,
                        baseline_name=None,
                        fold_map=fold_map,
                    ),
                }
            )
    return rounded(pd.DataFrame(rows))


def save_tables(prefix: str, tables: dict[str, pd.DataFrame]) -> None:
    path_map = {
        "overall_allphysio": PRES_DIR / f"{prefix}_overall_allphysio_table.csv",
        "modality_ablation": PRES_DIR / f"{prefix}_modality_ablation_table.csv",
        "exercise_ablation": PRES_DIR / f"{prefix}_exercise_ablation_table.csv",
        "device_ablation": PRES_DIR / f"{prefix}_device_ablation_table.csv",
        "overall_allphysio_plus_exercise": PRES_DIR / f"{prefix}_overall_allphysio_plus_exercise_table.csv",
        "overall_acc_eda_plus_exercise": PRES_DIR / f"{prefix}_overall_acc_eda_plus_exercise_table.csv",
    }
    for key, table in tables.items():
        table.to_csv(path_map[key], index=False)


def main() -> None:
    print("[INFO] Loading harmonized table from:", INPUT_CSV)
    df = load_data()
    feature_sets = build_feature_sets(df)
    all_feature_cols = feature_sets[ALL_FEATURE_SET]
    eda_acc_cols = sorted(set(feature_sets["EDA_only"] + feature_sets["ACC_only"]))
    fold_map = subject_folds(df)

    fold_map_df = (
        pd.DataFrame({"subject": sorted(fold_map), "fold": [fold_map[s] for s in sorted(fold_map)]})
        .sort_values(["fold", "subject"])
        .reset_index(drop=True)
    )
    fold_map_df.to_csv(PRES_DIR / "harmonized_subject_folds.csv", index=False)

    fixed_tables = {
        "overall_allphysio": overall_table(
            df,
            cols=all_feature_cols,
            with_exercise=False,
            evaluator=eval_fixed,
            fold_map=fold_map,
        ),
        "modality_ablation": modality_table(
            df,
            feature_sets=feature_sets,
            evaluator=eval_fixed,
            fold_map=fold_map,
        ),
        "exercise_ablation": exercise_table(
            df,
            cols=all_feature_cols,
            evaluator=eval_fixed,
            fold_map=fold_map,
        ),
        "device_ablation": device_table(
            df,
            cols=all_feature_cols,
            evaluator=eval_fixed,
            fold_map=fold_map,
        ),
        "overall_allphysio_plus_exercise": overall_table(
            df,
            cols=all_feature_cols,
            with_exercise=True,
            evaluator=eval_fixed,
            fold_map=fold_map,
        ),
        "overall_acc_eda_plus_exercise": overall_table(
            df,
            cols=eda_acc_cols,
            with_exercise=True,
            evaluator=eval_fixed,
            fold_map=fold_map,
        ),
    }
    save_tables("harmonized_fixed", fixed_tables)
    print("[OK] Saved unified fixed-parameter tables.")

    nested_tables = {
        "overall_allphysio": overall_table(
            df,
            cols=all_feature_cols,
            with_exercise=False,
            evaluator=eval_nested,
            fold_map=fold_map,
        ),
        "modality_ablation": modality_table(
            df,
            feature_sets=feature_sets,
            evaluator=eval_nested,
            fold_map=fold_map,
        ),
        "exercise_ablation": exercise_table(
            df,
            cols=all_feature_cols,
            evaluator=eval_nested,
            fold_map=fold_map,
        ),
        "device_ablation": device_table(
            df,
            cols=all_feature_cols,
            evaluator=eval_nested,
            fold_map=fold_map,
        ),
        "overall_allphysio_plus_exercise": overall_table(
            df,
            cols=all_feature_cols,
            with_exercise=True,
            evaluator=eval_nested,
            fold_map=fold_map,
        ),
        "overall_acc_eda_plus_exercise": overall_table(
            df,
            cols=eda_acc_cols,
            with_exercise=True,
            evaluator=eval_nested,
            fold_map=fold_map,
        ),
    }
    save_tables("harmonized_nested", nested_tables)
    print("[OK] Saved unified nested tables.")


if __name__ == "__main__":
    main()
