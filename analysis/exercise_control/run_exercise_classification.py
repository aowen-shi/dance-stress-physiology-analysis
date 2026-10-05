import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.shared.harmonized_eval import (
    GROUP_COL,
    INPUT_CSV,
    OUT_DIR,
    SEED,
    build_feature_sets,
    load_data,
)


PRES_DIR = OUT_DIR / "exercise_control"
PRES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "exercise"
OUTER_SPLITS = 5
INNER_SPLITS = 4
N_ITER_RF = 2
N_ITER_HGB = 3
SEARCH_N_JOBS = 1
MODALITY_ORDER = ["ACC_only", "BVP_only", "EDA_only", "TEMP_only", "ALL"]
FAMILIES = ("RidgeCls", "RandomForestCls", "HistGBCls")
NESTED_FAMILIES = FAMILIES
PRIMARY_SCORE = "f1_macro"
CLASS_ORDER = ["Ex1", "Ex2", "Ex3"]


def subject_folds(df: pd.DataFrame) -> dict[str, int]:
    y = df[TARGET_COL].astype(str).to_numpy()
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


def preprocessor(cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
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
        ],
        remainder="drop",
    )


def get_model(name: str):
    if name == "RidgeCls":
        return RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=SEED)
    if name == "RandomForestCls":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        )
    if name == "HistGBCls":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=SEED,
        )
    raise ValueError("Unknown model: " + name)


def tuner(name: str, cols: list[str], n_inner: int):
    pipe = Pipeline(
        [
            ("prep", preprocessor(cols)),
            ("model", get_model(name)),
        ]
    )
    inner_cv = GroupKFold(n_splits=n_inner)

    if name == "RidgeCls":
        return GridSearchCV(
            pipe,
            param_grid={
                "model__alpha": [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0],
                "model__class_weight": [None, "balanced"],
            },
            scoring=PRIMARY_SCORE,
            cv=inner_cv,
            n_jobs=SEARCH_N_JOBS,
            refit=True,
            verbose=0,
        )

    if name == "RandomForestCls":
        return RandomizedSearchCV(
            pipe,
            param_distributions={
                "model__n_estimators": [100, 200],
                "model__max_depth": [8, 16, None],
                "model__min_samples_leaf": [1, 5],
                "model__min_samples_split": [2, 10],
                "model__max_features": ["sqrt", 1.0],
                "model__class_weight": [None, "balanced_subsample"],
            },
            n_iter=N_ITER_RF,
            scoring=PRIMARY_SCORE,
            cv=inner_cv,
            n_jobs=SEARCH_N_JOBS,
            random_state=SEED,
            refit=True,
            verbose=0,
        )

    return GridSearchCV(
        pipe,
        param_grid={
            "model__learning_rate": [0.05],
            "model__max_depth": [3, None],
            "model__max_leaf_nodes": [31],
            "model__min_samples_leaf": [10, 20],
            "model__l2_regularization": [0.0, 1.0],
        },
        scoring=PRIMARY_SCORE,
        cv=inner_cv,
        n_jobs=SEARCH_N_JOBS,
        refit=True,
        verbose=0,
    )


def majority_guess(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    majority = str(train[TARGET_COL].astype(str).value_counts().idxmax())
    return np.full(len(test), majority, dtype=object)


def score_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    return {
        "acc_mean": float(df["accuracy"].mean()),
        "acc_std": float(df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std(ddof=0)),
        "bal_acc_mean": float(df["balanced_acc"].mean()),
        "bal_acc_std": float(df["balanced_acc"].std(ddof=0)),
    }


def eval_fixed(
    df: pd.DataFrame,
    cols: list[str],
    family: str | None,
    baseline: bool,
    fold_map: dict[str, int],
) -> dict[str, float]:
    rows = []
    for fold in fold_ids(df, fold_map):
        test_mask = df[GROUP_COL].astype(str).map(fold_map).eq(fold)
        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()
        if test.empty or train.empty:
            continue

        y_test = test[TARGET_COL].astype(str).to_numpy()
        if baseline:
            pred = majority_guess(train, test)
        else:
            pipe = Pipeline([
                ("prep", preprocessor(cols)),
                ("model", get_model(family)),
            ])
            pipe.fit(train[cols], train[TARGET_COL].astype(str).to_numpy())
            pred = pipe.predict(test[cols])
        rows.append(score_fold(y_test, pred))
    return summarize(rows)


def eval_nested(
    df: pd.DataFrame,
    cols: list[str],
    family: str | None,
    baseline: bool,
    fold_map: dict[str, int],
) -> dict[str, float]:
    rows = []
    for fold in fold_ids(df, fold_map):
        test_mask = df[GROUP_COL].astype(str).map(fold_map).eq(fold)
        train = df.loc[~test_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        if test.empty or train.empty:
            continue

        y_tr = train[TARGET_COL].astype(str).to_numpy()
        y_te = test[TARGET_COL].astype(str).to_numpy()

        if baseline:
            pred = majority_guess(train, test)
        else:
            train_groups = train[GROUP_COL].astype(str).to_numpy()
            n_inner = min(INNER_SPLITS, int(pd.Series(train_groups).nunique()))
            if n_inner < 2:
                raise ValueError("Not enough training subjects for inner GroupKFold.")
            search = tuner(family, cols=cols, n_inner=n_inner)
            search.fit(train[cols], y_tr, groups=train_groups)
            pred = search.best_estimator_.predict(test[cols])
        rows.append(score_fold(y_te, pred))
    return summarize(rows)


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std", "bal_acc_mean", "bal_acc_std"]:
        out[col] = out[col].round(4)
    return out


def best_row(rows: list[dict]) -> dict:
    return (
        pd.DataFrame(rows)
        .sort_values(["macro_f1_mean", "acc_mean", "bal_acc_mean"], ascending=[False, False, False])
        .iloc[0]
        .to_dict()
    )


def overall_table(
    df: pd.DataFrame,
    cols: list[str],
    evaluator,
    fold_map: dict[str, int],
    families: tuple[str, ...],
) -> pd.DataFrame:
    rows = [
        {
            "family": "Baseline",
            "model_name": "baseline_global_majority",
            **evaluator(
                df,
                cols=[],
                family=None,
                baseline=True,
                fold_map=fold_map,
            ),
        }
    ]
    for family in families:
        rows.append(
            {
                "family": family,
                "model_name": family,
                **evaluator(
                    df,
                    cols=cols,
                    family=family,
                    baseline=False,
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
    families: tuple[str, ...],
) -> pd.DataFrame:
    baseline = evaluator(
        df,
        cols=[],
        family=None,
        baseline=True,
        fold_map=fold_map,
    )
    display_rows = [
        {
            "Modality": "Baseline",
            "Best model": "global_majority",
            "Accuracy": round(baseline["acc_mean"], 4),
            "Acc std": round(baseline["acc_std"], 4),
            "Macro F1": round(baseline["macro_f1_mean"], 4),
            "Macro F1 std": round(baseline["macro_f1_std"], 4),
            "Balanced Acc": round(baseline["bal_acc_mean"], 4),
            "Balanced Acc std": round(baseline["bal_acc_std"], 4),
        }
    ]
    for feature_set in MODALITY_ORDER:
        rows = []
        for family in families:
            metrics = evaluator(
                df,
                cols=feature_sets[feature_set],
                family=family,
                baseline=False,
                fold_map=fold_map,
            )
            rows.append({"family": family, **metrics})
        best = best_row(rows)
        display_rows.append(
            {
                "Modality": feature_set,
                "Best model": best["family"],
                "Accuracy": round(best["acc_mean"], 4),
                "Acc std": round(best["acc_std"], 4),
                "Macro F1": round(best["macro_f1_mean"], 4),
                "Macro F1 std": round(best["macro_f1_std"], 4),
                "Balanced Acc": round(best["bal_acc_mean"], 4),
                "Balanced Acc std": round(best["bal_acc_std"], 4),
            }
        )
    return pd.DataFrame(display_rows)


def save_tables(prefix: str, tables: dict[str, pd.DataFrame]) -> None:
    path_map = {
        "overall_allphysio": PRES_DIR / f"{prefix}_overall_allphysio_table.csv",
        "modality_ablation": PRES_DIR / f"{prefix}_modality_ablation_table.csv",
        "overall_acc_eda": PRES_DIR / f"{prefix}_overall_acc_eda_table.csv",
    }
    for key, table in tables.items():
        table.to_csv(path_map[key], index=False)


def main() -> None:
    print("[INFO] Loading harmonized table from:", INPUT_CSV)
    df = load_data().copy()
    df[TARGET_COL] = df[TARGET_COL].astype(str)

    feature_sets = build_feature_sets(df)
    all_cols = feature_sets["ALL"]
    eda_acc_cols = sorted(set(feature_sets["EDA_only"] + feature_sets["ACC_only"]))
    fold_map = subject_folds(df)

    label_counts = df[TARGET_COL].value_counts().reindex(CLASS_ORDER)
    label_df = pd.DataFrame({"exercise_class": label_counts.index, "count": label_counts.values.astype(int)})
    label_df.to_csv(PRES_DIR / "harmonized_exercise_classification_label_counts.csv", index=False)

    fixed_tables = {
        "overall_allphysio": overall_table(df, all_cols, evaluator=eval_fixed, fold_map=fold_map, families=FAMILIES),
        "modality_ablation": modality_table(df, feature_sets, evaluator=eval_fixed, fold_map=fold_map, families=FAMILIES),
        "overall_acc_eda": overall_table(df, eda_acc_cols, evaluator=eval_fixed, fold_map=fold_map, families=FAMILIES),
    }
    save_tables("harmonized_exercise_cls_fixed", fixed_tables)
    print("[OK] Saved harmonized fixed exercise-classification tables.")

    nested_tables = {
        "overall_allphysio": overall_table(df, all_cols, evaluator=eval_nested, fold_map=fold_map, families=NESTED_FAMILIES),
        "modality_ablation": modality_table(df, feature_sets, evaluator=eval_nested, fold_map=fold_map, families=NESTED_FAMILIES),
        "overall_acc_eda": overall_table(df, eda_acc_cols, evaluator=eval_nested, fold_map=fold_map, families=NESTED_FAMILIES),
    }
    save_tables("harmonized_exercise_cls_nested", nested_tables)
    print("[OK] Saved harmonized nested exercise-classification tables.")


if __name__ == "__main__":
    main()
