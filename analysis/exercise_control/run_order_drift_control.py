import numpy as np
import pandas as pd

from analysis.exercise_control import run_exercise_classification as m


def remove_order_trend(df, cols):
    out = df.copy()
    order = out[m.TARGET_COL].map({"Ex1": 1.0, "Ex2": 2.0, "Ex3": 3.0})
    for col in cols:
        residual = out[col].astype(float).copy()
        for _, idx in out.groupby(m.GROUP_COL).groups.items():
            idx_list = list(idx)
            x = order.loc[idx_list].to_numpy(dtype=float)
            y = out.loc[idx_list, col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3 and np.unique(x[mask]).size >= 3:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                residual.loc[idx_list] = y - (slope * x + intercept)
            elif mask.sum() >= 2:
                residual.loc[idx_list] = y - np.nanmean(y[mask])
        out[col] = residual
    return out


def main():
    df = m.load_data().copy()
    df[m.TARGET_COL] = df[m.TARGET_COL].astype(str)
    feature_sets = m.build_feature_sets(df)
    fold_map = m.subject_folds(df)

    autonomic_modalities = ["BVP_only", "EDA_only", "TEMP_only"]
    residual_rows = []
    for modality in autonomic_modalities:
        cols = feature_sets[modality]
        residual_df = remove_order_trend(df, cols)
        for family in m.NESTED_FAMILIES:
            metrics = m.eval_nested(
                residual_df,
                cols=cols,
                family=family,
                baseline=False,
                fold_map=fold_map,
            )
            residual_rows.append({"Modality": modality, "model": family, **metrics})

    residual_table = m.rounded(pd.DataFrame(residual_rows))
    residual_table.to_csv(
        m.PRES_DIR / "harmonized_exercise_cls_nested_autonomic_linear_order_residualized_table.csv",
        index=False,
    )

    best_rows = []
    for modality, rows in residual_table.groupby("Modality"):
        best = rows.sort_values(["macro_f1_mean", "acc_mean", "bal_acc_mean"], ascending=[False, False, False]).iloc[0]
        best_rows.append(
            {
                "Modality": modality,
                "Best model": best["model"],
                "Accuracy": best["acc_mean"],
                "Acc std": best["acc_std"],
                "Macro F1": best["macro_f1_mean"],
                "Macro F1 std": best["macro_f1_std"],
                "Balanced Acc": best["bal_acc_mean"],
                "Balanced Acc std": best["bal_acc_std"],
            }
        )
    best_residual = pd.DataFrame(best_rows)
    best_residual.to_csv(
        m.PRES_DIR / "harmonized_exercise_cls_nested_autonomic_linear_order_residualized_best_table.csv",
        index=False,
    )

    print("[OK] Saved the manuscript exercise-order drift-control tables.")
    print(best_residual.to_csv(index=False))


if __name__ == "__main__":
    main()
