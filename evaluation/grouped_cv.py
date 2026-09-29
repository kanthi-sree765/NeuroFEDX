"""
TRACK B (rigorous variant): GROUPED 5-FOLD CROSS-VALIDATION.

Same leakage controls as evaluation/leakage_free.py (participant-level
grouping, train-only imputation/Pearson/SMOTE), but averaged over 5
patient-disjoint folds instead of a single split, for a more stable
estimate of true generalization performance. This is the single most
rigorous evaluation in this project.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR, KNN_NEIGHBORS, SMOTE_RANDOM_STATE
from preprocessing.preprocess_common import pearson_prune, smote_resample
from models.random_forest import train_rf, evaluate


N_FOLDS = 5


def main():
    print("=" * 80)
    print("GROUPED 5-FOLD CV -- MOST RIGOROUS LEAKAGE-FREE ESTIMATE")
    print("=" * 80)

    base = pd.read_csv(RESULTS_DIR / "base_dataframe_unimputed.csv")
    feature_cols = [c for c in base.columns
                    if c not in ("OASISID", "diagnosis_class", "MRI_matched")]

    groups = base["OASISID"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_summaries = []
    all_per_class = []
    subject_check = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(base, groups=groups)):
        train_df = base.iloc[train_idx].reset_index(drop=True)
        test_df = base.iloc[test_idx].reset_index(drop=True)

        train_subj = set(train_df["OASISID"])
        test_subj = set(test_df["OASISID"])
        overlap = train_subj & test_subj
        subject_check.append({"fold": fold, "train_subjects": len(train_subj),
                               "test_subjects": len(test_subj), "overlap": len(overlap)})
        assert len(overlap) == 0, f"Fold {fold}: subject leakage detected!"

        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        X_train_imp = imputer.fit_transform(train_df[feature_cols])
        X_test_imp = imputer.transform(test_df[feature_cols])
        train_imp_df = pd.DataFrame(X_train_imp, columns=feature_cols)
        test_imp_df = pd.DataFrame(X_test_imp, columns=feature_cols)

        survivors, dropped = pearson_prune(train_imp_df, feature_cols, verbose=False)

        X_train = train_imp_df[survivors].values
        X_test = test_imp_df[survivors].values
        y_train = train_df["diagnosis_class"].values
        y_test = test_df["diagnosis_class"].values

        X_train_res, y_train_res = smote_resample(
            X_train, y_train, k_neighbors=5, random_state=SMOTE_RANDOM_STATE + fold
        )

        rf = train_rf(X_train_res, y_train_res)
        summary, per_class, cm, labels = evaluate(
            rf, X_test, y_test, label=f"grouped_cv_fold{fold}"
        )
        summary["fold"] = fold
        summary["n_dropped_features"] = len(dropped)
        fold_summaries.append(summary)
        per_class["fold"] = fold
        all_per_class.append(per_class)

        print(f"Fold {fold}: train_subj={len(train_subj)} test_subj={len(test_subj)} "
              f"overlap={len(overlap)} | acc={summary['accuracy']:.4f} "
              f"bal_acc={summary['balanced_accuracy']:.4f} "
              f"macro_f1={summary['macro_f1']:.4f}")

    results_df = pd.DataFrame(fold_summaries)
    per_class_df = pd.concat(all_per_class, ignore_index=True)
    subject_check_df = pd.DataFrame(subject_check)

    results_df.to_csv(RESULTS_DIR / "grouped_cv_full_results.csv", index=False)
    per_class_df.to_csv(RESULTS_DIR / "grouped_cv_full_per_class.csv", index=False)
    subject_check_df.to_csv(RESULTS_DIR / "grouped_cv_subject_overlap_check.csv", index=False)

    print("\n" + "=" * 80)
    print("GROUPED 5-FOLD CV SUMMARY (mean +/- std across folds)")
    print("=" * 80)
    numeric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                     "macro_precision", "macro_recall", "auc_macro_ovr"]
    means = results_df[numeric_cols].mean()
    stds = results_df[numeric_cols].std()
    for col in numeric_cols:
        print(f"  {col:20s}: {means[col]:.4f} +/- {stds[col]:.4f}")

    print("\nPer-class recall averaged across folds (this is where the "
          "paper's 98%+ recall claim breaks down for minority classes):")
    per_class_avg = per_class_df.groupby("class")[["precision", "recall", "f1"]].mean()
    print(per_class_avg.to_string())

    print(f"\nSaved: grouped_cv_full_results.csv, grouped_cv_full_per_class.csv, "
          f"grouped_cv_subject_overlap_check.csv")


if __name__ == "__main__":
    main()
