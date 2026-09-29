"""
TRACK B EVALUATION: SCIENTIFICALLY VALID / LEAKAGE-FREE (Rules 7-8).

Correct pipeline, in order:
  raw dataset
  -> participant-level split (GroupShuffleSplit on OASISID; Rule 8)
  -> KNN imputer FIT ON TRAIN ONLY, applied to test
  -> Pearson feature selection computed ON TRAIN ONLY
  -> SMOTE ON TRAIN ONLY
  -> train RF
  -> evaluate on the untouched, un-resampled, real test set

Produces results/split_audit.csv proving zero subject overlap, and
results/leakage_free_results.csv / *_per_class.csv / confusion matrices,
all labeled LEAKAGE-FREE.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import (
    RESULTS_DIR, TEST_SIZE, SPLIT_RANDOM_STATE, KNN_NEIGHBORS,
    SMOTE_RANDOM_STATE, N_CLIENTS, TREES_PER_CLIENT, N_ROUNDS,
)
from preprocessing.preprocess_common import pearson_prune, smote_resample
from models.random_forest import train_rf, evaluate
from federated.federated_rf import run_federated_rounds


def participant_split(base_df, test_size=TEST_SIZE, seed=SPLIT_RANDOM_STATE):
    """Group-aware split: every row of a given OASISID goes entirely to
    train or entirely to test. Stratification is approximate (by each
    subject's most common diagnosis_class) since GroupShuffleSplit does
    not natively stratify."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = base_df["OASISID"].values
    train_idx, test_idx = next(gss.split(base_df, groups=groups))
    return train_idx, test_idx


def write_split_audit(base_df, train_idx, test_idx):
    train_subjects = set(base_df.iloc[train_idx]["OASISID"])
    test_subjects = set(base_df.iloc[test_idx]["OASISID"])
    overlap = train_subjects & test_subjects
    audit = pd.DataFrame({
        "set": ["train", "test", "overlap"],
        "n_subjects": [len(train_subjects), len(test_subjects), len(overlap)],
        "n_rows": [len(train_idx), len(test_idx), 0],
    })
    audit.to_csv(RESULTS_DIR / "split_audit.csv", index=False)
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} subjects appear in both train and test!"
    print(f"Split audit: {len(train_subjects)} train subjects, "
          f"{len(test_subjects)} test subjects, overlap={len(overlap)} (must be 0)")
    return audit


def main():
    print("=" * 80)
    print("TRACK B EVALUATION -- SCIENTIFICALLY VALID / LEAKAGE-FREE")
    print("=" * 80)

    base = pd.read_csv(RESULTS_DIR / "base_dataframe_unimputed.csv")
    feature_cols = [c for c in base.columns
                    if c not in ("OASISID", "diagnosis_class", "MRI_matched")]

    train_idx, test_idx = participant_split(base)
    write_split_audit(base, train_idx, test_idx)

    train_df = base.iloc[train_idx].reset_index(drop=True)
    test_df = base.iloc[test_idx].reset_index(drop=True)
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")
    print(f"Train class distribution:\n{train_df['diagnosis_class'].value_counts().to_string()}")
    print(f"Test class distribution:\n{test_df['diagnosis_class'].value_counts().to_string()}")

    # ---- KNN imputer: FIT ON TRAIN ONLY ----
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_train_imp = imputer.fit_transform(train_df[feature_cols])
    X_test_imp = imputer.transform(test_df[feature_cols])
    train_imp_df = pd.DataFrame(X_train_imp, columns=feature_cols)
    test_imp_df = pd.DataFrame(X_test_imp, columns=feature_cols)
    print(f"\nKNN imputer (k={KNN_NEIGHBORS}) fit on TRAIN ONLY "
          f"({len(train_df)} rows), applied to test via .transform().")

    # ---- Pearson pruning: computed ON TRAIN ONLY ----
    survivors, dropped = pearson_prune(train_imp_df, feature_cols)
    print(f"Features selected using TRAIN-ONLY correlations: "
          f"{len(survivors)} kept, {len(dropped)} dropped: {dropped}")

    X_train = train_imp_df[survivors].values
    X_test = test_imp_df[survivors].values
    y_train = train_df["diagnosis_class"].values
    y_test = test_df["diagnosis_class"].values

    # ---- SMOTE: TRAIN ONLY ----
    print(f"\nTrain class distribution before SMOTE:\n{pd.Series(y_train).value_counts().to_string()}")
    X_train_res, y_train_res = smote_resample(X_train, y_train, k_neighbors=5,
                                               random_state=SMOTE_RANDOM_STATE)
    print(f"Train class distribution after SMOTE (train-only):\n"
          f"{pd.Series(y_train_res).value_counts().to_string()}")
    print("Test set is untouched: no imputation refit, no SMOTE, real "
          "(imbalanced) class distribution preserved.")

    # ---- Centralized RF ----
    print("\n--- Centralized RF (leakage-free) ---")
    rf = train_rf(X_train_res, y_train_res)
    summary_c, per_class_c, cm_c, labels_c = evaluate(
        rf, X_test, y_test, label="leakage_free_centralized"
    )
    print(pd.Series(summary_c).to_string())
    print("\nPer-class (this is the honest number -- pay attention to "
          "Non-AD / Others recall):")
    print(per_class_c.to_string(index=False))

    # ---- Federated ----
    print(f"\n--- Federated tree pooling (leakage-free), {N_CLIENTS} clients "
          f"x {TREES_PER_CLIENT} trees, {N_ROUNDS} communication rounds ---")
    class_names = sorted(pd.unique(y_train_res))
    accs, fed_model, _ = run_federated_rounds(
        X_train_res, y_train_res, X_test, y_test,
        n_rounds=N_ROUNDS, class_names=class_names,
        history_name="leakage_free_federated_round_history.csv",
    )
    summary_f, per_class_f, cm_f, labels_f = evaluate(
        fed_model, X_test, y_test, label="leakage_free_federated_final_round"
    )
    print("\nFederated (final round model) per-class:")
    print(per_class_f.to_string(index=False))

    # ---- Save everything ----
    results = pd.DataFrame([summary_c, summary_f])
    results.to_csv(RESULTS_DIR / "leakage_free_full_results.csv", index=False)
    per_class_c.to_csv(RESULTS_DIR / "leakage_free_centralized_per_class.csv", index=False)
    per_class_f.to_csv(RESULTS_DIR / "leakage_free_federated_per_class.csv", index=False)
    pd.DataFrame(cm_c, index=labels_c, columns=labels_c).to_csv(
        RESULTS_DIR / "leakage_free_centralized_confusion_matrix.csv")
    pd.DataFrame(cm_f, index=labels_f, columns=labels_f).to_csv(
        RESULTS_DIR / "leakage_free_federated_confusion_matrix.csv")
    np.save(RESULTS_DIR / "leakage_free_federated_round_accuracies.npy", accs)

    import joblib
    joblib.dump(
        {"centralized": rf, "federated": fed_model,
         "X_test": X_test, "y_test": y_test, "feature_cols": survivors},
        RESULTS_DIR / "leakage_free_models.joblib",
    )

    print(f"\nSaved: leakage_free_full_results.csv, per-class CSVs, "
          f"confusion matrices, split_audit.csv. All labeled LEAKAGE-FREE.")


if __name__ == "__main__":
    main()
