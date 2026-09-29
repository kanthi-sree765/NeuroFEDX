"""
TRACK A EVALUATION: PAPER-FAITHFUL / POTENTIALLY LEAKAGE-AFFECTED.

Trains and evaluates on results/paper_faithful_ml_ready.csv, which was
built by whole-dataset KNN -> Pearson -> SMOTE (preprocess_paper_faithful.py).
A plain stratified train_test_split is used here because SMOTE has
already destroyed per-subject identity for synthetic rows -- this is
consistent with, and included specifically to reproduce, the paper's
apparent methodology, NOT presented as a scientifically valid estimate
of real-world performance. See evaluation/leakage_free.py and
evaluation/grouped_cv.py for the valid estimates.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import (
    RESULTS_DIR, TEST_SIZE, SPLIT_RANDOM_STATE, CLASS_NAMES,
    N_CLIENTS, TREES_PER_CLIENT, N_ROUNDS,
)
from models.random_forest import train_rf, evaluate
from federated.federated_rf import run_federated_rounds


def main():
    print("=" * 80)
    print("TRACK A EVALUATION -- PAPER-FAITHFUL / POTENTIALLY LEAKAGE-AFFECTED")
    print("=" * 80)

    df = pd.read_csv(RESULTS_DIR / "paper_faithful_ml_ready.csv")
    feature_cols = [c for c in df.columns if c != "diagnosis_class"]
    X = df[feature_cols].values
    y = df["diagnosis_class"].values
    class_names = sorted(pd.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_RANDOM_STATE
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print("[LEAKAGE CAVEAT: because SMOTE ran before this split, some "
          "synthetic training rows are near-duplicate interpolations of "
          "real rows that ended up in the test set.]")

    # ---- Centralized RF ----
    print("\n--- Centralized RF (paper-faithful hyperparameters) ---")
    rf = train_rf(X_train, y_train)
    summary_c, per_class_c, cm_c, labels_c = evaluate(
        rf, X_test, y_test, label="paper_faithful_centralized"
    )
    print(pd.Series(summary_c).to_string())
    print("\nPer-class:")
    print(per_class_c.to_string(index=False))

    # ---- Federated RF-FL interpretation: iterative 100-tree pooling ----
    print(f"\n--- Iterative federated tree pooling: {N_CLIENTS} clients x "
          f"{TREES_PER_CLIENT} trees = {N_CLIENTS * TREES_PER_CLIENT} "
          f"global trees, {N_ROUNDS} communication rounds ---")
    round_accuracies, fed_model, _ = run_federated_rounds(
        X_train, y_train, X_test, y_test,
        n_rounds=N_ROUNDS, class_names=class_names,
        history_name="paper_faithful_federated_round_history.csv",
    )
    summary_f, per_class_f, cm_f, labels_f = evaluate(
        fed_model, X_test, y_test, label="paper_faithful_federated_final_round"
    )
    print("\nFederated (final round model) per-class:")
    print(per_class_f.to_string(index=False))

    # ---- Save everything ----
    results = pd.DataFrame([summary_c, summary_f])
    results.to_csv(RESULTS_DIR / "paper_faithful_results.csv", index=False)

    per_class_c.to_csv(RESULTS_DIR / "paper_faithful_centralized_per_class.csv", index=False)
    per_class_f.to_csv(RESULTS_DIR / "paper_faithful_federated_per_class.csv", index=False)
    pd.DataFrame(cm_c, index=labels_c, columns=labels_c).to_csv(
        RESULTS_DIR / "paper_faithful_centralized_confusion_matrix.csv")
    pd.DataFrame(cm_f, index=labels_f, columns=labels_f).to_csv(
        RESULTS_DIR / "paper_faithful_federated_confusion_matrix.csv")
    np.save(RESULTS_DIR / "paper_faithful_federated_round_accuracies.npy", round_accuracies)

    print(f"\nSaved: paper_faithful_results.csv, per-class CSVs, confusion "
          f"matrices, round-accuracy array. All labeled PAPER-FAITHFUL.")

    # Persist the fitted models + test split for reuse by Tree SHAP
    import joblib
    joblib.dump(
        {"centralized": rf, "federated": fed_model,
         "X_test": X_test, "y_test": y_test, "feature_cols": feature_cols},
        RESULTS_DIR / "paper_faithful_models.joblib",
    )


if __name__ == "__main__":
    main()
