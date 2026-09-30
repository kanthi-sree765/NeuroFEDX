"""
Contribution 1: CDR-based severity prediction.

Adds a three-category severity task alongside the existing five-class diagnosis
pipeline. The diagnosis pipeline is not modified.

Severity target:
    CDRTOT == 0   -> 0
    CDRTOT == 0.5 -> 1
    CDRTOT >= 1   -> 2

Scientific evaluation is participant-disjoint and keeps imputation, Pearson
selection, and SMOTE strictly training-only. CDR-derived variables that define
or directly encode the target are excluded from the severity feature matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, f1_score, confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import (
    RESULTS_DIR, KNN_NEIGHBORS, PEARSON_THRESHOLD,
    SMOTE_RANDOM_STATE, TEST_SIZE, SPLIT_RANDOM_STATE,
    N_CLIENTS, TREES_PER_CLIENT, N_ROUNDS, RF_PARAMS,
)
from preprocessing.preprocess_common import (
    load_feature_mapping, pearson_prune, smote_resample,
)
from models.random_forest import train_rf
from federated.federated_rf import run_federated_rounds

SEVERITY_LABELS = [0, 1, 2]
SEVERITY_NAMES = {0: "Category 0", 1: "Category 1", 2: "Category 2"}
TARGET_DERIVED_COLUMNS = {
    "CDRTOT", "CDRSUM", "memory", "orient", "judgment", "commun",
    "homehobb", "perscare",
}


def severity_from_cdr(series: pd.Series) -> pd.Series:
    """Map only the specified CDRTOT values; invalid values become NA."""
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[x == 0] = 0
    out.loc[x == 0.5] = 1
    out.loc[x >= 1] = 2
    return out.astype("Int64")


def build_severity_dataframe():
    """Build severity-labelled data from the existing merged multimodal table."""
    merged_path = RESULTS_DIR / "merged_oasis3.csv"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"{merged_path} not found. Run preprocessing/merge.py first."
        )
    merged = pd.read_csv(merged_path, low_memory=False)
    fmap = load_feature_mapping()
    usable = fmap[fmap["status"].isin(["VERIFIED", "INTERPRETED"])].copy()

    # Exclude every CDR-derived target/target-encoding variable by source name.
    usable = usable[~usable["oasis_column"].isin(TARGET_DERIVED_COLUMNS)].copy()

    rename_map = {
        "AgeatEntry": "AgeAtEntry",
        "HEIGHT": "Height",
        "WEIGHT": "Weight",
        "lhCorticalWhiteMatterVol": "LhCorticalWhiteMatterVol",
    }
    df = merged.rename(columns=rename_map).copy()

    keep = {"OASISID": "OASISID"}
    missing = []
    for _, row in usable.iterrows():
        paper_name = row["paper_feature"]
        oasis_col = row["oasis_column"]
        candidate = paper_name if paper_name in df.columns else oasis_col
        if candidate in df.columns:
            keep[candidate] = paper_name
        else:
            missing.append((paper_name, oasis_col))
    if missing:
        raise ValueError(f"Severity feature columns missing from merged data: {missing}")

    out = df[list(keep.keys())].rename(columns=keep)
    out["severity_label"] = severity_from_cdr(df["CDRTOT"])
    out["CDRTOT_source"] = pd.to_numeric(df["CDRTOT"], errors="coerce")

    invalid = out["severity_label"].isna()
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"Dropping {n_invalid} rows with invalid/missing CDRTOT for severity target.")
        out = out.loc[~invalid].copy()

    feature_cols = [c for c in out.columns if c not in {"OASISID", "severity_label", "CDRTOT_source"}]
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["severity_label"] = out["severity_label"].astype(int)

    # Keep source CDRTOT in the labelled artifact for auditability, never in X.
    label_path = RESULTS_DIR / "severity_labelled_dataset.csv"
    out.to_csv(label_path, index=False)

    counts = out["severity_label"].value_counts().sort_index()
    pd.DataFrame({
        "severity_label": counts.index,
        "severity_name": [SEVERITY_NAMES[i] for i in counts.index],
        "n_rows": counts.values,
    }).to_csv(RESULTS_DIR / "severity_class_distribution.csv", index=False)

    pd.DataFrame({
        "excluded_target_derived_column": sorted(TARGET_DERIVED_COLUMNS),
        "reason": [
            "defines severity target" if c == "CDRTOT" else "direct CDR-derived target component"
            for c in sorted(TARGET_DERIVED_COLUMNS)
        ],
    }).to_csv(RESULTS_DIR / "severity_excluded_features.csv", index=False)

    print(f"Severity-labelled dataset: {len(out)} rows, {out['OASISID'].nunique()} subjects")
    print("Severity distribution:")
    print(counts.to_string())
    print(f"Excluded target-derived variables: {sorted(TARGET_DERIVED_COLUMNS)}")
    print(f"Severity predictive features before imputation: {len(feature_cols)}")
    return out, feature_cols


def participant_split(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["OASISID"].values))
    train_subj = set(df.iloc[train_idx]["OASISID"])
    test_subj = set(df.iloc[test_idx]["OASISID"])
    overlap = train_subj & test_subj
    assert not overlap, f"Severity participant leakage: {len(overlap)} overlapping subjects"
    audit = pd.DataFrame({
        "set": ["train", "test", "overlap"],
        "n_subjects": [len(train_subj), len(test_subj), len(overlap)],
        "n_rows": [len(train_idx), len(test_idx), 0],
    })
    audit.to_csv(RESULTS_DIR / "severity_split_audit.csv", index=False)
    return train_idx, test_idx


def metric_summary(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    labels = np.array(SEVERITY_LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    try:
        auc = roc_auc_score(
            label_binarize(y_test, classes=labels), proba,
            average="macro", multi_class="ovr"
        )
    except Exception:
        auc = float("nan")
    summary = {
        "label": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_macro_ovr": auc,
        "n_test": len(y_test),
    }
    per_class = pd.DataFrame({
        "severity_label": labels,
        "severity_name": [SEVERITY_NAMES[i] for i in labels],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return summary, per_class, cm, y_pred, proba


def main():
    df, feature_cols = build_severity_dataframe()
    train_idx, test_idx = participant_split(df)
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_train_imp = imputer.fit_transform(train[feature_cols])
    X_test_imp = imputer.transform(test[feature_cols])
    train_imp = pd.DataFrame(X_train_imp, columns=feature_cols)
    test_imp = pd.DataFrame(X_test_imp, columns=feature_cols)

    survivors, dropped = pearson_prune(train_imp, feature_cols, threshold=PEARSON_THRESHOLD)
    X_train = train_imp[survivors].to_numpy()
    X_test = test_imp[survivors].to_numpy()
    y_train = train["severity_label"].to_numpy()
    y_test = test["severity_label"].to_numpy()

    X_res, y_res = smote_resample(
        X_train, y_train, k_neighbors=5, random_state=SMOTE_RANDOM_STATE
    )

    pd.DataFrame({"selected_feature": survivors}).to_csv(
        RESULTS_DIR / "severity_selected_features.csv", index=False
    )
    pd.DataFrame({"dropped_feature": dropped}).to_csv(
        RESULTS_DIR / "severity_pearson_dropped_features.csv", index=False
    )

    print(f"Severity train/test: {len(train)} / {len(test)} rows; "
          f"subjects={train.OASISID.nunique()} / {test.OASISID.nunique()}; overlap=0")
    print(f"Features: {len(feature_cols)} -> {len(survivors)} after training-only Pearson; dropped={dropped}")
    print("Training-only SMOTE distribution:")
    print(pd.Series(y_res).value_counts().sort_index().to_string())

    rf = train_rf(X_res, y_res)
    summary_c, per_c, cm_c, pred_c, prob_c = metric_summary(
        rf, X_test, y_test, "severity_centralized"
    )

    accs, fed, _ = run_federated_rounds(
        X_res, y_res, X_test, y_test,
        n_rounds=N_ROUNDS,
        n_clients=N_CLIENTS,
        total_trees=100,
        class_names=SEVERITY_LABELS,
        client_seed=42,
        history_name="severity_federated_round_history.csv",
        client_distribution_name="severity_client_distribution.csv",
    )
    summary_f, per_f, cm_f, pred_f, prob_f = metric_summary(
        fed, X_test, y_test, "severity_federated_final_round"
    )

    pd.DataFrame([summary_c, summary_f]).to_csv(RESULTS_DIR / "severity_metrics.csv", index=False)
    per_c.to_csv(RESULTS_DIR / "severity_centralized_per_class.csv", index=False)
    per_f.to_csv(RESULTS_DIR / "severity_federated_per_class.csv", index=False)
    pd.DataFrame(cm_c, index=SEVERITY_LABELS, columns=SEVERITY_LABELS).to_csv(
        RESULTS_DIR / "severity_centralized_confusion_matrix.csv"
    )
    pd.DataFrame(cm_f, index=SEVERITY_LABELS, columns=SEVERITY_LABELS).to_csv(
        RESULTS_DIR / "severity_federated_confusion_matrix.csv"
    )

    predictions = pd.DataFrame({
        "OASISID": test["OASISID"].values,
        "severity_true": y_test,
        "severity_pred_centralized": pred_c,
        "severity_prob_c0": prob_c[:, 0],
        "severity_prob_c1": prob_c[:, 1],
        "severity_prob_c2": prob_c[:, 2],
        "severity_pred_federated": pred_f,
        "severity_fed_prob_c0": prob_f[:, 0],
        "severity_fed_prob_c1": prob_f[:, 1],
        "severity_fed_prob_c2": prob_f[:, 2],
    })
    predictions.to_csv(RESULTS_DIR / "severity_predictions.csv", index=False)

    bundle = {
        "centralized": rf,
        "federated": fed,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": survivors,
        "severity_names": SEVERITY_NAMES,
    }
    joblib.dump(bundle, RESULTS_DIR / "severity_models.joblib")

    metadata = {
        "target": "severity_label",
        "definition": {"0": "CDRTOT == 0", "1": "CDRTOT == 0.5", "2": "CDRTOT >= 1"},
        "n_rows": int(len(df)),
        "n_subjects": int(df.OASISID.nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_subjects": int(train.OASISID.nunique()),
        "test_subjects": int(test.OASISID.nunique()),
        "subject_overlap": 0,
        "excluded_target_derived": sorted(TARGET_DERIVED_COLUMNS),
        "n_input_features_before_pearson": len(feature_cols),
        "n_input_features_after_pearson": len(survivors),
        "pearson_threshold": PEARSON_THRESHOLD,
        "knn_neighbors": KNN_NEIGHBORS,
        "smote": "training_only, k_neighbors=5, random_state=42",
        "rf_params": RF_PARAMS,
        "federated_method": "existing round-wise RF tree pooling; not parameter-vector FedAvg",
        "federated_clients": N_CLIENTS,
        "federated_rounds": N_ROUNDS,
        "global_trees": 100,
    }
    (RESULTS_DIR / "severity_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSeverity centralized metrics:")
    print(pd.Series(summary_c).to_string())
    print("\nSeverity federated final-round metrics:")
    print(pd.Series(summary_f).to_string())
    print("\nSaved Contribution 1 artifacts under results/severity_*")


if __name__ == "__main__":
    main()
