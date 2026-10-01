"""
Generate patient-level application results from the completed C2 experiment.

IMPORTANT:
- Does NOT modify the research implementation.
- Uses the exact saved C2 client assignments.
- Uses the already-saved C2 global model.
- Rebuilds only the client-specific personalization models,
  because those were intentionally not persisted by Contribution 2.
- Produces patient-level aggregated predictions for the application.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from config.model_config import (
    RESULTS_DIR,
    KNN_NEIGHBORS,
    SMOTE_RANDOM_STATE,
)

from preprocessing.preprocess_common import smote_resample
from federated.personalized_rf import personalize_global_model


APPLICATION_DIR = RESULTS_DIR / "application"
APPLICATION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Exact C2 configuration
# ---------------------------------------------------------------------

SEED = 42
PERSONALIZATION_TREES = 10


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_selected_features(task_dir: Path):
    """Load the exact features selected by the completed C2 run."""
    df = pd.read_csv(task_dir / "selected_features.csv")
    return df["selected_feature"].tolist()


def prepare_c2_data(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    label_col: str,
    selected_features: list[str],
):
    """
    Reproduce the C2 preprocessing needed for application inference.

    C2:
      1. Uses only subjects present in sample_assignments.csv.
      2. Fits KNN imputation on federation + personalization rows.
      3. Applies the same imputer to test rows.
      4. Uses the exact selected feature list saved by C2.
    """

    assignments = assignments[
        ["OASISID", "client_id", "split"]
    ].copy()

    assignments["OASISID"] = assignments["OASISID"].astype(str)
    df["OASISID"] = df["OASISID"].astype(str)

    # Only subjects belonging to the C2 experiment.
    work = df.merge(
        assignments,
        on="OASISID",
        how="inner",
        validate="many_to_one",
    )

    if len(work) == 0:
        raise RuntimeError(
            f"No rows matched C2 assignments for {label_col}."
        )

    train_mask = work["split"].isin(
        ["federation", "personalization"]
    )

    test_mask = work["split"].eq("test")

    train_rows = work.loc[train_mask].copy()
    test_rows = work.loc[test_mask].copy()

    if train_rows.empty:
        raise RuntimeError(
            f"No federation/personalization rows found for {label_col}."
        )

    if test_rows.empty:
        raise RuntimeError(
            f"No test rows found for {label_col}."
        )

    # -----------------------------------------------------------------
    # EXACT C2 KNN imputation
    # -----------------------------------------------------------------

    # The C2 feature universe is everything except identifiers/labels.
    excluded = {
        "OASISID",
        label_col,
        "MRI_matched",
        "CDRTOT_source",
        "client_id",
        "split",
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    imputer = KNNImputer(
        n_neighbors=KNN_NEIGHBORS
    )

    X_train_imp = imputer.fit_transform(
        train_rows[feature_cols]
    )

    X_test_imp = imputer.transform(
        test_rows[feature_cols]
    )

    train_imp = pd.DataFrame(
        X_train_imp,
        columns=feature_cols,
        index=train_rows.index,
    )

    test_imp = pd.DataFrame(
        X_test_imp,
        columns=feature_cols,
        index=test_rows.index,
    )

    # Keep only the exact C2 selected features.
    missing_features = [
        f for f in selected_features
        if f not in train_imp.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing selected features for {label_col}: "
            f"{missing_features}"
        )

    train_out = train_rows[
        ["OASISID", label_col, "client_id", "split"]
    ].copy()

    test_out = test_rows[
        ["OASISID", label_col, "client_id", "split"]
    ].copy()

    for feature in selected_features:
        train_out[feature] = train_imp[feature]
        test_out[feature] = test_imp[feature]

    return (
        train_out.reset_index(drop=True),
        test_out.reset_index(drop=True),
    )


def patient_true_label(group: pd.DataFrame, label_col: str):
    """Return the subject-level majority label, matching C2 subject logic."""
    counts = group[label_col].value_counts()

    if counts.empty:
        return None

    return counts.index[0]


def probability_columns(prefix: str, classes):
    """Create consistent probability column names."""
    columns = []

    for cls in classes:
        safe = str(cls).replace(" ", "_")
        columns.append(
            f"{prefix}_prob_{safe}"
        )

    return columns


def generate_patient_predictions(
    train_proc: pd.DataFrame,
    test_proc: pd.DataFrame,
    label_col: str,
    selected_features: list[str],
    global_model,
    task_name: str,
):
    """
    Rebuild each client's personalization model and generate
    patient-level predictions.

    If a patient has multiple rows, probabilities are averaged
    across that patient's rows before choosing the final prediction.
    """

    labels = np.asarray(
        global_model.classes_
    )

    results = []

    client_ids = sorted(
        test_proc["client_id"].unique()
    )

    # ---------------------------------------------------------------
    # Build personalized model for each client.
    # ---------------------------------------------------------------

    personalized_models = {}

    for cid in client_ids:

        personalization = train_proc[
            (train_proc["client_id"] == cid)
            & (train_proc["split"] == "personalization")
        ].copy()

        if personalization.empty:
            raise RuntimeError(
                f"No personalization data for client {cid} "
                f"in {task_name}."
            )

        Xp = personalization[
            selected_features
        ].to_numpy()

        yp = personalization[
            label_col
        ].to_numpy()

        # EXACT C2 SMOTE.
        Xp_resampled, yp_resampled = smote_resample(
            Xp,
            yp,
            k_neighbors=5,
            random_state=SMOTE_RANDOM_STATE,
        )

        # EXACT C2 personalization:
        # global model + 10 local trees.
        personalized, _ = personalize_global_model(
            global_model,
            Xp_resampled,
            yp_resampled,
            n_trees=PERSONALIZATION_TREES,
            seed=SEED + int(cid),
        )

        personalized_models[int(cid)] = personalized

    # ---------------------------------------------------------------
    # Probability column names.
    # ---------------------------------------------------------------

    global_prob_cols = probability_columns(
        "global",
        labels,
    )

    personalized_prob_cols = probability_columns(
        "personalized",
        labels,
    )

    local_prob_cols = probability_columns(
        "local",
        labels,
    )

    # ---------------------------------------------------------------
    # Generate patient-level results.
    # ---------------------------------------------------------------

    for oasis_id, patient_rows in test_proc.groupby(
        "OASISID",
        sort=True,
    ):

        # C2 assignment guarantees one client per subject.
        client_ids_for_patient = (
            patient_rows["client_id"]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(client_ids_for_patient) != 1:
            raise RuntimeError(
                f"Patient {oasis_id} has inconsistent client assignments: "
                f"{client_ids_for_patient}"
            )

        client_id = int(
            client_ids_for_patient[0]
        )

        personalized_model = personalized_models[
            client_id
        ]

        X_patient = patient_rows[
            selected_features
        ].to_numpy()

        y_patient = patient_rows[
            label_col
        ].to_numpy()

        # -----------------------------------------------------------
        # Global model
        # -----------------------------------------------------------

        global_proba_rows = (
            global_model.predict_proba(
                X_patient
            )
        )

        global_proba = global_proba_rows.mean(
            axis=0
        )

        global_index = int(
            np.argmax(global_proba)
        )

        global_prediction = labels[
            global_index
        ]

        # -----------------------------------------------------------
        # Personalized model
        # -----------------------------------------------------------

        personalized_proba_rows = (
            personalized_model.predict_proba(
                X_patient
            )
        )

        personalized_proba = (
            personalized_proba_rows.mean(axis=0)
        )

        personalized_index = int(
            np.argmax(personalized_proba)
        )

        personalized_prediction = labels[
            personalized_index
        ]

        # -----------------------------------------------------------
        # Local-only model
        #
        # This is generated only for comparison in the dashboard.
        # It uses the same personalization-side training data as C2.
        # -----------------------------------------------------------

        from models.random_forest import train_rf

        personalization = train_proc[
            (train_proc["client_id"] == client_id)
            & (train_proc["split"] == "personalization")
        ]

        Xp = personalization[
            selected_features
        ].to_numpy()

        yp = personalization[
            label_col
        ].to_numpy()

        Xp_local, yp_local = smote_resample(
            Xp,
            yp,
            k_neighbors=5,
            random_state=SMOTE_RANDOM_STATE,
        )

        local_model = train_rf(
            Xp_local,
            yp_local,
        )

        local_proba_rows = (
            local_model.predict_proba(
                X_patient
            )
        )

        local_proba = local_proba_rows.mean(
            axis=0
        )

        local_index = int(
            np.argmax(local_proba)
        )

        local_prediction = labels[
            local_index
        ]

        # -----------------------------------------------------------
        # True patient label
        # -----------------------------------------------------------

        true_label = patient_true_label(
            patient_rows,
            label_col,
        )

        row = {
            "OASISID": str(oasis_id),
            "client_id": client_id,
            "n_rows": int(len(patient_rows)),
            "true_label": true_label,

            "global_prediction": global_prediction,
            "global_confidence": float(
                global_proba[global_index]
            ),

            "personalized_prediction": personalized_prediction,
            "personalized_confidence": float(
                personalized_proba[
                    personalized_index
                ]
            ),

            "local_prediction": local_prediction,
            "local_confidence": float(
                local_proba[local_index]
            ),

            "global_correct": bool(
                global_prediction == true_label
            ),

            "personalized_correct": bool(
                personalized_prediction == true_label
            ),

            "local_correct": bool(
                local_prediction == true_label
            ),
        }

        # Global probabilities.
        for col, value in zip(
            global_prob_cols,
            global_proba,
        ):
            row[col] = float(value)

        # Personalized probabilities.
        for col, value in zip(
            personalized_prob_cols,
            personalized_proba,
        ):
            row[col] = float(value)

        # Local probabilities.
        for col, value in zip(
            local_prob_cols,
            local_proba,
        ):
            row[col] = float(value)

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------

def generate_task(
    task_name: str,
    data_file: Path,
    label_col: str,
):
    print()
    print("=" * 80)
    print(f"GENERATING APPLICATION RESULTS: {task_name.upper()}")
    print("=" * 80)

    task_dir = RESULTS_DIR / "contribution2" / task_name

    assignment_file = (
        task_dir / "sample_assignments.csv"
    )

    model_file = (
        task_dir / "global_model.joblib"
    )

    selected_file = (
        task_dir / "selected_features.csv"
    )

    print(f"Data:        {data_file}")
    print(f"Assignments: {assignment_file}")
    print(f"Model:       {model_file}")

    # ---------------------------------------------------------------
    # Load exact C2 artifacts.
    # ---------------------------------------------------------------

    df = pd.read_csv(
        data_file,
        low_memory=False,
    )

    assignments = pd.read_csv(
        assignment_file,
    )

    bundle = joblib.load(
        model_file
    )

    global_model = bundle["global_model"]

    selected_features = load_selected_features(
        task_dir
    )

    print(
        f"Selected features: {len(selected_features)}"
    )

    print(
        f"Assignment subjects: "
        f"{assignments['OASISID'].nunique()}"
    )

    # ---------------------------------------------------------------
    # Reproduce C2 preprocessing.
    # ---------------------------------------------------------------

    train_proc, test_proc = prepare_c2_data(
        df=df,
        assignments=assignments,
        label_col=label_col,
        selected_features=selected_features,
    )

    print(
        f"Training-side rows: {len(train_proc)}"
    )

    print(
        f"Test rows: {len(test_proc)}"
    )

    print(
        f"Test patients: {test_proc['OASISID'].nunique()}"
    )

    # ---------------------------------------------------------------
    # Generate predictions.
    # ---------------------------------------------------------------

    patient_results = generate_patient_predictions(
        train_proc=train_proc,
        test_proc=test_proc,
        label_col=label_col,
        selected_features=selected_features,
        global_model=global_model,
        task_name=task_name,
    )

    # ---------------------------------------------------------------
    # Save.
    # ---------------------------------------------------------------

    output_file = (
        APPLICATION_DIR
        / f"{task_name}_predictions.csv"
    )

    patient_results.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Saved: {output_file}")
    print(
        f"Patients saved: {len(patient_results)}"
    )

    print()
    print(
        patient_results[
            [
                "OASISID",
                "client_id",
                "true_label",
                "global_prediction",
                "personalized_prediction",
                "local_prediction",
            ]
        ].head(10).to_string(index=False)
    )

    return patient_results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    diagnosis_results = generate_task(
        task_name="diagnosis",
        data_file=RESULTS_DIR
        / "base_dataframe_unimputed.csv",
        label_col="diagnosis_class",
    )

    severity_results = generate_task(
        task_name="severity",
        data_file=RESULTS_DIR
        / "severity_labelled_dataset.csv",
        label_col="severity_label",
    )

    # ---------------------------------------------------------------
    # Create the 48-patient application cohort.
    #
    # These are patients present in BOTH C2 test sets, which is what
    # the current /api/patients endpoint exposes.
    # ---------------------------------------------------------------

    combined = diagnosis_results.merge(
        severity_results,
        on="OASISID",
        how="inner",
        suffixes=(
            "_diagnosis",
            "_severity",
        ),
    )

    combined_file = (
        APPLICATION_DIR
        / "patients.csv"
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    print()
    print("=" * 80)
    print("APPLICATION RESULT GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Diagnosis patients: "
        f"{len(diagnosis_results)}"
    )

    print(
        f"Severity patients: "
        f"{len(severity_results)}"
    )

    print(
        f"Patients available for combined dashboard: "
        f"{len(combined)}"
    )

    print()
    print(
        f"Saved combined file:\n{combined_file}"
    )


if __name__ == "__main__":
    main()