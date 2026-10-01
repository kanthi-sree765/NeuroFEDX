"""
Patient-level TreeSHAP explanations for the NeuroFEDX application.

This module:
- Reuses the completed C2 experiment artifacts.
- Reconstructs the same personalization models used by the application.
- Computes genuine TreeSHAP explanations for a selected test patient.
- Aggregates SHAP values across all rows belonging to that patient.
- Maps features to Clinical / Psychological / MRI modalities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.model_config import (
    RESULTS_DIR,
    KNN_NEIGHBORS,
    SMOTE_RANDOM_STATE,
)

from preprocessing.preprocess_common import smote_resample
from federated.personalized_rf import personalize_global_model
from models.random_forest import train_rf

from explainability.contribution3 import (
    tree_union_shap,
    modality_map_for,
    modality_aggregate,
)


# ---------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------

def json_safe(value):
    """Convert NumPy/Pandas values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return [
            json_safe(v)
            for v in value.tolist()
        ]

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


SEED = 42
PERSONALIZATION_TREES = 10

APPLICATION_DIR = RESULTS_DIR / "application"


# ---------------------------------------------------------------------
# Load C2 artifacts
# ---------------------------------------------------------------------

def load_task_artifacts(task: str):
    """Load the exact C2 model, assignments and selected features."""

    task_dir = RESULTS_DIR / "contribution2" / task

    bundle = joblib.load(
        task_dir / "global_model.joblib"
    )

    global_model = bundle["global_model"]

    selected_features = list(
        bundle["selected_features"]
    )

    labels = np.asarray(
        bundle["labels"]
    )

    assignments = pd.read_csv(
        task_dir / "sample_assignments.csv"
    )

    if task == "diagnosis":
        data_file = (
            RESULTS_DIR / "base_dataframe_unimputed.csv"
        )
        label_col = "diagnosis_class"

    elif task == "severity":
        data_file = (
            RESULTS_DIR / "severity_labelled_dataset.csv"
        )
        label_col = "severity_label"

    else:
        raise ValueError(
            f"Unknown task: {task}"
        )

    df = pd.read_csv(
        data_file,
        low_memory=False,
    )

    return (
        df,
        assignments,
        global_model,
        selected_features,
        labels,
        label_col,
    )


# ---------------------------------------------------------------------
# Reproduce C2 preprocessing
# ---------------------------------------------------------------------

def prepare_task_data(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    selected_features: list[str],
    label_col: str,
):
    """Reproduce the C2 training-side imputation and test preprocessing."""

    assignments = assignments[
        ["OASISID", "client_id", "split"]
    ].copy()

    assignments["OASISID"] = (
        assignments["OASISID"].astype(str)
    )

    df = df.copy()

    df["OASISID"] = (
        df["OASISID"].astype(str)
    )

    work = df.merge(
        assignments,
        on="OASISID",
        how="inner",
        validate="many_to_one",
    )

    train_mask = work["split"].isin(
        ["federation", "personalization"]
    )

    test_mask = work["split"].eq("test")

    train_rows = work.loc[
        train_mask
    ].copy()

    test_rows = work.loc[
        test_mask
    ].copy()

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

    train_out = train_rows[
        [
            "OASISID",
            label_col,
            "client_id",
            "split",
        ]
    ].copy()

    test_out = test_rows[
        [
            "OASISID",
            label_col,
            "client_id",
            "split",
        ]
    ].copy()

    for feature in selected_features:
        train_out[feature] = train_imp[
            feature
        ]

        test_out[feature] = test_imp[
            feature
        ]

    return (
        train_out.reset_index(drop=True),
        test_out.reset_index(drop=True),
    )


# ---------------------------------------------------------------------
# Reconstruct personalization models
# ---------------------------------------------------------------------

def reconstruct_personalized_models(
    train_proc: pd.DataFrame,
    selected_features: list[str],
    label_col: str,
    global_model,
):
    """
    Rebuild the same 10-tree client personalization models
    used by the completed C2 application.
    """

    personalized_models = {}
    local_models = {}

    for cid in sorted(
        train_proc["client_id"].unique()
    ):

        personalization = train_proc[
            (train_proc["client_id"] == cid)
            &
            (
                train_proc["split"]
                == "personalization"
            )
        ].copy()

        if personalization.empty:
            continue

        Xp = personalization[
            selected_features
        ].to_numpy()

        yp = personalization[
            label_col
        ].to_numpy()

        Xp_resampled, yp_resampled = (
            smote_resample(
                Xp,
                yp,
                k_neighbors=5,
                random_state=SMOTE_RANDOM_STATE,
            )
        )

        personalized, local_rf = (
            personalize_global_model(
                global_model,
                Xp_resampled,
                yp_resampled,
                n_trees=PERSONALIZATION_TREES,
                seed=SEED + int(cid),
            )
        )

        personalized_models[int(cid)] = (
            personalized
        )

        local_models[int(cid)] = local_rf

    return (
        personalized_models,
        local_models,
    )


# ---------------------------------------------------------------------
# Get trees belonging to a model
# ---------------------------------------------------------------------

def global_tree_specs(global_model):
    """
    Build the tree specification expected by
    contribution3.tree_union_shap().
    """

    specs = []

    for tree in global_model.estimators_:
        specs.append(
            (
                tree,
                np.asarray(
                    global_model.classes_
                ),
            )
        )

    return specs


def personalized_tree_specs(
    global_model,
    local_model,
):
    """
    Build tree specifications for:
        100 global trees + 10 local trees.
    """

    specs = []

    for tree in global_model.estimators_:
        specs.append(
            (
                tree,
                np.asarray(
                    global_model.classes_
                ),
            )
        )

    for tree in local_model.estimators_:
        specs.append(
            (
                tree,
                np.asarray(
                    local_model.classes_
                ),
            )
        )

    return specs


# ---------------------------------------------------------------------
# Explain one patient
# ---------------------------------------------------------------------

def explain_patient_task(
    oasis_id: str,
    task: str,
):
    """
    Generate patient-level explanations for one task.

    Returns:
        prediction information,
        top feature explanations,
        modality contributions.
    """

    (
        df,
        assignments,
        global_model,
        selected_features,
        labels,
        label_col,
    ) = load_task_artifacts(task)

    train_proc, test_proc = prepare_task_data(
        df,
        assignments,
        selected_features,
        label_col,
    )

    oasis_id = str(oasis_id)

    patient_rows = test_proc[
        test_proc["OASISID"].astype(str)
        == oasis_id
    ].copy()

    if patient_rows.empty:
        raise ValueError(
            f"Patient {oasis_id} is not present "
            f"in the {task} C2 test set."
        )

    client_ids = (
        patient_rows["client_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(client_ids) != 1:
        raise ValueError(
            f"Patient {oasis_id} has inconsistent "
            f"client assignments."
        )

    client_id = int(
        client_ids[0]
    )

    personalized_models, local_models = (
        reconstruct_personalized_models(
            train_proc,
            selected_features,
            label_col,
            global_model,
        )
    )

    if client_id not in personalized_models:
        raise ValueError(
            f"No personalized model found for "
            f"client {client_id}."
        )

    personalized_model = (
        personalized_models[client_id]
    )

    local_model = (
        local_models[client_id]
    )

    X_patient = patient_rows[
        selected_features
    ].to_numpy()

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    global_prob_rows = (
        global_model.predict_proba(
            X_patient
        )
    )

    personalized_prob_rows = (
        personalized_model.predict_proba(
            X_patient
        )
    )

    local_prob_rows = (
        local_model.predict_proba(
            X_patient
        )
    )

    global_prob = (
        global_prob_rows.mean(axis=0)
    )

    personalized_prob = (
        personalized_prob_rows.mean(axis=0)
    )

    local_prob = (
        local_prob_rows.mean(axis=0)
    )

    global_idx = int(
        np.argmax(global_prob)
    )

    personalized_idx = int(
        np.argmax(personalized_prob)
    )

    local_idx = int(
        np.argmax(local_prob)
    )

    global_prediction = labels[
        global_idx
    ]

    personalized_prediction = labels[
        personalized_idx
    ]

    local_prediction = labels[
        local_idx
    ]

    # ---------------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------------

    global_specs = global_tree_specs(
        global_model
    )

    personalized_specs = (
        personalized_tree_specs(
            global_model,
            local_model,
        )
    )

    global_values, global_expected = (
        tree_union_shap(
            global_specs,
            X_patient,
            labels,
        )
    )

    personalized_values, personalized_expected = (
        tree_union_shap(
            personalized_specs,
            X_patient,
            labels,
        )
    )

    # ---------------------------------------------------------------
    # Explain the predicted class
    # ---------------------------------------------------------------

    def build_feature_rows(
        values,
        prediction,
        probabilities,
    ):

        class_index = int(
            np.flatnonzero(
                labels == prediction
            )[0]
        )

        shap_matrix = values[
            class_index
        ]

        rows = []

        for row_index in range(
            len(patient_rows)
        ):

            feature_values = X_patient[
                row_index
            ]

            shap_values = shap_matrix[
                row_index
            ]

            for feature, value, shap_value in zip(
                selected_features,
                feature_values,
                shap_values,
            ):

                if shap_value > 0:
                    direction = (
                        "toward predicted class"
                    )

                elif shap_value < 0:
                    direction = (
                        "away from predicted class"
                    )

                else:
                    direction = "neutral"

                rows.append(
                    {
                        "feature": feature,
                        "feature_value": float(
                            value
                        ),
                        "shap_value": float(
                            shap_value
                        ),
                        "abs_shap_value": float(
                            abs(shap_value)
                        ),
                        "direction": direction,
                    }
                )

        return rows

    global_feature_rows = (
        build_feature_rows(
            global_values,
            global_prediction,
            global_prob,
        )
    )

    personalized_feature_rows = (
        build_feature_rows(
            personalized_values,
            personalized_prediction,
            personalized_prob,
        )
    )

    # ---------------------------------------------------------------
    # Aggregate repeated patient rows
    # ---------------------------------------------------------------

    def aggregate_features(rows):

        frame = pd.DataFrame(rows)

        grouped = (
            frame.groupby(
                "feature",
                as_index=False,
            )
            .agg(
                feature_value=(
                    "feature_value",
                    "mean",
                ),
                shap_value=(
                    "shap_value",
                    "mean",
                ),
            )
        )

        grouped["abs_shap_value"] = (
            grouped["shap_value"].abs()
        )

        grouped["direction"] = np.where(
            grouped["shap_value"] > 0,
            "toward predicted class",
            np.where(
                grouped["shap_value"] < 0,
                "away from predicted class",
                "neutral",
            ),
        )

        grouped = grouped.sort_values(
            "abs_shap_value",
            ascending=False,
        )

        return grouped

    global_features = aggregate_features(
        global_feature_rows
    )

    personalized_features = (
        aggregate_features(
            personalized_feature_rows
        )
    )

    # ---------------------------------------------------------------
    # Modality contribution
    # ---------------------------------------------------------------

    modality_map, modality_status = (
        modality_map_for(
            selected_features
        )
    )

    def calculate_modalities(
        feature_frame,
    ):

        rows = []

        for _, r in feature_frame.iterrows():

            rows.append(
                {
                    "feature": r["feature"],
                    "shap_value": float(
                        r["shap_value"]
                    ),
                }
            )

        # IMPORTANT:
        # Use the actual feature name from each row.
        # feature_frame is sorted by SHAP importance,
        # so passing selected_features here would
        # incorrectly pair features with modalities.

        feature_list = [
            row["feature"]
            for row in rows
        ]

        aggregated = modality_aggregate(
            rows,
            feature_list,
            modality_map,
        )

        result = []

        for (
            modality,
            abs_sum,
            signed_sum,
            normalized,
        ) in aggregated:

            if pd.isna(normalized):
                normalized = 0.0

            result.append(
                {
                    "modality": modality,
                    "absolute_shap": float(
                        abs_sum
                    ),
                    "signed_shap": float(
                        signed_sum
                    ),
                    "contribution": float(
                        normalized
                    ),
                }
            )

        return result

    global_modalities = (
        calculate_modalities(
            global_features
        )
    )

    personalized_modalities = (
        calculate_modalities(
            personalized_features
        )
    )

    # ---------------------------------------------------------------
    # Final response
    # ---------------------------------------------------------------

    result = {
        "OASISID": oasis_id,
        "task": task,
        "client_id": client_id,
        "n_rows": int(
            len(patient_rows)
        ),

        "true_label": (
            patient_rows[label_col]
            .value_counts()
            .index[0]
        ),

        "global": {
            "prediction": global_prediction,
            "confidence": float(
                global_prob[global_idx]
            ),
            "probabilities": {
                str(label): float(
                    global_prob[i]
                )
                for i, label in enumerate(
                    labels
                )
            },
            "top_features": (
                global_features
                .head(5)
                .to_dict("records")
            ),
            "modality_contribution": (
                global_modalities
            ),
        },

        "personalized": {
            "prediction": personalized_prediction,
            "confidence": float(
                personalized_prob[
                    personalized_idx
                ]
            ),
            "probabilities": {
                str(label): float(
                    personalized_prob[i]
                )
                for i, label in enumerate(
                    labels
                )
            },
            "top_features": (
                personalized_features
                .head(5)
                .to_dict("records")
            ),
            "modality_contribution": (
                personalized_modalities
            ),
        },

        "mapping_status": {
            feature: modality_status.get(
                feature,
                "MISSING_FROM_MAPPING",
            )
            for feature in selected_features
        },
    }

    # Convert NumPy/Pandas scalar values
    # before FastAPI receives the result.
    return json_safe(result)


# ---------------------------------------------------------------------
# Simple command-line test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import json

    patient = "OAS30091"

    print(
        f"Generating explanation for {patient}..."
    )

    diagnosis = explain_patient_task(
        patient,
        "diagnosis",
    )

    severity = explain_patient_task(
        patient,
        "severity",
    )

    print(
        json.dumps(
            {
                "diagnosis": diagnosis,
                "severity": severity,
            },
            indent=2,
            default=str,
        )
    )

