from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.explain_patient import explain_patient_task


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

APPLICATION_DIR = ROOT / "results" / "application"

PATIENTS_FILE = APPLICATION_DIR / "patients.csv"
DIAGNOSIS_FILE = APPLICATION_DIR / "diagnosis_predictions.csv"
SEVERITY_FILE = APPLICATION_DIR / "severity_predictions.csv"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="NeuroFEDX API",
    description="Alzheimer's Federated Learning Research Dashboard API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD APPLICATION RESULTS
# ============================================================

if not PATIENTS_FILE.exists():
    raise RuntimeError(
        f"Application patient file not found: {PATIENTS_FILE}"
    )

if not DIAGNOSIS_FILE.exists():
    raise RuntimeError(
        f"Diagnosis results file not found: {DIAGNOSIS_FILE}"
    )

if not SEVERITY_FILE.exists():
    raise RuntimeError(
        f"Severity results file not found: {SEVERITY_FILE}"
    )


PATIENTS = pd.read_csv(PATIENTS_FILE)
DIAGNOSIS_RESULTS = pd.read_csv(DIAGNOSIS_FILE)
SEVERITY_RESULTS = pd.read_csv(SEVERITY_FILE)


# Make sure OASISID is always treated as text.
PATIENTS["OASISID"] = PATIENTS["OASISID"].astype(str)
DIAGNOSIS_RESULTS["OASISID"] = DIAGNOSIS_RESULTS["OASISID"].astype(str)
SEVERITY_RESULTS["OASISID"] = SEVERITY_RESULTS["OASISID"].astype(str)


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """Convert pandas/numpy values into JSON-safe Python values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def get_probability_dict(row, prefix):
    """Extract probability columns and safely handle NaN values."""

    result = {}

    for column in row.index:
        if column.startswith(prefix + "_prob_"):
            class_name = column[len(prefix + "_prob_"):]
            value = row[column]

            if pd.isna(value):
                result[class_name] = None
            else:
                result[class_name] = float(value)

    return result

def build_task_result(row, task_name):
    """Build the API response for one task."""

    if task_name == "diagnosis":
        return {
            "client_id": int(row["client_id"]),
            "n_rows": int(row["n_rows"]),
            "true_label": clean_value(row["true_label"]),

            "global": {
                "prediction": clean_value(
                    row["global_prediction"]
                ),
                "confidence": clean_value(
    			row["global_confidence"]
		),                
		"probabilities": get_probability_dict(
                    row,
                    "global",
                ),
                "correct": bool(
                    row["global_correct"]
                ),
            },

            "personalized": {
                "prediction": clean_value(
                    row["personalized_prediction"]
                ),
                "confidence": clean_value(
                    row["personalized_confidence"]
                ),
                "probabilities": get_probability_dict(
                    row,
                    "personalized",
                ),
                "correct": bool(
                    row["personalized_correct"]
                ),
            },

            "local": {
                "prediction": clean_value(
                    row["local_prediction"]
                ),
                "confidence": clean_value(
                    row["local_confidence"]
                ),
                "probabilities": get_probability_dict(
                    row,
                    "local",
                ),
                "correct": bool(
                    row["local_correct"]
                ),
            },
        }

    if task_name == "severity":
        return {
            "client_id": int(row["client_id"]),
            "n_rows": int(row["n_rows"]),
            "true_label": int(row["true_label"]),

            "global": {
                "prediction": int(
                    row["global_prediction"]
                ),
                "confidence": float(
                    row["global_confidence"]
                ),
                "probabilities": get_probability_dict(
                    row,
                    "global",
                ),
                "correct": bool(
                    row["global_correct"]
                ),
            },

            "personalized": {
                "prediction": int(
                    row["personalized_prediction"]
                ),
                "confidence": float(
                    row["personalized_confidence"]
                ),
                "probabilities": get_probability_dict(
                    row,
                    "personalized",
                ),
                "correct": bool(
                    row["personalized_correct"]
                ),
            },

            "local": {
                "prediction": int(
                    row["local_prediction"]
                ),
                "confidence": float(
                    row["local_confidence"]
                ),
                "probabilities": get_probability_dict(
                    row,
                    "local",
                ),
                "correct": bool(
                    row["local_correct"]
                ),
            },
        }

    raise ValueError(
        f"Unknown task: {task_name}"
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message": "NeuroFEDX API is running",
        "status": "ok",
    }


@app.get("/api/patients")
def get_patients():

    patients = []

    for _, row in PATIENTS.iterrows():

        patients.append({
            "OASISID": str(row["OASISID"]),

            "diagnosis_client_id": int(
                row["client_id_diagnosis"]
            ),

            "severity_client_id": int(
                row["client_id_severity"]
            ),
        })

    return {
        "count": len(patients),
        "patients": patients,
    }


@app.get("/api/patient/{oasis_id}")
def get_patient(oasis_id: str):

    diagnosis = DIAGNOSIS_RESULTS[
        DIAGNOSIS_RESULTS["OASISID"] == oasis_id
    ]

    severity = SEVERITY_RESULTS[
        SEVERITY_RESULTS["OASISID"] == oasis_id
    ]

    if diagnosis.empty and severity.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {oasis_id} was not found.",
        )

    result = {
        "OASISID": oasis_id,
        "diagnosis": None,
        "severity": None,
    }

    if not diagnosis.empty:

        diagnosis_row = diagnosis.iloc[0]

        result["diagnosis"] = build_task_result(
            diagnosis_row,
            "diagnosis",
        )

    if not severity.empty:

        severity_row = severity.iloc[0]

        result["severity"] = build_task_result(
            severity_row,
            "severity",
        )

    return result

# ============================================================
# PATIENT EXPLANATION
# ============================================================

@app.get("/api/patient/{oasis_id}/explanation")
def get_patient_explanation(oasis_id: str):

    oasis_id = str(oasis_id)

    if oasis_id not in set(PATIENTS["OASISID"]):
        raise HTTPException(
            status_code=404,
            detail=f"Patient {oasis_id} was not found.",
        )

    try:
        diagnosis = explain_patient_task(
            oasis_id,
            "diagnosis",
        )

        severity = explain_patient_task(
            oasis_id,
            "severity",
        )

        return {
            "OASISID": oasis_id,
            "diagnosis": diagnosis,
            "severity": severity,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {exc}",
        )