"""
Central Random Forest trainer, used by every track. Imports RF_PARAMS
from config/model_config.py so the hyperparameters are defined in ONE
place (Rule 6) -- no script downstream hardcodes max_depth, n_estimators,
etc.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    f1_score, confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RF_PARAMS, CLASS_NAMES


def train_rf(X_train, y_train, **overrides):
    params = dict(RF_PARAMS)
    params.update(overrides)
    rf = RandomForestClassifier(n_jobs=-1, **params)
    rf.fit(X_train, y_train)
    return rf


def evaluate(model, X_test, y_test, class_names=None, label=""):
    """Returns a dict of overall metrics plus a per-class DataFrame and
    the confusion matrix, computed honestly (no reweighting tricks)."""
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    classes_in_model = model.classes_
    if class_names is None:
        class_names = list(classes_in_model)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes_in_model, zero_division=0
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    try:
        y_test_bin = label_binarize(y_test, classes=classes_in_model)
        auc = roc_auc_score(y_test_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=classes_in_model)

    per_class = pd.DataFrame({
        "class": classes_in_model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    summary = dict(
        label=label,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=precision.mean(),
        macro_recall=recall.mean(),
        auc_macro_ovr=auc,
        n_test=len(y_test),
    )

    return summary, per_class, cm, classes_in_model
