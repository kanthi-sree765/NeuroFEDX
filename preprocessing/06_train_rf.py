"""
Random Forest training on the ML-ready dataset, matching the base
paper's exact configuration (Jahan et al., PLOS ONE 2023 / IEEE
Access 2025):

  - 100 trees
  - Gini impurity
  - max_depth = 20
  - min_samples_split = 2
  - min_samples_leaf = 1
  - max_features = 'sqrt'
  - bootstrap = True
  - random_state = 42
  - 80/20 train/test split
  - Stratified 10-fold cross-validation

Paper's reported results (for comparison):
  Training accuracy:  100%
  Testing accuracy:   98.84%
  10-fold CV accuracy: 98.81%
  Precision: 98.94%   Recall: 98.79%   F1: 98.75%   AUC: 99.97%

Needs: pip install scikit-learn

Run from the preprocessing/ folder:
    python 06_train_rf.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

INPUT_FILE = "../results/ml_ready_dataset.csv"
LABEL_COL = "diagnosis_class"

RF_PARAMS = dict(
    n_estimators=100,
    criterion="gini",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    class_weight=None,   # paper: "Weight one is meant to be assigned to all classes"
)


def main():
    print("=" * 80)
    print("LOADING ML-READY DATASET")
    print("=" * 80)
    df = pd.read_csv(INPUT_FILE)
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df[LABEL_COL].value_counts().to_string()}")

    X = df.drop(columns=[LABEL_COL])
    y_raw = df[LABEL_COL]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"\nClass encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # --------------------------------------------------------
    # Train/test split (80/20, stratified)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT (80/20, stratified)")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST")
    print("=" * 80)
    print("Parameters:", RF_PARAMS)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTraining accuracy: {train_acc:.4%}")
    print(f"Testing accuracy:  {test_acc:.4%}")

    # --------------------------------------------------------
    # Test-set metrics (precision / recall / F1 / AUC)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SET METRICS")
    print("=" * 80)

    precision = precision_score(y_test, test_pred, average="weighted")
    recall = recall_score(y_test, test_pred, average="weighted")
    f1 = f1_score(y_test, test_pred, average="weighted")

    test_proba = rf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    auc = roc_auc_score(y_test_bin, test_proba, average="weighted", multi_class="ovr")

    print(f"Precision (weighted): {precision:.4%}")
    print(f"Recall (weighted):    {recall:.4%}")
    print(f"F1-score (weighted):  {f1:.4%}")
    print(f"AUC (weighted, OvR):  {auc:.4%}")

    print("\nClassification report:")
    print(classification_report(y_test, test_pred, target_names=le.classes_))

    print("\nConfusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())

    # --------------------------------------------------------
    # Stratified 10-fold cross-validation
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRATIFIED 10-FOLD CROSS-VALIDATION")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    rf_cv = RandomForestClassifier(**RF_PARAMS)
    cv_scores = cross_val_score(rf_cv, X, y, cv=skf, scoring="accuracy")

    print(f"Per-fold accuracy: {np.round(cv_scores, 4).tolist()}")
    print(f"Mean CV accuracy:  {cv_scores.mean():.4%}")
    print(f"Std CV accuracy:   {cv_scores.std():.4%}")

    # --------------------------------------------------------
    # Feature importance (quick preview -- full SHAP comes later)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TOP 15 FEATURES BY RF IMPORTANCE")
    print("=" * 80)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(15).to_string())

    # --------------------------------------------------------
    # Final comparison against the paper
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON WITH THE PAPER")
    print("=" * 80)
    print(f"{'Metric':30s} {'This run':>12s} {'Paper':>12s}")
    print(f"{'Training accuracy':30s} {train_acc:11.2%} {'100%':>12s}")
    print(f"{'Testing accuracy':30s} {test_acc:11.2%} {'98.84%':>12s}")
    print(f"{'10-fold CV accuracy':30s} {cv_scores.mean():11.2%} {'98.81%':>12s}")
    print(f"{'Precision':30s} {precision:11.2%} {'98.94%':>12s}")
    print(f"{'Recall':30s} {recall:11.2%} {'98.79%':>12s}")
    print(f"{'F1-score':30s} {f1:11.2%} {'98.75%':>12s}")
    print(f"{'AUC':30s} {auc:11.2%} {'99.97%':>12s}")

    print("\nDon't be alarmed if these numbers differ somewhat from the")
    print("paper's -- differences in dataset size (23,275 rows here vs.")
    print("their smaller pool), the 4 missing Trail-test features, and")
    print("natural variation are all expected. What matters is being in")
    print("the same ballpark and understanding WHY any gap exists.")
    print("=" * 80)


if __name__ == "__main__":
    main()