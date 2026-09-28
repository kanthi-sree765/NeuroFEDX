"""
TRACK A: PAPER-FAITHFUL / POTENTIALLY LEAKAGE-AFFECTED preprocessing.

Reproduces the sequence implied by the paper's text (Section IV.B):
    dataset -> KNN impute -> Pearson feature selection -> SMOTE -> model

All three steps are fit on the FULL dataset, before any train/test
split. The paper does not state the order relative to splitting, and
neither did the original project code -- this is exactly the ambiguity
flagged in the audit. This script keeps that ambiguity and reproduces
the leaky reading explicitly, so it can be labeled and never confused
with the corrected track.

Output: results/paper_faithful_ml_ready.csv (imputed, pruned, SMOTE-
balanced -- ready for a plain train_test_split downstream).

DO NOT use this file's accuracy as "the" result. Use it side-by-side
with the leakage-free track (evaluation/leakage_free.py) and the
grouped-CV track (evaluation/grouped_cv.py).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR, KNN_NEIGHBORS, SMOTE_RANDOM_STATE
from preprocessing.preprocess_common import pearson_prune, smote_resample


def main():
    base = pd.read_csv(RESULTS_DIR / "base_dataframe_unimputed.csv")
    feature_cols = [c for c in base.columns
                    if c not in ("OASISID", "diagnosis_class", "MRI_matched")]

    print("=" * 80)
    print("TRACK A -- PAPER-FAITHFUL (whole-dataset KNN -> Pearson -> SMOTE)")
    print("=" * 80)
    print(f"Starting candidates: {len(feature_cols)} features, {len(base)} rows")

    # ---- Step 1: KNN imputation on the WHOLE dataset ----
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_imputed = imputer.fit_transform(base[feature_cols])
    imputed_df = pd.DataFrame(X_imputed, columns=feature_cols)
    print(f"KNN imputation (k={KNN_NEIGHBORS}) fit on all {len(base)} rows "
          f"[LEAKAGE POINT: whole-dataset fit, matches original project + "
          f"most plausible reading of the paper].")

    # ---- Step 2: Pearson pruning on the WHOLE dataset ----
    survivors, dropped = pearson_prune(imputed_df, feature_cols)
    print(f"[LEAKAGE POINT: Pearson correlations computed on all rows, "
          f"including what will later be the SMOTE-derived test-adjacent "
          f"data -- this matches the paper's stated one-shot preprocessing.]")

    final_df = imputed_df[survivors].copy()
    final_df["diagnosis_class"] = base["diagnosis_class"].values

    # ---- Step 3: SMOTE on the WHOLE dataset ----
    X = final_df[survivors].values
    y = final_df["diagnosis_class"].values
    print(f"\nClass distribution before SMOTE:\n{pd.Series(y).value_counts().to_string()}")

    X_res, y_res = smote_resample(X, y, k_neighbors=5, random_state=SMOTE_RANDOM_STATE)
    print(f"\n[LEAKAGE POINT: SMOTE fit on the full dataset. Synthetic rows "
          f"are interpolations of real rows that will later be split into "
          f"BOTH train and test -- this is the single largest leakage "
          f"source identified in the audit.]")
    print(f"Class distribution after SMOTE:\n{pd.Series(y_res).value_counts().to_string()}")

    out = pd.DataFrame(X_res, columns=survivors)
    out["diagnosis_class"] = y_res

    out_path = RESULTS_DIR / "paper_faithful_ml_ready.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (label: PAPER-FAITHFUL / POTENTIALLY LEAKAGE-AFFECTED)")
    print(f"Final shape: {out.shape}")


if __name__ == "__main__":
    main()
