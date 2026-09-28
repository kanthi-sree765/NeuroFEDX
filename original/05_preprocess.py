"""
Week 2 pipeline: build the final ML-ready dataset from
merged_oasis3_fixed.csv.

Steps (matching the base paper's order):
  1. Group the 38 raw dx1 diagnosis labels into 5 classes:
     CN, AD, Non-AD, Uncertain, Others
  2. Select the clinical + psychological + MRI feature columns
  3. KNN imputation (K=2) to fill missing values
  4. Pearson correlation pruning (drop one of any pair with |r| > 0.95)
  5. SMOTE balancing across the 5 classes

Needs: pip install scikit-learn imbalanced-learn

Run from the preprocessing/ folder:
    python 05_preprocess.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError(
        "imbalanced-learn is not installed. Run:\n"
        "  pip install imbalanced-learn scikit-learn"
    )

INPUT_FILE = "../results/merged_oasis3_fixed.csv"
OUTPUT_FILE = "../results/ml_ready_dataset.csv"

PEARSON_THRESHOLD = 0.95

# ============================================================
# 1. DIAGNOSIS -> 5-CLASS GROUPING
# ============================================================
#
# This mapping is taken directly from Table 2 of the actual paper
# (Jahan et al., PLOS ONE 2023), which explicitly lists every raw
# diagnosis label under its 5-class grouping. This replaces an
# earlier version of this script that guessed 3 of these mappings
# incorrectly:
#   - "AD dem cannot be primary" -> paper says AD (not Others)
#   - "Incipient demt PTP"       -> paper says Non-AD (not Others)
#   - "0.5 in memory only"       -> paper says Others (not Uncertain)
#
# Only 2 raw values are genuinely in the paper's "Others" bucket:
# "." (no diagnosis recorded) and "0.5 in memory only".
#
# A few labels present in this OASIS-3 release were not in the
# paper's own Table 2 listing (their subject pool was smaller and
# may not have included every rare label). These are marked
# NEEDS VERIFICATION below and assigned by pattern (any "AD dem..."
# label is AD) rather than guessed individually.

DX1_TO_CLASS = {
    "Cognitively normal": "CN",
    "AD Dementia": "AD",
    "uncertain dementia": "Uncertain",
    "Unc: ques. Impairment": "Uncertain",
    "AD dem w/depresss, not contribut": "AD",
    ".": "Others",
    "Incipient demt PTP": "Non-AD",
    "0.5 in memory only": "Others",
    "DLBD, primary": "Non-AD",
    "AD dem w/depresss, contribut": "AD",
    "Non AD dem, Other primary": "Non-AD",
    "uncertain, possible NON AD dem": "Uncertain",
    "Frontotemporal demt. prim": "Non-AD",
    "Vascular Demt, primary": "Non-AD",
    "AD dem w/CVD not contrib": "AD",
    "AD dem Language dysf after": "AD",
    "AD dem Language dysf prior": "AD",
    "AD dem w/PDI after AD dem not contrib": "AD",
    "Dementia/PD, primary": "Non-AD",
    "AD dem Language dysf with": "AD",
    "AD dem distrubed social, after": "AD",
    "AD dem w/CVD contribut": "AD",
    "AD dem w/oth (list B) contribut": "AD",
    "Unc: impair reversible": "Uncertain",
    "AD dem cannot be primary": "AD",
    "AD dem w/oth (list B) not contrib": "AD",
    "AD dem distrubed social, with": "AD",
    "Vascular Demt, secondary": "Non-AD",
    "Incipient Non-AD dem": "Non-AD",
    "AD dem w/PDI after AD dem contribut": "AD",
    "AD dem visuospatial, with": "AD",
    "AD dem w/oth unusual features/demt on": "AD",
    "AD dem/FLD prior to AD dem": "AD",              # NEEDS VERIFICATION (mixed dementia, not in paper's Table 2)
    "AD dem w/oth unusual feat/subs demt": "AD",
    "AD dem distrubed social, prior": "AD",
    "AD dem visuospatial, prior": "AD",               # NEEDS VERIFICATION (not in paper's Table 2)
    "AD dem w/oth unusual features": "AD",            # NEEDS VERIFICATION (not in paper's Table 2)
    "AD dem visuospatial, after": "AD",
}

NEEDS_VERIFICATION = [
    "AD dem/FLD prior to AD dem",
    "AD dem visuospatial, prior",
    "AD dem w/oth unusual features",
]


# ============================================================
# 2. FEATURE COLUMNS
# ============================================================
# Twin columns not already coalesced by the merge script (WAIS,
# MEMTIME, BOSTON use different names depending on UDS version).
EXTRA_TWIN_COLUMNS = [
    ("WAIS", "digsym", "WAIS"),
    ("MEMTIME", "lmdelay", "MEMTIME"),
    ("BOSTON", "bnt", "BOSTON"),
]

CLINICAL_CANDIDATES = [
    "AgeatEntry", "Judgment", "Commun", "Memory", "Orient", "Perscare",
    "MMSE", "APOE", "CDRSUM", "HEIGHT", "WEIGHT", "Homehobb",
]

# Full 17-variable psychological list from the paper's Table 4.
# TRAILARR, TRAILALI, TRAILBRR, TRAILBLI are included here even though
# they were already confirmed unavailable in this OASIS-3 release --
# keeping them in the list means the script's own MISSING report
# documents that gap clearly rather than silently omitting it.
PSYCH_CANDIDATES = [
    "DIGIF", "DIGIFLEN", "DIGIB", "DIGIBLEN", "LOGIMEM", "MEMUNITS",
    "ANIMALS", "VEG", "TRAILA", "TRAILARR", "TRAILALI", "TRAILB",
    "TRAILBRR", "TRAILBLI", "WAIS", "MEMTIME", "BOSTON",
]

# Exact 10-variable MRI list from the paper's Table 4 -- NOT every
# FreeSurfer region. An earlier version of this script auto-detected
# every column containing "Vol" (114 columns: hippocampus, amygdala,
# every named cortical region, etc.) instead of just these 10 global
# summary volumes, which inflated the feature count to 141 and made
# the reproduction not comparable to the paper.
MRI_CANDIDATES = [
    "IntraCranialVol", "lhCortexVol", "RhCortexVol", "CortexVol",
    "SubCortGrayVol", "TotalGrayVol", "SupraTentorialVol",
    "LhCorticalWhiteMatterVol", "RhCorticalWhiteMatterVol",
    "CorticalWhiteMatterVol",
]


def find_column(df, name):
    """Case-insensitive column lookup. Returns the actual column name
    if found, else None."""
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(name.lower())


def main():
    print("=" * 80)
    print("LOADING MERGED DATASET")
    print("=" * 80)
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Rows: {len(df)}")

    # --------------------------------------------------------
    # Diagnosis grouping
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("GROUPING DIAGNOSES INTO 5 CLASSES")
    print("=" * 80)

    unmapped = set(df["dx1"].dropna().unique()) - set(DX1_TO_CLASS.keys())
    if unmapped:
        print("\nWARNING: dx1 values found that are NOT in the mapping dict:")
        for v in unmapped:
            print("  -", v)
        print("These rows will be dropped. Add them to DX1_TO_CLASS if they")
        print("should be kept.")

    df["diagnosis_class"] = df["dx1"].map(DX1_TO_CLASS)

    print("\nClass distribution:")
    print(df["diagnosis_class"].value_counts(dropna=False).to_string())

    verification_rows = df["dx1"].isin(NEEDS_VERIFICATION).sum()
    print(f"\n{verification_rows} rows used a NEEDS-VERIFICATION mapping "
          f"(see script comments) -- flag these in your report.")

    before = len(df)
    df = df.dropna(subset=["diagnosis_class"]).reset_index(drop=True)
    print(f"\nDropped {before - len(df)} rows with no diagnosis class.")
    print(f"Remaining rows: {len(df)}")

    # --------------------------------------------------------
    # Coalesce the remaining twin columns (WAIS/MEMTIME/BOSTON)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("COALESCING REMAINING TWIN COLUMNS (WAIS, MEMTIME, BOSTON)")
    print("=" * 80)
    for paper_name, primary_col, fallback_col in EXTRA_TWIN_COLUMNS:
        primary_actual = find_column(df, primary_col)
        fallback_actual = find_column(df, fallback_col)
        cols_present = [c for c in (primary_actual, fallback_actual) if c]
        if not cols_present:
            print(f"  {paper_name}: not found under either name -- skipped")
            continue
        if len(cols_present) == 1:
            df[paper_name] = df[cols_present[0]]
        else:
            df[paper_name] = df[primary_actual].fillna(df[fallback_actual])
        print(f"  {paper_name}: coalesced from {cols_present} "
              f"-> {df[paper_name].notna().sum()} non-null")

    # --------------------------------------------------------
    # Select feature columns
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("SELECTING FEATURE COLUMNS")
    print("=" * 80)

    feature_cols = []
    print("\nClinical:")
    for name in CLINICAL_CANDIDATES:
        actual = find_column(df, name)
        if actual:
            feature_cols.append(actual)
            print(f"  FOUND   {name:20s} -> {actual}")
        else:
            print(f"  MISSING {name:20s} -- not in dataset, skipped")

    print("\nPsychological:")
    for name in PSYCH_CANDIDATES:
        actual = find_column(df, name)
        if actual and actual not in feature_cols:
            feature_cols.append(actual)
            print(f"  FOUND   {name:20s} -> {actual}")
        else:
            print(f"  MISSING {name:20s} -- not in dataset, skipped")

    print("\nMRI / FreeSurfer (exact 10 variables from the paper's Table 4):")
    for name in MRI_CANDIDATES:
        actual = find_column(df, name)
        if actual and actual not in feature_cols:
            feature_cols.append(actual)
            print(f"  FOUND   {name:25s} -> {actual}")
        else:
            print(f"  MISSING {name:25s} -- not in dataset, skipped")

    print(f"\nTotal features selected: {len(feature_cols)}")

    # Coerce everything to numeric (APOE etc. may be read as object/string)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --------------------------------------------------------
    # KNN imputation
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("KNN IMPUTATION (K=2)")
    print("=" * 80)

    missing_before = df[feature_cols].isna().sum().sum()
    print(f"Total missing values before imputation: {missing_before}")

    imputer = KNNImputer(n_neighbors=2)
    imputed_array = imputer.fit_transform(df[feature_cols])
    df_imputed = pd.DataFrame(imputed_array, columns=feature_cols, index=df.index)

    missing_after = df_imputed.isna().sum().sum()
    print(f"Total missing values after imputation: {missing_after}")

    # --------------------------------------------------------
    # Pearson correlation pruning
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"PEARSON CORRELATION PRUNING (threshold = {PEARSON_THRESHOLD})")
    print("=" * 80)

    corr_matrix = df_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > PEARSON_THRESHOLD].tolist()
        for other in high_corr:
            if other not in to_drop and col not in to_drop:
                to_drop.add(col)
                print(f"  Dropping '{col}' (correlated {upper.loc[other, col]:.3f} "
                      f"with '{other}')")

    kept_cols = [c for c in feature_cols if c not in to_drop]
    print(f"\nDropped {len(to_drop)} redundant features.")
    print(f"Remaining features: {len(kept_cols)}")

    X = df_imputed[kept_cols]
    y = df["diagnosis_class"]
    unbalanced_df = X.copy()
    unbalanced_df["diagnosis_class"] = y.values
    unbalanced_df.to_csv(OUTPUT_FILE.replace(".csv", "_unbalanced.csv"), index=False)
    pd.DataFrame({"OASISID": df["OASISID"].values}).to_csv(
    OUTPUT_FILE.replace(".csv", "_groups.csv"), index=False)

    # --------------------------------------------------------
    # SMOTE balancing
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("SMOTE BALANCING")
    print("=" * 80)

    print("\nClass distribution before SMOTE:")
    print(y.value_counts().to_string())

    min_class_size = y.value_counts().min()
    k_neighbors = max(1, min(5, min_class_size - 1))
    if k_neighbors < 5:
        print(f"\nSmallest class has only {min_class_size} rows -- using "
              f"k_neighbors={k_neighbors} for SMOTE instead of the default 5.")

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts().to_string())

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    final_df = X_resampled.copy()
    final_df["diagnosis_class"] = y_resampled
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Saved: {OUTPUT_FILE}")
    print("\nThis file is ready for Random Forest / Federated Learning.")


if __name__ == "__main__":
    main()