"""
Shared preprocessing step used by BOTH tracks (Rule 7).

This module only does what is safe to do before any train/test split:
  - apply the verified diagnosis mapping (results/diagnosis_mapping.csv)
  - select the columns corresponding to the VERIFIED/INTERPRETED features in
    results/final_feature_mapping.csv (UNAVAILABLE features are simply
    not present as columns -- nothing is imputed or invented for them)
  - keep OASISID as a group key for participant-level splitting

It deliberately does NOT impute, does NOT run Pearson pruning, and does
NOT run SMOTE, because doing any of those on the full dataset before a
split is exactly the leakage this project is correcting. Those steps
are applied separately by:
  - preprocessing/preprocess_paper_faithful.py  (whole-dataset, as the
    original project + most plausible reading of the paper did)
  - evaluation/leakage_free.py and evaluation/grouped_cv.py (fit on
    training folds only)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR

RENAME_TO_PAPER = {
    "AgeatEntry": "AgeAtEntry",
    "CDRSUM": "Sumbox",
    "HEIGHT": "Height",
    "WEIGHT": "Weight",
    "lhCorticalWhiteMatterVol": "LhCorticalWhiteMatterVol",
}


def load_diagnosis_mapping():
    df = pd.read_csv(RESULTS_DIR / "diagnosis_mapping.csv")
    return dict(zip(df["OASIS_DX"], df["Paper_Class"]))


def load_feature_mapping():
    return pd.read_csv(RESULTS_DIR / "final_feature_mapping.csv")


def get_available_feature_columns():
    """Return all available paper features before Pearson selection.

    The five MRI variables that the paper reports as being removed at the
    0.95 Pearson threshold must remain in this pre-selection table.  They
    are available features in the source data; excluding them here would
    bypass the Pearson stage rather than reproduce it.  Only the five
    genuinely unavailable variables are excluded at this stage.
    """
    fmap = load_feature_mapping()
    usable = fmap[fmap["status"].isin(["VERIFIED", "INTERPRETED"])]
    return usable[["paper_feature", "oasis_column"]]


def build_base_dataframe(merged_df):
    """From the merged clinical+psych+MRI table, produce a dataframe with:
       OASISID, one column per available paper feature (paper-name headers),
       diagnosis_class (mapped), MRI_matched (bool).
    Rows whose dx1 cannot be mapped are dropped (with a printed count).
    No imputation, no scaling, no SMOTE."""

    dx_map = load_diagnosis_mapping()
    feature_table = get_available_feature_columns()

    df = merged_df.copy()
    df = df.rename(columns=RENAME_TO_PAPER)

    # Some paper-name columns already match after rename; others are raw
    # OASIS column names untouched. Build the final column list from
    # oasis_column, but present under the paper's own feature name.
    keep_cols = {"OASISID": "OASISID"}
    missing = []
    for _, row in feature_table.iterrows():
        paper_name, oasis_col = row["paper_feature"], row["oasis_column"]
        candidate = paper_name if paper_name in df.columns else oasis_col
        if candidate in df.columns:
            keep_cols[candidate] = paper_name
        else:
            missing.append((paper_name, oasis_col))

    if missing:
        print(f"WARNING: {len(missing)} mapped features not found as "
              f"columns in the merged dataframe (unexpected -- check "
              f"merge.py output): {missing}")

    out = df[list(keep_cols.keys())].rename(columns=keep_cols)

    df["dx1_mapped"] = df["dx1"].map(dx_map)
    unmapped_mask = df["dx1_mapped"].isna() & df["dx1"].notna()
    n_unmapped = int(unmapped_mask.sum())
    if n_unmapped:
        print(f"Dropping {n_unmapped} rows with a dx1 value not in "
              f"diagnosis_mapping.csv.")

    out["diagnosis_class"] = df["dx1_mapped"]
    out["MRI_matched"] = df["MRI_matched"] if "MRI_matched" in df.columns else np.nan

    before = len(out)
    out = out.dropna(subset=["diagnosis_class"]).reset_index(drop=True)
    print(f"Rows with a usable diagnosis label: {len(out)}/{before}")

    # numeric coercion for everything except identifiers/labels
    non_numeric = {"OASISID", "diagnosis_class", "MRI_matched"}
    for col in out.columns:
        if col not in non_numeric:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    print(f"Base dataframe: {out.shape[0]} rows x {out.shape[1]} columns "
          f"({out.shape[1] - 3} candidate features + OASISID + label + MRI_matched)")
    print(f"Class distribution:\n{out['diagnosis_class'].value_counts().to_string()}")

    return out


def pearson_prune(df, feature_cols, threshold=None, verbose=True):
    """
    Genuine Pearson-correlation pruning (not a hardcoded list): for every
    pair of features with |correlation| > threshold, drop the SECOND
    one encountered (stable, deterministic given feature_cols order).

    This is run independently on whatever data is passed in -- the
    whole dataset for the paper-faithful track, or the training fold
    only for the leakage-free / grouped-CV tracks -- so the exact set
    of dropped features is a genuine measurement, not the paper's list
    copied in.
    """
    from config.model_config import PEARSON_THRESHOLD
    threshold = threshold if threshold is not None else PEARSON_THRESHOLD

    corr = df[feature_cols].corr().abs()
    to_drop = []
    kept = list(feature_cols)
    for i, f1 in enumerate(feature_cols):
        if f1 in to_drop:
            continue
        for f2 in feature_cols[i + 1:]:
            if f2 in to_drop:
                continue
            if corr.loc[f1, f2] > threshold:
                to_drop.append(f2)

    survivors = [c for c in feature_cols if c not in to_drop]
    if verbose:
        print(f"Pearson pruning at threshold {threshold}: dropped "
              f"{len(to_drop)}/{len(feature_cols)} features: {to_drop}")
    return survivors, to_drop


def smote_resample(X, y, k_neighbors=5, random_state=42):
    """
    Minimal from-scratch SMOTE (imbalanced-learn is unavailable in this
    offline environment). Standard algorithm (Chawla et al. 2002):
    for each minority-class sample, pick a random one of its k nearest
    same-class neighbors and interpolate a synthetic point along the
    line between them. Every class is oversampled up to the size of
    the largest class, matching sklearn-imblearn's default behaviour
    and the paper's stated result (all classes brought to the majority
    class's count).
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    X_parts = [X]
    y_parts = [y]

    for cls, n in zip(classes, counts):
        n_needed = target - n
        if n_needed <= 0:
            continue
        X_cls = X[y == cls]
        k = min(k_neighbors, len(X_cls) - 1)
        if k < 1:
            # Can't interpolate with fewer than 2 samples; duplicate instead.
            idx = rng.randint(0, len(X_cls), size=n_needed)
            synthetic = X_cls[idx]
        else:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X_cls)
            _, neighbors = nn.kneighbors(X_cls)
            base_idx = rng.randint(0, len(X_cls), size=n_needed)
            neighbor_choice = rng.randint(1, k + 1, size=n_needed)  # skip self (col 0)
            neighbor_idx = neighbors[base_idx, neighbor_choice]
            gaps = rng.uniform(0, 1, size=(n_needed, X.shape[1]))
            synthetic = X_cls[base_idx] + gaps * (X_cls[neighbor_idx] - X_cls[base_idx])

        X_parts.append(synthetic)
        y_parts.append(np.full(n_needed, cls))

    X_res = np.vstack(X_parts)
    y_res = np.concatenate(y_parts)
    perm = rng.permutation(len(y_res))
    return X_res[perm], y_res[perm]


if __name__ == "__main__":
    merged = pd.read_csv(RESULTS_DIR / "merged_oasis3.csv", low_memory=False)
    base = build_base_dataframe(merged)
    out_path = RESULTS_DIR / "base_dataframe_unimputed.csv"
    base.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
