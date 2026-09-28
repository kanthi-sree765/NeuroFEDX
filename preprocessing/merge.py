"""
Merge OASIS-3 UDS forms + FreeSurfer into one clinical+psychological+MRI
table (Rule 2).

CORRECTED 2026-09-26 vs. the original 03_merge_datasets_fixed.py:
  - MRI_TOLERANCE_DAYS is no longer 10**9 (unlimited). It is now a
    configurable parameter (config.model_config.MRI_TOLERANCE_DAYS_DEFAULT
    = 365 days) chosen because the paper describes imaging/cognitive
    visits recurring "every two to three years" -- the only documented
    cadence available to anchor a choice on. See
    evaluation/mri_tolerance_sensitivity.py for the sweep across
    180/365/540/730/1095 days that keeps this choice honest (not picked
    for accuracy).
  - Every row's matching decision is now written to
    results/session_matching_audit.csv (clinical subject/visit, MRI
    subject/visit, gap in days, matched true/false, rule used).

Can be run standalone (uses the default tolerance) or imported by the
sensitivity-sweep script with a different tolerance.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import (
    DATA_DIR, RESULTS_DIR, UDS_TOLERANCE_DAYS, MRI_TOLERANCE_DAYS_DEFAULT,
)

FILES = {
    "demographics": "OASIS3_demographics.csv",
    "udsa1": "OASIS3_UDSa1_participant_demo.csv",
    "udsb1": "OASIS3_UDSb1_physical_eval.csv",
    "udsb4": "OASIS3_UDSb4_cdr.csv",
    "udsc1": "OASIS3_UDSc1_cognitive_assessments.csv",
    "udsd1": "OASIS3_UDSd1_diagnoses.csv",
    "freesurfer": "OASIS3_Freesurfer_output.csv",
}

TWIN_COLUMNS = [
    ("DIGIF", "digforct", "DIGIF"),
    ("DIGIFLEN", "digforsl", None),
    ("DIGIB", "digbacct", "DIGIB"),
    ("DIGIBLEN", "digbacls", None),
    ("TRAILA", "tma", "TRAILA"),
    ("TRAILB", "tmb", "trailb"),
    ("LOGIMEM", "logmem", "LOGIMEM"),
]


def extract_day(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r"_d(\d+)$", expand=False),
        errors="coerce",
    )


def combine_duplicate_records(df, name, key_columns=("OASISID", "visit_day")):
    key_columns = list(key_columns)
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if not duplicate_mask.any():
        return df, []
    duplicates = df.loc[duplicate_mask].copy()
    value_columns = [c for c in df.columns if c not in key_columns]
    combined_rows, conflicts = [], []
    for key, group in duplicates.groupby(key_columns, dropna=False):
        combined = {key_columns[0]: key[0], key_columns[1]: key[1]}
        for column in value_columns:
            values = group[column].dropna()
            if len(values) == 0:
                combined[column] = np.nan
                continue
            if len(values) == 1:
                combined[column] = values.iloc[0]
                continue
            unique_values = values.astype(str).unique()
            combined[column] = values.iloc[0]
            if len(unique_values) != 1:
                conflicts.append({"OASISID": key[0], "visit_day": key[1],
                                   "variable": column,
                                   "values": " | ".join(unique_values)})
        combined_rows.append(combined)
    combined_duplicates = pd.DataFrame(combined_rows)
    non_duplicates = df.loc[~duplicate_mask].copy()
    result = pd.concat([non_duplicates, combined_duplicates], ignore_index=True)
    result = result.sort_values(key_columns).reset_index(drop=True)
    return result, conflicts


def nearest_visit_merge(base_df, base_day_col, other_df, other_day_col,
                         subject_col, tolerance_days, label, keep_all_gaps=None):
    """Attach nearest-in-time other_df row per subject, within tolerance.
    If keep_all_gaps is a list, every candidate (subject, gap, matched)
    decision is appended to it for audit purposes (only the nearest per
    base row, which is what actually gets used)."""
    other_cols = [c for c in other_df.columns
                  if c not in (subject_col, other_day_col)
                  and c not in base_df.columns]

    out_frames = []
    all_gaps = []

    for subject, base_group in base_df.groupby(subject_col, sort=False):
        base_group = base_group.reset_index(drop=True)
        other_group = other_df[other_df[subject_col] == subject].reset_index(drop=True)

        if other_group.empty:
            filler = pd.DataFrame(np.nan, index=base_group.index, columns=other_cols)
            filler[f"{label}_gap_days"] = np.nan
            out_frames.append(pd.concat([base_group, filler], axis=1))
            if keep_all_gaps is not None:
                for _, brow in base_group.iterrows():
                    keep_all_gaps.append({
                        "OASISID": subject,
                        "clinical_visit_day": brow[base_day_col],
                        "mri_visit_day": np.nan,
                        "gap_days": np.nan,
                        "matched": False,
                        "reason": "subject has zero MRI/FreeSurfer records",
                    })
            continue

        base_days = base_group[base_day_col].to_numpy()
        other_days = other_group[other_day_col].to_numpy()
        distances = np.abs(base_days[:, None] - other_days[None, :])

        nearest_idx = distances.argmin(axis=1)
        nearest_gap = distances[np.arange(len(base_group)), nearest_idx]

        matched_block = other_group.iloc[nearest_idx][other_cols].reset_index(drop=True).copy()
        within_tol = nearest_gap <= tolerance_days

        for col in other_cols:
            matched_block.loc[~within_tol, col] = np.nan
        matched_block[f"{label}_gap_days"] = np.where(within_tol, nearest_gap, np.nan)

        all_gaps.extend(nearest_gap[within_tol].tolist())
        out_frames.append(pd.concat([base_group, matched_block], axis=1))

        if keep_all_gaps is not None:
            nearest_mri_day = other_days[nearest_idx]
            for i in range(len(base_group)):
                keep_all_gaps.append({
                    "OASISID": subject,
                    "clinical_visit_day": int(base_days[i]),
                    "mri_visit_day": int(nearest_mri_day[i]),
                    "gap_days": int(nearest_gap[i]),
                    "matched": bool(within_tol[i]),
                    "reason": "within tolerance" if within_tol[i]
                              else f"nearest MRI is {int(nearest_gap[i])} days away "
                                   f"(exceeds {tolerance_days}-day tolerance)",
                })

    merged = pd.concat(out_frames, ignore_index=True).copy()
    return merged, pd.Series(all_gaps, name=f"{label}_gap_days")


def coalesce_twins(df):
    for paper_name, primary_col, fallback_col in TWIN_COLUMNS:
        cols_present = [c for c in (primary_col, fallback_col) if c and c in df.columns]
        if not cols_present:
            continue
        if len(cols_present) == 1:
            df[paper_name] = df[cols_present[0]]
        else:
            df[paper_name] = df[primary_col].fillna(df[fallback_col])
    return df


def build_merged_dataset(mri_tolerance_days=MRI_TOLERANCE_DAYS_DEFAULT,
                          uds_tolerance_days=UDS_TOLERANCE_DAYS,
                          write_audit=True, verbose=True):
    """Runs the full merge and returns (final_df, summary_dict).
    Writes results/merged_oasis3.csv and, if write_audit,
    results/session_matching_audit.csv."""

    def log(*a):
        if verbose:
            print(*a)

    raw = {}
    for name, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path, low_memory=False)
        raw[name] = df

    prepared_uds = {}
    all_conflicts = []
    for key in ["udsa1", "udsb1", "udsb4", "udsc1", "udsd1"]:
        df = raw[key].copy()
        df["visit_day"] = extract_day(df["OASIS_session_label"])
        missing_key = df[["OASISID", "visit_day"]].isna().any(axis=1)
        df = df.loc[~missing_key].copy()
        df["visit_day"] = df["visit_day"].astype("int64")
        df = df.drop_duplicates(keep="first").copy()
        df, conflicts = combine_duplicate_records(df, key)
        for c in conflicts:
            c["form"] = key
        all_conflicts.extend(conflicts)
        prepared_uds[key] = df

    if all_conflicts and write_audit:
        pd.DataFrame(all_conflicts).to_csv(RESULTS_DIR / "uds_conflicting_values.csv", index=False)

    clinical = prepared_uds["udsb4"].rename(columns={"visit_day": "anchor_day"})
    for key in ["udsa1", "udsb1", "udsc1", "udsd1"]:
        other = prepared_uds[key]
        clinical, gaps = nearest_visit_merge(
            base_df=clinical, base_day_col="anchor_day", other_df=other,
            other_day_col="visit_day", subject_col="OASISID",
            tolerance_days=uds_tolerance_days, label=key,
        )
    clinical = clinical.rename(columns={"anchor_day": "visit_day"})

    has_psych = clinical["udsc1_gap_days"].notna()
    clinical = clinical.loc[has_psych].reset_index(drop=True)

    clinical = coalesce_twins(clinical)

    demographics = raw["demographics"].copy()
    demo_cols = [c for c in demographics.columns
                 if c != "OASISID" and c not in clinical.columns]
    clinical = clinical.merge(demographics[["OASISID"] + demo_cols], on="OASISID", how="left")

    fs = raw["freesurfer"].copy()
    fs.rename(columns={"Subject": "OASISID"}, inplace=True)
    if "FS QC Status" in fs.columns:
        fs["_qc_norm"] = fs["FS QC Status"].astype(str).str.strip().str.lower()
        fs = fs[fs["_qc_norm"] != "quarantined"].drop(columns="_qc_norm")
    fs["MRI_day"] = extract_day(fs["MR_session"])
    fs = fs.dropna(subset=["MRI_day"]).copy()
    fs["MRI_day"] = fs["MRI_day"].astype("int64")

    audit_rows = [] if write_audit else None
    final, mri_gaps = nearest_visit_merge(
        base_df=clinical, base_day_col="visit_day", other_df=fs,
        other_day_col="MRI_day", subject_col="OASISID",
        tolerance_days=mri_tolerance_days, label="MRI",
        keep_all_gaps=audit_rows,
    )
    final["MRI_matched"] = final["MRI_gap_days"].notna()

    RESULTS_DIR.mkdir(exist_ok=True)
    merged_file = RESULTS_DIR / "merged_oasis3.csv"
    final.to_csv(merged_file, index=False)

    if write_audit and audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        audit_df["matching_rule"] = (
            f"nearest-day match, MRI_TOLERANCE_DAYS={mri_tolerance_days}"
        )
        audit_df.to_csv(RESULTS_DIR / "session_matching_audit.csv", index=False)

    n_rows = len(final)
    n_subjects = final["OASISID"].nunique()
    n_matched = int(final["MRI_matched"].sum())
    n_matched_subjects = final.loc[final["MRI_matched"], "OASISID"].nunique()

    summary = dict(
        mri_tolerance_days=mri_tolerance_days,
        n_rows=n_rows,
        n_subjects=n_subjects,
        n_mri_matched_rows=n_matched,
        n_mri_missing_rows=n_rows - n_matched,
        n_mri_matched_subjects=n_matched_subjects,
        mri_match_rate=n_matched / n_rows if n_rows else float("nan"),
        median_matched_gap_days=float(final.loc[final["MRI_matched"], "MRI_gap_days"].median())
            if n_matched else float("nan"),
        pct_matched_gap_over_365d=float(
            (final.loc[final["MRI_matched"], "MRI_gap_days"] > 365).mean()
        ) if n_matched else float("nan"),
    )

    log("=" * 80)
    log(f"MERGE COMPLETE (MRI_TOLERANCE_DAYS={mri_tolerance_days})")
    log("=" * 80)
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\nPaper reference values: rows=3342, MRI-matched=3220, "
        f"missing=122, subjects=810, MRI-matched-subjects=799")
    log(f"Saved: {merged_file}")

    return final, summary


if __name__ == "__main__":
    build_merged_dataset()
