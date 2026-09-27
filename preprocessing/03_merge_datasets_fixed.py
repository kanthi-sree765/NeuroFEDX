"""
Fixed merge pipeline for reproducing Jahan et al. (IEEE Access 2025 /
PLoS ONE 2023) on OASIS-3.

CHANGES IN THIS VERSION vs. 03_merge_datasets_fixed.py (first pass):

  - Added a FILTER step after merging the UDS forms: only rows where
    a UDSc1 (psychological/cognitive) record was found within the
    tolerance window are kept. Without this, the previous version
    counted every clinical visit even when no cognitive test was
    matched nearby, which inflated the row count (8,621) far past
    what the paper means by "3,342 clinical+psychological instances"
    -- the paper's number specifically means "both present together".
  - Fixed a diagnostic-only bug where the per-form match count printed
    during the merge loop was wrong (it also caused a harmless but
    noisy DeprecationWarning).
  - Suppressed pandas' cosmetic "DataFrame is highly fragmented"
    PerformanceWarning -- it does not affect correctness, only speed,
    and was cluttering the output.

WHAT THIS FIXES vs. the ORIGINAL preprocessing/03_merge_datasets.py:

  1. The original script joined the five UDS forms (UDSa1, UDSb1,
     UDSb4, UDSc1, UDSd1) to each other using an EXACT match on
     "visit_day" (the dNNNN suffix in OASIS_session_label). Different
     forms filled out at the same clinic visit are frequently logged
     under slightly different day offsets, so an exact match silently
     produced two half-empty rows instead of one complete row.
     Result: only 394 / 2681 rows had both the CDR block (UDSb4) and
     the cognitive block (UDSc1) populated together.

     FIX: forms are merged with a NEAREST-DAY match per subject,
     within a configurable tolerance window (UDS_TOLERANCE_DAYS), the
     same technique the original script already used for MRI-to-
     clinical matching -- just applied one step earlier, between the
     UDS forms themselves.

  2. The original script anchored the pipeline on the FreeSurfer (MRI)
     table and pulled clinical data toward it, which structurally
     cannot reproduce the paper's "122 missing MRI instances" (that
     number only makes sense if you start from the clinical+
     psychological table and see how many rows fail to find MRI).

     FIX: anchor on the merged clinical+psychological table, then
     left-join FreeSurfer onto it. Unmatched rows keep NaN MRI columns
     and an MRI_matched=False flag, mirroring the paper's reporting.

  3. OASIS-3 carries both UDS-2 and UDS-3 era column names for several
     cognitive tests (e.g. DIGIF vs digforct, TRAILA vs tma, LOGIMEM
     vs logmem). The original script's column-merge step kept
     whichever name it met first and silently dropped rows that only
     had the other name filled in.

     FIX: known UDS-2/UDS-3 twin columns are coalesced into one column
     immediately after the UDS forms are merged.

  4. FreeSurfer rows with FS QC Status == "Quarantined" are dropped
     (2 rows) -- OASIS-3 documentation states these failed QC and
     should not be used. "Passed" and "Passed with edits" (both
     casings seen in the export) are kept.

This script does not invent any data. Where OASIS-3's current release
no longer carries a paper variable (TRAILARR, TRAILALI, TRAILBRR,
TRAILBLI), the corresponding column is left absent and flagged in the
run report -- do not fabricate it.

Run from the preprocessing/ directory, same as the original:
    python 03_merge_datasets_fixed.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "../data"
RESULTS_DIR = "../results"

# Tolerance for matching different UDS forms to the same clinical visit.
# The paper does not state this explicitly; 30 days is a conservative
# starting point. Check udsc1_gap_days.describe() in the printed
# report before widening this.
UDS_TOLERANCE_DAYS = 30

# Tolerance for matching an MRI/FreeSurfer session to the nearest
# clinical+psychological visit.
#
# CHANGED: was 180 days. Real-world result at 180 days: only 28.8% of
# clinical+psych rows found an MRI match, vs. the paper's 96.3%. This
# is because MRI scans happen far less often than clinical visits
# (~2 scans per person vs. ~4.7 clinical visits per person in this
# release) -- most clinical visits simply aren't within 180 days of
# *any* MRI scan, even when the person has MRI data elsewhere in
# their record.
#
# The paper's own language -- "missing MRI instances" for the
# (3342 - 3220 = 122) gap -- reads most naturally as "this person has
# no MRI scan at all", not "this person has a scan, but it's too far
# away in time to count". So the cutoff is effectively removed here:
# a clinical+psych row is matched to whichever MRI scan that subject
# has that is closest in time, no matter the gap. A row only stays
# unmatched if the subject has zero MRI/FreeSurfer records.
#
# Set back to something like 180 or 365 if you want a stricter,
# same-era-only match instead -- check MRI_gap_days.describe() in the
# printed report either way to see what gaps you're actually keeping.
MRI_TOLERANCE_DAYS = 10**9

FILES = {
    "demographics": "OASIS3_demographics.csv",
    "udsa1": "OASIS3_UDSa1_participant_demo.csv",
    "udsb1": "OASIS3_UDSb1_physical_eval.csv",
    "udsb4": "OASIS3_UDSb4_cdr.csv",
    "udsc1": "OASIS3_UDSc1_cognitive_assessments.csv",
    "udsd1": "OASIS3_UDSd1_diagnoses.csv",
    "freesurfer": "OASIS3_Freesurfer_output.csv",
}

# UDS-2 / UDS-3 twin columns to coalesce after the UDS forms are
# merged. Format: (paper_name, uds3_or_alt_col, uds2_col)
TWIN_COLUMNS = [
    ("DIGIF", "digforct", "DIGIF"),
    ("DIGIFLEN", "digforsl", None),
    ("DIGIB", "digbacct", "DIGIB"),
    ("DIGIBLEN", "digbacls", None),
    ("TRAILA", "tma", "TRAILA"),
    ("TRAILB", "tmb", "trailb"),
    ("LOGIMEM", "logmem", "LOGIMEM"),
]


# ============================================================
# HELPERS
# ============================================================

def extract_day(series):
    """Extract dXXXX from an OASIS session label, e.g.
    OAS30001_UDSb4_d0339 -> 339."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"_d(\d+)$", expand=False),
        errors="coerce",
    )


def combine_duplicate_records(df, name, key_columns=("OASISID", "visit_day")):
    """Combine exact duplicate subject-visit rows within one file.
    If values conflict, keep the first and log the conflict rather
    than stopping."""
    key_columns = list(key_columns)
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)

    if not duplicate_mask.any():
        print(f"{name}: no duplicate subject-visit records.")
        return df, []

    duplicates = df.loc[duplicate_mask].copy()
    value_columns = [c for c in df.columns if c not in key_columns]

    combined_rows = []
    conflicts = []

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
            if len(unique_values) == 1:
                combined[column] = values.iloc[0]
            else:
                combined[column] = values.iloc[0]
                conflicts.append(
                    {
                        "OASISID": key[0],
                        "visit_day": key[1],
                        "variable": column,
                        "values": " | ".join(unique_values),
                    }
                )
        combined_rows.append(combined)

    combined_duplicates = pd.DataFrame(combined_rows)
    non_duplicates = df.loc[~duplicate_mask].copy()
    result = pd.concat([non_duplicates, combined_duplicates], ignore_index=True)
    result = result.sort_values(key_columns).reset_index(drop=True)

    print(f"{name}: {len(conflicts)} conflicting duplicate values (first value kept).")
    return result, conflicts


def nearest_visit_merge(base_df, base_day_col, other_df, other_day_col,
                         subject_col, tolerance_days, label):
    """
    For each subject in base_df, attach the columns of other_df from
    the nearest visit (by day offset) within `tolerance_days`. Rows in
    base_df with no other_df record for that subject, or no record
    within tolerance, get NaN for the new columns.

    Returns (merged_df, gap_days_series_for_matched_rows).
    """
    other_cols = [c for c in other_df.columns
                  if c not in (subject_col, other_day_col)
                  and c not in base_df.columns]

    out_frames = []
    all_gaps = []

    for subject, base_group in base_df.groupby(subject_col, sort=False):
        base_group = base_group.reset_index(drop=True)
        other_group = other_df[other_df[subject_col] == subject].reset_index(drop=True)

        if other_group.empty:
            filler = pd.DataFrame(
                np.nan, index=base_group.index, columns=other_cols
            )
            filler[f"{label}_gap_days"] = np.nan
            out_frames.append(pd.concat([base_group, filler], axis=1))
            continue

        base_days = base_group[base_day_col].to_numpy()
        other_days = other_group[other_day_col].to_numpy()
        distances = np.abs(base_days[:, None] - other_days[None, :])

        nearest_idx = distances.argmin(axis=1)
        nearest_gap = distances[np.arange(len(base_group)), nearest_idx]

        matched = other_group.iloc[nearest_idx][other_cols].reset_index(drop=True).copy()
        within_tol = nearest_gap <= tolerance_days

        for col in other_cols:
            matched.loc[~within_tol, col] = np.nan

        matched[f"{label}_gap_days"] = np.where(within_tol, nearest_gap, np.nan)

        all_gaps.extend(nearest_gap[within_tol].tolist())
        out_frames.append(pd.concat([base_group, matched], axis=1))

    merged = pd.concat(out_frames, ignore_index=True).copy()
    return merged, pd.Series(all_gaps, name=f"{label}_gap_days")


def coalesce_twins(df):
    """Combine UDS-2/UDS-3 twin columns into one column per paper
    variable name. Reports how many rows each twin recovers."""
    for paper_name, primary_col, fallback_col in TWIN_COLUMNS:
        cols_present = [c for c in (primary_col, fallback_col) if c and c in df.columns]
        if not cols_present:
            print(f"  {paper_name}: neither {primary_col} nor {fallback_col} present -- skipped")
            continue

        before_each = {c: df[c].notna().sum() for c in cols_present}

        if len(cols_present) == 1:
            df[paper_name] = df[cols_present[0]]
        else:
            df[paper_name] = df[primary_col]
            df[paper_name] = df[paper_name].fillna(df[fallback_col])

        after = df[paper_name].notna().sum()
        print(f"  {paper_name}: " + ", ".join(f"{c}={n}" for c, n in before_each.items())
              + f"  -> coalesced={after}")

    return df


# ============================================================
# LOAD RAW FILES
# ============================================================

print("=" * 80)
print("LOADING OASIS-3 DATA")
print("=" * 80)

raw = {}
for name, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    raw[name] = df
    uid_col = "OASISID" if "OASISID" in df.columns else "Subject"
    print(f"{name:15s} rows={len(df):6d}  participants={df[uid_col].nunique()}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# PREPARE EACH UDS FORM: derive visit_day, dedupe
# ============================================================

print("\n" + "=" * 80)
print("PREPARING UDS FORMS")
print("=" * 80)

prepared_uds = {}
all_conflicts = []

for key in ["udsa1", "udsb1", "udsb4", "udsc1", "udsd1"]:
    df = raw[key].copy()
    print(f"\n{key}")

    if "OASIS_session_label" not in df.columns:
        raise ValueError(f"{key} has no OASIS_session_label column.")

    df["visit_day"] = extract_day(df["OASIS_session_label"])
    missing_key = df[["OASISID", "visit_day"]].isna().any(axis=1)
    print("  records without valid subject/day:", int(missing_key.sum()))
    df = df.loc[~missing_key].copy()
    df["visit_day"] = df["visit_day"].astype("int64")

    before = len(df)
    df = df.drop_duplicates(keep="first").copy()
    print("  exact duplicate rows removed:", before - len(df))

    df, conflicts = combine_duplicate_records(df, key)
    for c in conflicts:
        c["form"] = key
    all_conflicts.extend(conflicts)

    prepared_uds[key] = df

if all_conflicts:
    pd.DataFrame(all_conflicts).to_csv(
        os.path.join(RESULTS_DIR, "uds_conflicting_values.csv"), index=False
    )
    print(f"\n{len(all_conflicts)} total duplicate-value conflicts logged to "
          f"results/uds_conflicting_values.csv")

# ============================================================
# MERGE UDS FORMS -- NEAREST VISIT, NOT EXACT DAY
# ============================================================

print("\n" + "=" * 80)
print(f"MERGING UDS FORMS (nearest visit, tolerance = {UDS_TOLERANCE_DAYS} days)")
print("=" * 80)

# Anchor on UDSb4: it carries the CDR block (Memory, Orient, Judgment,
# Commun, Homehobb, Perscare, MMSE, CDRSUM) that defines a clinical
# visit in the paper's methodology.
clinical = prepared_uds["udsb4"].rename(columns={"visit_day": "anchor_day"})

for key in ["udsa1", "udsb1", "udsc1", "udsd1"]:
    other = prepared_uds[key]
    clinical, gaps = nearest_visit_merge(
        base_df=clinical,
        base_day_col="anchor_day",
        other_df=other,
        other_day_col="visit_day",
        subject_col="OASISID",
        tolerance_days=UDS_TOLERANCE_DAYS,
        label=key,
    )
    if len(gaps):
        print(f"  merged {key}: {len(gaps)} rows matched within tolerance "
              f"(median gap {gaps.median():.1f} days)")
    else:
        print(f"  merged {key}: 0 rows matched within tolerance")

clinical = clinical.rename(columns={"anchor_day": "visit_day"})

print(f"\nClinical rows after UDS merge (before psychological filter): {len(clinical)}")
print(f"Unique participants: {clinical['OASISID'].nunique()}")

# ============================================================
# FILTER: KEEP ONLY ROWS WHERE UDSc1 (PSYCHOLOGICAL) WAS MATCHED
# ============================================================
#
# This is the step that was missing before. The paper's "3,342
# clinical+psychological instances" specifically means rows where
# BOTH blocks are present together -- not every clinical visit
# regardless of whether a nearby cognitive test exists. Without this
# filter the row count includes clinical-only visits, which inflates
# the total far past what the paper reports and understates the MRI
# match rate (matching is attempted against a padded, partly-empty
# base).

print("\n" + "=" * 80)
print("FILTERING TO ROWS WITH BOTH CLINICAL (UDSb4) AND PSYCHOLOGICAL (UDSc1) DATA")
print("=" * 80)

before_filter = len(clinical)
has_psych = clinical["udsc1_gap_days"].notna()

print(f"Rows before filter: {before_filter}")
print(f"Rows with a matched UDSc1 record within {UDS_TOLERANCE_DAYS}-day tolerance: "
      f"{int(has_psych.sum())}")

clinical = clinical.loc[has_psych].reset_index(drop=True)

print(f"Rows after filter: {len(clinical)}")
print(f"Unique participants after filter: {clinical['OASISID'].nunique()}")

if len(clinical):
    print("\nudsc1 match gap statistics (days) for kept rows:")
    print(clinical["udsc1_gap_days"].describe())

# ============================================================
# COALESCE UDS-2 / UDS-3 TWIN COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("COALESCING UDS-2 / UDS-3 TWIN COLUMNS")
print("=" * 80)

clinical = coalesce_twins(clinical)

# ============================================================
# ADD DEMOGRAPHICS (subject-level, no visit_day)
# ============================================================

print("\n" + "=" * 80)
print("ADDING DEMOGRAPHICS")
print("=" * 80)

demographics = raw["demographics"].copy()
if demographics["OASISID"].duplicated().any():
    raise ValueError("Demographics contains duplicate OASISID values.")

demo_cols = [c for c in demographics.columns
             if c != "OASISID" and c not in clinical.columns]
clinical = clinical.merge(
    demographics[["OASISID"] + demo_cols], on="OASISID", how="left"
)

print("Clinical/psychological rows:", len(clinical))
print("Clinical/psychological participants:", clinical["OASISID"].nunique())
print("Unique visits:", clinical[["OASISID", "visit_day"]].drop_duplicates().shape[0])

# ============================================================
# PREPARE FREESURFER: drop quarantined, derive MRI day
# ============================================================

print("\n" + "=" * 80)
print("PREPARING FREESURFER")
print("=" * 80)

fs = raw["freesurfer"].copy()
fs.rename(columns={"Subject": "OASISID"}, inplace=True)

if "FS QC Status" in fs.columns:
    before = len(fs)
    fs["_qc_norm"] = fs["FS QC Status"].astype(str).str.strip().str.lower()
    fs = fs[fs["_qc_norm"] != "quarantined"].drop(columns="_qc_norm")
    print(f"Dropped {before - len(fs)} quarantined FreeSurfer rows.")

fs["MRI_day"] = extract_day(fs["MR_session"])
missing_mri_day = fs["MRI_day"].isna().sum()
if missing_mri_day > 0:
    raise ValueError("FreeSurfer contains invalid MR_session values.")
fs["MRI_day"] = fs["MRI_day"].astype("int64")

print("FreeSurfer rows after QC filter:", len(fs))
print("FreeSurfer participants:", fs["OASISID"].nunique())

# ============================================================
# ATTACH FREESURFER TO CLINICAL+PSYCHOLOGICAL (anchor flipped)
# ============================================================

print("\n" + "=" * 80)
print(f"ATTACHING MRI TO CLINICAL+PSYCHOLOGICAL (tolerance = {MRI_TOLERANCE_DAYS} days)")
print("=" * 80)
print("NOTE: anchor is clinical+psychological (as in the paper), not MRI.")

final, mri_gaps = nearest_visit_merge(
    base_df=clinical,
    base_day_col="visit_day",
    other_df=fs,
    other_day_col="MRI_day",
    subject_col="OASISID",
    tolerance_days=MRI_TOLERANCE_DAYS,
    label="MRI",
)

final["MRI_matched"] = final["MRI_gap_days"].notna()

print("\nMRI_gap_days distribution for matched rows (no cap applied -- see")
print("MRI_TOLERANCE_DAYS comment above for why):")
if final["MRI_matched"].any():
    print(final.loc[final["MRI_matched"], "MRI_gap_days"].describe())

# ============================================================
# SAVE OUTPUT
# ============================================================

merged_file = os.path.join(RESULTS_DIR, "merged_oasis3_fixed.csv")
final.to_csv(merged_file, index=False)

# ============================================================
# FINAL REPORT -- compare against the paper's reported numbers
# ============================================================

print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

n_rows = len(final)
n_subjects = final["OASISID"].nunique()
n_matched = int(final["MRI_matched"].sum())
n_matched_subjects = final.loc[final["MRI_matched"], "OASISID"].nunique()
n_missing_mri = n_rows - n_matched

print(f"{'Metric':40s} {'This run':>12s} {'Paper (2023)':>14s}")
print(f"{'Clinical+psychological rows':40s} {n_rows:12d} {3342:14d}")
print(f"{'MRI-matched rows':40s} {n_matched:12d} {3220:14d}")
print(f"{'Missing MRI rows':40s} {n_missing_mri:12d} {122:14d}")
print(f"{'Unique clinical/psych participants':40s} {n_subjects:12d} {810:14d}")
print(f"{'Unique MRI-matched participants':40s} {n_matched_subjects:12d} {799:14d}")

if "dx1" in final.columns:
    print(f"\nRows with a diagnosis label (dx1): {final['dx1'].notna().sum()} / {n_rows}")

known_unavailable = ["TRAILARR", "TRAILALI", "TRAILBRR", "TRAILBLI"]
print(f"\nPaper variables confirmed unavailable in this OASIS-3 release: "
      f"{known_unavailable}")

print(f"\nSaved: {merged_file}")
print("\nRatios worth checking against the paper (should be in the same")
print("ballpark even though absolute counts differ due to dataset growth):")
paper_visits_per_subject = 3342 / 810
this_visits_per_subject = n_rows / n_subjects if n_subjects else float("nan")
print(f"  Visits per participant -- paper: {paper_visits_per_subject:.2f}   "
      f"this run: {this_visits_per_subject:.2f}")
paper_mri_rate = 3220 / 3342
this_mri_rate = n_matched / n_rows if n_rows else float("nan")
print(f"  MRI match rate         -- paper: {paper_mri_rate:.1%}   "
      f"this run: {this_mri_rate:.1%}")
print("=" * 80)
