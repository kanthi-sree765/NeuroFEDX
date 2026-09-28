import pandas as pd
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "../data"
RESULTS_DIR = "../results"

# MRI ↔ UDS matching window
MAX_SESSION_GAP = 180


FILES = {
    "demographics": "OASIS3_demographics.csv",
    "udsa1": "OASIS3_UDSa1_participant_demo.csv",
    "udsb1": "OASIS3_UDSb1_physical_eval.csv",
    "udsb4": "OASIS3_UDSb4_cdr.csv",
    "udsc1": "OASIS3_UDSc1_cognitive_assessments.csv",
    "udsd1": "OASIS3_UDSd1_diagnoses.csv",
    "freesurfer": "OASIS3_Freesurfer_output.csv",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_day(series):
    """
    Extract dXXXX from an OASIS session label.

    Examples:
        OAS30001_UDSb4_d0339 -> 339
        OAS30001_MR_d0129    -> 129
    """

    return pd.to_numeric(
        series.astype(str)
        .str.extract(r"_d(\d+)$", expand=False),
        errors="coerce"
    )


def combine_duplicate_records(df, name):
    """
    Combine duplicate OASIS subject-visit records.

    For duplicate OASISID + visit_day records:
      - If all values are NaN -> keep NaN
      - If only one non-null value exists -> keep it
      - If multiple identical non-null values exist -> keep the value
      - If different non-null values exist -> keep the first value
        and record the conflict in an audit file.

    The pipeline does NOT stop because of a conflict.
    """

    key_columns = [
        "OASISID",
        "visit_day"
    ]

    duplicate_mask = df.duplicated(
        subset=key_columns,
        keep=False
    )

    if not duplicate_mask.any():

        print(
            f"{name}: no duplicate subject-visit records."
        )

        return df

    duplicates = df.loc[
        duplicate_mask
    ].copy()

    duplicate_keys = (
        duplicates[
            key_columns
        ]
        .drop_duplicates()
    )

    print(
        f"{name}: {len(duplicate_keys)} "
        f"duplicate subject-visit keys found."
    )

    value_columns = [
        c for c in df.columns
        if c not in key_columns
    ]

    combined_rows = []
    conflicts = []

    # ========================================================
    # COMBINE EACH DUPLICATE SUBJECT + VISIT
    # ========================================================

    for key, group in duplicates.groupby(
        key_columns,
        dropna=False
    ):

        combined = {
            "OASISID": key[0],
            "visit_day": key[1]
        }

        for column in value_columns:

            values = group[column].dropna()

            # ------------------------------------------------
            # No value in either record
            # ------------------------------------------------

            if len(values) == 0:

                combined[column] = np.nan

                continue

            # ------------------------------------------------
            # Only one non-null value
            # ------------------------------------------------

            if len(values) == 1:

                combined[column] = values.iloc[0]

                continue

            # ------------------------------------------------
            # Multiple non-null values
            # ------------------------------------------------

            unique_values = (
                values.astype(str)
                .unique()
            )

            # ------------------------------------------------
            # Same value in all records
            # ------------------------------------------------

            if len(unique_values) == 1:

                combined[column] = values.iloc[0]

            # ------------------------------------------------
            # Different values
            # ------------------------------------------------

            else:

                # Preserve the first value so that the
                # dataset can continue through preprocessing.
                combined[column] = values.iloc[0]

                conflicts.append({
                    "OASISID": key[0],
                    "visit_day": key[1],
                    "variable": column,
                    "values": " | ".join(
                        unique_values
                    ),
                    "action": "First non-null value retained; conflict audited"
                })

        combined_rows.append(
            combined
        )

    combined_duplicates = pd.DataFrame(
        combined_rows
    )

    # ========================================================
    # NON-DUPLICATE RECORDS
    # ========================================================

    non_duplicates = df.loc[
        ~duplicate_mask
    ].copy()

    # ========================================================
    # COMBINE
    # ========================================================

    result = pd.concat(
        [
            non_duplicates,
            combined_duplicates
        ],
        ignore_index=True
    )

    result = result.sort_values(
        [
            "OASISID",
            "visit_day"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE CONFLICT AUDIT
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    if conflicts:

        conflict_df = pd.DataFrame(
            conflicts
        )

        conflict_file = os.path.join(
            RESULTS_DIR,
            f"{name}_conflicting_values.csv"
        )

        conflict_df.to_csv(
            conflict_file,
            index=False
        )

        print(
            f"{name}: {len(conflicts)} "
            f"non-identical values found in duplicate records."
        )

        print(
            f"Conflict audit saved to: {conflict_file}"
        )

    else:

        print(
            f"{name}: no conflicting non-null values."
        )

    print(
        f"{name}: duplicate records successfully combined."
    )

    print(
        f"{name}: rows after duplicate handling = "
        f"{len(result)}"
    )

    return result

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING OASIS-3 DATA")
print("=" * 80)

data = {}

for name, filename in FILES.items():

    path = os.path.join(
        DATA_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    data[name] = df

    if "OASISID" in df.columns:

        participants = df[
            "OASISID"
        ].nunique()

    else:

        participants = df[
            "Subject"
        ].nunique()

    print(
        f"{name:15s} "
        f"rows={len(df):5d} "
        f"participants={participants}"
    )


# ============================================================
# PREPARE DEMOGRAPHICS
# ============================================================

print("\n" + "=" * 80)
print("PREPARING DEMOGRAPHICS")
print("=" * 80)

demographics = data[
    "demographics"
].copy()

if demographics[
    "OASISID"
].duplicated().any():

    raise ValueError(
        "Demographics contains duplicate OASISID values."
    )

print(
    "Demographics rows:",
    len(demographics)
)

print(
    "Unique participants:",
    demographics[
        "OASISID"
    ].nunique()
)


# ============================================================
# PREPARE UDS DATA
# ============================================================

print("\n" + "=" * 80)
print("PREPARING UDS DATA")
print("=" * 80)

uds_mapping = {
    "UDSa1": "udsa1",
    "UDSb1": "udsb1",
    "UDSb4": "udsb4",
    "UDSc1": "udsc1",
    "UDSd1": "udsd1"
}

prepared_uds = {}


for name, data_key in uds_mapping.items():

    df = data[
        data_key
    ].copy()

    print(
        f"\n{name}"
    )

    # --------------------------------------------------------
    # Extract visit day
    # --------------------------------------------------------

    if "OASIS_session_label" not in df.columns:

        raise ValueError(
            f"{name} does not contain "
            f"OASIS_session_label."
        )

    df["visit_day"] = extract_day(
        df["OASIS_session_label"]
    )

    # --------------------------------------------------------
    # Remove records without usable key
    # --------------------------------------------------------

    missing_key = df[
        [
            "OASISID",
            "visit_day"
        ]
    ].isna().any(axis=1)

    print(
        "Records without valid subject/day:",
        missing_key.sum()
    )

    df = df.loc[
        ~missing_key
    ].copy()

    df["visit_day"] = df[
        "visit_day"
    ].astype("int64")

    # --------------------------------------------------------
    # Remove completely identical rows
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        keep="first"
    ).copy()

    exact_removed = (
        before - len(df)
    )

    print(
        "Exact duplicate rows removed:",
        exact_removed
    )

    # --------------------------------------------------------
    # Combine remaining duplicate subject-visits
    # --------------------------------------------------------

    df = combine_duplicate_records(
        df,
        name
    )

    prepared_uds[
        name
    ] = df


# ============================================================
# MERGE UDS DOMAINS
# ============================================================

print("\n" + "=" * 80)
print("MERGING UDS DOMAINS")
print("=" * 80)

# Start from UDSb4.
clinical = prepared_uds[
    "UDSb4"
].copy()


for name in [
    "UDSa1",
    "UDSb1",
    "UDSc1",
    "UDSd1"
]:

    df = prepared_uds[
        name
    ].copy()

    # Only add columns that are not already present.
    columns_to_add = [
        c for c in df.columns
        if c not in [
            "OASISID",
            "visit_day"
        ]
        and c not in clinical.columns
    ]

    df_to_merge = df[
        [
            "OASISID",
            "visit_day"
        ] + columns_to_add
    ]

    print(
        f"Merging {name}: "
        f"{len(df_to_merge)} rows, "
        f"{len(columns_to_add)} new columns"
    )

    clinical = clinical.merge(
        df_to_merge,
        on=[
            "OASISID",
            "visit_day"
        ],
        how="outer",
        validate="one_to_one"
    )

    print(
        "Rows after merge:",
        len(clinical)
    )


# ============================================================
# ADD DEMOGRAPHICS
# ============================================================

print("\n" + "=" * 80)
print("ADDING DEMOGRAPHICS")
print("=" * 80)

demo_columns = [
    c for c in demographics.columns
    if c != "OASISID"
    and c not in clinical.columns
]

clinical = clinical.merge(
    demographics[
        [
            "OASISID"
        ] + demo_columns
    ],
    on="OASISID",
    how="left",
    validate="many_to_one"
)

print(
    "Clinical/psychological rows:",
    len(clinical)
)

print(
    "Clinical/psychological participants:",
    clinical[
        "OASISID"
    ].nunique()
)

print(
    "Clinical/psychological unique visits:",
    clinical[
        [
            "OASISID",
            "visit_day"
        ]
    ].drop_duplicates().shape[0]
)


# ============================================================
# PREPARE FREESURFER
# ============================================================

print("\n" + "=" * 80)
print("PREPARING FREESURFER")
print("=" * 80)

fs = data[
    "freesurfer"
].copy()

fs.rename(
    columns={
        "Subject": "OASISID"
    },
    inplace=True
)

fs["MRI_day"] = extract_day(
    fs["MR_session"]
)

missing_mri_day = fs[
    "MRI_day"
].isna().sum()

print(
    "FreeSurfer rows:",
    len(fs)
)

print(
    "FreeSurfer participants:",
    fs[
        "OASISID"
    ].nunique()
)

print(
    "Records without valid MRI day:",
    missing_mri_day
)

if missing_mri_day > 0:

    raise ValueError(
        "FreeSurfer contains invalid MR_session values."
    )

fs["MRI_day"] = fs[
    "MRI_day"
].astype("int64")


# ============================================================
# MRI → UDS MATCHING
# ============================================================

print("\n" + "=" * 80)
print("MATCHING MRI SESSIONS TO UDS VISITS")
print("=" * 80)

print(
    f"Maximum allowed session gap: "
    f"{MAX_SESSION_GAP} days"
)

uds_columns = [
    c for c in clinical.columns
    if c not in [
        "OASISID",
        "visit_day"
    ]
]

matched_parts = []


for subject, mri_group in fs.groupby(
    "OASISID",
    sort=False
):

    mri_group = mri_group.reset_index(
        drop=True
    )

    uds_group = clinical[
        clinical[
            "OASISID"
        ] == subject
    ].copy()

    uds_group = uds_group.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # No UDS record
    # --------------------------------------------------------

    if uds_group.empty:

        missing_uds = pd.DataFrame(
            np.nan,
            index=mri_group.index,
            columns=uds_columns
        )
    
        missing_uds["UDS_visit_day"] = np.nan
        missing_uds["session_gap_days"] = np.nan
        missing_uds["MRI_matched"] = False
    
        output = pd.concat(
            [
                mri_group,
                missing_uds
            ],
            axis=1
        )
    
        matched_parts.append(output)
    
        continue

    # --------------------------------------------------------
    # Calculate distances
    # --------------------------------------------------------

    mri_days = mri_group[
        "MRI_day"
    ].to_numpy()

    uds_days = uds_group[
        "visit_day"
    ].to_numpy()

    distances = np.abs(
        mri_days[:, None]
        -
        uds_days[None, :]
    )

    # --------------------------------------------------------
    # Nearest UDS visit
    # --------------------------------------------------------

    nearest_indices = distances.argmin(
        axis=1
    )

    nearest_distances = distances[
        np.arange(
            len(mri_group)
        ),
        nearest_indices
    ]

    nearest_uds = uds_group.iloc[
        nearest_indices
    ].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Build matched UDS dataframe
    # --------------------------------------------------------

    uds_result = nearest_uds[
        uds_columns
    ].copy()

    uds_result[
        "UDS_visit_day"
    ] = nearest_uds[
        "visit_day"
    ].values

    uds_result[
        "session_gap_days"
    ] = nearest_distances

    uds_result[
        "MRI_matched"
    ] = (
        nearest_distances
        <= MAX_SESSION_GAP
    )

    # --------------------------------------------------------
    # Clear UDS values outside matching window
    # --------------------------------------------------------

    unmatched = ~uds_result[
        "MRI_matched"
    ]

    for col in uds_columns:

        uds_result.loc[
            unmatched,
            col
        ] = np.nan

    uds_result.loc[
        unmatched,
        "UDS_visit_day"
    ] = np.nan

    # --------------------------------------------------------
    # Combine MRI + UDS
    # --------------------------------------------------------

    output = pd.concat(
        [
            mri_group,
            uds_result
        ],
        axis=1
    )

    matched_parts.append(
        output
    )


# ============================================================
# COMBINE MATCHED RECORDS
# ============================================================

matched = pd.concat(
    matched_parts,
    ignore_index=True
)


# ============================================================
# MATCHING REPORT
# ============================================================

print("\n" + "=" * 80)
print("MRI / UDS MATCHING RESULTS")
print("=" * 80)

total_mri = len(matched)

matched_count = int(
    matched[
        "MRI_matched"
    ].sum()
)

unmatched_count = (
    total_mri - matched_count
)

matched_subjects = matched.loc[
    matched[
        "MRI_matched"
    ],
    "OASISID"
].nunique()

print(
    "Total MRI records:",
    total_mri
)

print(
    "MRI records matched:",
    matched_count
)

print(
    "MRI records unmatched:",
    unmatched_count
)

print(
    "Unique subjects with matched MRI:",
    matched_subjects
)

if matched_count > 0:

    print(
        "\nSession gap statistics:"
    )

    print(
        matched.loc[
            matched[
                "MRI_matched"
            ],
            "session_gap_days"
        ].describe()
    )


# ============================================================
# SAVE MERGED DATA
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

merged_file = os.path.join(
    RESULTS_DIR,
    "merged_oasis3.csv"
)

matched.to_csv(
    merged_file,
    index=False
)


# ============================================================
# SAVE SESSION MATCHING AUDIT
# ============================================================

audit = matched[
    [
        "OASISID",
        "MR_session",
        "MRI_day",
        "UDS_visit_day",
        "session_gap_days",
        "MRI_matched"
    ]
].copy()

audit_file = os.path.join(
    RESULTS_DIR,
    "session_matching_audit.csv"
)

audit.to_csv(
    audit_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MERGE COMPLETE")
print("=" * 80)

print(
    f"UDS merged rows: {len(clinical)}"
)

print(
    f"UDS participants: "
    f"{clinical['OASISID'].nunique()}"
)

print(
    f"MRI records: {total_mri}"
)

print(
    f"MRI-UDS matched: {matched_count}"
)

print(
    f"MRI-UDS unmatched: {unmatched_count}"
)

print(
    f"Maximum session gap: "
    f"{MAX_SESSION_GAP} days"
)

print(
    "\nFiles created:"
)

print(
    f"  {merged_file}"
)

print(
    f"  {audit_file}"
)

print("=" * 80)