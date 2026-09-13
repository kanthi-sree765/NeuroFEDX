import pandas as pd
from pathlib import Path
import re

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# FILES
# ============================================================

FILES = {
    "Demographics": DATA_DIR / "OASIS3_demographics.csv",
    "UDSa1": DATA_DIR / "OASIS3_UDSa1_participant_demo.csv",
    "UDSb1": DATA_DIR / "OASIS3_UDSb1_physical_eval.csv",
    "UDSb4": DATA_DIR / "OASIS3_UDSb4_cdr.csv",
    "UDSd1": DATA_DIR / "OASIS3_UDSd1_diagnoses.csv",
    "Psychometrics": DATA_DIR / "OASIS3_UDSc1_cognitive_assessments.csv",
    "FreeSurfer": DATA_DIR / "OASIS3_Freesurfer_output.csv",
}

# Your actual dictionary file
DICTIONARY_FILE = PROJECT_ROOT / "data" / "OASIS3_Clinical and Psychometrics_dictionary.xlsx"


# ============================================================
# PAPER FEATURES
# ============================================================

paper_features = [
    "DIGIF",
    "DIGIFLEN",
    "DIGIB",
    "DIGIBLEN",
    "LOGIMEM",
    "MEMUNITS",
    "ANIMALS",
    "VEG",
    "TRAILA",
    "TRAILARR",
    "TRAILALI",
    "TRAILB",
    "TRAILBRR",
    "TRAILBLI",
    "WAIS",
    "MEMTIME",
    "BOSTON",
    "lhCortexVol",
    "IntraCranialVol",
    "SupraTentorialVol",
    "LhCorticalWhiteMatterVol",
    "SubCortGrayVol",
    "AgeAtEntry",
    "Homehobb",
    "Judgment",
    "Commun",
    "Memory",
    "Orient",
    "Perscare",
    "MMSE",
    "APOE",
    "Sumbox",
    "Height",
    "Weight",
]

# Remove accidental duplicate names while preserving order
paper_features = list(dict.fromkeys(paper_features))


# ============================================================
# CANDIDATE MAPPINGS
# ============================================================
#
# IMPORTANT:
# These are ONLY candidate mappings.
# They are NOT automatically marked Verified.
#
# The script checks the real CSV columns and dictionary.
# ============================================================

candidate_mappings = {

    # Psychological
    "DIGIF": ("Psychometrics", "digforct"),
    "DIGIFLEN": ("Psychometrics", "digforsl"),
    "DIGIB": ("Psychometrics", "digbacct"),
    "DIGIBLEN": ("Psychometrics", "digbacls"),

    "LOGIMEM": ("Psychometrics", "LOGIMEM"),
    "MEMUNITS": ("Psychometrics", "MEMUNITS"),
    "ANIMALS": ("Psychometrics", "ANIMALS"),
    "VEG": ("Psychometrics", "VEG"),

    "TRAILA": ("Psychometrics", "tma"),
    "TRAILARR": ("Psychometrics", None),
    "TRAILALI": ("Psychometrics", None),

    "TRAILB": ("Psychometrics", "tmb"),
    "TRAILBRR": ("Psychometrics", None),
    "TRAILBLI": ("Psychometrics", None),

    "WAIS": ("Psychometrics", "digsym"),
    "MEMTIME": ("Psychometrics", "lmdelay"),
    "BOSTON": ("Psychometrics", "bnt"),

    # Clinical
    "AgeAtEntry": ("Demographics", "AgeatEntry"),
    "APOE": ("Demographics", "APOE"),

    "Height": ("UDSb1", "HEIGHT"),
    "Weight": ("UDSb1", "WEIGHT"),

    "MMSE": ("UDSb4", "MMSE"),
    "Memory": ("UDSb4", "memory"),
    "Orient": ("UDSb4", "orient"),
    "Judgment": ("UDSb4", "judgment"),
    "Commun": ("UDSb4", "commun"),
    "Homehobb": ("UDSb4", "homehobb"),
    "Perscare": ("UDSb4", "perscare"),
    "Sumbox": ("UDSb4", "CDRSUM"),

    # FreeSurfer
    "lhCortexVol": ("FreeSurfer", "lhCortexVol"),
    "IntraCranialVol": ("FreeSurfer", "IntraCranialVol"),
    "SupraTentorialVol": ("FreeSurfer", "SupraTentorialVol"),
    "LhCorticalWhiteMatterVol": (
        "FreeSurfer",
        "lhCorticalWhiteMatterVol"
    ),
    "SubCortGrayVol": (
        "FreeSurfer",
        "SubCortGrayVol"
    ),
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):
    """
    Converts names to a common comparison form.

    Example:
        TRAILARR
        TrailArr
        trail_arr

    all become:

        trailarr
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    return re.sub(r"[^a-z0-9]", "", value)


# ============================================================
# LOAD CSV COLUMN NAMES
# ============================================================

print("=" * 80)
print("LOADING OASIS DATA STRUCTURE")
print("=" * 80)

datasets = {}

for source, path in FILES.items():

    if not path.exists():
        print(f"\nWARNING: Missing file:")
        print(path)
        continue

    try:

        df = pd.read_csv(path, nrows=5)

        datasets[source] = {
            "path": path,
            "columns": list(df.columns),
            "normalized": {
                normalize(column): column
                for column in df.columns
            }
        }

    except Exception as e:

        print(f"\nERROR reading {path}")
        print(e)

print(
    f"\nDatasets successfully inspected: {len(datasets)}"
)


# ============================================================
# LOAD EXCEL DICTIONARY
# ============================================================

print("\n" + "=" * 80)
print("LOADING OASIS DICTIONARY")
print("=" * 80)

if not DICTIONARY_FILE.exists():

    print("\nERROR:")
    print("Dictionary file not found:")
    print(DICTIONARY_FILE)

    raise SystemExit


try:

    dictionary_sheets = pd.read_excel(
        DICTIONARY_FILE,
        sheet_name=None,
        engine="openpyxl"
    )

except Exception as e:

    print("\nERROR loading dictionary:")
    print(e)

    raise SystemExit


print(
    f"\nDictionary sheets found: "
    f"{len(dictionary_sheets)}"
)

for sheet_name in dictionary_sheets:
    print(f"  - {sheet_name}")


# ============================================================
# BUILD FAST DICTIONARY INDEX
# ============================================================
#
# Instead of searching every cell repeatedly, we create
# one index containing normalized values.
# ============================================================

print("\nBuilding dictionary index...")

dictionary_index = {}

for sheet_name, sheet in dictionary_sheets.items():

    if sheet.empty:
        continue

    for column in sheet.columns:

        # Convert entire column at once
        normalized_values = (
            sheet[column]
            .astype(str)
            .map(normalize)
        )

        for row_index, normalized_value in normalized_values.items():

            if not normalized_value:
                continue

            if normalized_value not in dictionary_index:

                dictionary_index[normalized_value] = []

            row = sheet.loc[row_index]

            evidence = " | ".join(
                str(value)
                for value in row.tolist()
                if str(value) != "nan"
            )

            dictionary_index[normalized_value].append({
                "sheet": sheet_name,
                "row": row_index + 2,
                "column": column,
                "evidence": evidence
            })


print(
    f"Dictionary index entries: "
    f"{len(dictionary_index)}"
)


# ============================================================
# DICTIONARY SEARCH
# ============================================================

def dictionary_lookup(variable):

    key = normalize(variable)

    return dictionary_index.get(key, [])


# ============================================================
# VERIFY FEATURES
# ============================================================

results = []

print("\n" + "=" * 80)
print("VERIFYING FEATURES")
print("=" * 80)

for feature in paper_features:

    source, candidate = candidate_mappings.get(
        feature,
        (None, None)
    )

    # --------------------------------------------------------
    # No mapping supplied
    # --------------------------------------------------------

    if source is None:

        results.append({
            "Paper_Feature": feature,
            "OASIS_Source": "",
            "Candidate_Column": "",
            "Actual_Column": "",
            "CSV_Column_Exists": False,
            "Dictionary_Match": False,
            "Dictionary_Sheet": "",
            "Dictionary_Row": "",
            "Dictionary_Evidence": "",
            "Status": "No candidate mapping"
        })

        continue


    # --------------------------------------------------------
    # Dataset missing
    # --------------------------------------------------------

    if source not in datasets:

        results.append({
            "Paper_Feature": feature,
            "OASIS_Source": source,
            "Candidate_Column": candidate or "",
            "Actual_Column": "",
            "CSV_Column_Exists": False,
            "Dictionary_Match": False,
            "Dictionary_Sheet": "",
            "Dictionary_Row": "",
            "Dictionary_Evidence": "",
            "Status": "Source dataset missing"
        })

        continue


    dataset = datasets[source]


    # --------------------------------------------------------
    # Check candidate column
    # --------------------------------------------------------

    actual_column = None

    if candidate:

        actual_column = dataset["normalized"].get(
            normalize(candidate)
        )

    csv_exists = actual_column is not None


    # --------------------------------------------------------
    # Check paper variable in dictionary
    # --------------------------------------------------------

    dictionary_matches = dictionary_lookup(feature)

    dictionary_found = len(dictionary_matches) > 0


    sheet_name = ""
    row_number = ""
    evidence = ""

    if dictionary_matches:

        match = dictionary_matches[0]

        sheet_name = match["sheet"]
        row_number = match["row"]
        evidence = match["evidence"]


    # --------------------------------------------------------
    # Special handling for the four unavailable variables
    # --------------------------------------------------------

    unavailable_features = {
        "TRAILARR",
        "TRAILALI",
        "TRAILBRR",
        "TRAILBLI"
    }

    if feature in unavailable_features:

        if dictionary_found and not csv_exists:

            status = "Dictionary variable found; CSV value unavailable"

        elif not dictionary_found and not csv_exists:

            status = "Not found in dictionary or CSV"

        else:

            status = "Requires manual investigation"


    # --------------------------------------------------------
    # Normal candidate mapping
    # --------------------------------------------------------

    else:

        if csv_exists and dictionary_found:

            status = "Candidate verified structurally"

        elif csv_exists and not dictionary_found:

            status = "CSV column found; dictionary variable not found"

        elif not csv_exists and dictionary_found:

            status = "Dictionary variable found; candidate CSV column missing"

        else:

            status = "Not verified"


    results.append({
        "Paper_Feature": feature,
        "OASIS_Source": source,
        "Candidate_Column": candidate or "",
        "Actual_Column": actual_column or "",
        "CSV_Column_Exists": csv_exists,
        "Dictionary_Match": dictionary_found,
        "Dictionary_Sheet": sheet_name,
        "Dictionary_Row": row_number,
        "Dictionary_Evidence": evidence,
        "Status": status
    })


# ============================================================
# CREATE AUDIT DATAFRAME
# ============================================================

audit_df = pd.DataFrame(results)


# ============================================================
# SAVE DETAILED AUDIT
# ============================================================

audit_file = RESULTS_DIR / "feature_mapping_audit.csv"

audit_df.to_csv(
    audit_file,
    index=False
)


# ============================================================
# SAVE SIMPLE MAPPING
# ============================================================

simple_df = audit_df[
    [
        "Paper_Feature",
        "OASIS_Source",
        "Candidate_Column",
        "Actual_Column",
        "Status"
    ]
].copy()

mapping_file = RESULTS_DIR / "feature_mapping.csv"

simple_df.to_csv(
    mapping_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE MAPPING AUDIT COMPLETE")
print("=" * 80)

print(
    f"\nTotal paper features checked: "
    f"{len(audit_df)}"
)

print(
    f"Candidate verified structurally: "
    f"{(audit_df['Status'] == 'Candidate verified structurally').sum()}"
)

print(
    f"Dictionary variable found; CSV value unavailable: "
    f"{(audit_df['Status'] == 'Dictionary variable found; CSV value unavailable').sum()}"
)

print(
    f"CSV column found; dictionary variable not found: "
    f"{(audit_df['Status'] == 'CSV column found; dictionary variable not found').sum()}"
)

print(
    f"Not verified: "
    f"{(audit_df['Status'] == 'Not verified').sum()}"
)

print("\n" + "-" * 80)
print("FEATURE RESULTS")
print("-" * 80)

for _, row in audit_df.iterrows():

    print(
        f"\n{row['Paper_Feature']}"
    )

    print(
        f"  Source: {row['OASIS_Source']}"
    )

    print(
        f"  Candidate: {row['Candidate_Column'] or 'NONE'}"
    )

    print(
        f"  Actual CSV column: "
        f"{row['Actual_Column'] or 'NOT FOUND'}"
    )

    print(
        f"  CSV exists: "
        f"{row['CSV_Column_Exists']}"
    )

    print(
        f"  Dictionary match: "
        f"{row['Dictionary_Match']}"
    )

    print(
        f"  Status: "
        f"{row['Status']}"
    )

    if row["Dictionary_Sheet"]:

        print(
            f"  Dictionary sheet: "
            f"{row['Dictionary_Sheet']}"
        )

        print(
            f"  Dictionary row: "
            f"{row['Dictionary_Row']}"
        )


# ============================================================
# FINAL FILE LOCATIONS
# ============================================================

print("\n" + "=" * 80)
print("FILES CREATED")
print("=" * 80)

print(f"\nDetailed audit:")
print(audit_file)

print(f"\nSimple mapping:")
print(mapping_file)

print("\nDone.")