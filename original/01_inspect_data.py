import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Project locations
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# --------------------------------------------------
# 2. Files we are going to inspect
# --------------------------------------------------

files = {
    "Demo": "OASIS3_demographics.csv",
    "UDSa1": "OASIS3_UDSa1_participant_demo.csv",
    "UDSb1": "OASIS3_UDSb1_physical_eval.csv",
    "UDSb4": "OASIS3_UDSb4_cdr.csv",
    "UDSd1": "OASIS3_UDSd1_diagnoses.csv",
    "Psychometrics": "OASIS3_UDSc1_cognitive_assessments.csv",
    "FreeSurfer": "OASIS3_Freesurfer_output.csv"
}

# --------------------------------------------------
# 3. Inspect every file
# --------------------------------------------------

for name, filename in files.items():

    file_path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"{name}: {filename}")
    print("=" * 70)

    if not file_path.exists():
        print("FILE NOT FOUND")
        print(f"Expected location: {file_path}")
        continue

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # Unique participants
    if "OASISID" in df.columns:
        print(f"Unique OASIS participants: {df['OASISID'].nunique():,}")

    elif "Subject" in df.columns:
        print(f"Unique subjects: {df['Subject'].nunique():,}")

    # Show important ID columns
    id_columns = [
        col for col in [
            "OASISID",
            "OASIS_session_label",
            "days_to_visit",
            "Subject",
            "MR_session"
        ]
        if col in df.columns
    ]

    print("\nID/session columns:")
    print(id_columns)

    # Missing-value summary
    missing = df.isna().sum()

    print(f"\nTotal missing values: {missing.sum():,}")

    print("\nColumns with missing values:")
    print(missing[missing > 0].sort_values(ascending=False).head(10))

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)