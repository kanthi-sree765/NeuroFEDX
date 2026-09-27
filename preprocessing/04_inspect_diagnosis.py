"""
Quick inspection of diagnosis-related columns in merged_oasis3_fixed.csv.

Run this BEFORE building the 5-class grouping (CN / AD / Non-AD /
Uncertain / Others). We need to see the actual diagnosis codes present
in this OASIS-3 release before mapping them -- guessing the mapping
risks silently mislabeling patients, which would invalidate everything
downstream.

Run from the preprocessing/ folder:
    python 04_inspect_diagnosis.py
"""

import pandas as pd

df = pd.read_csv("../results/merged_oasis3_fixed.csv", low_memory=False)

print("=" * 80)
print(f"Total rows: {len(df)}")
print("=" * 80)

# Find every column whose name looks diagnosis-related, so we don't
# miss one (dx1..dx5, NORMCOG, DEMENTED, etc. depending on UDS version)
candidate_cols = [c for c in df.columns if any(
    key in c.lower() for key in ["dx", "normcog", "demented", "diag"]
)]

print("\nDiagnosis-related columns found:")
for c in candidate_cols:
    print(" -", c)

for col in candidate_cols:
    print("\n" + "-" * 80)
    print(f"Column: {col}")
    print(f"Non-null: {df[col].notna().sum()} / {len(df)}")
    print(f"Unique values: {df[col].nunique()}")
    print(df[col].value_counts(dropna=False).head(40).to_string())

print("\n" + "=" * 80)
print("Next: send me this full printout. I'll build the 5-class")
print("grouping (CN / AD / Non-AD / Uncertain / Others) from the")
print("actual codes shown above -- not guessed.")
print("=" * 80)
