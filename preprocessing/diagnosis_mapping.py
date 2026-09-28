"""
Diagnosis -> 5-class mapping (Rule 5).

Source of truth: Table 2 of the companion paper Jahan et al., PLOS ONE
2023 ("Explainable AI-based Alzheimer's prediction and management using
multimodal data"), which explicitly lists every raw dx1 label under its
5-class grouping. The IEEE Access 2025 paper being reproduced does not
itself republish this table -- it only states the 5-class scheme and
final class counts -- so this mapping is one level removed from the
primary paper. That provenance is recorded in every row below.

Produces results/diagnosis_mapping.csv with columns:
    OASIS_DX, Paper_Class, Evidence, Status
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR

MERGED_FILE = RESULTS_DIR / "merged_oasis3.csv"

# (OASIS_DX, Paper_Class, Evidence, Status)
DX_TABLE = [
    ("Cognitively normal", "CN", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD Dementia", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("uncertain dementia", "Uncertain", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("Unc: ques. Impairment", "Uncertain", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem w/depresss, not contribut", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    (".", "Others", "PLOS ONE 2023 Table 2: 'no diagnosis recorded' -> Others", "VERIFIED"),
    ("Incipient demt PTP", "Non-AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("0.5 in memory only", "Others", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("DLBD, primary", "Non-AD", "Non-AD dementia category (Dementia with Lewy Bodies), per paper text", "VERIFIED"),
    ("AD dem w/depresss, contribut", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("Non AD dem, Other primary", "Non-AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("uncertain, possible NON AD dem", "Uncertain", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("Frontotemporal demt. prim", "Non-AD", "Non-AD dementia category (frontotemporal), per paper text", "VERIFIED"),
    ("Vascular Demt, primary", "Non-AD", "Non-AD dementia category (vascular), per paper text", "VERIFIED"),
    ("AD dem w/CVD not contrib", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem Language dysf after", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem Language dysf prior", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem w/PDI after AD dem not contrib", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("Dementia/PD, primary", "Non-AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem Language dysf with", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem distrubed social, after", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem w/CVD contribut", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem w/oth (list B) contribut", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("Unc: impair reversible", "Uncertain", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem cannot be primary", "AD", "PLOS ONE 2023 Table 2, direct listing (corrected from an earlier "
        "guess of 'Others' in this project's first draft)", "VERIFIED"),
    ("AD dem w/oth (list B) not contrib", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem distrubed social, with", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("Vascular Demt, secondary", "Non-AD", "Non-AD dementia category (vascular), per paper text", "VERIFIED"),
    ("Incipient Non-AD dem", "Non-AD", "PLOS ONE 2023 Table 2, direct listing (corrected from an earlier "
        "guess of 'Others' in this project's first draft)", "VERIFIED"),
    ("AD dem w/PDI after AD dem contribut", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem visuospatial, with", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem w/oth unusual features/demt on", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem/FLD prior to AD dem", "", "Not in PLOS ONE 2023 Table 2 (mixed AD/frontotemporal presentation); "
        "assigned by pattern ('AD dem...' prefix -> AD) consistent with every "
        "other AD-prefixed label in the table.", "NEEDS_VERIFICATION"),
    ("AD dem w/oth unusual feat/subs demt", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem distrubed social, prior", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
    ("AD dem visuospatial, prior", "", "Not in PLOS ONE 2023 Table 2; assigned by pattern "
        "('AD dem...' prefix -> AD).", "NEEDS_VERIFICATION"),
    ("AD dem w/oth unusual features", "", "Not in PLOS ONE 2023 Table 2; assigned by pattern "
        "('AD dem...' prefix -> AD).", "NEEDS_VERIFICATION"),
    ("AD dem visuospatial, after", "AD", "PLOS ONE 2023 Table 2, direct listing", "VERIFIED"),
]


def main():
    df = pd.DataFrame(DX_TABLE, columns=["OASIS_DX", "Paper_Class", "Evidence", "Status"])
    out_path = RESULTS_DIR / "diagnosis_mapping.csv"
    df.to_csv(out_path, index=False)

    print("=" * 80)
    print("DIAGNOSIS MAPPING")
    print("=" * 80)
    print(df["Status"].value_counts().to_string())
    print(f"\nClass distribution of mapping table:")
    print(df["Paper_Class"].value_counts().to_string())
    print(f"\nSaved: {out_path}")

    needs_verification = df[df["Status"] == "NEEDS_VERIFICATION"]
    if needs_verification["Paper_Class"].notna().any() and (needs_verification["Paper_Class"] != "").any():
        raise AssertionError("NEEDS_VERIFICATION diagnosis labels must not be assigned a model class.")
    print(f"\n{len(needs_verification)} labels marked NEEDS_VERIFICATION "
          f"(assigned by pattern, not confirmed against the companion "
          f"paper's table -- they were not present in it):")
    for _, r in needs_verification.iterrows():
        print(f"  - '{r['OASIS_DX']}' -> {r['Paper_Class']}")

    # Cross-check against what actually appears in the current merged data,
    # if it exists yet.
    if MERGED_FILE.exists():
        merged = pd.read_csv(MERGED_FILE, low_memory=False)
        if "dx1" in merged.columns:
            actual_values = set(merged["dx1"].dropna().unique())
            mapped_values = set(df["OASIS_DX"])
            unmapped = actual_values - mapped_values
            if unmapped:
                print(f"\nWARNING: {len(unmapped)} dx1 values in the current "
                      f"merged dataset are NOT in this mapping table:")
                for v in sorted(unmapped):
                    print(f"  - {v}")
            else:
                print("\nAll dx1 values present in the current merged "
                      "dataset are covered by this mapping table.")


if __name__ == "__main__":
    main()
