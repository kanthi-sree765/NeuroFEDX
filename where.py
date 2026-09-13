import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Paper feature mapping
# --------------------------------------------------

mapping = [

    # Psychological / cognitive
    ["DIGIF", "Psychometrics", "digforct", "Renamed from UDS-3 variable", "Verified"],
    ["DIGIFLEN", "Psychometrics", "digforsl", "Renamed from UDS-3 variable", "Verified"],
    ["DIGIB", "Psychometrics", "digbacct", "Renamed from UDS-3 variable", "Verified"],
    ["DIGIBLEN", "Psychometrics", "digbacls", "Renamed from UDS-3 variable", "Verified"],

    ["LOGIMEM", "Psychometrics", "LOGIMEM", "Direct", "Verified"],
    ["MEMUNITS", "Psychometrics", "MEMUNITS", "Direct", "Verified"],
    ["ANIMALS", "Psychometrics", "ANIMALS", "Direct", "Verified"],
    ["VEG", "Psychometrics", "VEG", "Direct", "Verified"],

    ["TRAILA", "Psychometrics", "tma", "Equivalent value verified against TRAILA", "Verified"],
    ["TRAILARR", "Psychometrics", "", "Not present in current OASIS export", "Unavailable"],
    ["TRAILALI", "Psychometrics", "", "Not present in current OASIS export", "Unavailable"],

    ["TRAILB", "Psychometrics", "tmb", "Equivalent value verified against trailb", "Verified"],
    ["TRAILBRR", "Psychometrics", "", "Not present in current OASIS export", "Unavailable"],
    ["TRAILBLI", "Psychometrics", "", "Not present in current OASIS export", "Unavailable"],

    ["WAIS", "Psychometrics", "digsym", "Requires verification of UDS definition", "Needs verification"],
    ["MEMTIME", "Psychometrics", "lmdelay", "Requires verification of UDS definition", "Needs verification"],
    ["BOSTON", "Psychometrics", "bnt", "Requires verification of UDS definition", "Needs verification"],

    # Clinical
    ["AgeAtEntry", "Demographics", "AgeatEntry", "Direct", "Verified"],
    ["APOE", "Demographics", "APOE", "Direct", "Verified"],

    ["Height", "UDSb1", "HEIGHT", "Direct", "Verified"],
    ["Weight", "UDSb1", "WEIGHT", "Direct", "Verified"],

    ["MMSE", "UDSb4", "MMSE", "Direct", "Verified"],
    ["Memory", "UDSb4", "memory", "Direct", "Verified"],
    ["Orient", "UDSb4", "orient", "Direct", "Verified"],
    ["Judgment", "UDSb4", "judgment", "Direct", "Verified"],
    ["Commun", "UDSb4", "commun", "Direct", "Verified"],
    ["Homehobb", "UDSb4", "homehobb", "Direct", "Verified"],
    ["Perscare", "UDSb4", "perscare", "Direct", "Verified"],
    ["Sumbox", "UDSb4", "CDRSUM", "Paper name mapped to CDRSUM", "Verified"],

    # MRI / FreeSurfer
    ["lhCortexVol", "FreeSurfer", "lhCortexVol", "Direct", "Verified"],
    ["IntraCranialVol", "FreeSurfer", "IntraCranialVol", "Direct", "Verified"],
    ["SupraTentorialVol", "FreeSurfer", "SupraTentorialVol", "Direct", "Verified"],
    ["LhCorticalWhiteMatterVol", "FreeSurfer", "lhCorticalWhiteMatterVol", "Direct", "Verified"],
    ["SubCortGrayVol", "FreeSurfer", "SubCortGrayVol", "Direct", "Verified"],
]

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

df = pd.DataFrame(
    mapping,
    columns=[
        "Paper_Feature",
        "OASIS_Source",
        "OASIS_Column",
        "Processing",
        "Status"
    ]
)

# --------------------------------------------------
# Save result
# --------------------------------------------------

output_file = RESULTS_DIR / "feature_mapping.csv"

df.to_csv(output_file, index=False)

print("=" * 70)
print("FEATURE MAPPING CREATED")
print("=" * 70)

print(f"\nTotal paper features documented: {len(df)}")
print(f"Verified: {(df['Status'] == 'Verified').sum()}")
print(f"Needs verification: {(df['Status'] == 'Needs verification').sum()}")
print(f"Unavailable in current export: {(df['Status'] == 'Unavailable').sum()}")

print(f"\nSaved to:")
print(output_file)