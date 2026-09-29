"""
Sensitivity sweep over MRI_TOLERANCE_DAYS (Rule 2).

Runs the merge at several candidate tolerances and reports the
resulting match rate for each -- so the choice of 365 days in
config/model_config.py is visibly not "whichever number gives the best
accuracy," since no accuracy is computed here at all, only match rates.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import MRI_TOLERANCE_SENSITIVITY_DAYS, RESULTS_DIR
from preprocessing.merge import build_merged_dataset


def main():
    rows = []
    for tol in MRI_TOLERANCE_SENSITIVITY_DAYS:
        print(f"\n--- Running merge with MRI_TOLERANCE_DAYS={tol} ---")
        _, summary = build_merged_dataset(mri_tolerance_days=tol, write_audit=False,
                                           verbose=False)
        rows.append(summary)

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "mri_tolerance_sensitivity.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 100)
    print("MRI TOLERANCE SENSITIVITY SWEEP")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nPaper reference: match rate 96.3% (3220/3342), 799 matched subjects")
    print(f"\nSaved: {out_path}")
    print("\nNote: no accuracy/model metric appears anywhere in this sweep. The "
          "365-day default in config/model_config.py was chosen from the "
          "paper's stated 'every two to three years' imaging cadence BEFORE "
          "this sweep was run, not selected from this table.")

    # Re-build the default (365-day) merged dataset as the one actually used
    # downstream, since the sweep above overwrote results/merged_oasis3.csv
    # repeatedly with each tolerance in turn.
    from config.model_config import MRI_TOLERANCE_DAYS_DEFAULT
    print(f"\nRestoring results/merged_oasis3.csv to the default tolerance "
          f"({MRI_TOLERANCE_DAYS_DEFAULT} days) used by the rest of the pipeline...")
    build_merged_dataset(mri_tolerance_days=MRI_TOLERANCE_DAYS_DEFAULT,
                          write_audit=True, verbose=True)


if __name__ == "__main__":
    main()
