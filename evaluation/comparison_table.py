"""
Final comparison table (Rule 14). Pulls together every experiment run
in this project and writes results/FINAL_COMPARISON_TABLE.csv +
prints a markdown version. No missing values are filled with guesses --
a row/column is left blank if that combination was never computed.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR


def pct(x):
    return "" if pd.isna(x) else f"{100 * x:.2f}"


def main():
    rows = []

    # Paper reported (from the paper text, Section V / Table 6)
    rows.append({
        "Experiment": "Paper reported (Table 6, FL approach)",
        "Accuracy": 98.93, "Balanced_Accuracy": "", "Macro_F1": 98.93,
        "Precision": 98.94, "Recall": 98.93, "AUC": 99.97,
    })

    pf = pd.read_csv(RESULTS_DIR / "paper_faithful_results.csv")
    for _, r in pf.iterrows():
        rows.append({
            "Experiment": f"Ours: PAPER-FAITHFUL -- {r['label']}",
            "Accuracy": pct(r["accuracy"]),
            "Balanced_Accuracy": pct(r["balanced_accuracy"]),
            "Macro_F1": pct(r["macro_f1"]),
            "Precision": pct(r["macro_precision"]),
            "Recall": pct(r["macro_recall"]),
            "AUC": pct(r["auc_macro_ovr"]),
        })

    lf = pd.read_csv(RESULTS_DIR / "leakage_free_full_results.csv")
    for _, r in lf.iterrows():
        rows.append({
            "Experiment": f"Ours: LEAKAGE-FREE (single split) -- {r['label']}",
            "Accuracy": pct(r["accuracy"]),
            "Balanced_Accuracy": pct(r["balanced_accuracy"]),
            "Macro_F1": pct(r["macro_f1"]),
            "Precision": pct(r["macro_precision"]),
            "Recall": pct(r["macro_recall"]),
            "AUC": pct(r["auc_macro_ovr"]),
        })

    cv = pd.read_csv(RESULTS_DIR / "grouped_cv_full_results.csv")
    cv_mean = cv[["accuracy", "balanced_accuracy", "macro_f1",
                  "macro_precision", "macro_recall", "auc_macro_ovr"]].mean()
    cv_std = cv[["accuracy", "balanced_accuracy", "macro_f1",
                 "macro_precision", "macro_recall", "auc_macro_ovr"]].std()
    rows.append({
        "Experiment": "Ours: GROUPED 5-FOLD CV (mean +/- std, most rigorous)",
        "Accuracy": f"{100*cv_mean['accuracy']:.2f} +/- {100*cv_std['accuracy']:.2f}",
        "Balanced_Accuracy": f"{100*cv_mean['balanced_accuracy']:.2f} +/- {100*cv_std['balanced_accuracy']:.2f}",
        "Macro_F1": f"{100*cv_mean['macro_f1']:.2f} +/- {100*cv_std['macro_f1']:.2f}",
        "Precision": f"{100*cv_mean['macro_precision']:.2f} +/- {100*cv_std['macro_precision']:.2f}",
        "Recall": f"{100*cv_mean['macro_recall']:.2f} +/- {100*cv_std['macro_recall']:.2f}",
        "AUC": f"{100*cv_mean['auc_macro_ovr']:.2f} +/- {100*cv_std['auc_macro_ovr']:.2f}",
    })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "FINAL_COMPARISON_TABLE.csv"
    df.to_csv(out_path, index=False)

    print("=" * 100)
    print("FINAL COMPARISON TABLE")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")

    # Per-class detail for the two most important (honest) evaluations
    print("\n" + "=" * 100)
    print("PER-CLASS RECALL: PAPER-FAITHFUL vs LEAKAGE-FREE vs GROUPED-CV")
    print("=" * 100)
    pf_pc = pd.read_csv(RESULTS_DIR / "paper_faithful_centralized_per_class.csv").set_index("class")["recall"]
    lf_pc = pd.read_csv(RESULTS_DIR / "leakage_free_centralized_per_class.csv").set_index("class")["recall"]
    cv_pc = pd.read_csv(RESULTS_DIR / "grouped_cv_full_per_class.csv").groupby("class")["recall"].mean()

    combined = pd.DataFrame({
        "paper_faithful_recall": pf_pc,
        "leakage_free_recall": lf_pc,
        "grouped_cv_mean_recall": cv_pc,
    })
    combined = (combined * 100).round(2)
    print(combined.to_string())
    combined.to_csv(RESULTS_DIR / "per_class_recall_comparison.csv")
    print(f"\nSaved: {RESULTS_DIR / 'per_class_recall_comparison.csv'}")


if __name__ == "__main__":
    main()
