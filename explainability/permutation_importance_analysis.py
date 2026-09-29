"""Optional supplementary permutation-importance analysis.

This analysis is intentionally separate from SHAP and must never be reported
as SHAP. It is useful as a robustness comparison only.
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR

N_REPEATS = 10
SEED = 42

def main():
    for track in ("paper_faithful", "leakage_free"):
        path = RESULTS_DIR / f"{track}_models.joblib"
        if not path.exists():
            print(f"SKIP {track}: {path} not found")
            continue
        bundle = joblib.load(path)
        X = pd.DataFrame(bundle["X_test"], columns=bundle["feature_cols"])
        y = bundle["y_test"]
        for name in ("centralized", "federated"):
            model = bundle[name]
            result = permutation_importance(
                model, X, y, scoring="accuracy",
                n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1,
            )
            out = pd.DataFrame({
                "feature": bundle["feature_cols"],
                "mean_accuracy_drop": result.importances_mean,
                "std_accuracy_drop": result.importances_std,
            }).sort_values("mean_accuracy_drop", ascending=False)
            out.to_csv(
                RESULTS_DIR / f"permutation_importance_{track}_{name}.csv",
                index=False,
            )

if __name__ == "__main__":
    main()
