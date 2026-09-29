"""
Actual SHAP / Tree SHAP explainability for NeuroFEDX.

This module intentionally requires the real `shap` package. It does NOT
fall back to permutation importance. If SHAP is unavailable, the command
fails with an actionable installation message rather than silently changing
the scientific method.

The paper describes SHAP explanations for the RF global model. We use
shap.TreeExplainer on the fitted sklearn RandomForestClassifier. For
multiclass classification, SHAP 0.50 returns
(n_samples, n_features, n_classes); older versions may return a list of
(n_samples, n_features) arrays. Both formats are normalized below.

Permutation importance, if desired, is a separate optional analysis and is
not used as a SHAP substitute.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR

SAMPLE_PER_CLASS = 20
TOP_N = 10
SEED = 42


def balanced_sample(y, per_class=SAMPLE_PER_CLASS, seed=SEED):
    rng = np.random.RandomState(seed)
    picks = []
    y = np.asarray(y)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n = min(per_class, len(idx))
        picks.extend(rng.choice(idx, size=n, replace=False))
    return np.asarray(sorted(picks), dtype=int)


def normalize_shap_values(values, n_samples, n_features, n_classes):
    """Return a list of class arrays, each shape (n_samples, n_features)."""
    if isinstance(values, list):
        arrays = [np.asarray(v) for v in values]
    else:
        arr = np.asarray(values)
        if arr.ndim == 3 and arr.shape == (n_samples, n_features, n_classes):
            arrays = [arr[:, :, k] for k in range(n_classes)]
        elif arr.ndim == 3 and arr.shape == (n_classes, n_samples, n_features):
            arrays = [arr[k] for k in range(n_classes)]
        elif arr.ndim == 2 and n_classes == 2:
            arrays = [arr, -arr]
        else:
            raise ValueError(
                f"Unsupported Tree SHAP output shape {arr.shape}; "
                f"expected multiclass 3-D output."
            )

    if len(arrays) != n_classes:
        raise ValueError(
            f"Tree SHAP returned {len(arrays)} class arrays; "
            f"model has {n_classes} classes."
        )
    for arr in arrays:
        if arr.shape != (n_samples, n_features):
            raise ValueError(
                f"Unexpected SHAP class array shape {arr.shape}; "
                f"expected {(n_samples, n_features)}."
            )
    return arrays


def explain_model(name, model, X_sample_df, class_names, track):
    import shap

    feature_names = list(X_sample_df.columns)
    X_sample = X_sample_df.to_numpy()
    n_samples, n_features = X_sample.shape
    n_classes = len(class_names)

    print(
        f"--- Tree SHAP: {track}/{name}; "
        f"trees={len(model.estimators_)}, samples={n_samples} ---"
    )

    start = time.time()
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(
        X_sample,
        check_additivity=True,
    )
    values = normalize_shap_values(
        raw_values, n_samples, n_features, n_classes
    )
    elapsed = time.time() - start

    # Explicit numerical additivity verification. TreeExplainer's own
    # check_additivity=True is the primary check; this records the observed
    # reconstruction error for the report.
    expected = np.asarray(explainer.expected_value)
    proba = model.predict_proba(X_sample)
    reconstruction_errors = []
    for class_index in range(n_classes):
        reconstructed = expected[class_index] + values[class_index].sum(axis=1)
        reconstruction_errors.append(
            float(np.max(np.abs(reconstructed - proba[:, class_index])))
        )
    max_error = max(reconstruction_errors)

    per_class = np.vstack([
        np.mean(np.abs(class_values), axis=0)
        for class_values in values
    ])
    importance = pd.DataFrame(
        per_class.T,
        index=feature_names,
        columns=class_names,
    )
    importance.insert(0, "overall_mean_abs_shap", per_class.mean(axis=0))
    importance = importance.sort_values(
        "overall_mean_abs_shap", ascending=False
    )

    out = RESULTS_DIR / f"shap_importance_{track}_{name}.csv"
    importance.to_csv(out)

    audit = pd.DataFrame([{
        "track": track,
        "model": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_trees": len(model.estimators_),
        "shap_package_version": getattr(shap, "__version__", "unknown"),
        "max_additivity_error": max_error,
        "runtime_seconds": elapsed,
        "method": "shap.TreeExplainer",
    }])
    audit_path = RESULTS_DIR / "shap_audit.csv"
    if audit_path.exists():
        pd.concat([pd.read_csv(audit_path), audit], ignore_index=True).to_csv(
            audit_path, index=False
        )
    else:
        audit.to_csv(audit_path, index=False)

    # Optional plots are generated with SHAP's own plotting API.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top["overall_mean_abs_shap"].values)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(f"{track}/{name} RF — Tree SHAP")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / f"shap_bar_{track}_{name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    for class_index, class_name in enumerate(class_names):
        shap.summary_plot(
            values[class_index],
            X_sample_df,
            feature_names=feature_names,
            max_display=15,
            show=False,
        )
        plt.title(f"{track}/{name} RF — Tree SHAP, class={class_name}")
        safe = str(class_name).replace("/", "-").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / f"shap_beeswarm_{track}_{name}_{safe}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print(
        f"Tree SHAP complete in {elapsed:.1f}s; "
        f"max additivity error={max_error:.3e}"
    )
    return importance


def run_for_track(track):
    import joblib

    bundle_path = RESULTS_DIR / f"{track}_models.joblib"
    if not bundle_path.exists():
        print(f"SKIP {track}: {bundle_path} does not exist.")
        return None

    bundle = joblib.load(bundle_path)
    X_test = np.asarray(bundle["X_test"])
    y_test = np.asarray(bundle["y_test"])
    feature_cols = list(bundle["feature_cols"])
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    picks = balanced_sample(y_test)
    X_sample_df = X_test_df.iloc[picks].reset_index(drop=True)
    class_names = list(bundle["centralized"].classes_)

    central = explain_model(
        "centralized",
        bundle["centralized"],
        X_sample_df,
        class_names,
        track,
    )
    federated = explain_model(
        "federated",
        bundle["federated"],
        X_sample_df,
        class_names,
        track,
    )

    from scipy.stats import spearmanr
    common = central.index.intersection(federated.index)
    rho, p_value = spearmanr(
        central.loc[common, "overall_mean_abs_shap"],
        federated.loc[common, "overall_mean_abs_shap"],
    )
    comparison = pd.DataFrame({
        "centralized_mean_abs_shap": central.loc[common, "overall_mean_abs_shap"],
        "federated_mean_abs_shap": federated.loc[common, "overall_mean_abs_shap"],
    })
    comparison["centralized_rank"] = comparison[
        "centralized_mean_abs_shap"
    ].rank(ascending=False, method="min")
    comparison["federated_rank"] = comparison[
        "federated_mean_abs_shap"
    ].rank(ascending=False, method="min")
    comparison.to_csv(
        RESULTS_DIR / f"shap_ranking_comparison_{track}.csv"
    )
    print(
        f"{track}: centralized/federated SHAP rank Spearman rho="
        f"{rho:.4f}, p={p_value:.3g}"
    )
    return central, federated


def main():
    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "The real SHAP package is required. Install it with "
            "`pip install 'shap>=0.50,<0.51'` in the project environment. "
            "Permutation importance is intentionally NOT used as a fallback."
        ) from exc

    print(f"Using SHAP {shap.__version__} / TreeExplainer")
    for track in ("paper_faithful", "leakage_free"):
        run_for_track(track)


if __name__ == "__main__":
    main()
