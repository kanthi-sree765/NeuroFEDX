"""
SHAP explainability for the NeuroFEDX Random Forest pipeline.

Explains TWO models, using the same data split and hyperparameters as
scripts 06 / 07:

  1. CENTRALIZED RF   - one forest (100 trees) trained on all training data.
  2. FEDERATED RF     - 10 IID clients, each trains a 100-tree forest on its
                        own slice; all trees are pooled into one global forest
                        (1000 trees), exactly like script 07.

For each model it computes SHAP values on a class-balanced sample of the
TEST set and reports which features drive predictions, overall and per
diagnosis class. It then compares the two feature rankings: if the federated
model relies on the same features as the centralized one, that is a useful
result for the report (federation did not change WHAT the model learned to
look at, only how well).

Needs: pip install shap matplotlib scikit-learn pandas numpy scipy

Run from the preprocessing/ folder:
    python 09_shap_explainability.py
Optional: python 09_shap_explainability.py path/to/ml_ready_dataset.csv

Outputs (in ../results/):
    shap_importance_centralized.csv / shap_importance_federated.csv
    shap_ranking_comparison.csv
    shap_bar_centralized.png / shap_bar_federated.png
    shap_beeswarm_<model>_<class>.png     (one per class per model)
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "../results/ml_ready_dataset.csv"
RESULTS_DIR = "../results"
LABEL_COL = "diagnosis_class"

# Must match 06 / 07
N_CLIENTS = 10
TREES_PER_CLIENT = 100
CENTRAL_TREES = 100
LOCAL_TREE_PARAMS = dict(
    criterion="gini",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
)

# SHAP on 1000 deep trees is slow, so explain a class-balanced test sample.
SAMPLE_PER_CLASS = 60        # 60 x 5 classes = 300 rows. Raise for more precision.
TOP_N = 10                   # how many top features to print
N_JOBS = -1
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def partition_clients(X, y, n_clients, seed=42):
    """IID, class-stratified split (same logic as script 07). Returns
    a list of (X_i, y_i) numpy shards."""
    rng = np.random.RandomState(seed)
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    for c in idx_by_class:
        rng.shuffle(idx_by_class[c])
    client_idx = [[] for _ in range(n_clients)]
    for c, idxs in idx_by_class.items():
        for i, part in enumerate(np.array_split(idxs, n_clients)):
            client_idx[i].extend(part.tolist())
    shards = []
    for idxs in client_idx:
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        shards.append((X[idxs], y[idxs]))
    return shards


def build_pooled_forest(shards, seed=1):
    """Train one local forest per client, then pool ALL trees into a single
    sklearn RandomForestClassifier object (so SHAP's TreeExplainer and
    predict_proba work on it directly). Only trees are shared, never data."""
    forests = []
    for i, (cx, cy) in enumerate(shards):
        rf = RandomForestClassifier(
            n_estimators=TREES_PER_CLIENT,
            random_state=1000 * seed + i,
            n_jobs=N_JOBS,
            **LOCAL_TREE_PARAMS,
        )
        rf.fit(cx, cy)
        forests.append(rf)

    pooled = forests[0]                       # reuse first forest as the shell
    all_trees = []
    for rf in forests:
        all_trees.extend(rf.estimators_)
    pooled.estimators_ = all_trees
    pooled.n_estimators = len(all_trees)
    return pooled


def balanced_sample(X_test, y_test, per_class, seed):
    rng = np.random.RandomState(seed)
    picks = []
    for c in np.unique(y_test):
        idx = np.where(y_test == c)[0]
        picks.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    picks = np.array(sorted(picks))
    return picks


def shap_per_class(shap_values, n_samples, n_features, n_classes):
    """Normalize SHAP output to a list of (n_samples, n_features) arrays,
    one per class. Older SHAP returns a list; newer SHAP returns a single
    array of shape (n_samples, n_features, n_classes)."""
    if isinstance(shap_values, list):
        out = [np.asarray(v) for v in shap_values]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3 and arr.shape == (n_samples, n_features, n_classes):
            out = [arr[:, :, k] for k in range(n_classes)]
        elif arr.ndim == 3 and arr.shape == (n_classes, n_samples, n_features):
            out = [arr[k] for k in range(n_classes)]
        else:
            raise ValueError(f"Unexpected SHAP output shape {arr.shape}")
    for v in out:
        assert v.shape == (n_samples, n_features), f"bad shape {v.shape}"
    return out


def explain_model(name, model, X_sample_df, class_names):
    """Run TreeExplainer, sanity-check additivity, save importances + plots.
    Returns a DataFrame with mean |SHAP| per feature (overall + per class)."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_names = list(X_sample_df.columns)
    X_s = X_sample_df.values
    n_samples, n_features = X_s.shape
    n_classes = len(class_names)

    print(f"\n--- Explaining {name} model "
          f"({len(model.estimators_)} trees, {n_samples} sample rows) ---")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_s, check_additivity=False)
    sv = shap_per_class(sv_raw, n_samples, n_features, n_classes)
    print(f"SHAP computed in {time.time() - t0:.0f}s")

    # Sanity check: base value + sum(SHAP) should equal predicted probability
    base = np.atleast_1d(explainer.expected_value)
    proba = model.predict_proba(X_s)
    errs = []
    for k in range(n_classes):
        recon = base[k] + sv[k].sum(axis=1)
        errs.append(np.abs(recon - proba[:, k]).max())
    print(f"Additivity check (max |base + sum(SHAP) - predicted prob|): {max(errs):.4f}")
    if max(errs) > 0.02:
        print("  WARNING: SHAP values do not add up to the model output. "
              "Do not trust these plots until this is investigated.")

    # Mean |SHAP| per feature: overall and per class
    per_class_imp = np.array([np.abs(v).mean(axis=0) for v in sv])   # (classes, features)
    overall = per_class_imp.mean(axis=0)
    imp_df = pd.DataFrame(per_class_imp.T, index=feature_names, columns=class_names)
    imp_df.insert(0, "overall", overall)
    imp_df = imp_df.sort_values("overall", ascending=False)
    imp_df.to_csv(f"{RESULTS_DIR}/shap_importance_{name}.csv")

    print(f"\nTop {TOP_N} features ({name}) by mean |SHAP| (averaged over classes):")
    print(imp_df["overall"].head(TOP_N).to_string(float_format=lambda v: f"{v:.4f}"))

    # Bar plot: overall importance, stacked by class contribution
    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(top))
    for cname in class_names:
        vals = top[cname].values / n_classes        # each class's share of the overall mean
        ax.barh(top.index, vals, left=left, label=cname)
        left += vals
    ax.set_xlabel("Mean |SHAP value| (contribution to overall importance)")
    ax.set_title(f"Top features - {name} RF")
    ax.legend(title="Class", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/shap_bar_{name}.png", dpi=150)
    plt.close(fig)

    # Beeswarm per class
    for k, cname in enumerate(class_names):
        plt.figure()
        shap.summary_plot(sv[k], X_sample_df, feature_names=feature_names,
                          max_display=15, show=False)
        plt.title(f"{name} RF - SHAP values for class '{cname}'")
        plt.tight_layout()
        safe = cname.replace("/", "-").replace(" ", "_")
        plt.savefig(f"{RESULTS_DIR}/shap_beeswarm_{name}_{safe}.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    print(f"Saved importance CSV and plots for {name} model to {RESULTS_DIR}/")
    return imp_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        import shap  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as e:
        print(f"Missing package: {e.name}. Install with:\n"
              f"    pip install shap matplotlib scipy")
        sys.exit(1)

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    df = pd.read_csv(INPUT_FILE)
    X_df = df.drop(columns=[LABEL_COL])
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    class_names = list(le.classes_)

    # Same split as 06 / 07
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_test = X_train_df.values, X_test_df.values
    print(f"Train: {X_train.shape}   Test: {X_test.shape}   Classes: {class_names}")

    # ---- Train both models ----
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    t0 = time.time()
    central = RandomForestClassifier(
        n_estimators=CENTRAL_TREES, random_state=SEED, n_jobs=N_JOBS,
        **LOCAL_TREE_PARAMS,
    ).fit(X_train, y_train)
    acc_c = accuracy_score(y_test, central.predict(X_test))
    print(f"Centralized RF ({CENTRAL_TREES} trees): test accuracy {acc_c:.2%}   "
          f"(expect ~98.9%)")

    shards = partition_clients(X_train, y_train, N_CLIENTS)
    federated = build_pooled_forest(shards)
    acc_f = accuracy_score(y_test, federated.predict(X_test))
    print(f"Federated pooled RF ({N_CLIENTS} clients x {TREES_PER_CLIENT} = "
          f"{len(federated.estimators_)} trees): test accuracy {acc_f:.2%}   "
          f"(expect ~94.5%)")
    print(f"Training took {time.time() - t0:.0f}s")

    # ---- Class-balanced test sample for SHAP ----
    picks = balanced_sample(X_test, y_test, SAMPLE_PER_CLASS, SEED)
    X_sample_df = X_test_df.iloc[picks].reset_index(drop=True)
    print(f"\nSHAP sample: {len(picks)} test rows "
          f"({SAMPLE_PER_CLASS} per class)")

    # ---- Explain both ----
    print("\n" + "=" * 80)
    print("SHAP EXPLANATIONS")
    print("=" * 80)
    imp_c = explain_model("centralized", central, X_sample_df, class_names)
    imp_f = explain_model("federated", federated, X_sample_df, class_names)

    # ---- Compare rankings ----
    print("\n" + "=" * 80)
    print("DO THE TWO MODELS USE THE SAME FEATURES?")
    print("=" * 80)
    from scipy.stats import spearmanr
    common = imp_c.index
    rho, p = spearmanr(imp_c.loc[common, "overall"], imp_f.loc[common, "overall"])
    top_c = set(imp_c.head(TOP_N).index)
    top_f = set(imp_f.head(TOP_N).index)
    overlap = len(top_c & top_f)
    print(f"Spearman rank correlation of feature importance: {rho:.3f} (p={p:.2g})")
    print(f"Overlap of top-{TOP_N} features: {overlap}/{TOP_N}")

    cmp_df = pd.DataFrame({
        "centralized_importance": imp_c["overall"],
        "federated_importance": imp_f.loc[common, "overall"],
    })
    cmp_df["centralized_rank"] = cmp_df["centralized_importance"].rank(ascending=False).astype(int)
    cmp_df["federated_rank"] = cmp_df["federated_importance"].rank(ascending=False).astype(int)
    cmp_df = cmp_df.sort_values("centralized_rank")
    cmp_df.to_csv(f"{RESULTS_DIR}/shap_ranking_comparison.csv")
    print("\nSide-by-side ranking (top 10 by centralized):")
    print(cmp_df.head(TOP_N).to_string(float_format=lambda v: f"{v:.4f}"))

    if rho > 0.8:
        print("\n-> Rankings are highly consistent: federation changed accuracy,\n"
              "   not which features the model relies on.")
    elif rho > 0.5:
        print("\n-> Rankings are moderately consistent; mention the differences.")
    else:
        print("\n-> Rankings differ substantially: worth investigating and reporting.")

    print(f"\nSaved comparison to {RESULTS_DIR}/shap_ranking_comparison.csv")
    print("\nREMINDER: SHAP shows what the MODEL relies on, not clinical causation.")


if __name__ == "__main__":
    main()