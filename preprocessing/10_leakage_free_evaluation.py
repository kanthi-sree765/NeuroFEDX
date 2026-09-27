"""
Leakage-free evaluation of the Random Forest and Federated RF pipeline.

PROBLEM: 05_preprocess.py runs SMOTE on the WHOLE dataset, and only
afterwards do scripts 06 / 07 split into train and test. SMOTE creates
synthetic patients by interpolating between real neighbours, so synthetic
test rows are blends of real training rows (and vice versa). The model is
therefore partly tested on data it has effectively already seen, which
inflates accuracy.

This script compares two protocols on the same data:

  PROTOCOL A - "paper-style" (leaky): load the already-balanced file
               (ml_ready_dataset.csv), split, train, test. This is what
               scripts 06 / 07 did, so it reproduces ~98.9% / ~94.5%.

  PROTOCOL B - leakage-free: load the UNBALANCED file, split the REAL
               patients first (80/20, stratified), apply SMOTE to the
               TRAINING part only, and test on untouched real patients.

The test set in B keeps the real class proportions, so plain accuracy can
be misleading; balanced accuracy and macro-F1 are the primary metrics.

For each protocol it evaluates: centralized RF, federated tree pooling with
10 clients (100 trees per client, IID, as in script 07), and for B also a
client sweep (1, 2, 5, 10, 20 clients) and a "no balancing at all" reference.

SMOTE note: APOE is a genotype CODE (22, 23, 24, 33, 34, 44), not a number.
Plain SMOTE interpolates it into invalid values (e.g. 22.378). Here, columns
listed in CATEGORICAL_COLS are handled with SMOTENC, which copies the most
common category among neighbours instead of interpolating.

REQUIRES the unbalanced file. Add these lines to 05_preprocess.py, right
after the line   y = df["diagnosis_class"]   and re-run 05:

    unbalanced_df = X.copy()
    unbalanced_df["diagnosis_class"] = y.values
    unbalanced_df.to_csv(OUTPUT_FILE.replace(".csv", "_unbalanced.csv"), index=False)

Needs: pip install scikit-learn imbalanced-learn pandas numpy

Run from the preprocessing/ folder:
    python 10_leakage_free_evaluation.py

Outputs: ../results/leakage_free_results.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, recall_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

BALANCED_FILE = "../results/ml_ready_dataset.csv"
UNBALANCED_FILE = "../results/ml_ready_dataset_unbalanced.csv"
OUT_CSV = "../results/leakage_free_results.csv"
LABEL_COL = "diagnosis_class"

# Columns that are category codes, not quantities (handled with SMOTENC)
CATEGORICAL_COLS = ["APOE"]

# Must match 06 / 07
CENTRAL_TREES = 100
TREES_PER_CLIENT = 100
FED_CLIENTS = 10
CLIENT_SWEEP = [1, 2, 5, 10, 20]
LOCAL_TREE_PARAMS = dict(
    criterion="gini",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
)
REPEATS = 3          # seeds averaged per setting
N_JOBS = -1
SPLIT_SEED = 42      # same split seed as 06 / 07


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def partition_clients(X, y, n_clients, seed):
    """IID, class-stratified split into n_clients shards (same as 07)."""
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


def fit_rf(X, y, n_trees, seed):
    return RandomForestClassifier(
        n_estimators=n_trees, random_state=seed, n_jobs=N_JOBS,
        **LOCAL_TREE_PARAMS,
    ).fit(X, y)


def aligned_proba(rf, X, n_classes):
    """predict_proba with columns aligned to the global class set."""
    proba = rf.predict_proba(X)
    out = np.zeros((X.shape[0], n_classes))
    for col, cls in enumerate(rf.classes_):
        out[:, int(cls)] = proba[:, col]
    return out


def evaluate(y_true, proba, n_classes):
    pred = np.argmax(proba, axis=1)
    res = dict(
        accuracy=accuracy_score(y_true, pred),
        balanced_acc=balanced_accuracy_score(y_true, pred),
        macro_f1=f1_score(y_true, pred, average="macro"),
        weighted_f1=f1_score(y_true, pred, average="weighted"),
    )
    try:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        res["auc"] = roc_auc_score(y_bin, proba, average="macro")
    except ValueError:
        res["auc"] = np.nan
    return res


def summarize(rows):
    keys = rows[0].keys()
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["accuracy_std"] = float(np.std([r["accuracy"] for r in rows]))
    return out


def run_centralized(Xtr, ytr, Xte, yte, n_classes):
    rows, first = [], None
    for r in range(REPEATS):
        rf = fit_rf(Xtr, ytr, CENTRAL_TREES, seed=r)
        proba = aligned_proba(rf, Xte, n_classes)
        rows.append(evaluate(yte, proba, n_classes))
        if r == 0:
            first = proba
    return summarize(rows), first


def run_federated(Xtr, ytr, Xte, yte, n_classes, n_clients):
    """Tree pooling: each client trains a full local forest; pooled forest's
    averaged probabilities equal the mean of the client forests' (equal
    tree counts)."""
    rows, first = [], None
    for r in range(REPEATS):
        shards = partition_clients(Xtr, ytr, n_clients, seed=r)
        total = np.zeros((Xte.shape[0], n_classes))
        for i, (cx, cy) in enumerate(shards):
            rf = fit_rf(cx, cy, TREES_PER_CLIENT, seed=1000 * r + i)
            total += aligned_proba(rf, Xte, n_classes)
        proba = total / len(shards)
        rows.append(evaluate(yte, proba, n_classes))
        if r == 0:
            first = proba
    return summarize(rows), first


def oversample(X, y, feature_names, seed=42):
    """SMOTE (SMOTENC for category-code columns) on the given data only."""
    from imblearn.over_sampling import SMOTE, SMOTENC
    min_size = int(np.bincount(y).min())
    if min_size < 2:
        raise ValueError("A class has fewer than 2 training rows; SMOTE cannot run.")
    k = max(1, min(5, min_size - 1))
    cat_idx = [feature_names.index(c) for c in CATEGORICAL_COLS if c in feature_names]
    if cat_idx:
        sm = SMOTENC(categorical_features=cat_idx, random_state=seed, k_neighbors=k)
    else:
        sm = SMOTE(random_state=seed, k_neighbors=k)
    return sm.fit_resample(X, y)


def fmt(res):
    return (f"acc {res['accuracy']:.2%} (+-{res['accuracy_std']:.2%})  "
            f"bal.acc {res['balanced_acc']:.2%}  macroF1 {res['macro_f1']:.2%}  "
            f"AUC {res['auc']:.2%}")


def per_class_report(name, y_true, proba, class_names):
    pred = np.argmax(proba, axis=1)
    rec = recall_score(y_true, pred, average=None, labels=list(range(len(class_names))),
                       zero_division=0)
    print(f"\n{name}: recall per class")
    print("  " + "   ".join(f"{c}: {r:.1%}" for c, r in zip(class_names, rec)))
    cm = confusion_matrix(y_true, pred, labels=list(range(len(class_names))))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(UNBALANCED_FILE):
        print(f"Missing {UNBALANCED_FILE}\n\n"
              "Add these 3 lines to 05_preprocess.py, right after the line\n"
              '    y = df["diagnosis_class"]\n'
              "and re-run 05_preprocess.py:\n\n"
              "    unbalanced_df = X.copy()\n"
              '    unbalanced_df["diagnosis_class"] = y.values\n'
              '    unbalanced_df.to_csv(OUTPUT_FILE.replace(".csv", "_unbalanced.csv"), index=False)\n')
        sys.exit(1)
    try:
        import imblearn  # noqa: F401
    except ImportError:
        print("Missing package. Install with:  pip install imbalanced-learn")
        sys.exit(1)

    results = []

    def record(protocol, model, res):
        results.append(dict(protocol=protocol, model=model, **res))

    # ======================================================================
    # PROTOCOL A - SMOTE before split (what 06 / 07 did)
    # ======================================================================
    print("=" * 80)
    print("PROTOCOL A - SMOTE applied BEFORE the split (leaky, as in 06/07)")
    print("=" * 80)
    dfa = pd.read_csv(BALANCED_FILE)
    le = LabelEncoder()
    ya = le.fit_transform(dfa[LABEL_COL])
    class_names = list(le.classes_)
    n_classes = len(class_names)
    Xa = dfa.drop(columns=[LABEL_COL]).values
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        Xa, ya, test_size=0.2, stratify=ya, random_state=SPLIT_SEED)
    print(f"Train {Xa_tr.shape}  Test {Xa_te.shape}")

    res, _ = run_centralized(Xa_tr, ya_tr, Xa_te, ya_te, n_classes)
    print(f"Centralized RF        : {fmt(res)}")
    record("A leaky", "centralized", res)
    res, _ = run_federated(Xa_tr, ya_tr, Xa_te, ya_te, n_classes, FED_CLIENTS)
    print(f"Federated ({FED_CLIENTS} clients)  : {fmt(res)}")
    record("A leaky", f"federated_{FED_CLIENTS}", res)

    # ======================================================================
    # PROTOCOL B - split real patients first, SMOTE on train only
    # ======================================================================
    print("\n" + "=" * 80)
    print("PROTOCOL B - split REAL patients first, SMOTE on TRAIN only (leakage-free)")
    print("=" * 80)
    dfb = pd.read_csv(UNBALANCED_FILE)
    yb = le.transform(dfb[LABEL_COL])          # same class encoding as A
    feature_names = [c for c in dfb.columns if c != LABEL_COL]
    Xb = dfb[feature_names].values

    print("Real class counts (whole dataset):")
    print(pd.Series(dfb[LABEL_COL]).value_counts().to_string())
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        Xb, yb, test_size=0.2, stratify=yb, random_state=SPLIT_SEED)
    print(f"\nReal train {Xb_tr.shape}   Real test {Xb_te.shape}  "
          f"(test keeps real class proportions)")
    print("Test class counts:",
          dict(zip(class_names, np.bincount(yb_te, minlength=n_classes).tolist())))

    t0 = time.time()
    Xb_sm, yb_sm = oversample(Xb_tr, yb_tr, feature_names)
    print(f"\nSMOTE on train only: {len(Xb_tr)} -> {len(Xb_sm)} rows "
          f"({len(Xb_sm) - len(Xb_tr)} synthetic)   [{time.time() - t0:.0f}s]")
    print("Train counts after SMOTE:",
          dict(zip(class_names, np.bincount(yb_sm, minlength=n_classes).tolist())))

    # Reference: no balancing at all
    res, _ = run_centralized(Xb_tr, yb_tr, Xb_te, yb_te, n_classes)
    print(f"\nCentralized, NO balancing      : {fmt(res)}")
    record("B leak-free", "centralized_no_smote", res)

    # Centralized with SMOTE on train
    res, proba_c = run_centralized(Xb_sm, yb_sm, Xb_te, yb_te, n_classes)
    print(f"Centralized, SMOTE on train    : {fmt(res)}")
    record("B leak-free", "centralized", res)

    # Federated sweep with SMOTE'd training set partitioned across clients
    print(f"\nClient sweep (SMOTE'd train set split IID, {TREES_PER_CLIENT} trees/client):")
    proba_f = None
    for n_clients in CLIENT_SWEEP:
        res, p = run_federated(Xb_sm, yb_sm, Xb_te, yb_te, n_classes, n_clients)
        print(f"  {n_clients:>2d} clients : {fmt(res)}")
        record("B leak-free", f"federated_{n_clients}", res)
        if n_clients == FED_CLIENTS:
            proba_f = p

    per_class_report("Centralized (B, SMOTE on train)", yb_te, proba_c, class_names)
    if proba_f is not None:
        per_class_report(f"Federated {FED_CLIENTS} clients (B)", yb_te, proba_f, class_names)

    # ======================================================================
    # Summary
    # ======================================================================
    tab = pd.DataFrame(results)
    tab.to_csv(OUT_CSV, index=False)

    def get(protocol, model, key):
        return float(tab[(tab.protocol == protocol) & (tab.model == model)][key].iloc[0])

    fed = f"federated_{FED_CLIENTS}"
    print("\n" + "=" * 80)
    print("SUMMARY (accuracy / balanced accuracy / macro-F1)")
    print("=" * 80)
    print(f"{'':28s}{'A: leaky':>26s}{'B: leak-free':>26s}")
    for label, model in (("Centralized RF", "centralized"),
                         (f"Federated ({FED_CLIENTS} clients)", fed)):
        a = (get("A leaky", model, "accuracy"), get("A leaky", model, "balanced_acc"),
             get("A leaky", model, "macro_f1"))
        b = (get("B leak-free", model, "accuracy"), get("B leak-free", model, "balanced_acc"),
             get("B leak-free", model, "macro_f1"))
        print(f"{label:28s}{a[0]:>10.2%}/{a[1]:.2%}/{a[2]:.2%}"
              f"   {b[0]:>8.2%}/{b[1]:.2%}/{b[2]:.2%}")

    drop = get("A leaky", "centralized", "accuracy") - get("B leak-free", "centralized", "balanced_acc")
    gap_a = get("A leaky", "centralized", "accuracy") - get("A leaky", fed, "accuracy")
    gap_b = get("B leak-free", "centralized", "balanced_acc") - get("B leak-free", fed, "balanced_acc")
    print(f"\nCentralized: leaky accuracy minus leak-free balanced accuracy = {drop:+.2%}")
    print(f"Centralized-vs-federated gap:  A (leaky) {gap_a:.2%}   B (leak-free) {gap_b:.2%}")
    print("\nPrimary metrics for the leak-free protocol: balanced accuracy and macro-F1\n"
          "(the real test set is class-imbalanced).")
    print(f"\nSaved full table to {OUT_CSV}")


if __name__ == "__main__":
    main()