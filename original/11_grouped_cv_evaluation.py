"""
Patient-grouped, stratified cross-validation (the honest evaluation).

Why this script exists
----------------------
Two separate leaks inflated the earlier results:

  1. SMOTE was applied before the train/test split (fixed in script 10).
  2. Rows are VISITS, not patients (OASIS-3 is longitudinal). Features such as
     AgeatEntry, APOE, IntraCranialVol and shared MRI volumes are nearly
     constant for one person, so with a row-level split the forest can
     recognise a patient it saw in training and repeat that patient's label.

This script fixes both:
  - every patient (OASISID) is entirely in train or entirely in test,
  - class balancing happens INSIDE each training fold only,
  - 5-fold CV, so every real patient (all 79 Non-AD and 45 Others rows) is
    tested exactly once and the rare-class numbers are far less noisy than a
    single 20% split.

Models compared (all use the same folds)
----------------------------------------
  central_plain     centralized RF, no balancing
  central_cw        centralized RF, class_weight="balanced_subsample"
                    (no synthetic data at all)
  central_smote     centralized RF, SMOTENC on the training fold
  fed10_cw          federated tree pooling, 10 clients; REAL training patients
                    are split across clients BY PATIENT; each client trains a
                    100-tree forest with class_weight="balanced_subsample".
                    No data leaves a client and no synthetic data is used.
                    Clients that never saw a rare class cannot vote for it,
                    which is a genuine non-IID effect of federation.
  fed10_smote_iid   the earlier pipeline: SMOTE the training fold, split IID
                    across 10 clients (server-side SMOTE, not truly federated;
                    kept for continuity with scripts 07 / 10).

It also runs a ROW-LEVEL 5-fold CV (patients may appear in train and test)
for central_plain and central_cw, to show how much of the score comes from
patient re-identification rather than diagnosis.

Metrics: pooled out-of-fold accuracy, BALANCED accuracy and macro-F1 (primary,
because the real data is 77% CN), AUC, per-class recall with counts, and the
per-fold standard deviation.

REQUIRES two extra files written by 05_preprocess.py. After the three lines
you added earlier (the ones that save the _unbalanced file), add:

    pd.DataFrame({"OASISID": df["OASISID"].values}).to_csv(
        OUTPUT_FILE.replace(".csv", "_groups.csv"), index=False)

and re-run 05_preprocess.py.

Needs: pip install scikit-learn imbalanced-learn pandas numpy

Run from the preprocessing/ folder:
    python 11_grouped_cv_evaluation.py

Outputs: ../results/grouped_cv_results.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

UNBALANCED_FILE = "../results/ml_ready_dataset_unbalanced.csv"
GROUPS_FILE = "../results/ml_ready_dataset_groups.csv"
OUT_CSV = "../results/grouped_cv_results.csv"
LABEL_COL = "diagnosis_class"
GROUP_COL = "OASISID"

CATEGORICAL_COLS = ["APOE"]          # category codes -> SMOTENC
N_SPLITS = 5
N_CLIENTS = 10
CENTRAL_TREES = 100
TREES_PER_CLIENT = 100
LOCAL_TREE_PARAMS = dict(
    criterion="gini",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
)
SEED = 42
N_JOBS = -1

GROUPED_MODELS = ["central_plain", "central_cw", "central_smote",
                  "fed10_cw", "fed10_smote_iid"]
ROW_LEVEL_MODELS = ["central_plain", "central_cw"]
CM_MODELS = ["central_cw", "central_smote", "fed10_cw"]   # confusion matrices to print
NEEDS_SMOTE = {"central_smote", "fed10_smote_iid"}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def fit_rf(X, y, n_trees, seed, class_weight=None):
    return RandomForestClassifier(
        n_estimators=n_trees, random_state=seed, n_jobs=N_JOBS,
        class_weight=class_weight, **LOCAL_TREE_PARAMS,
    ).fit(X, y)


def aligned_proba(rf, X, n_classes):
    """predict_proba aligned to the global class set. A forest that never saw
    a class contributes probability 0 for it."""
    proba = rf.predict_proba(X)
    out = np.zeros((X.shape[0], n_classes))
    for col, cls in enumerate(rf.classes_):
        out[:, int(cls)] = proba[:, col]
    return out


def safe_oversample(X, y, feature_names, n_classes, seed=SEED):
    """SMOTE (SMOTENC for category-code columns) on the given TRAINING data."""
    from imblearn.over_sampling import SMOTE, SMOTENC
    counts = np.bincount(y, minlength=n_classes)
    if counts.min() < 2:
        print(f"    WARNING: a class has <2 training rows {counts.tolist()}; "
              f"skipping SMOTE in this fold")
        return X, y
    k = max(1, min(5, int(counts.min()) - 1))
    cat_idx = [feature_names.index(c) for c in CATEGORICAL_COLS if c in feature_names]
    if cat_idx:
        sm = SMOTENC(categorical_features=cat_idx, random_state=seed, k_neighbors=k)
    else:
        sm = SMOTE(random_state=seed, k_neighbors=k)
    return sm.fit_resample(X, y)


def partition_iid(X, y, n_clients, seed):
    """Class-stratified IID split by ROWS (used for the SMOTE'd training set)."""
    rng = np.random.RandomState(seed)
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    for c in idx_by_class:
        rng.shuffle(idx_by_class[c])
    client_idx = [[] for _ in range(n_clients)]
    for c, idxs in idx_by_class.items():
        for i, part in enumerate(np.array_split(idxs, n_clients)):
            client_idx[i].extend(part.tolist())
    return [(X[np.array(ix)], y[np.array(ix)]) for ix in client_idx]


def partition_by_patient(X, y, groups, n_clients, seed):
    """Split REAL training rows across clients so each patient lives on exactly
    one client (like separate sites), roughly balancing classes."""
    sgkf = StratifiedGroupKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    shards = []
    for _, idx in sgkf.split(X, y, groups):
        shards.append((X[idx], y[idx]))
    return shards


def federated_proba(shards, Xte, n_classes, seed, class_weight=None):
    """Tree pooling. With equal trees per client, the pooled forest's averaged
    probabilities equal the mean of the client forests' probabilities."""
    total = np.zeros((Xte.shape[0], n_classes))
    used = 0
    for i, (cx, cy) in enumerate(shards):
        if len(np.unique(cy)) < 2:
            continue                       # a client with a single class cannot learn a boundary
        rf = fit_rf(cx, cy, TREES_PER_CLIENT, seed=1000 * seed + i, class_weight=class_weight)
        total += aligned_proba(rf, Xte, n_classes)
        used += 1
    return total / max(used, 1)


def predict_model(name, ctx, n_classes):
    Xtr, ytr, gtr = ctx["Xtr"], ctx["ytr"], ctx["gtr"]
    Xte = ctx["Xte"]
    if name == "central_plain":
        return aligned_proba(fit_rf(Xtr, ytr, CENTRAL_TREES, SEED), Xte, n_classes)
    if name == "central_cw":
        rf = fit_rf(Xtr, ytr, CENTRAL_TREES, SEED, class_weight="balanced_subsample")
        return aligned_proba(rf, Xte, n_classes)
    if name == "central_smote":
        return aligned_proba(fit_rf(ctx["Xsm"], ctx["ysm"], CENTRAL_TREES, SEED), Xte, n_classes)
    if name == "fed10_cw":
        shards = partition_by_patient(Xtr, ytr, gtr, N_CLIENTS, SEED)
        return federated_proba(shards, Xte, n_classes, SEED, class_weight="balanced_subsample")
    if name == "fed10_smote_iid":
        shards = partition_iid(ctx["Xsm"], ctx["ysm"], N_CLIENTS, SEED)
        return federated_proba(shards, Xte, n_classes, SEED)
    raise ValueError(name)


def fold_scores(yte, proba):
    pred = np.argmax(proba, axis=1)
    labels = np.unique(yte)                # score only classes present in this test fold
    return (balanced_accuracy_score(yte, pred),
            f1_score(yte, pred, labels=labels, average="macro", zero_division=0))


def pooled_metrics(y, proba, n_classes):
    pred = np.argmax(proba, axis=1)
    res = dict(
        accuracy=accuracy_score(y, pred),
        balanced_acc=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y, pred, average="weighted", zero_division=0),
    )
    try:
        y_bin = label_binarize(y, classes=list(range(n_classes)))
        res["auc"] = roc_auc_score(y_bin, proba, average="macro")
    except ValueError:
        res["auc"] = np.nan
    return res


def run_cv(title, splits, X, y, groups, feature_names, class_names, models):
    n, k = len(y), len(class_names)
    oof = {m: np.zeros((n, k)) for m in models}
    fold_bal = {m: [] for m in models}
    fold_f1 = {m: [] for m in models}
    shared = []

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for f, (tr, te) in enumerate(splits, 1):
        t0 = time.time()
        shared.append(float(np.isin(groups[te], groups[tr]).mean()))
        ctx = dict(Xtr=X[tr], ytr=y[tr], gtr=groups[tr], Xte=X[te])
        if any(m in NEEDS_SMOTE for m in models):
            ctx["Xsm"], ctx["ysm"] = safe_oversample(ctx["Xtr"], ctx["ytr"], feature_names, k)
        for m in models:
            proba = predict_model(m, ctx, k)
            oof[m][te] = proba
            b, f1 = fold_scores(y[te], proba)
            fold_bal[m].append(b)
            fold_f1[m].append(f1)
        tc = np.bincount(y[te], minlength=k).tolist()
        print(f"  fold {f}/{len(splits)}: test rows {len(te)}, per-class {dict(zip(class_names, tc))}, "
              f"test rows from patients also in train: {shared[-1]:.1%}   [{time.time()-t0:.0f}s]")
    print(f"Average share of test rows whose patient also appears in training: "
          f"{np.mean(shared):.1%}")

    rows = []
    for m in models:
        res = pooled_metrics(y, oof[m], k)
        res["balanced_acc_fold_std"] = float(np.std(fold_bal[m]))
        res["scheme"] = title.split(":")[0]
        res["model"] = m
        pred = np.argmax(oof[m], axis=1)
        for c, cname in enumerate(class_names):
            res[f"recall_{cname}"] = float(((pred == c) & (y == c)).sum() / max((y == c).sum(), 1))
        rows.append(res)
    return rows, oof


def print_table(rows):
    print(f"\n{'model':18s}{'accuracy':>10s}{'bal.acc':>10s}{'macroF1':>10s}"
          f"{'AUC':>9s}{'fold std(bal)':>15s}")
    for r in rows:
        print(f"{r['model']:18s}{r['accuracy']:>10.2%}{r['balanced_acc']:>10.2%}"
              f"{r['macro_f1']:>10.2%}{r['auc']:>9.2%}{r['balanced_acc_fold_std']:>15.2%}")


def print_recall(rows, y, class_names):
    counts = np.bincount(y, minlength=len(class_names))
    print("\nPer-class recall (out-of-fold, every real row tested once):")
    print(f"{'model':18s}" + "".join(f"{c + f' (n={counts[i]})':>20s}"
                                      for i, c in enumerate(class_names)))
    for r in rows:
        print(f"{r['model']:18s}" + "".join(f"{r['recall_' + c]:>20.1%}" for c in class_names))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not (os.path.exists(UNBALANCED_FILE) and os.path.exists(GROUPS_FILE)):
        print("Missing the unbalanced file and/or the patient-ID file.\n\n"
              "In 05_preprocess.py, right after the 3 lines that save the _unbalanced\n"
              "file, add:\n\n"
              "    pd.DataFrame({\"OASISID\": df[\"OASISID\"].values}).to_csv(\n"
              "        OUTPUT_FILE.replace(\".csv\", \"_groups.csv\"), index=False)\n\n"
              "then re-run 05_preprocess.py.")
        sys.exit(1)
    try:
        import imblearn  # noqa: F401
    except ImportError:
        print("Missing package. Install with:  pip install imbalanced-learn")
        sys.exit(1)

    print("=" * 80)
    print("LOADING REAL (UNBALANCED) DATA AND PATIENT IDS")
    print("=" * 80)
    df = pd.read_csv(UNBALANCED_FILE)
    gdf = pd.read_csv(GROUPS_FILE)
    if len(df) != len(gdf):
        print(f"Row mismatch: {len(df)} rows vs {len(gdf)} patient IDs. Re-run 05_preprocess.py.")
        sys.exit(1)

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    class_names = list(le.classes_)
    k = len(class_names)
    feature_names = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_names].values
    groups = pd.factorize(gdf[GROUP_COL])[0]

    print(f"Rows (visits): {len(df)}   Patients: {len(np.unique(groups))}   "
          f"Features: {len(feature_names)}")
    print(f"Visits per patient: mean {len(df)/len(np.unique(groups)):.1f}, "
          f"max {np.bincount(groups).max()}")
    print(f"\n{'class':12s}{'visits':>8s}{'patients':>10s}")
    for c, cname in enumerate(class_names):
        print(f"{cname:12s}{(y == c).sum():>8d}{len(np.unique(groups[y == c])):>10d}")
    if min(len(np.unique(groups[y == c])) for c in range(k)) < N_SPLITS:
        print("\nNOTE: a class has fewer patients than folds, so some test folds will\n"
              "contain none of it. Pooled out-of-fold results still cover every patient.")

    all_rows = []

    # ---- Grouped (patient-level) CV -------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    g_splits = list(sgkf.split(X, y, groups))
    for tr, te in g_splits:                                     # sanity: no patient overlap
        assert not np.isin(groups[te], groups[tr]).any(), "patient appears in train and test"
    g_rows, g_oof = run_cv("GROUPED: patient-level 5-fold CV (honest)",
                           g_splits, X, y, groups, feature_names, class_names, GROUPED_MODELS)
    print_table(g_rows)
    print_recall(g_rows, y, class_names)
    for m in CM_MODELS:
        pred = np.argmax(g_oof[m], axis=1)
        cm = confusion_matrix(y, pred, labels=list(range(k)))
        print(f"\nConfusion matrix, {m} (rows=true, cols=predicted):")
        print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    all_rows += g_rows

    # ---- Row-level CV, for comparison -----------------------------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    r_splits = list(skf.split(X, y))
    r_rows, _ = run_cv("ROW-LEVEL: visit-level 5-fold CV (patients can be in train AND test)",
                       r_splits, X, y, groups, feature_names, class_names, ROW_LEVEL_MODELS)
    print_table(r_rows)
    print_recall(r_rows, y, class_names)
    all_rows += r_rows

    # ---- Summary ---------------------------------------------------------
    tab = pd.DataFrame(all_rows)
    tab.to_csv(OUT_CSV, index=False)

    def val(scheme, model, key):
        sel = tab[(tab.scheme == scheme) & (tab.model == model)]
        return float(sel[key].iloc[0])

    print("\n" + "=" * 80)
    print("SUMMARY - balanced accuracy / macro-F1")
    print("=" * 80)
    print(f"{'model':18s}{'row-level CV':>22s}{'patient-grouped CV':>24s}")
    for m in ROW_LEVEL_MODELS:
        print(f"{m:18s}{val('ROW-LEVEL', m, 'balanced_acc'):>13.2%}/{val('ROW-LEVEL', m, 'macro_f1'):.2%}"
              f"{val('GROUPED', m, 'balanced_acc'):>14.2%}/{val('GROUPED', m, 'macro_f1'):.2%}")
    for m in GROUPED_MODELS:
        if m not in ROW_LEVEL_MODELS:
            print(f"{m:18s}{'-':>22s}{val('GROUPED', m, 'balanced_acc'):>14.2%}/"
                  f"{val('GROUPED', m, 'macro_f1'):.2%}")
    gap = val("ROW-LEVEL", "central_cw", "balanced_acc") - val("GROUPED", "central_cw", "balanced_acc")
    print(f"\nPatient re-identification effect (central_cw, balanced accuracy): {gap:+.2%}")
    print("Report the GROUPED numbers as the honest performance estimate.")
    print(f"\nSaved full table to {OUT_CSV}")


if __name__ == "__main__":
    main()