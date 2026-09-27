"""
Diagnostics for the federated-vs-centralized accuracy gap.

Question being answered: why does federated tree pooling get ~94.5%
when the centralized RF gets ~98.95%?

Three quick checks (whole script should take a few minutes):

  A. LEARNING CURVE - train a normal centralized RF (same params as
     07_federated_learning.py) on 5%, 10%, 25%, 50%, 100% of the
     training data. The 10% row is a FLOOR for 10-client federation
     (one model that sees a single client's slice); the 100% row is
     the CEILING (centralized). Pooling ten different slices should
     land between the two.

  B. CLIENT SWEEP - federated tree pooling with 1, 2, 5, 10, 20
     clients (100 trees per client, IID split, same as script 07).
     1 client should reproduce the centralized result (sanity check).
     If accuracy falls steadily as clients increase (fewer rows per
     client), the gap is caused by data fragmentation per tree, not
     by a bug. Gives an accuracy-vs-clients curve for the report.

  C. DUPLICATE / LEAKAGE CHECK - counts test rows that also appear
     EXACTLY in the training set. A high number would mean the class
     balancing step duplicated rows before the train/test split, which
     inflates the centralized score and hurts small partitions.
     (Synthetic-oversampling methods such as SMOTE create near
     duplicates, not exact ones, so 0 exact matches does not fully
     rule that out.)

Run from the preprocessing/ folder:
    python 08_federated_diagnostics.py
Optional: python 08_federated_diagnostics.py path/to/ml_ready_dataset.csv

Outputs: ../results/federated_diagnostics.csv
         ../results/federated_diagnostics.png (if matplotlib installed)
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
LABEL_COL = "diagnosis_class"
OUT_CSV = "../results/federated_diagnostics.csv"
OUT_PNG = "../results/federated_diagnostics.png"

# Must match 07_federated_learning.py
TREES_PER_CLIENT = 100
LOCAL_TREE_PARAMS = dict(
    criterion="gini",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
)

TRAIN_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
CLIENT_COUNTS = [1, 2, 5, 10, 20]
REPEATS = 3          # repeats per setting, averaged (different seeds)
N_JOBS = -1          # use all CPU cores


def partition_clients(X, y, n_clients, seed):
    """IID, class-stratified split into n_clients shards (numpy arrays).
    Same logic as script 07."""
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
    rf = RandomForestClassifier(
        n_estimators=n_trees, random_state=seed, n_jobs=N_JOBS,
        **LOCAL_TREE_PARAMS,
    )
    rf.fit(X, y)
    return rf


def federated_accuracy(shards, X_test, y_test, n_classes, seed):
    """Tree pooling: every client trains a full local forest, the server
    pools all trees. With equal trees per client, the pooled forest's
    averaged predict_proba equals the mean of each client forest's
    predict_proba, so we compute it that way (much faster)."""
    total = np.zeros((X_test.shape[0], n_classes))
    for i, (cx, cy) in enumerate(shards):
        rf = fit_rf(cx, cy, TREES_PER_CLIENT, seed * 1000 + i)
        proba = rf.predict_proba(X_test)
        # Align columns to the global class set in case a shard lacked a class
        aligned = np.zeros_like(total)
        for col, cls in enumerate(rf.classes_):
            aligned[:, int(cls)] = proba[:, col]
        total += aligned
    total /= len(shards)
    return accuracy_score(y_test, np.argmax(total, axis=1))


def main():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    df = pd.read_csv(INPUT_FILE)
    X_df = df.drop(columns=[LABEL_COL])
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    n_classes = len(le.classes_)

    # Same split as scripts 06 / 07
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_test = X_train_df.values, X_test_df.values
    print(f"Train: {X_train.shape}   Test: {X_test.shape}   Classes: {list(le.classes_)}")

    rows = []

    # ------------------------------------------------------------------
    # A. Learning curve (centralized RF on a fraction of the train set)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("A. LEARNING CURVE - centralized RF (100 trees) on a fraction of train data")
    print("=" * 80)
    print(f"{'Train fraction':>15s} {'Rows':>8s} {'Accuracy (mean)':>17s} {'Std':>8s}")
    for frac in TRAIN_FRACTIONS:
        accs = []
        for r in range(REPEATS if frac < 1.0 else 1):
            if frac < 1.0:
                X_sub, _, y_sub, _ = train_test_split(
                    X_train, y_train, train_size=frac,
                    stratify=y_train, random_state=100 + r,
                )
            else:
                X_sub, y_sub = X_train, y_train
            rf = fit_rf(X_sub, y_sub, 100, seed=r)
            accs.append(accuracy_score(y_test, rf.predict(X_test)))
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
        n_rows = int(round(len(X_train) * frac))
        print(f"{frac:>15.0%} {n_rows:>8d} {mean_acc:>17.2%} {std_acc:>8.2%}")
        rows.append(dict(experiment="learning_curve", setting=f"{frac:.0%} of train data",
                         x=frac, accuracy=mean_acc, std=std_acc))

    # ------------------------------------------------------------------
    # B. Client sweep (federated tree pooling)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"B. CLIENT SWEEP - federated tree pooling, {TREES_PER_CLIENT} trees/client, IID split")
    print("=" * 80)
    print(f"{'Clients':>8s} {'Rows/client':>12s} {'Pooled trees':>13s} "
          f"{'Accuracy (mean)':>17s} {'Std':>8s} {'Time':>8s}")
    for n_clients in CLIENT_COUNTS:
        t0 = time.time()
        accs = []
        for r in range(REPEATS):
            shards = partition_clients(X_train, y_train, n_clients, seed=r)
            accs.append(federated_accuracy(shards, X_test, y_test, n_classes, seed=r))
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
        print(f"{n_clients:>8d} {len(X_train)//n_clients:>12d} "
              f"{n_clients*TREES_PER_CLIENT:>13d} {mean_acc:>17.2%} "
              f"{std_acc:>8.2%} {time.time()-t0:>7.0f}s")
        rows.append(dict(experiment="client_sweep", setting=f"{n_clients} clients",
                         x=n_clients, accuracy=mean_acc, std=std_acc))

    # ------------------------------------------------------------------
    # C. Exact-duplicate leakage check
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("C. DUPLICATE / LEAKAGE CHECK")
    print("=" * 80)
    train_hashes = set(pd.util.hash_pandas_object(X_train_df, index=False))
    test_hashes = pd.util.hash_pandas_object(X_test_df, index=False)
    n_dup_test = int(test_hashes.isin(train_hashes).sum())
    n_dup_full = int(df.drop(columns=[LABEL_COL]).duplicated().sum())
    print(f"Test rows that exactly match a training row : {n_dup_test} / {len(X_test_df)} "
          f"({n_dup_test/len(X_test_df):.1%})")
    print(f"Duplicate feature-rows in the full dataset  : {n_dup_full} / {len(df)} "
          f"({n_dup_full/len(df):.1%})")
    rows.append(dict(experiment="duplicates", setting="test rows also in train",
                     x=np.nan, accuracy=n_dup_test / len(X_test_df), std=np.nan))

    # ------------------------------------------------------------------
    # Interpretation help
    # ------------------------------------------------------------------
    lc = {r["x"]: r["accuracy"] for r in rows if r["experiment"] == "learning_curve"}
    fed = {r["x"]: r["accuracy"] for r in rows if r["experiment"] == "client_sweep"}
    print("\n" + "=" * 80)
    print("HOW TO READ THIS")
    print("=" * 80)
    floor, ceiling = lc[0.10], lc[1.00]
    print(f"Floor   (centralized RF on 10% of data) : {floor:.2%}")
    print(f"Federated, 10 clients                   : {fed[10]:.2%}")
    print(f"Ceiling (centralized RF, all data)      : {ceiling:.2%}")

    # 1) Sanity: 1 client should behave like centralized
    if abs(fed[1] - ceiling) > 0.01:
        print("-> WARNING: 1-client federation differs from centralized by >1 point.\n"
              "   Check pooling logic and hyperparameters vs 06_train_rf.py.")
    else:
        print("-> Sanity OK: 1-client federation matches centralized.")

    # 2) Position of the 10-client result
    if floor - 0.005 <= fed[10] <= ceiling + 0.005:
        print("-> 10-client result sits between floor and ceiling, as expected for\n"
              "   pooling slices of the data.")
    else:
        print("-> 10-client result is outside the floor/ceiling range: unexpected,\n"
              "   check the pooling code.")

    # 3) Trend with client count
    sweep = [fed[k] for k in sorted(fed)]
    if all(a >= b - 0.003 for a, b in zip(sweep, sweep[1:])):
        print("-> Accuracy falls steadily as clients (and thus rows-per-client)\n"
              "   increase: the gap is a data-fragmentation effect, not a bug.")
    else:
        print("-> Accuracy does not fall steadily with client count; the gap may have\n"
              "   another cause. Look at hyperparameters and the data-split step.")

    # 4) Leakage
    if n_dup_test / len(X_test_df) > 0.05:
        print("-> Many test rows are exact copies of training rows: the class-balancing\n"
              "   step probably duplicated rows before the split. The centralized\n"
              "   accuracy is likely inflated. Fix by splitting first, balancing after.")
    else:
        print("-> Few exact duplicates. Also check whether balancing (e.g. SMOTE) was\n"
              "   applied BEFORE the train/test split, since that leaves near-duplicates.")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to {OUT_CSV}")

    # Optional plot for the report
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        xs = sorted(lc)
        axes[0].plot([x * 100 for x in xs], [lc[x] * 100 for x in xs], marker="o")
        axes[0].set_xlabel("Training data used (%)")
        axes[0].set_ylabel("Test accuracy (%)")
        axes[0].set_title("Centralized RF learning curve")
        axes[0].grid(alpha=0.3)

        xs = sorted(fed)
        axes[1].plot(xs, [fed[x] * 100 for x in xs], marker="o", color="tab:orange")
        axes[1].set_xlabel("Number of clients")
        axes[1].set_ylabel("Test accuracy (%)")
        axes[1].set_title("Federated tree pooling vs. number of clients")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        print(f"Saved plot to {OUT_PNG}")
    except ImportError:
        print("(matplotlib not installed - skipped plot; pip install matplotlib to enable)")


if __name__ == "__main__":
    main()