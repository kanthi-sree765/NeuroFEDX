"""
Federated Learning on top of the Random Forest pipeline, matching the
base paper's stated setup: 10 clients, 50 communication rounds.

IMPORTANT DESIGN NOTE (flagged earlier in this project's planning):
Standard Federated Learning (FedAvg) works by averaging NUMERICAL
WEIGHTS across clients each round -- this is built for neural
networks. A Random Forest has no such weights; it's a collection of
decision trees. The paper does not specify how it federates RF, so
this script uses the standard, defensible interpretation for
federated tree ensembles: TREE POOLING.

How it works here:
  - The forest's total tree budget (100, matching the paper's single
    RF) is split evenly across the 10 clients (10 trees each).
  - Each "communication round", every client trains its 10 trees
    locally on its own data slice (bootstrap-sampled, so trees differ
    round to round) and sends the trained trees (not raw data) to the
    server.
  - The server POOLS all 100 trees into one global forest and
    evaluates it on the held-out test set. No numerical weights are
    averaged; the trees themselves are the "model update".
  - This repeats for 50 rounds so you can see how stable/consistent
    the federated approach is round to round.

This is a legitimate, commonly used way to federate tree ensembles
(sometimes called a "federated forest"). Document this design choice
explicitly in your report -- it's a defensible interpretation, not
the only possible one, since the paper itself doesn't specify.

Needs: pip install scikit-learn

Run from the preprocessing/ folder:
    python 07_federated_learning.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

INPUT_FILE = "../results/ml_ready_dataset.csv"
LABEL_COL = "diagnosis_class"

N_CLIENTS = 10
# CHANGED: each client now trains a FULL local forest (100 trees),
# instead of the total tree budget being split 10 ways (10 trees each).
#
# Why: the first version of this script produced 94.29% federated
# accuracy vs. 98.95% centralized -- a real ~4.7-point gap. But the
# actual paper being reproduced (Jahan et al., IEEE Access 2025)
# reports federated accuracy of 98.93%, essentially matching its own
# centralized/non-federated result (98.81%, from the companion 2023
# paper). So a large federated accuracy drop is NOT what the original
# work found -- something about our aggregation setup was too weak.
#
# The paper never specifies its exact aggregation mechanism (this
# ambiguity was flagged from the start of this project). Splitting a
# fixed 100-tree budget across 10 clients was one reasonable reading;
# giving each client a FULL forest and pooling everything is another,
# and is the standard way most "federated forest" implementations
# work in practice. This version tries that instead.
TREES_PER_CLIENT = 100
TOTAL_TREES = N_CLIENTS * TREES_PER_CLIENT   # 1000, informational only
N_ROUNDS = 50
PRINT_EVERY = 5            # avoid flooding the console with 50 lines

LOCAL_TREE_PARAMS = dict(
    criterion="gini",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
)


def partition_clients(X, y, n_clients, seed=42):
    """Split (X, y) into n_clients roughly equal, class-stratified
    shards -- an IID partition. Returns a list of (X_i, y_i)."""
    rng = np.random.RandomState(seed)
    indices_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    for c in indices_by_class:
        rng.shuffle(indices_by_class[c])

    client_indices = [[] for _ in range(n_clients)]
    for c, idxs in indices_by_class.items():
        splits = np.array_split(idxs, n_clients)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    shards = []
    for idxs in client_indices:
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        shards.append((X.iloc[idxs].reset_index(drop=True),
                        y[idxs]))
    return shards


def pooled_predict_proba(trees, X, n_classes):
    """Average predict_proba across a pooled list of DecisionTree
    estimators, correctly aligning each tree's local class columns to
    the full global class set (a client's bootstrap sample could in
    principle miss a rare class)."""
    X_values = X.values if hasattr(X, "values") else X
    n_samples = X_values.shape[0]
    total = np.zeros((n_samples, n_classes))
    for tree in trees:
        local_proba = tree.predict_proba(X_values)
        aligned = np.zeros((n_samples, n_classes))
        for col_idx, class_label in enumerate(tree.classes_):
            aligned[:, int(class_label)] = local_proba[:, col_idx]
        total += aligned
    return total / len(trees)


def main():
    print("=" * 80)
    print("LOADING ML-READY DATASET")
    print("=" * 80)
    df = pd.read_csv(INPUT_FILE)
    print(f"Shape: {df.shape}")

    X = df.drop(columns=[LABEL_COL])
    y_raw = df[LABEL_COL]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"Class encoding: {dict(zip(le.classes_, range(n_classes)))}")

    # Same split as the centralized RF script, so results are
    # directly comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    # --------------------------------------------------------
    # Partition training data across 10 clients
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"PARTITIONING TRAINING DATA ACROSS {N_CLIENTS} CLIENTS (IID)")
    print("=" * 80)

    client_shards = partition_clients(X_train, y_train, N_CLIENTS)
    client_shards = [(cx.values, cy) for cx, cy in client_shards]
    for i, (cx, cy) in enumerate(client_shards):
        counts = pd.Series(cy).value_counts().sort_index()
        print(f"  Client {i+1}: {len(cx)} rows, class counts "
              f"{dict(zip(le.classes_, counts.reindex(range(n_classes), fill_value=0)))}")

    # --------------------------------------------------------
    # Federated training loop
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"FEDERATED TRAINING ({N_ROUNDS} rounds, tree pooling, "
          f"{TREES_PER_CLIENT} trees/client -> {TOTAL_TREES} total)")
    print("=" * 80)

    round_accuracies = []
    final_pooled_trees = None

    for round_num in range(1, N_ROUNDS + 1):
        pooled_trees = []
        for client_idx, (cx, cy) in enumerate(client_shards):
            local_rf = RandomForestClassifier(
                n_estimators=TREES_PER_CLIENT,
                random_state=1000 * round_num + client_idx,
                **LOCAL_TREE_PARAMS,
            )
            local_rf.fit(cx, cy)
            pooled_trees.extend(local_rf.estimators_)

        proba = pooled_predict_proba(pooled_trees, X_test, n_classes)
        preds = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, preds)
        round_accuracies.append(acc)

        if round_num % PRINT_EVERY == 0 or round_num == 1:
            print(f"  Round {round_num:3d}/{N_ROUNDS}: "
                  f"global forest accuracy = {acc:.4%}")

        final_pooled_trees = pooled_trees  # keep the last round's forest

    # --------------------------------------------------------
    # Final evaluation (last round's global model)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL GLOBAL MODEL EVALUATION (last round)")
    print("=" * 80)

    final_proba = pooled_predict_proba(final_pooled_trees, X_test, n_classes)
    final_preds = np.argmax(final_proba, axis=1)

    final_acc = accuracy_score(y_test, final_preds)
    final_precision = precision_score(y_test, final_preds, average="weighted")
    final_recall = recall_score(y_test, final_preds, average="weighted")
    final_f1 = f1_score(y_test, final_preds, average="weighted")
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    final_auc = roc_auc_score(y_test_bin, final_proba, average="weighted", multi_class="ovr")

    print(f"Accuracy:  {final_acc:.4%}")
    print(f"Precision: {final_precision:.4%}")
    print(f"Recall:    {final_recall:.4%}")
    print(f"F1-score:  {final_f1:.4%}")
    print(f"AUC:       {final_auc:.4%}")

    print("\nClassification report:")
    print(classification_report(y_test, final_preds, target_names=le.classes_))

    print("\nConfusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, final_preds)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

    # --------------------------------------------------------
    # Round-to-round stability summary
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("ROUND-TO-ROUND STABILITY")
    print("=" * 80)
    round_acc_arr = np.array(round_accuracies)
    print(f"Mean accuracy across all {N_ROUNDS} rounds: {round_acc_arr.mean():.4%}")
    print(f"Std across rounds:                          {round_acc_arr.std():.4%}")
    print(f"Best round:  {round_acc_arr.argmax()+1} ({round_acc_arr.max():.4%})")
    print(f"Worst round: {round_acc_arr.argmin()+1} ({round_acc_arr.min():.4%})")

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON: CENTRALIZED RF vs FEDERATED RF vs PAPER (federated)")
    print("=" * 80)
    print("NOTE: the paper's own reported federated numbers are used here")
    print("(98.93% / 98.94% / 98.93% / 98.93% / 99.97%) -- NOT the earlier")
    print("centralized-only figures. The actual 2025 FL paper reports")
    print("federated accuracy essentially matching its centralized version.")
    print(f"{'Metric':25s} {'Federated (this run)':>22s} {'Paper (federated)':>18s}")
    print(f"{'Accuracy':25s} {final_acc:21.2%} {'98.93%':>18s}")
    print(f"{'Precision':25s} {final_precision:21.2%} {'98.94%':>18s}")
    print(f"{'Recall':25s} {final_recall:21.2%} {'98.93%':>18s}")
    print(f"{'F1-score':25s} {final_f1:21.2%} {'98.93%':>18s}")
    print(f"{'AUC':25s} {final_auc:21.2%} {'99.97%':>18s}")

    print("\nCompare this against your centralized (non-federated) RF result")
    print("from 06_train_rf.py -- the federated version splitting the same")
    print("data across 10 clients should land close to the centralized")
    print("number. A large gap would suggest the client partition is too")
    print("small/skewed for 10 trees per client to learn well locally.")
    print("=" * 80)

    # Save pooled trees info for the SHAP step (next script)
    np.save("../results/federated_round_accuracies.npy", round_acc_arr)
    print("\nSaved round-by-round accuracy history to "
          "results/federated_round_accuracies.npy")


if __name__ == "__main__":
    main()