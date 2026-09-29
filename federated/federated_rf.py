"""
Federated Random Forest implementation for the NeuroFEDX reproduction.

IMPORTANT TERMINOLOGY
---------------------
The paper calls its RF aggregation "FedAvg" and describes averaging local
RF "weights", but it never defines an RF parameterization or a tree/weight
aggregation equation. Standard FedAvg is defined for numeric parameter vectors;
a fitted RandomForestClassifier is an ensemble of discrete decision trees, so
there is no literal node/weight average that can be performed without inventing
an operation.

This module therefore implements the closest technically defensible RF-FL
interpretation supported by the paper's stated constraints and by federated
random-forest practice:

  * 10 fixed IID clients.
  * Each client trains 10 local trees per communication round.
  * The server pools those 100 trees into the global forest.
  * The global forest is replaced/updated every round.
  * The global forest is ALWAYS exactly 100 trees.
  * Client partitions are fixed across rounds; tree bootstrap/random seeds vary.
  * No raw client data are sent to the server.

This IS round-wise federated tree pooling. It is NOT literal numerical FedAvg.
The global forest is replaced after each round but is NOT used to initialize
local training in the next round; sklearn RandomForest trees do not expose a
paper-defined update operation from a fitted global forest.
The distinction is intentional and is documented in REPRODUCTION_REPORT.md.

The paper's Algorithm 1/2 additionally says that the global RF is sent back
and clients train "the updated RF". A sklearn decision tree cannot be updated
from a fitted global tree without defining a new training/aggregation rule.
We therefore do not pretend that this part is reproduced. The implemented
round is the defensible RF-specific analogue: fresh local tree updates from
the fixed private client data, followed by a 100-tree global aggregation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import (
    RF_PARAMS,
    N_CLIENTS,
    TREES_PER_CLIENT,
    GLOBAL_FOREST_SIZE,
    N_ROUNDS,
    FEDERATION_METHOD,
    RESULTS_DIR,
)

LOCAL_TREE_PARAMS = {
    k: v for k, v in RF_PARAMS.items()
    if k not in ("n_estimators", "random_state", "class_weight")
}


def partition_clients_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = N_CLIENTS,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create one fixed, class-stratified IID client partition."""
    rng = np.random.RandomState(seed)
    client_indices = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c).copy()
        rng.shuffle(idx)
        for client_id, part in enumerate(np.array_split(idx, n_clients)):
            client_indices[client_id].extend(part.tolist())

    shards = []
    for idxs in client_indices:
        idxs = np.asarray(idxs, dtype=int)
        rng.shuffle(idxs)
        shards.append((X[idxs], y[idxs]))
    return shards


def _tree_budget(n_clients: int, total_trees: int) -> list[int]:
    """Split a global tree budget as evenly as possible."""
    base, remainder = divmod(total_trees, n_clients)
    return [base + (i < remainder) for i in range(n_clients)]


def _fit_client_forest(
    X_client: np.ndarray,
    y_client: np.ndarray,
    n_trees: int,
    seed: int,
) -> RandomForestClassifier:
    if n_trees <= 0:
        raise ValueError("Every client must receive at least one tree.")
    model = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=seed,
        n_jobs=-1,
        **LOCAL_TREE_PARAMS,
    )
    model.fit(X_client, y_client)
    return model


def aggregate_tree_pool(
    local_forests: Iterable[RandomForestClassifier],
    global_tree_count: int = GLOBAL_FOREST_SIZE,
) -> RandomForestClassifier:
    """Pool local trees into a fitted sklearn RF with exactly global_tree_count."""
    forests = list(local_forests)
    if not forests:
        raise ValueError("At least one local forest is required.")

    trees = [tree for forest in forests for tree in forest.estimators_]
    if len(trees) != global_tree_count:
        raise ValueError(
            f"Aggregation produced {len(trees)} trees; expected "
            f"{global_tree_count}."
        )

    # All local forests share the same feature/class space. Start from the
    # first fitted forest so sklearn metadata (classes_, n_features_in_, etc.)
    # remains valid, then replace only the tree ensemble.
    global_model = forests[0]
    global_model.estimators_ = trees
    global_model.n_estimators = len(trees)
    return global_model


def train_federated_round(
    client_shards: list[tuple[np.ndarray, np.ndarray]],
    round_number: int,
    total_trees: int = GLOBAL_FOREST_SIZE,
    base_seed: int = 42,
) -> RandomForestClassifier:
    """Train one RF-FL communication round on a fixed client partition."""
    budgets = _tree_budget(len(client_shards), total_trees)
    local_forests = []

    for client_id, ((X_client, y_client), n_trees) in enumerate(
        zip(client_shards, budgets)
    ):
        seed = base_seed + round_number * 10_000 + client_id
        local_forests.append(
            _fit_client_forest(X_client, y_client, n_trees, seed)
        )

    return aggregate_tree_pool(local_forests, total_trees)


def write_client_distribution(
    shards: list[tuple[np.ndarray, np.ndarray]],
    class_names: Iterable,
    out_path=None,
) -> pd.DataFrame:
    rows = []
    class_names = list(class_names)
    for client_id, (_, y_client) in enumerate(shards):
        counts = pd.Series(y_client).value_counts()
        row = {"client_id": client_id, "num_samples": len(y_client)}
        for cname in class_names:
            row[cname] = int(counts.get(cname, 0))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = out_path or (RESULTS_DIR / "client_distribution.csv")
    df.to_csv(out_path, index=False)
    return df


def run_federated_rounds(
    X_train,
    y_train,
    X_test,
    y_test,
    n_rounds: int = N_ROUNDS,
    n_clients: int = N_CLIENTS,
    total_trees: int = GLOBAL_FOREST_SIZE,
    class_names=None,
    client_seed: int = 42,
    verbose: bool = True,
    history_name: str = "federated_round_history.csv",
    client_distribution_name: str = "client_distribution.csv",
):
    """
    Run the configured round-wise interpreted RF-FL procedure.

    Client partitions are created ONCE and reused across all rounds. Each round
    creates fresh local trees and replaces the 100-tree global ensemble.
    The global forest is not reused as a local training state, so these are
    repeated communication-style tree-pooling rounds, not literal FedAvg
    continuation from the previous global parameters.

    Returns:
        round_accuracies, final_global_model, client_shards
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if total_trees != GLOBAL_FOREST_SIZE:
        raise ValueError(
            f"Paper-faithful global forest must remain {GLOBAL_FOREST_SIZE} trees."
        )

    client_shards = partition_clients_iid(
        X_train, y_train, n_clients=n_clients, seed=client_seed
    )
    if class_names is not None:
        write_client_distribution(client_shards, class_names, RESULTS_DIR / client_distribution_name)

    round_accuracies = []
    final_model = None

    for round_number in range(1, n_rounds + 1):
        final_model = train_federated_round(
            client_shards,
            round_number=round_number,
            total_trees=total_trees,
            base_seed=client_seed,
        )
        accuracy = accuracy_score(y_test, final_model.predict(X_test))
        round_accuracies.append(accuracy)

        if verbose and (round_number == 1 or round_number % 5 == 0):
            print(
                f"Round {round_number:02d}/{n_rounds}: "
                f"accuracy={accuracy:.4%}; "
                f"global_trees={len(final_model.estimators_)}; "
                f"method={FEDERATION_METHOD}"
            )

    history = pd.DataFrame({
        "round": np.arange(1, n_rounds + 1),
        "accuracy": round_accuracies,
        "global_tree_count": [total_trees] * n_rounds,
        "method": [FEDERATION_METHOD] * n_rounds,
    })
    history.to_csv(
        RESULTS_DIR / history_name, index=False
    )

    return np.asarray(round_accuracies), final_model, client_shards
