"""Contribution 2: client-specific personalization on top of the existing RF-FL forest.

The base NeuroFEDX global model remains the existing round-wise RF tree-pooling
forest. A personalized client model is represented as the union of the global
trees and additional trees trained only from that client's personalization set.
No probability averaging between separate global/local models is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.model_config import RF_PARAMS, GLOBAL_FOREST_SIZE, N_CLIENTS, N_ROUNDS
from federated.federated_rf import train_federated_round


@dataclass
class ClientPartition:
    client_id: int
    federation_indices: np.ndarray
    personalization_indices: np.ndarray
    test_indices: np.ndarray


class PooledRandomForest:
    """RF tree pool with explicit parent-forest class alignment for non-IID clients."""
    def __init__(self, forests, classes):
        self.forests = list(forests)
        self.tree_specs = [(tree, np.asarray(forest.classes_))
                           for forest in self.forests for tree in forest.estimators_]
        self.estimators_ = [t for t,_ in self.tree_specs]
        self.classes_ = np.asarray(classes)
        self.n_estimators = len(self.estimators_)
        self.n_features_in_ = self.forests[0].n_features_in_

    def _aligned_tree_proba(self, tree, parent_classes, X):
        p = tree.predict_proba(X)
        out = np.zeros((len(X), len(self.classes_)), dtype=float)
        for j, c in enumerate(np.asarray(parent_classes)):
            k = np.flatnonzero(self.classes_ == c)
            if len(k): out[:, k[0]] = p[:, j]
        return out

    def predict_proba(self, X):
        total = np.zeros((len(X), len(self.classes_)), dtype=float)
        for tree, parent_classes in self.tree_specs:
            total += self._aligned_tree_proba(tree, parent_classes, X)
        return total / len(self.tree_specs)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class PersonalizedRandomForest:
    """A tree-union ensemble: global trees plus client-specific local trees.

    The wrapper aligns per-tree class probabilities to the global class order,
    which allows personalization trees trained on a client that lacks a rare
    class to remain valid without fabricating samples from another client.
    """

    def __init__(self, global_model: RandomForestClassifier, local_model: RandomForestClassifier):
        self.global_model = global_model
        self.local_model = local_model
        self.global_estimators_ = list(global_model.estimators_)
        self.local_estimators_ = list(local_model.estimators_)
        if hasattr(global_model, "tree_specs"):
            self.global_tree_specs = list(global_model.tree_specs)
        else:
            self.global_tree_specs = [(t, np.asarray(global_model.classes_)) for t in self.global_estimators_]
        self.local_tree_specs = [(t, np.asarray(local_model.classes_)) for t in self.local_estimators_]
        self.tree_specs = self.global_tree_specs + self.local_tree_specs
        self.estimators_ = [t for t,_ in self.tree_specs]
        self.classes_ = np.asarray(global_model.classes_)
        self.n_features_in_ = global_model.n_features_in_
        self.n_estimators = len(self.estimators_)

    @property
    def n_global_trees(self):
        return len(self.global_estimators_)

    @property
    def n_local_trees(self):
        return len(self.local_estimators_)

    def _aligned_tree_proba(self, tree, parent_classes, X):
        p = tree.predict_proba(X)
        out = np.zeros((len(X), len(self.classes_)), dtype=float)
        for j, c in enumerate(np.asarray(parent_classes)):
            matches = np.flatnonzero(self.classes_ == c)
            if len(matches): out[:, matches[0]] = p[:, j]
        return out

    def predict_proba(self, X):
        X = np.asarray(X)
        total = np.zeros((len(X), len(self.classes_)), dtype=float)
        for tree, parent_classes in self.tree_specs:
            total += self._aligned_tree_proba(tree, parent_classes, X)
        total /= len(self.estimators_)
        return total

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


def fit_local_rf(X, y, n_trees, seed):
    """Train personalization trees strictly from one client's private data."""
    params = dict(RF_PARAMS)
    params["n_estimators"] = int(n_trees)
    params["random_state"] = int(seed)
    return RandomForestClassifier(n_jobs=-1, **params).fit(X, y)


def personalize_global_model(global_model, X_personalization, y_personalization,
                             n_trees=10, seed=42):
    local = fit_local_rf(X_personalization, y_personalization, n_trees, seed)
    return PersonalizedRandomForest(global_model, local), local


def train_global_on_shards(client_shards, n_rounds=N_ROUNDS,
                            total_trees=GLOBAL_FOREST_SIZE, seed=42):
    """Run the unchanged RF tree-pooling algorithm on supplied heterogeneous shards."""
    model = None
    histories = []
    for round_number in range(1, n_rounds + 1):
        model = train_federated_round(
            client_shards,
            round_number=round_number,
            total_trees=total_trees,
            base_seed=seed,
        )
        histories.append({"round": round_number, "global_tree_count": len(model.estimators_)})
    return model, histories


def train_non_iid_federated_round(client_shards, round_number, total_trees=100, base_seed=42, classes=None):
    """Same round-wise tree pooling, with explicit class alignment for non-IID shards."""
    from federated.federated_rf import _tree_budget, _fit_client_forest
    budgets=_tree_budget(len(client_shards), total_trees)
    forests=[]
    for cid, ((Xc,yc),nt) in enumerate(zip(client_shards,budgets)):
        seed=base_seed + round_number*10000 + cid
        forests.append(_fit_client_forest(Xc,yc,nt,seed))
    if classes is None: classes=np.unique(np.concatenate([np.asarray(y) for _,y in client_shards]))
    return PooledRandomForest(forests, classes), forests


def train_non_iid_federated(client_shards, n_rounds=50, total_trees=100, seed=42, classes=None):
    model=None; history=[]; last_forests=None
    for rnd in range(1,n_rounds+1):
        model,last_forests=train_non_iid_federated_round(client_shards,rnd,total_trees,seed,classes)
        history.append({"round":rnd,"global_tree_count":len(model.estimators_)})
    return model,last_forests,history
