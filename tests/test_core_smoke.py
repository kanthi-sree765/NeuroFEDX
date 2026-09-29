"""Offline smoke tests for the frozen NeuroFEDX core.

These tests intentionally use synthetic data. They verify configuration,
feature-stage semantics, RF parameters, 100-tree aggregation, and Tree SHAP.
They do not claim to reproduce OASIS results without the authorized data.
"""
import numpy as np
import pandas as pd

from config.model_config import RF_PARAMS, GLOBAL_FOREST_SIZE, N_CLIENTS, N_ROUNDS, PEARSON_THRESHOLD, KNN_NEIGHBORS
from models.random_forest import train_rf
from federated.federated_rf import run_federated_rounds
from preprocessing.preprocess_common import pearson_prune


def test_rf_config():
    assert RF_PARAMS["n_estimators"] == 100
    assert RF_PARAMS["criterion"] == "gini"
    assert RF_PARAMS["max_depth"] == 30
    assert RF_PARAMS["min_samples_split"] == 2
    assert RF_PARAMS["min_samples_leaf"] == 1
    assert RF_PARAMS["max_features"] == "sqrt"
    assert RF_PARAMS["bootstrap"] is True
    assert RF_PARAMS["random_state"] == 42
    assert GLOBAL_FOREST_SIZE == 100
    assert N_CLIENTS == 10
    assert N_ROUNDS == 50
    assert PEARSON_THRESHOLD == 0.95
    assert KNN_NEIGHBORS == 2


def test_pearson_is_real_selection():
    rng = np.random.RandomState(42)
    x = rng.normal(size=120)
    df = pd.DataFrame({"a": x, "b": x * 0.999 + rng.normal(scale=0.001, size=120), "c": rng.normal(size=120)})
    kept, dropped = pearson_prune(df, list(df.columns), threshold=0.95, verbose=False)
    assert "b" in dropped
    assert "a" in kept and "c" in kept


def test_federated_100_tree_rounds():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(300, 6))
    y = np.tile(np.arange(5), 60)
    # The smoke test uses a small number of rounds for speed; the production
    # configuration remains fixed at 50 rounds.
    accs, model, shards = run_federated_rounds(
        X[:240], y[:240], X[240:], y[240:],
        n_rounds=2, n_clients=10, total_trees=100,
        verbose=False, history_name="_smoke_round_history.csv",
    )
    assert len(accs) == 2
    assert len(model.estimators_) == 100
    assert len(shards) == 10
    # Remove the temporary history created by the smoke test.
    from config.model_config import RESULTS_DIR
    (RESULTS_DIR / "_smoke_round_history.csv").unlink(missing_ok=True)


def test_tree_shap():
    import shap
    rng = np.random.RandomState(11)
    X = rng.normal(size=(160, 5))
    y = np.tile(np.arange(5), 32)
    model = train_rf(X, y)
    sample = X[:20]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample, check_additivity=True)
    # SHAP 0.50 returns a 3-D array for multiclass sklearn RF.
    arr = np.asarray(values)
    assert arr.shape == (20, 5, 5)
