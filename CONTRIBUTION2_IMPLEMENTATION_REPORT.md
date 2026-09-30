# Contribution 2 Implementation Report

## A. Inspected architecture

The stabilized Contribution-1 project was inspected before implementation, including:

- `config/model_config.py`
- `preprocessing/`
- `models/random_forest.py`
- `federated/federated_rf.py`
- `evaluation/leakage_free.py`
- `evaluation/severity_prediction.py`
- `evaluation/severity_grouped_cv.py`
- `explainability/shap_analysis.py`
- `explainability/severity_shap.py`
- existing tests
- existing diagnosis/severity results and audits

The frozen base federated implementation is round-wise RF tree pooling: 10 clients, 50 rounds, 10 trees/client/round, and a 100-tree global forest. It is not called FedAvg.

## B. Integration point

Contribution 2 was added as an extension module. The original federated implementation and existing Contribution-1 scripts were left unchanged. Contribution 2 consumes the same cleaned feature representations and the existing RF tree-training primitives.

## C. New/changed files

New:

- `federated/personalized_rf.py`
- `evaluation/contribution2.py`
- `explainability/contribution2_shap.py`
- `tests/test_contribution2.py`
- `CONTRIBUTION2_README.md`
- `CONTRIBUTION2_IMPLEMENTATION_REPORT.md`
- `results/contribution2/` artifacts

Existing `README.md` and `CHANGELOG.md` were appended with Contribution-2 documentation. No existing source/result file from the Contribution-1 ZIP was changed.

## D. Client experiment

- Clients: 10
- Dirichlet concentration: `alpha=1.0`
- Random seed: 42
- Minimum client rows: 50
- Federation training: 60% of each client's participant set
- Personalization training: 20%
- Client held-out test: 20%
- Participants remain intact within one client and one split.

The original leakage-safe outer split has 1,030 training subjects and 258 held-out subjects with zero overlap. The Contribution-2 client experiment uses only the 1,030-subject outer training population; the outer test population is never used by the C2 models.

## E. Personalization definition

For client `i`:

`P_i = G ∪ L_i`

where `G` is the exact 100-tree global tree pool and `L_i` contains 10 additional trees trained only on the client's personalization-training data. The global model is identical across clients. Local trees differ by client data and seed.

Because non-IID clients can lack rare diagnosis classes, the C2 tree-pool wrapper aligns each tree's probability columns using its parent RF's class order. This avoids the sklearn internal remapping of tree class indices and preserves correct class probabilities.

## F. Diagnosis results

Across the 10 client test sets:

| Model | Accuracy mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| Local-only | 94.17% | 5.26% | 80.00% | 100.00% |
| Global federated | 95.62% | 2.32% | 92.98% | 100.00% |
| Personalized federated | 95.54% | 2.45% | 92.11% | 100.00% |

Personalization gains were calculated per client against the global model. The observed mean accuracy gain was approximately **-0.088 percentage points**. Thus this experiment does not show a universal accuracy improvement from personalization.

## G. Severity results

Across the 10 client test sets:

| Model | Accuracy mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| Local-only | 74.99% | 10.84% | 58.62% | 92.38% |
| Global federated | 76.71% | 11.46% | 60.61% | 94.29% |
| Personalized federated | 77.01% | 11.63% | 60.61% | 94.29% |

Observed mean personalization accuracy gain was approximately **+0.308 percentage points**, with both positive and negative client-level gains. This is reported descriptively and is not treated as proof of superiority.

## H. Leakage/reproducibility controls

Saved artifacts include:

- `client_distributions.csv`
- `sample_assignments.csv`
- `outer_split_audit.csv`
- `personalization_tree_audit.csv`
- model-specific metrics
- per-class metrics
- confusion matrices
- personalization gains
- global round history
- RF/global-model metadata

The final client tests are never used for model fitting or tuning.

## I. SHAP

Actual `shap.TreeExplainer` was used. For the class-aligned tree union, component TreeSHAP values were weighted by the exact number of trees in each component.

Maximum recorded additivity errors:

- diagnosis: approximately `2.16e-15`
- severity: approximately `1.11e-15`

These are numerical reconstruction errors, consistent with successful TreeSHAP additivity.

## J. Validation

Full project test suite:

`13 passed`

This includes the frozen core tests, Contribution-1 tests, and Contribution-2 structural/leakage tests.

## K. Reproduction command

From the project root:

```text
python evaluation/contribution2.py
python -m pytest -q
```

The run regenerates `results/contribution2/` without changing the frozen base-paper result files.

## L. Limitations

- This is a controlled client-heterogeneity experiment, not a claim of universal personalization benefit.
- The RF federation remains the project's documented tree-pooling interpretation, not parameter-vector FedAvg.
- No privacy guarantee is claimed; no differential privacy, secure aggregation, or cryptographic protection was added.
- Contribution 3 multimodal explanation is not implemented.
