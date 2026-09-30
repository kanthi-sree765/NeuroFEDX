# NeuroFEDX Contribution 1 — CDR-based Severity Prediction

## Scope

This contribution adds a second prediction task to the existing NeuroFEDX system. The original five-class `diagnosis_class` task remains unchanged. This contribution predicts a CDR-based severity category from the cleaned multimodal representation.

Contribution 2 (personalized federated learning) and Contribution 3 (multimodal explanation/modality analysis) are **not implemented** here.

## Target definition

The OASIS-3 variable `CDRTOT` creates `severity_label` exactly as follows:

| CDRTOT | severity_label | Name |
|---|---:|---|
| 0 | 0 | Category 0 |
| 0.5 | 1 | Category 1 |
| >= 1 | 2 | Category 2 |

No clinical disease-stage terminology is assigned to these categories.

## Observed data

The actual merged dataset used for this implementation contains:

- 6,036 rows
- 1,288 unique participants
- Category 0: 4,680 rows
- Category 1: 1,016 rows
- Category 2: 340 rows

There were no missing/invalid `CDRTOT` values in this dataset.

## Target leakage prevention

The following variables are excluded from the severity feature matrix:

- `CDRTOT` — directly creates the target
- `CDRSUM`
- `memory`
- `orient`
- `judgment`
- `commun`
- `homehobb`
- `perscare`

These are CDR-derived variables that directly define or encode the target. The source `CDRTOT` is retained only in the severity-labelled audit artifact and is never passed to the model.

The 34 available/interpreted base-paper features therefore become **27 severity-eligible features** before imputation and Pearson selection.

## Feature selection

The severity model starts with 27 eligible multimodal features. Pearson selection is fitted on the training data only at `|r| > 0.95`.

Five features are removed in the participant-level holdout:

- `CortexVol`
- `RhCortexVol`
- `TotalGrayVol`
- `CorticalWhiteMatterVol`
- `RhCorticalWhiteMatterVol`

The resulting severity model uses **22 features**.

## Leakage-free split and preprocessing

The primary severity evaluation uses an 80/20 participant-level split with random state 42:

- Train: 4,819 rows / 1,030 participants
- Test: 1,217 rows / 258 participants
- Participant overlap: **0**

Processing order:

1. Participant-level split.
2. KNN imputation (`k=2`) fitted on training data only.
3. Pearson selection fitted on training data only.
4. SMOTE (`k_neighbors=5`, random state 42) applied to training data only.
5. RF trained on the resampled training set.
6. Test data remain untouched and are evaluated as observed.

Training-only SMOTE produced 3,741 samples per severity class in the training matrix.

## Model

The existing authoritative Random Forest configuration is reused:

- 100 trees
- Gini criterion
- maximum depth 30
- minimum samples split 2
- minimum samples leaf 1
- `max_features="sqrt"`
- bootstrap=True
- random state 42
- no class weighting

## Holdout results

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro OVR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Severity centralized RF | 81.02% | 67.06% | 62.82% | 67.06% | 64.42% | 80.83% | 86.95% |
| Severity federated RF, final round | 77.90% | 68.37% | 60.48% | 68.37% | 63.54% | 79.00% | 86.23% |

### Per-class centralized performance

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Category 0 | 90.55% | 90.84% | 90.70% | 939 |
| Category 1 | 47.34% | 41.59% | 44.28% | 214 |
| Category 2 | 50.57% | 68.75% | 58.28% | 64 |

### Per-class federated performance

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Category 0 | 91.94% | 85.09% | 88.38% | 939 |
| Category 1 | 40.55% | 48.13% | 44.02% | 214 |
| Category 2 | 48.94% | 71.88% | 58.23% | 64 |

## Grouped 5-fold CV

All five folds have zero participant overlap.

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 79.72% | 1.73% |
| Balanced Accuracy | 66.77% | 1.37% |
| Macro Precision | 62.82% | 1.51% |
| Macro Recall | 66.77% | 1.37% |
| Macro F1 | 64.43% | 0.75% |
| Weighted F1 | 79.82% | 1.63% |
| Macro OVR AUC | 84.96% | 2.60% |

## Federated severity extension

The existing RF federated framework is reused without changing its methodology. It remains the project's documented **round-wise federated tree pooling** interpretation, not parameter-vector FedAvg.

- 10 clients
- 50 rounds
- 10 trees/client
- 100-tree global forest
- fixed client partition
- no global model reuse as local training state

The final-round federated severity result is reported separately from the base-paper diagnosis FL result.

## TreeSHAP

The existing real `shap.TreeExplainer` implementation was extended to the severity models.

- SHAP version: 0.50.0
- Centralized model: 60 balanced test samples
- Federated model: 60 balanced test samples
- Maximum additivity error, centralized: `1.082e-13`
- Maximum additivity error, federated: `1.277e-15`

Top overall mean absolute SHAP features for the centralized severity model were `MMSE`, `MEMUNITS`, `VEG`, `LOGIMEM`, and `WAIS`.

These are model explanations, not causal or clinical claims.

## Federated vs diagnosis distinction

This contribution does not replace `diagnosis_class`. The existing five-class diagnosis prediction remains intact. Severity is a separate three-class target generated from `CDRTOT`.

## Limitations

- The three severity categories are task-defined CDR-based categories, not formally validated clinical stages.
- Severity target construction necessarily depends on `CDRTOT`; target-derived CDR variables are therefore excluded from prediction.
- The federated implementation inherits the existing RF tree-pooling interpretation and should not be described as literal numerical FedAvg.
- The severity task is an experimental research contribution and is not clinically validated.

## Reproduction commands

From the project root, with the authorized merged OASIS output available at `results/merged_oasis3.csv`:

```powershell
python evaluation/severity_prediction.py
python evaluation/severity_grouped_cv.py
python explainability/severity_shap.py
python -m pytest -q tests/test_core_smoke.py tests/test_severity_contribution.py
```

The existing diagnosis commands and results remain unchanged.
