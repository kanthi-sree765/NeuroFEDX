# NeuroFEDX

Reproduction and verification implementation for Jahan et al., *Federated Explainable AI-Based Alzheimer’s Disease Prediction With Multimodal Data*, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3547343.

## Start here

- `REPRODUCTION_REPORT.md` — authoritative final audit and A–D reproducibility classification.
- `CHANGELOG.md` — implementation changes and their reasons.
- `results/FINAL_COMPARISON_TABLE.csv` — existing OASIS result comparison; see the final report for its run-status caveat.
- `MISSING_OASIS_DATA_REQUEST.md` — exact request for the five unavailable OASIS variables.

## Scientific status

The code is stabilized, but the project is **not declared an exact reproduction** of the paper. The paper's RF-FedAvg aggregation is underspecified, five requested OASIS variables are unavailable, three diagnosis labels remain unresolved, and the paper does not specify an MRI matching tolerance.

The project never substitutes the five unavailable variables and never maps `MEMTIME` to `lmdelay`.

## Authoritative structure

```text
config/              Single source of truth for RF, FL, MRI and preprocessing settings
preprocessing/       merge.py, feature_mapping.py, diagnosis_mapping.py,
                     preprocess_common.py, preprocess_paper_faithful.py
models/              random_forest.py
federated/           federated_rf.py — RF-specific tree pooling, not literal FedAvg
explainability/      shap_analysis.py — real shap.TreeExplainer
                     permutation_importance_analysis.py — supplementary only
evaluation/          paper_faithful.py, leakage_free.py, grouped_cv.py,
                     mri_tolerance_sensitivity.py, comparison_table.py
results/             audited result and sensitivity files
original/            historical pre-fix scripts; not part of the active pipeline
tests/               offline smoke tests; no OASIS data required
```

Obsolete numbered preprocessing scripts were removed from the active package because they contained conflicting RF/FL settings or the old forbidden MEMTIME mapping. Their historical copies remain under `original/`.

## Feature pipeline

```text
39 paper candidates
        ↓
5 unavailable — never replaced
        ↓
34 available/interpreted features
        ↓
KNN (k=2)
        ↓
Pearson |r| > 0.95
        ↓
SMOTE
        ↓
Random Forest (100 trees)
        ↓
RF-specific federated tree pooling / evaluation
        ↓
Tree SHAP
```

The five MRI features that the paper reports as Pearson-pruned remain in the 34-feature pre-Pearson set so that the Pearson stage actually performs the selection.

## Running with authorized OASIS data

```text
python preprocessing/merge.py
python preprocessing/feature_mapping.py
python preprocessing/diagnosis_mapping.py
python preprocessing/preprocess_common.py
python preprocessing/preprocess_paper_faithful.py
python evaluation/paper_faithful.py
python evaluation/leakage_free.py
python evaluation/grouped_cv.py
python explainability/shap_analysis.py
python evaluation/mri_tolerance_sensitivity.py
python evaluation/comparison_table.py
pytest -q tests/test_core_smoke.py
```

Raw OASIS data are intentionally excluded from the repository/package and must be supplied through an authorized local data directory.

## Contribution 1 — CDR-based severity prediction

NeuroFEDX now includes an additional, non-diagnostic CDR-based severity task alongside the existing five-class `diagnosis_class` task.

Severity target (`severity_label`):
- Category 0: `CDRTOT == 0`
- Category 1: `CDRTOT == 0.5`
- Category 2: `CDRTOT >= 1`

`CDRTOT` is used only to create the target. The CDR-derived variables `CDRTOT`, `CDRSUM`, `memory`, `orient`, `judgment`, `commun`, `homehobb`, and `perscare` are excluded from the severity feature matrix to prevent target leakage.

The scientifically valid severity track uses participant-level splitting, training-only KNN imputation, training-only Pearson selection, training-only SMOTE, untouched test data, grouped five-fold CV, and the existing RF/tree-pooling framework. Severity SHAP uses the project's real `shap.TreeExplainer` implementation.

See `CONTRIBUTION1_SEVERITY_REPORT.md` and `results/severity_*` for the implementation and results.

## Contribution 2 — Personalized Federated Learning

Contribution 2 extends the existing NeuroFEDX federated RF system with client-specific local adaptation. It does not replace the 100-tree global forest. For each client, the personalized model is the union of the existing 100 global trees and 10 additional trees trained only on that client's private personalization-training data.

The experiment uses a fixed 10-client Dirichlet non-IID allocation (`alpha=1.0`, seed 42), followed by participant-intact 60% federation / 20% personalization / 20% held-out client-test splits. Training-side KNN imputation, Pearson selection, and SMOTE remain separated from the held-out client tests.

See `CONTRIBUTION2_README.md` and `results/contribution2/` for methodology, audits, metrics, personalization gains, and TreeSHAP results.

## Contribution 3 — Multimodal Patient-Level Explainability

Contribution 3 extends the existing TreeSHAP layer without introducing a new prediction model. It explains the existing Contribution-2 global and personalized diagnosis/severity models at feature and modality levels using the project's existing feature mapping. Outputs are under `results/contribution3/`; methodology and reproducibility details are in `CONTRIBUTION3_README.md` and `CONTRIBUTION3_IMPLEMENTATION_REPORT.md`.

## Contribution 4 — Computational, Communication, and Latency Evaluation

Contribution 4 is an evaluation-only layer over the existing Base Paper + C1 + C2 + C3 system. It measures centralized RF, the existing 10-client/50-round round-wise RF tree-pooling global model, C2 personalization, serialized communication payloads, memory/RSS, inference latency, round/client costs, and the existing C3 TreeSHAP latency. It does not change model behavior or previous contribution results.

See `CONTRIBUTION4_README.md`, `CONTRIBUTION4_IMPLEMENTATION_REPORT.md`, and `results/contribution4/`.
