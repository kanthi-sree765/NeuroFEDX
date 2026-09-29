# Contribution 3 Implementation Report

## Scope

Contribution 3 was added to the existing NeuroFEDX project through Contribution 2. It is a post-hoc explainability extension and does not introduce a prediction model. Existing Base Paper, Contribution 1, and Contribution 2 implementations remain in place.

The uploaded Contribution 3 specification requires actual TreeSHAP, feature-to-modality aggregation, patient-level explanations, global-vs-personalized comparison, reproducibility, and validation without retraining on test data.

## Existing architecture inspected

- `preprocessing/feature_mapping.py`
- `results/final_feature_mapping.csv`
- `models/random_forest.py`
- `federated/federated_rf.py`
- `federated/personalized_rf.py`
- `evaluation/contribution2.py`
- `explainability/shap_analysis.py`
- `explainability/contribution2_shap.py`
- `results/contribution2/diagnosis/*`
- `results/contribution2/severity/*`
- `results/base_dataframe_unimputed.csv`
- `results/severity_labelled_dataset.csv`
- existing tests and project documentation

## Integration

New primary module:

`explainability/contribution3.py`

It consumes the already trained Contribution-2 global forest and reconstructs the already-defined client personalization forests from the recorded private personalization partitions. It does not fit a new predictive architecture.

## Configuration

- seed: `42`
- client-level C3 patient examples: clients `0` and `5`
- one reproducibly selected held-out client-test sample per selected client and task
- TOP_K_FEATURES_PER_MODALITY: `5`
- TreeSHAP: `shap.TreeExplainer`
- additivity tolerance: `1e-8`
- existing C2: 10 clients, 50 rounds, 100 global trees, 10 personalization trees

The client-level modality summaries use the existing C2 TreeSHAP aggregate, which already explains five held-out samples per client. The smaller C3 patient-example pass is intentionally separated from that all-client aggregate so patient-level explanations remain computationally tractable and reproducible without changing C2.

## Feature-to-modality mapping

Existing `final_feature_mapping.csv` is used without modification.

Diagnosis selected features: 29
- Clinical / Demographic: 12
- Psychological / Cognitive: 12
- MRI: 5
- UNMAPPED: 0

Severity selected features: 22
- Clinical / Demographic: 5
- Psychological / Cognitive: 12
- MRI: 5
- UNMAPPED: 0

The six existing interpreted mappings remain interpreted; they are not silently upgraded. The five unavailable paper variables remain unavailable and are not substituted.

## SHAP methodology

For each explained sample, the predicted class is the class being explained. Feature-level values include feature value, signed SHAP value, absolute SHAP value, direction, and predicted probability.

For modality `m`:

`I_m = Σ_j |SHAP_j|`

`C_m = I_m / Σ_k I_k`

Signed modality contribution is also stored as:

`S_m = Σ_j SHAP_j`

These are model-attributed contributions, not causal, biological, or clinical importance claims.

Because the Contribution-2 global forest is a pool of client forests and rare classes may be absent from individual client forests, C3 uses the recorded `(tree, parent-class-space)` metadata and computes TreeSHAP on the real 10-tree RF components before exact class-aligned pooling. Personalized explanations add the client's actual personalization RF component.

## Validation

Maximum absolute SHAP additivity error:

- Diagnosis: `1.11e-16`
- Severity: `6.38e-16`

Both are far below the `1e-8` validation tolerance.

All tests:

`17 passed`

This includes the existing project tests, Contribution 1 tests, Contribution 2 tests, and Contribution 3 tests.

## Global vs personalized patient examples

Diagnosis patient examples showed the same predicted class for both global and personalized models in the two selected examples. Feature-ranking Spearman correlations were approximately `0.866` and `0.954`; top-5 Jaccard overlap was `1.00` and `0.667`; top-10 overlap was `0.818` and `1.00`.

Severity patient examples also retained the same predicted category. Spearman correlations were approximately `0.965` and `0.898`; top-5 overlap was `1.00` for both examples; top-10 overlap was `0.818` and `1.00`.

These are sample-level findings only and are not generalized claims about personalization.

## Modality findings

For the two selected diagnosis patient examples, personalized normalized SHAP contributions averaged approximately:

- Clinical / Demographic: `71.35%`
- Psychological / Cognitive: `22.19%`
- MRI: `6.46%`

For the two selected severity patient examples:

- Clinical / Demographic: `30.50%`
- Psychological / Cognitive: `62.47%`
- MRI: `7.03%`

These percentages describe the evaluated model explanations only.

The all-client personalized summaries derived from the existing C2 TreeSHAP aggregate show mean normalized contributions of approximately:

Diagnosis:
- Clinical / Demographic: `65.88%`
- Psychological / Cognitive: `19.05%`
- MRI: `15.07%`

Severity:
- Clinical / Demographic: `41.34%`
- Psychological / Cognitive: `45.46%`
- MRI: `13.20%`

These should be interpreted as SHAP-based model-attributed contribution distributions, not medical causation.

## Outputs

`results/contribution3/diagnosis/`
- `patient_level_explanations.csv`
- `modality_contributions.csv`
- `top_features.csv`
- `global_vs_personalized_explanations.csv`
- `explanation_similarity.csv`
- `client_modality_summary.csv`
- `patient_json/`
- `plots/`

`results/contribution3/severity/` contains the corresponding severity outputs.

`results/contribution3/summary/` contains:
- `contribution3_summary.csv`
- `modality_summary.csv`
- `explanation_similarity_summary.csv`
- `feature_modality_mapping_audit.csv`
- `shap_audit.csv`

`results/contribution3/config/contribution3_config.json` stores the reproducibility configuration.

## Limitations

1. Patient-level C3 examples are deliberately a small reproducible subset (one sample each from clients 0 and 5). The all-client modality summaries use the existing C2 TreeSHAP aggregate for five held-out samples per client.
2. A Spearman correlation is reported only when both explained models predict the same class. If predicted classes differ, direct same-class ranking correlation is not mathematically equivalent and is left unavailable.
3. SHAP is an explanation of the fitted model and does not establish causality, biological mechanism, or clinical importance.
4. The project's interpreted feature mappings remain interpreted and are not upgraded to verified status.
