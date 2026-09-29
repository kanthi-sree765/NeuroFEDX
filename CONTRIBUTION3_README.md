# Contribution 3 — Multimodal Patient-Level Explainability for Personalized Federated Alzheimer’s Prediction

Contribution 3 is a **post-hoc explainability extension**, not a new prediction model. It consumes the already trained Contribution-2 global federated RF and client-personalized RF models and preserves the existing TreeSHAP implementation.

## Method

For each selected client-test sample, the model's **predicted class** is explained with TreeSHAP. Feature-level SHAP values are aggregated using the existing `final_feature_mapping.csv` into Clinical / Demographic, Psychological / Cognitive, MRI, and explicitly UNMAPPED when applicable.

For modality `m`:

`I_m = Σ_j |SHAP_j|`

`C_m = I_m / Σ_k I_k`

Both absolute and signed sums are retained. Normalized values are model-attributed SHAP contribution proportions, not clinical or biological causality.

Top-K is configurable (`TOP_K_FEATURES_PER_MODALITY=5`). Global and personalized explanations are compared for the same client-test samples using predicted class/probability, feature SHAP differences, Spearman rank correlation when the explained classes match, top-5/top-10 Jaccard overlap, and modality contribution correlation.

## Feature mapping

The project's existing mapping is used without silently changing it. Features marked `INTERPRETED` remain documented as interpreted. Features absent from the mapping are assigned `UNMAPPED` rather than guessed. The five unavailable paper variables remain unavailable and are not substituted.

## Sample selection

Five client-test samples per client are selected with a fixed seed (`42 + client_id`). Selection is based only on the pre-existing held-out client-test partition and is not tuned for favorable explanations.

## Leakage controls

No model is fitted on client-test data. Explanation generation is post-hoc. Preprocessing for the explanation inputs reproduces the existing C2 training-side KNN imputation/Pearson selection procedure; test rows are transformed only after the training-side fit. No model hyperparameter tuning is performed.

## Outputs

`results/contribution3/diagnosis/` and `severity/` contain patient-level explanations, modality contributions, top features, global-vs-personalized feature comparisons, similarity statistics, client summaries, machine-readable per-sample JSON, and plots. `results/contribution3/summary/` contains configuration/audits and overall summaries.

## Validation

Every explained model records the maximum absolute TreeSHAP additivity error. Tests verify feature dimensions, mapping coverage, signed/absolute aggregation, normalization, predicted-class explanation, global/personalized separation, and existing test compatibility.

## Reproduce

From the project root with the environment active:

```text
python -m evaluation.contribution2
python -m explainability.contribution3
python -m pytest -q
```

The first command regenerates the Contribution-2 models/results consumed by C3; the second generates the Contribution-3 explanation outputs; the final command validates the integrated project.
