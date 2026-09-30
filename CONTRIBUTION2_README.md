# Contribution 2 — Personalized Federated Learning using Client-Specific Local Adaptation

Contribution 2 extends the existing NeuroFEDX RF tree-pooling federation without replacing the frozen base-paper experiment or Contribution 1.

## Method

For each task, the leakage-safe participant training population is partitioned into 10 clients with a fixed Dirichlet class allocation (`alpha=1.0`, seed 42). Participants remain intact within one client. Each client's assigned training participants are split into:

- 60% federation-training data
- 20% personalization-training data
- 20% local held-out test data

The local held-out test data are never used for fitting or tuning.

The existing RF configuration is preserved. The global model is produced by the same 50-round, 10-client, 100-tree round-wise RF tree-pooling mechanism. For non-IID clients, the implementation explicitly aligns tree class outputs because individual sklearn trees can use local class indices when a rare class is absent from a client.

The personalized model is the literal tree union:

`Personalized_i = Global_100_trees ∪ Local_i_10_trees`

No averaging of separate global/local probabilities is used as the personalization method.

## Tasks

The experiment is run first for the original five-class `diagnosis_class` target and then for the existing three-category `severity_label` target.

## Leakage controls

- participant-level outer train/test separation is preserved
- participants are not split across client federation/personalization/test partitions
- KNN imputation is fitted only on federation + personalization training rows
- Pearson selection is fitted only on federation + personalization training rows
- SMOTE is applied separately to training-side federation and personalization data only
- client test rows are untouched until evaluation

## Evaluation

For each client and each model (local-only, global federated, personalized federated), accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, AUC when valid, confusion matrices, and per-class metrics are saved. Personalization gains are computed as personalized minus global metrics per client.

## SHAP

Actual `shap.TreeExplainer` is retained. Because non-IID RF clients may lack rare classes, Contribution 2's global/personalized tree unions use a class-aligned component representation. TreeSHAP values are combined according to the exact tree counts in the union, and additivity is explicitly audited.

## Limitations

The experiment is a controlled methodology study on the available OASIS-derived project table. It does not establish privacy guarantees, clinical validity, or superiority of personalization. Results are reported as observed rather than tuned to a target accuracy.
