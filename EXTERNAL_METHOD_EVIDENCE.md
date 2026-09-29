# External Methodology Evidence

## Base paper
S. Jahan et al., “Federated Explainable AI-Based Alzheimer’s Disease
Prediction With Multimodal Data,” IEEE Access 13, 43435–43454 (2025).
DOI: 10.1109/ACCESS.2025.3547343.

The paper states:
- 10 FL clients.
- 50 communication epochs/rounds.
- RF as the global model.
- 100 trees, Gini splitting, max depth 30, sqrt feature selection,
  bootstrap samples, random state 42.
- FedAvg terminology and algorithms that describe exchanging/averaging
  model parameters/weights.
- SHAP explainability for the RF global model.
- 39 multimodal candidate features, with five MRI features removed by
  Pearson correlation at 0.95.
- KNN imputation with k=2 and SMOTE balancing.

## RF-FL ambiguity
The paper does not define a numeric RF parameter vector, tree-weight averaging
equation, tree serialization/selection rule, or how a received global RF is
updated locally. Therefore this project implements iterative federated tree
pooling as a technical interpretation, keeps the global forest at exactly
100 trees, and never labels the implementation as literal FedAvg.

## Missing UDS features
NACC/UDS documentation identifies:
- MEMTIME as time elapsed between Logical Memory immediate and delayed tests.
- TRAILARR/TRAILALI/TRAILBRR/TRAILBLI as Trail Making A/B error and correct-line
  sub-items.

NACC UDS v2/v3 literature also documents the version transition from DIGIF/
DIGIB to DIGFORCT/DIGBACCT and DIGIFLEN/DIGIBLEN to DIGFORSL/DIGBACLS.
Knight ADRC/ADSP-HC documentation identifies `tma` and `tmb` as Trail Making
A/B in seconds. These mappings are therefore documented version/center
interpretations, not exact literal variable matches.

## MRI tolerance
The paper does not specify a numeric MRI-to-clinical visit day tolerance.
The project's 365-day value is an engineering default. Sensitivity results
for 180/365/540/730/1095 days are retained. No value is described as the
paper's actual matching rule.
