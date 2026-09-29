# NeuroFEDX — Final Base-Paper Verification & Stabilization Audit

**Base paper:** S. Jahan et al., *Federated Explainable AI-Based Alzheimer’s Disease Prediction With Multimodal Data*, IEEE Access 13 (2025), 43435–43454.  
**DOI:** 10.1109/ACCESS.2025.3547343

## 1. BASE PAPER IMPLEMENTATION STATUS

### Complete components

The authoritative implementation now has:

- OASIS-3 multimodal data-level fusion: clinical + psychological/cognitive + FreeSurfer-derived MRI features.
- 39 documented candidate features.
- 5 unavailable features kept unavailable: `TRAILARR`, `TRAILALI`, `TRAILBRR`, `TRAILBLI`, `MEMTIME`.
- 28 exact/verified mappings + 6 explicitly **INTERPRETED** mappings + 5 unavailable mappings.
- KNN imputation with `k=2`.
- Pearson feature selection with threshold `0.95`.
- SMOTE balancing after the Pearson stage.
- Random Forest with the paper's stated 100-tree configuration.
- 10 federated clients.
- A 100-tree RF-specific federated tree-pooling interpretation.
- 50 round executions with a fixed client partition.
- Real `shap.TreeExplainer`; permutation importance is supplementary only.
- Participant-level leakage-free evaluation.
- Participant-grouped 5-fold CV.
- MRI matching tolerance kept configurable and explicitly identified as an engineering assumption.

### Remaining limitations

The project cannot honestly claim an exact reproduction of the paper because:

1. Five paper variables are absent from the supplied OASIS export.
2. The paper calls its RF procedure FedAvg but does not define an RF parameter/tree aggregation equation sufficient to reproduce literal numerical FedAvg.
3. Three diagnosis labels remain unresolved in the available mapping evidence and are no longer assigned a model class.
4. The paper does not specify a numeric MRI-to-visit matching tolerance.
5. The final ZIP excludes the raw OASIS dataset, so the newly corrected preprocessing/diagnosis logic could not be rerun end-to-end in this audit environment.

The last point is a **rerun requirement**, not a reason to invent or substitute data.

### A–D classification

| Component | Classification | Determination |
|---|---|---|
| RF hyperparameters | **A — EXACTLY REPRODUCED FROM PAPER** | 100 trees; Gini; depth 30; split 2; leaf 1; sqrt features; bootstrap; random state 42. |
| KNN `k=2` | **A — EXACTLY REPRODUCED FROM PAPER** | Implemented centrally in the authoritative configuration. |
| Pearson threshold 0.95 | **A — EXACTLY REPRODUCED FROM PAPER** | Actual correlation-based pruning; threshold is not tuned to results. |
| SHAP / TreeExplainer | **A — EXACTLY REPRODUCED FROM PAPER** | Actual `shap.TreeExplainer` is used on the RF. |
| SMOTE | **B — REPRODUCED WITH DOCUMENTED ASSUMPTION** | Standard SMOTE behavior is implemented locally; exact package/version behavior is not claimed. |
| Client count = 10 | **A — EXACTLY REPRODUCED FROM PAPER** | Ten clients are configured. |
| Global forest = 100 trees | **A — EXACTLY REPRODUCED FROM PAPER** | Every RF-FL round contains exactly 100 global trees. |
| Client partition | **B — REPRODUCED WITH DOCUMENTED ASSUMPTION** | The paper does not specify the actual client partition; this project uses one fixed class-stratified IID partition. |
| RF aggregation | **C — TECHNICALLY INTERPRETED BECAUSE PAPER IS AMBIGUOUS** | RF-specific tree pooling is used; no invented numeric weight averaging. |
| 50 rounds | **C — TECHNICALLY INTERPRETED BECAUSE PAPER IS AMBIGUOUS** | 50 round-wise tree-pooling executions are performed, but the global RF is not reused as the local training state because the paper gives no RF update rule. |
| Six feature mappings | **C — TECHNICALLY INTERPRETED BECAUSE PAPER IS AMBIGUOUS** | `DIGIF→DIGFORCT`, `DIGIFLEN→DIGFORSL`, `DIGIB→DIGBACCT`, `DIGIBLEN→DIGBACLS`, `TRAILA→tma`, `TRAILB→tmb`. |
| Five unavailable features | **D — IMPOSSIBLE TO REPRODUCE FROM AVAILABLE INFORMATION** | They are absent from the supplied OASIS export and are never substituted. |
| Three unresolved diagnosis labels | **D — IMPOSSIBLE TO REPRODUCE FROM AVAILABLE INFORMATION** | Their class assignment cannot be proven from the available mapping source, so they are excluded rather than guessed. |
| MRI matching rule | **C — TECHNICALLY INTERPRETED BECAUSE PAPER IS AMBIGUOUS** | 365 days is an engineering default, not claimed as a paper rule. |

---

## 2. WHAT YOU FIXED

Only genuine inconsistencies found during the final audit were changed:

1. **Feature-stage ordering:** the five MRI features that the paper reports as Pearson-pruned are now retained in the pre-Pearson candidate set. They are no longer silently removed before Pearson selection. The intended sequence is now **39 → 5 unavailable → 34 available → Pearson → final selected features**.
2. **Diagnosis mapping:** the three `NEEDS_VERIFICATION` labels no longer receive an invented `AD` class. Their `Paper_Class` is blank and the authoritative preprocessing drops unresolved diagnosis rows.
3. **Obsolete active scripts:** conflicting numbered scripts containing old `max_depth=20`, 100 trees/client, the old federated behavior, and the forbidden `MEMTIME→lmdelay` mapping were removed from the active `preprocessing/` package. Their historical copies remain under `original/`.
4. **Federated terminology:** the implementation is now described as **round-wise federated tree pooling with no global-model reuse**, not as literal FedAvg and not as iterative parameter averaging.
5. **Smoke tests:** an offline test suite was added for the authoritative RF configuration, Pearson pruning, 100-tree/10-client pooling, and Tree SHAP.
6. **Documentation:** the README, changelog, and this report were aligned with the actual code.

No RF hyperparameter was changed to improve accuracy, no missing feature was substituted, and no result was optimized toward 98.93%.

---

## 3. WHAT YOU VERIFIED

### Dataset

The project uses OASIS-3 clinical, psychological/cognitive, and FreeSurfer-derived MRI data. The supplied final ZIP does **not** contain the raw OASIS dataset, so the final corrected preprocessing code could not be executed against raw data in this audit environment.

Existing audited artifacts show 6,036 usable rows and 1,288 subjects in the previously generated model dataset. These existing results are retained as historical/reference outputs and are not silently relabeled as results of the newly corrected preprocessing logic.

### Features

The authoritative feature mapping contains exactly **39 candidates**:

- **28 VERIFIED**
- **6 INTERPRETED**
- **5 UNAVAILABLE**

The five unavailable features are exactly:

- `TRAILARR`
- `TRAILALI`
- `TRAILBRR`
- `TRAILBLI`
- `MEMTIME`

`MEMTIME` is never mapped to `lmdelay`.

The six interpreted mappings remain explicitly interpreted:

- `DIGIF → DIGFORCT`
- `DIGIFLEN → DIGFORSL`
- `DIGIB → DIGBACCT`
- `DIGIBLEN → DIGBACLS`
- `TRAILA → tma`
- `TRAILB → tmb`

The corrected code now keeps all 34 available features through the pre-Pearson stage. The five MRI variables that the paper says are eliminated at Pearson threshold 0.95 are therefore actually available to the Pearson procedure rather than being removed beforehand.

### Diagnosis

The mapping contains 38 known labels plus three unresolved labels. The three unresolved labels are:

- `AD dem/FLD prior to AD dem`
- `AD dem visuospatial, prior`
- `AD dem w/oth unusual features`

They are now **NEEDS_VERIFICATION with no assigned model class**. The project does not invent their interpretation.

### MRI

The authoritative matcher uses nearest-session matching within a configurable tolerance. The default is:

`MRI_TOLERANCE_DAYS_DEFAULT = 365`

This is explicitly an **engineering assumption**. The paper does not establish 365 days as its matching rule.

Existing sensitivity results for the prior audited data are:

| Tolerance | Rows matched | Match rate | Median matched gap |
|---:|---:|---:|---:|
| 180 days | 1,741 | 28.84% | 79 days |
| 365 days | 2,805 | 46.47% | 128 days |
| 540 days | 3,423 | 56.71% | 174 days |
| 730 days | 3,973 | 65.82% | 223 days |
| 1095 days | 4,566 | 75.65% | 275 days |

These values are sensitivity results, not evidence that the paper used 365 days.

### KNN

`KNNImputer(n_neighbors=2)` is the authoritative setting.

- Paper-faithful track: whole-dataset preprocessing is retained as the documented leakage-affected interpretation.
- Leakage-free track: imputer is fitted on training data only and transformed onto test data.

### Pearson

The authoritative implementation performs actual pairwise Pearson-correlation pruning at `|r| > 0.95`.

- Paper-faithful track: selection is calculated from the full preprocessing dataset.
- Leakage-free track: selection is calculated from training data only.
- The threshold is not adjusted to reproduce a target accuracy.

The corrected pipeline now exposes all 34 available candidate features to this stage.

### SMOTE

SMOTE is applied after Pearson selection.

- Paper-faithful track: full-data SMOTE is retained as the documented paper-style/leakage-affected interpretation.
- Leakage-free track: SMOTE is applied only to the training data.
- The test set is never SMOTE-resampled.

The local SMOTE implementation is treated as a documented implementation assumption, not as proof of the authors' exact library/version.

### RF

The authoritative `RF_PARAMS` are:

```text
n_estimators=100
criterion="gini"
max_depth=30
min_samples_split=2
min_samples_leaf=1
max_features="sqrt"
bootstrap=True
random_state=42
```

A code scan found no remaining active `max_depth=20` implementation.

### Federated learning

The paper explicitly uses the terminology FedAvg, 10 clients, and 50 epochs/rounds, but it does not define a numerical RF parameter vector or a tree/node aggregation equation. The published methodology describes broadcasting a global RF, local RF training, weighted aggregation, and redistribution, but does not specify how a fitted Random Forest's discrete trees are averaged.

The project therefore uses and documents:

- 10 fixed clients.
- One fixed class-stratified IID partition.
- 10 local trees/client/round.
- 100 global trees.
- Fresh local tree fitting each round.
- Server-side pooling of those 100 trees.
- No raw data exchange.
- Global forest replaced after each round.
- **Global forest is not used as the local training initialization in the next round.**

That last point is deliberate: there is no defensible RF continuation/update operation specified by the paper. Therefore the project does not call this literal FedAvg.

### 50 rounds

The code executes the configured 50 round loop and records every round in separate histories for the two evaluation tracks.

A round means:

`fixed client partition → fresh local RF trees → server pools 100 trees → global test evaluation`

The next round repeats that procedure with new local-tree seeds. The previous global trees are not locally retrained or averaged into new trees.

### SHAP

The primary explainability implementation is real `shap.TreeExplainer`.

Existing successful audits recorded:

| Track / model | Samples | Features | Trees | Max additivity error |
|---|---:|---:|---:|---:|
| Paper-faithful / centralized | 100 | 29 | 100 | 3.862e-13 |
| Paper-faithful / federated | 100 | 29 | 100 | 2.665e-15 |
| Leakage-free / centralized | 92 | 29 | 100 | 5.529e-14 |
| Leakage-free / federated | 92 | 29 | 100 | 4.297e-14 |

SHAP package version in that successful run: `0.50.0`.

Permutation importance remains supplementary and is never presented as SHAP.

### Leakage-free evaluation

The participant-level split audit shows zero subject overlap in the existing audited result set:

- Train subjects: 1,030
- Test subjects: 258
- Overlap: 0

Grouped 5-fold subject-overlap audits also show zero overlap in every fold.

The leakage-free track fits KNN and Pearson only on training data, applies SMOTE only to training data, and evaluates the untouched test set.

### Offline verification tests

The new smoke suite passes:

`4 passed in 3.34s`

It verifies:

- exact RF hyperparameters;
- Pearson pruning is genuinely computed;
- 100-tree / 10-client RF pooling works;
- multiclass Tree SHAP executes with additivity checking.

This is a code-level verification only; it is not a replacement for the missing OASIS end-to-end rerun.

---

## 4. FINAL RESULTS

The existing OASIS result artifacts from the previous full run are shown transparently below. **They must not be described as the final results of the newly corrected 34-feature preprocessing code until the authorized OASIS data are rerun.**

| Experiment | Accuracy | Balanced Accuracy | Macro-F1 | Precision | Recall | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Paper reported — FL | 98.93% | — | 98.93% | 98.94% | 98.93% | 99.97% |
| Existing paper-faithful centralized run | 97.83% | 97.83% | 97.82% | 97.84% | 97.83% | 99.88% |
| Existing paper-faithful RF-FL final round | 91.08% | 91.08% | 91.04% | 91.09% | 91.08% | 99.01% |
| Existing leakage-free centralized run | 94.08% | 54.22% | 52.25% | 50.58% | 54.22% | 92.91% |
| Existing leakage-free RF-FL final round | 93.34% | 58.70% | 58.02% | 58.59% | 58.70% | 93.32% |
| Existing grouped 5-fold CV | 94.35% ± 1.18 | 53.70% ± 1.08 | 52.30% ± 1.24 | 51.97% ± 2.32 | 53.70% ± 1.08 | 93.52% ± 1.73 |

These values were not tuned toward the paper's 98.93%.

A clean final OASIS run is still required after the feature-stage and unresolved-diagnosis corrections. The project intentionally does not fabricate that run without the raw data.

---

## 5. REMAINING LIMITATIONS

Only genuine limitations remain:

1. **Five unavailable OASIS features:** `TRAILARR`, `TRAILALI`, `TRAILBRR`, `TRAILBLI`, `MEMTIME`. They remain unavailable and are not replaced.
2. **Three unresolved diagnosis labels:** their class assignment is not supported by the available mapping evidence and they are therefore excluded rather than guessed.
3. **RF-FedAvg ambiguity:** the paper specifies FedAvg terminology but not a mathematically reproducible RF aggregation mechanism.
4. **50-round RF state update ambiguity:** the paper does not define how a fitted global RF is converted into the starting state of the next local RF training round.
5. **MRI matching ambiguity:** no numeric MRI-to-visit tolerance is specified by the paper.
6. **OASIS snapshot mismatch:** the supplied historical result artifacts come from a newer/larger OASIS export than the paper's reported snapshot; exact sample counts therefore cannot be expected to match automatically.
7. **Final corrected OASIS rerun:** raw OASIS data are intentionally absent from the project ZIP, so the newly corrected feature-stage and unresolved-diagnosis logic has not been rerun against raw OASIS in this environment.

No limitation is addressed by substituting one of the five unavailable variables.

---

## 6. FINAL VERDICT

**CODE IMPLEMENTATION: STABILIZED.**

The authoritative code now reflects the requested scientific constraints and the offline smoke tests pass. The five unavailable variables remain unavailable, the RF configuration is centralized, real Tree SHAP is used, stale conflicting scripts have been removed from the active pipeline, and the RF-FL interpretation is explicitly separated from literal FedAvg.

**BASE-PAPER IMPLEMENTATION: NOT YET READY TO FREEZE AS A FINAL RESULT PACKAGE.**

The remaining required action is a **single clean end-to-end OASIS run using the authorized raw dataset** so that the corrected 34-feature → Pearson pipeline and the unresolved-diagnosis exclusion regenerate every final result and audit from the same code/data/configuration. No result will be fabricated in place of that run.

### Reproduction commands after the authorized OASIS data are restored

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

The raw OASIS dataset must remain outside Git and outside the final project ZIP.

---

# CONTRIBUTION 1 — CDR-BASED SEVERITY PREDICTION (2026-09-27)

This section documents the first novel contribution added after the base-paper reproduction was stabilized. It does not replace or modify the existing five-class diagnosis task.

## Target

`severity_label` is created from OASIS-3 `CDRTOT`:

- `CDRTOT == 0` -> Category 0
- `CDRTOT == 0.5` -> Category 1
- `CDRTOT >= 1` -> Category 2

Observed counts in the actual merged dataset: 4680 / 1016 / 340 respectively (6036 rows, 1288 subjects).

## Leakage prevention

The severity feature matrix excludes `CDRTOT`, `CDRSUM`, `memory`, `orient`, `judgment`, `commun`, `homehobb`, and `perscare`. These are target-defining or directly target-derived CDR variables. The severity model therefore cannot read the CDR score used to construct its target.

The scientifically valid severity track performs participant-level splitting first, then training-only KNN imputation (`k=2`), training-only Pearson selection (`|r| > 0.95`), and training-only SMOTE. The test set is never SMOTEd and is evaluated untouched.

## Severity model

The existing RF configuration is reused: 100 trees, Gini, `max_depth=30`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features='sqrt'`, bootstrap, random state 42, no class weighting.

Twenty-seven severity-eligible multimodal features enter preprocessing; five are removed by training-only Pearson selection, leaving 22 model features.

## Holdout results

Centralized severity RF: accuracy 81.02%, balanced accuracy 67.06%, macro-F1 64.42%, weighted F1 80.83%, macro OVR AUC 86.95%.

Existing RF tree-pooling severity extension, final round: accuracy 77.90%, balanced accuracy 68.37%, macro-F1 63.54%, weighted F1 79.00%, macro OVR AUC 86.23%.

Train/test participants: 1030 / 258; participant overlap = 0.

## Grouped 5-fold CV

Accuracy 79.72% ± 1.73%; balanced accuracy 66.77% ± 1.37%; macro-F1 64.43% ± 0.75%; macro OVR AUC 84.96% ± 2.60%. Every fold has zero participant overlap.

## SHAP

Real `shap.TreeExplainer` was used. Maximum additivity error was `1.082e-13` for the centralized severity model and `1.277e-15` for the federated severity model.

## Federated terminology

The severity extension uses the existing round-wise federated Random Forest tree-pooling implementation: 10 clients, 50 rounds, 10 local trees/client, and a 100-tree global forest. It is not described as literal parameter-vector FedAvg.

## Research-integrity status

No metric, split, random seed, class weighting, SMOTE parameter, RF hyperparameter, or target definition was tuned to reach a target accuracy. The severity task is an experimental research contribution and is not claimed to be clinically validated.
