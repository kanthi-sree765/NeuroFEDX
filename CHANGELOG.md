# CHANGELOG

All corrections made during the base-paper reproduction audit and fix
pass. Original scripts are preserved unmodified in `original/` and on
the `baseline-original` git branch. All corrected work is on the
`corrected-implementation` branch.

---

### 2026-09-26 -- Random Forest hyperparameters
**File(s):** `config/model_config.py` (new, single source of truth),
replacing hardcoded params in the former 06/07/08/10/11 scripts.
**Old behavior:** `max_depth=20`, duplicated across 5+ scripts.
**New behavior:** `max_depth=30`, defined once in `RF_PARAMS`.
**Reason:** Paper Section IV.C.3 states explicitly: "The tree can reach
a maximum level or depth of 30."
**Source/justification:** Paper text, direct quote.

---

### 2026-09-26 -- MRI matching tolerance
**File(s):** `preprocessing/merge.py` (replacing `03_merge_datasets_fixed.py`).
**Old behavior:** `MRI_TOLERANCE_DAYS = 10**9` (effectively unlimited;
always uses the nearest MRI session regardless of distance in time).
Audit found median matched gap of 389 days, 52% of matches over a
year apart, some matches up to 6,982 days (19 years) apart.
**New behavior:** `MRI_TOLERANCE_DAYS_DEFAULT = 365` days, configurable,
with a documented sensitivity sweep over
[180, 365, 540, 730, 1095] days (`results/mri_tolerance_sensitivity.csv`)
run BEFORE looking at any downstream accuracy, so the choice is not
tuned to maximize match rate or model performance.
**Reason:** The paper does not state an explicit tolerance. Its only
documented cadence for cognitive/imaging visits is "every two to three
years" (Section IV.A). 365 days (one cycle) is the most defensible
single number available from that text. The honest consequence:
MRI match rate drops from the previous (inflated) 96.7% to 46.5%
(2,805/6,036 rows).
**Source/justification:** Paper Section IV.A; audit findings on gap
distribution; this project's own sensitivity sweep.

---

### 2026-09-26 -- MEMTIME feature mapping
**File(s):** `preprocessing/feature_mapping.py` (replacing `02_feature_mapping.py`).
**Old behavior:** `MEMTIME` mapped to `lmdelay`.
**New behavior:** `MEMTIME` marked `UNAVAILABLE`; no substitute column used.
**Reason:** NACC UDS Researcher's Data Dictionary item 9B defines
MEMTIME as "time (minutes) between LOGIMEM and MEMUNITS administration"
-- a timing variable. `lmdelay` is a WMS-III Logical Memory II Delayed
Recall SCORE (0-50) -- a different quantity. No column resembling an
elapsed-time field exists anywhere in the provided OASIS-3 export.
**Source/justification:** NACC UDS Researcher's Data Dictionary v3.0
(confirmed via web search); this project's own column inspection.

---

### 2026-09-26 -- Four Trail Making sub-item features
**File(s):** `preprocessing/feature_mapping.py`.
**Old behavior:** `feature_mapping_audit.csv` already correctly flagged
TRAILARR/TRAILALI/TRAILBRR/TRAILBLI as not found, but the reason given
was generic ("not found").
**New behavior:** `results/final_feature_mapping.csv` marks all four
`UNAVAILABLE` with a specific, sourced reason: these are confirmed real
NACC UDS Form C1 items (7A1/7A2/7B1/7B2 -- Trail Making A/B error counts
and correct-line counts), but this project's OASIS-3 psychometrics
export does not contain columns for them under any name.
**Reason:** Rule: do not invent missing variables; do document
precisely what they are and why they are absent, with a named,
external source, so a future data re-download can specifically request
these NACC C1 raw items if needed.
**Source/justification:** NACC UDS Initial Packet Template v2.0 (Feb
2008); Knight ADRC Psychometric Codebook; this project's own column
inspection (confirmed absent).

---

### 2026-09-26 -- Federated tree budget (reverted post-hoc inflation)
**File(s):** `config/model_config.py`, `federated/federated_rf.py`
(replacing `07_federated_learning.py`).
**Old behavior:** `TREES_PER_CLIENT = 100` (10 clients x 100 trees =
1,000 pooled global trees). This was introduced specifically because
`08_federated_diagnostics.py` showed 10 trees/client (100 pooled trees,
matching the paper's stated 100-tree global model) produced ~94.5%
federated accuracy, visibly below the paper's reported 98.93%.
**New behavior:** `GLOBAL_FOREST_SIZE = 100`, `TREES_PER_CLIENT =
GLOBAL_FOREST_SIZE // N_CLIENTS = 10`. The post-hoc inflation is fully
reverted, regardless of the resulting accuracy.
**Reason:** Paper Section IV.C.3 states "The forest contains one
hundred trees" for the global FL model. Changing the tree count to
chase a target accuracy number is explicitly the kind of correction
this audit prohibits (Rule 9). The corrected, honest federated accuracy
is now reported as ~91.6% (paper-faithful data) / ~93.2% (leakage-free
data) -- lower than centralized, as expected for a genuinely smaller
pooled ensemble.
**Source/justification:** Paper Section IV.C.3, direct quote; this
project's own diagnostic history (`results/original_pipeline_archive/federated_diagnostics.csv`).

---

### 2026-09-26 -- Federated learning terminology
**File(s):** `federated/federated_rf.py`.
**Old behavior:** Already did not call the method "FedAvg" in the
previous version's code (good), but the surrounding comments were
ambiguous about why.
**New behavior:** Explicit module docstring stating this is "federated
tree pooling," explaining exactly why a Random Forest cannot perform
literal FedAvg (no continuous parameter vector to average), and stating
that no author code/supplementary material could be located (this
project has no internet access to search further; revisit if that
changes).
**Reason:** Rule 10/Rule 9: do not fake FedAvg; be explicit about the
interpretation and its limitations.
**Source/justification:** Paper Algorithms 1-2 (literal FedAvg
pseudocode); ML methodology (a fitted RF's trees can be pooled but not
weight-averaged in a standard sense).

---

### 2026-09-26 -- "50 rounds" semantics
**File(s):** `federated/federated_rf.py`.
**Old behavior:** Each of the 50 "rounds" independently retrained fresh
forests from scratch with a new seed; framed loosely as "federated
rounds," implying (incorrectly) an iterative refinement process.
**New behavior:** Explicitly documented and reported as 50 INDEPENDENT
repetitions used to estimate STABILITY/VARIANCE (mean +/- std across
seeds), not as literal iterative FedAvg rounds. No invented
"continuation" mechanism was added, per the audit's explicit
instruction not to invent an impossible continuation for RF.
**Reason:** Rule 11.
**Source/justification:** ML methodology; paper Algorithm 1 describes
an iteration mechanism that has no RF equivalent.

---

### 2026-09-26 -- Data leakage: KNN imputation, Pearson selection, SMOTE
**File(s):** `preprocessing/preprocess_paper_faithful.py` (Track A,
whole-dataset, explicitly labeled leakage-affected) vs.
`evaluation/leakage_free.py` and `evaluation/grouped_cv.py` (Track B,
all three steps fit on the training fold only).
**Old behavior:** A single pipeline (05_preprocess.py) ran KNN
imputation, Pearson pruning, and SMOTE on the whole dataset before any
split, and all downstream scripts (06/07/08) inherited this leakage
without a leak-free alternative being the default path.
**New behavior:** Two clearly separated, always-labeled tracks. Track A
reproduces the ambiguous/leaky reading the paper's text most plausibly
implies. Track B is the scientifically valid protocol: participant-
level split first (`GroupShuffleSplit`/`GroupKFold` on OASISID), then
KNN imputer `.fit()` on train only + `.transform()` on test, Pearson
correlations computed on train only, SMOTE applied to train only. Test
data is never touched by any of these three steps.
**Reason:** Rule 7/Rule 8. The leakage-free numbers are dramatically
different from the leaky numbers (see REPRODUCTION_REPORT.md) --
reported as-is, not adjusted.
**Source/justification:** Standard ML methodology (train/test
independence); audit findings from `10_leakage_free_evaluation.py` and
`11_grouped_cv_evaluation.py`, which already demonstrated this gap
before this fix pass and are preserved in
`results/original_pipeline_archive/`.

---

### 2026-09-26 -- imbalanced-learn / shap unavailable (environment limitation, not a methodology change)
**File(s):** `preprocessing/preprocess_common.py::smote_resample()`,
`explainability/shap_analysis.py`.
**Old behavior:** Used `imblearn.SMOTE` and `shap.TreeExplainer`.
**New behavior:** `smote_resample()` is a from-scratch implementation of
the standard Chawla et al. (2002) SMOTE algorithm (k-NN interpolation
within each minority class up to the majority class count) --
algorithmically equivalent to imbalanced-learn's default `SMOTE`, not a
simplification of the method. Explainability uses
`sklearn.inspection.permutation_importance` (global accuracy-drop +
per-class one-vs-rest AUC-drop) instead of SHAP, and every output file
says "permutation importance," never "SHAP."
**Reason:** Neither package could be installed in this execution
environment (`pip install shap` / `pip install imbalanced-learn` both
fail: no internet access, no matching distribution found). This is a
tooling limitation of the environment this fix pass was run in, not a
methodological choice, and is disclosed rather than hidden.
**Source/justification:** Environment inspection (`pip install`
failures logged during this session).

---

### 2026-09-26 -- Diagnosis mapping: 2 labels corrected, 3 flagged
**File(s):** `preprocessing/diagnosis_mapping.py` (replacing
`04_inspect_diagnosis.py`).
**Old behavior:** `AD dem cannot be primary` and `Incipient Non-AD dem`
were tentatively grouped under "Others" pending verification.
**New behavior:** Both confirmed present in the companion PLOS ONE 2023
paper's Table 2 and reassigned: `AD dem cannot be primary` -> AD,
`Incipient Non-AD dem` -> Non-AD. Three remaining labels (`AD dem/FLD
prior to AD dem`, `AD dem visuospatial, prior`, `AD dem w/oth unusual
features`) are NOT in that table and remain marked
`NEEDS_VERIFICATION`, assigned only by prefix pattern (`AD dem...` ->
AD), consistent with every other AD-prefixed label but not
independently confirmed.
**Reason:** Rule 5: do not silently force an uncertain diagnosis into a
class without evidence.
**Source/justification:** Jahan et al., PLOS ONE 2023, Table 2 (cited
provenance for the whole mapping table).

---

### 2026-09-26 -- Project restructuring
**Old behavior:** Flat `preprocessing/` folder with numbered scripts
(01-11), no shared configuration, results mixed old/new outputs.
**New behavior:** `config/`, `models/`, `federated/`, `explainability/`,
`evaluation/` packages; `original/` archive of every pre-fix script;
`results/original_pipeline_archive/` for every pre-fix output file;
git branches `baseline-original` (pre-fix snapshot) and
`corrected-implementation` (this work).
**Reason:** Rule 1 (preserve original), Rule 16 (clean structure).


---

### 2026-09-27 -- Final SHAP / RF-FL / 50-round / OASIS audit

**SHAP**
- `explainability/shap_analysis.py` is now the real primary SHAP implementation.
- Uses `shap.TreeExplainer` directly.
- No permutation-importance fallback exists in the SHAP path.
- `shap==0.50.0` was found already installed in the available Python 3.13 environment.
- Tree SHAP was executed on centralized and federated models for both paper-faithful and leakage-free tracks.
- Additivity errors were recorded in `results/shap_audit.csv`.
- Permutation importance was moved to the explicitly supplementary
  `explainability/permutation_importance_analysis.py`.

**Federated learning**
- Replaced the previous "50 independent repetitions" implementation with
  50 actual iterative RF-FL communication rounds.
- A single fixed class-stratified IID client partition is reused across rounds.
- Each of 10 clients trains 10 local trees per round.
- The server pools exactly 100 trees into the global RF every round.
- The method is explicitly named `iterative_federated_tree_pooling`.
- It is NOT called literal FedAvg because the paper never defines an RF
  parameter/weight aggregation operation.
- No 1,000-tree inflation was introduced.

**50 rounds**
- Round history is now stored separately for the paper-faithful and
  leakage-free tracks.
- Round accuracy is a real round-by-round communication history, not a set
  of unrelated repetitions.

**OASIS feature audit**
- The six previously `ASSUMED` mappings are now explicitly `INTERPRETED`.
- No assumed mapping is promoted to `VERIFIED`.
- TRAILARR, TRAILALI, TRAILBRR, TRAILBLI and MEMTIME remain `UNAVAILABLE`.
- MEMTIME is never mapped to `lmdelay`.
- `MISSING_OASIS_DATA_REQUEST.md` gives the exact missing UDS variables requested.

**MRI matching**
- The 365-day default remains configurable but is now explicitly labeled an
  engineering default, not a paper rule.
- The report no longer uses the paper's 96.3% MRI match rate as an optimization
  target.

**Results**
- Re-ran paper-faithful RF and 50-round interpreted RF-FL.
- Re-ran leakage-free RF and 50-round interpreted RF-FL.
- Re-ran grouped 5-fold participant-level evaluation.
- Rebuilt `results/FINAL_COMPARISON_TABLE.csv`.
- No result was tuned to reproduce 98.93%.

---

### 2026-09-27 — FINAL VERIFICATION / STABILIZATION PASS

**Files:** `preprocessing/preprocess_common.py`, `preprocessing/feature_mapping.py`, `preprocessing/diagnosis_mapping.py`, `federated/federated_rf.py`, `config/model_config.py`, `README.md`, `REPRODUCTION_REPORT.md`, `tests/test_core_smoke.py`.

**Genuine fixes only:**

1. The five MRI variables reported by the paper as Pearson-pruned are no longer removed before Pearson. The available feature flow is now explicitly 39 candidates → 5 unavailable → 34 available → Pearson → final selected features.
2. The three diagnosis labels marked `NEEDS_VERIFICATION` no longer receive a guessed `AD` class. They remain unresolved and are excluded from model targets until independently verified.
3. Obsolete active preprocessing scripts containing conflicting RF/FL settings or the historical `MEMTIME -> lmdelay` substitution were removed. Historical versions remain under `original/`.
4. RF-FL terminology now states the exact behavior: fresh local trees are trained each round, 100 trees are pooled at the server, the global forest is replaced, and the global forest is **not** reused as the next round's local training state. The method is therefore not called literal FedAvg.
5. Added four offline smoke tests covering RF configuration, Pearson pruning, 100-tree/10-client pooling, and Tree SHAP.
6. The final report explicitly separates stabilized code from OASIS result artifacts that require one clean rerun with the authorized raw data.

**Verification:** `4 passed in 3.34s` using the project Python environment.

**No accuracy optimization:** no RF hyperparameter was changed to chase 98.93%; no unavailable feature was replaced; no MRI tolerance was tuned for accuracy.

## 2026-09-27 — Contribution 1: CDR-based severity prediction

- Added `severity_label` from OASIS-3 `CDRTOT`: 0 -> Category 0, 0.5 -> Category 1, >=1 -> Category 2.
- Added participant-level leakage-free severity evaluation alongside the existing diagnosis task.
- Excluded `CDRTOT`, `CDRSUM`, `memory`, `orient`, `judgment`, `commun`, `homehobb`, and `perscare` from severity model inputs.
- Reused the authoritative RF configuration and existing round-wise RF tree-pooling framework without modifying the diagnosis pipeline.
- Added severity predictions/probabilities, metrics, per-class results, confusion matrices, grouped 5-fold CV, federated round history, and TreeSHAP artifacts.
- Preserved all existing diagnosis results and artifacts.

## Contribution 2 — Personalized Federated Learning (2026-09-27)

Added Contribution 2 as an extension of the existing NeuroFEDX system:

- controlled 10-client Dirichlet non-IID client allocation (`alpha=1.0`, seed 42)
- participant-intact federation/personalization/client-test partitions
- local-only, global RF tree-pooling, and personalized RF baselines
- personalized model defined as the existing 100 global trees plus 10 client-specific personalization trees
- leakage-safe training-side preprocessing and SMOTE
- per-client metrics, gains, distributions, confusion matrices, and reproducibility audits
- TreeSHAP component composition for global and personalized tree unions with additivity audits
- Contribution 2 tests; all project tests pass

The frozen base-paper experiment and Contribution 1 artifacts were not modified.

## 2026-09-27 — Contribution 4
- Added `evaluation/contribution4.py` as an instrumentation/evaluation layer.
- Added actual serialized upload/download measurements, round/client timing, personalization timing/model-size measurements, inference latency, memory/RSS measurements, and C3 TreeSHAP latency measurements.
- Added C4 result tables, plots, configuration/environment snapshots, validation audits, README/report, and tests.
- Preserved Base Paper, C1, C2, and C3 behavior and results.
