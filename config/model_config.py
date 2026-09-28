"""
Central configuration for the NeuroFEDX reproduction pipeline.

Every script that trains a Random Forest, runs federated tree pooling,
or matches MRI sessions imports its parameters from here. Do not
hard-code these values anywhere else -- if a parameter needs to change,
change it once, here, and note why in CHANGELOG.md.
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================
# RANDOM FOREST HYPERPARAMETERS
# ============================================================
# Source: paper Section IV.C.3 ("RANDOM FOREST GLOBAL MODEL DESCRIPTION")
#   - "The forest contains one hundred trees."
#   - "Gini is used to evaluate a split's quality."
#   - "at least two samples are needed" [to split an internal node]
#   - "a single sample present at minimum for a leaf node to exist"
#   - "The square root function is used to determine how many
#      attributes to look at" (max_features='sqrt')
#   - "The tree can reach a maximum level or depth of 30"
#   - "the random state is maintained at 42"
#
# CORRECTED 2026-09-26: max_depth was 20 in the original project code.
# The paper explicitly states 30. This is the single authoritative
# value now; every script below imports RF_PARAMS from here.

RF_PARAMS = dict(
    n_estimators=100,
    criterion="gini",
    max_depth=30,          # CORRECTED from 20 -> matches paper's stated value
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    class_weight=None,     # paper: "All five classes are supposed to be
                            # allocated weight one" i.e. no reweighting
                            # in the paper-faithful track.
)

# ============================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================
# Source: paper Sections IV.C.1-3.
#   - "the number of clients is kept at 10"
#   - "The forest contains one hundred trees" (this is the GLOBAL
#     model's size, not per-client)
#   - "this process will continue for 50 epochs"
#
# The paper's Algorithm 1/2 describe literal FedAvg (weight averaging),
# which a Random Forest cannot natively perform -- there is no
# continuous parameter vector to average. This project therefore
# implements FEDERATED TREE POOLING as the documented, defensible
# interpretation, and does NOT call it FedAvg.
#
# CORRECTED 2026-09-26: the previous version of this project raised
# TREES_PER_CLIENT from 10 to 100 (yielding a 1,000-tree pooled global
# model) specifically because 10 trees/client produced a federated
# accuracy visibly below the paper's reported number. That change is
# reverted here. The global forest is restored to the paper's stated
# 100 trees, split evenly across the 10 clients, regardless of what
# accuracy this produces.

N_CLIENTS = 10
GLOBAL_FOREST_SIZE = 100                      # paper: "one hundred trees"
TREES_PER_CLIENT = GLOBAL_FOREST_SIZE // N_CLIENTS   # 10 trees/client, NOT 100
N_ROUNDS = 50
FEDERATION_METHOD = "roundwise_federated_tree_pooling_no_global_reuse"  # interpreted RF-FL; not literal FedAvg

# ============================================================
# MRI / VISIT MATCHING
# ============================================================
# The paper does not state an explicit day-tolerance for matching an
# MRI session to a clinical/psychological visit. Sections II/IV of the
# paper describe imaging and cognitive testing recurring "every two to
# three years," which is the only documented cadence to anchor a
# tolerance choice on.
#
# CORRECTED 2026-09-26: MRI_TOLERANCE_DAYS was previously 10**9
# (effectively unlimited), which the audit showed produces matches up
# to 19 years apart (median matched gap = 389 days; 52% of matches
# exceeded 1 year). A configurable, documented tolerance replaces it.
# This value is NOT tuned for accuracy -- see
# evaluation/mri_tolerance_sensitivity.py for the sensitivity sweep
# that keeps that claim honest.

MRI_TOLERANCE_DAYS_DEFAULT = 365   # ENGINEERING DEFAULT; paper does NOT specify this value
MRI_TOLERANCE_SENSITIVITY_DAYS = [180, 365, 540, 730, 1095]  # for the sweep only

UDS_TOLERANCE_DAYS = 30  # matching different UDS forms to the same clinical visit
                          # (unchanged from the audited version; not implicated
                          # in the MRI-matching finding)

# ============================================================
# PEARSON FEATURE-SELECTION THRESHOLD
# ============================================================
# Paper: "at the threshold of 0.95 Pearson correlation" for the final
# multimodal feature set. Confirmed consistent with current code; not
# changed by this correction pass.

PEARSON_THRESHOLD = 0.95

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42

# ============================================================
# KNN IMPUTATION
# ============================================================

KNN_NEIGHBORS = 2   # paper Table 2: lowest RMSE at 2 neighbors

# ============================================================
# SMOTE
# ============================================================

SMOTE_RANDOM_STATE = 42
SMOTE_CATEGORICAL_COLS = ["APOE"]  # genotype code, not a continuous quantity

# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = ["AD", "CN", "Non-AD", "Others", "Uncertain"]
