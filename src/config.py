"""Global configuration: seeds, paths, CV protocol and search spaces.

Every script imports from here so that the protocol is defined in exactly one place.
"""
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
N_JOBS = -1

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
LOGS = RESULTS / "logs"
REPORT = ROOT / "report"
REPORT_TABLES = REPORT / "tables"
REPORT_FIGURES = REPORT / "figures"

for _p in (DATA_RAW, DATA_SPLITS, TABLES, FIGURES, MODELS, LOGS, REPORT_TABLES, REPORT_FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

BANK_RAW_CSV = DATA_RAW / "bank_full_ucimlrepo.csv"
SKLEARN_DATA_HOME = DATA_RAW / "sklearn"

# --------------------------------------------------------------------------- #
# Partitioning (classification: Bank Marketing; regression: California Housing)
# --------------------------------------------------------------------------- #
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Optional consistent, stratified subsample of the classification data drawn
# BEFORE partitioning (None = use all rows). Applied identically to every model.
CLS_SUBSAMPLE_N = None

# --------------------------------------------------------------------------- #
# Classification features
# --------------------------------------------------------------------------- #
TARGET_CLS = "y"
LEAKY_FEATURE = "duration"  # only known after the call ends -> target leakage
NUM_FEATURES = ["age", "balance", "day_of_week", "campaign", "pdays", "previous", "prev_contacted"]
CAT_FEATURES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
CLASS_NAMES = ["no", "yes"]  # 0 / 1

# --------------------------------------------------------------------------- #
# Cross-validation protocol
# --------------------------------------------------------------------------- #
CV_FOLDS = 5
# Selection metric. Macro-F1 weights both classes equally, so a model cannot
# score well by predicting only the majority class (~88% "no").
SCORING = "f1_macro"
REPEATED_CV_REPEATS = 3  # Q5a: 5 folds x 3 repeats

# --------------------------------------------------------------------------- #
# Q2 search spaces
# --------------------------------------------------------------------------- #
LR_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "clf__class_weight": [None, "balanced"],
}
KNN_GRID = {
    "clf__n_neighbors": [1, 3, 5, 9, 15, 31, 51],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2],
}
# NB: representation is part of the search (Gaussian on scaled numeric + one-hot
# vs. Bernoulli on discretised numeric + one-hot), see q2_classifiers.py
NB_GAUSS_VAR_SMOOTHING = [1e-9, 1e-6, 1e-3, 1e-1]
NB_BERN_ALPHA = [0.01, 0.1, 1.0, 10.0]
NB_BERN_BINS = [3, 5, 10]

# --------------------------------------------------------------------------- #
# Q3 search spaces (regression)
# --------------------------------------------------------------------------- #
REG_ALPHAS = np.logspace(-4, 3, 36)
REG_LASSO_ALPHAS = np.logspace(-5, 0, 31)
REG_POLY_DEGREES = [1, 2, 3, 4]
REG_TREE_DEPTHS = list(range(1, 21))
REG_DT_GRID = {"reg__max_depth": [4, 6, 8, 10, 12, 15, None], "reg__min_samples_leaf": [1, 5, 20]}
REG_RF_GRID = {"reg__max_depth": [10, 20, None], "reg__max_features": [0.33, 1.0], "reg__min_samples_leaf": [1, 5]}
REG_RF_N_ESTIMATORS = 200

# --------------------------------------------------------------------------- #
# Q4 search spaces (trees / ensembles)
# --------------------------------------------------------------------------- #
DT_GRID = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [3, 5, 8, 12, None],
    "clf__min_samples_leaf": [1, 5, 20, 50],
    "clf__ccp_alpha": [0.0, 1e-4, 1e-3],
    "clf__class_weight": [None, "balanced"],
}
RF_N_ESTIMATORS_GRID_FIXED = 100  # trees used during the grid search (n_estimators studied separately by a curve)
RF_GRID = {
    "clf__max_depth": [12, None],
    "clf__max_features": ["sqrt", 0.5],
    "clf__min_samples_leaf": [1, 5],
    "clf__class_weight": [None, "balanced_subsample"],
}
RF_N_ESTIMATORS_CURVE = [10, 25, 50, 100, 200, 400]
RF_DEPTH_CURVE = [2, 4, 6, 8, 10, 14, 18, 24, None]
RF_MAX_FEATURES_CURVE = [1, 2, "sqrt", 0.25, 0.5, 0.75, 1.0]

ADA_MAX_ESTIMATORS = 400
ADA_LEARNING_RATES = [0.05, 0.1, 0.5, 1.0]
GB_MAX_ESTIMATORS = 400
GB_LEARNING_RATES = [0.05, 0.1, 0.2]
GB_MAX_DEPTH = 3
BOOST_SAMPLE_WEIGHTING = [False, True]  # balanced sample weights on/off
BOOST_STAGE_GRID = [10, 25, 50, 100, 150, 200, 300, 400]  # n_estimators values evaluated via staged predictions

# --------------------------------------------------------------------------- #
# Q5 robustness / bootstrap / multi-criteria
# --------------------------------------------------------------------------- #
NOISE_SEVERITIES = [0.0, 0.1, 0.25, 0.5, 1.0]  # Gaussian noise SD in units of each feature's training SD
MISSING_RATES = [0.0, 0.1, 0.2, 0.4]
ROBUSTNESS_REPEATS = 5
BOOTSTRAP_B = 10_000
N_DIRICHLET = 5_000
DIRICHLET_CENTER_CONC = 20  # centred sensitivity: Dirichlet(alpha = 20 * base weights)

# Ordinal 1 (worst) - 5 (best) scores for criteria that cannot be measured from runs.
# TODO(student): review and justify these subjective scores in the report.
INTERPRETABILITY_SCORE = {"LogReg": 4, "NaiveBayes": 4, "kNN": 2, "DecisionTree": 5,
                          "RandomForest": 2, "AdaBoost": 2, "GradientBoosting": 2}
IMPLEMENTATION_SIMPLICITY = {"LogReg": 5, "NaiveBayes": 4, "kNN": 4, "DecisionTree": 4,
                             "RandomForest": 3, "AdaBoost": 3, "GradientBoosting": 3}

# Base weights for the multi-criteria table (must sum to 1). Set by the student.
# TODO(student): justify these weights in the report.
MCDA_BASE_WEIGHTS = {
    "performance": 0.40,
    "robustness": 0.20,
    "stability": 0.10,
    "interpretability": 0.15,
    "train_cost": 0.05,
    "inference_cost": 0.05,
    "implementation": 0.05,
}

# Q5e: fit frozen candidates on "train" or "trainval" before the single test evaluation.
FINAL_REFIT_ON = "trainval"
