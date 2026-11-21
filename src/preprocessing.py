"""ColumnTransformer builders per model family.

All transformers are returned UNFITTED and are always placed inside a Pipeline,
so imputation/scaling/encoding statistics are learned on training folds only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from . import config as C


def _cat_pipe():
    # Missing categorical values ("unknown" in the raw data) are kept as their own
    # level: in poutcome/contact they are informative (e.g. poutcome missing =
    # never contacted before), so most-frequent imputation would destroy signal.
    # handle_unknown="ignore" -> unseen categories at predict time become all-zero.
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def make_preprocessor(family: str, num_features=None, cat_features=None, n_bins=10):
    """family in {"scaled", "tree", "discretized"}.

    scaled      -> median impute + StandardScaler + one-hot   (LogReg, kNN, GaussianNB)
    tree        -> median impute + one-hot, no scaling        (trees / ensembles)
    discretized -> median impute + quantile bins (one-hot) + one-hot (BernoulliNB)
    """
    num = list(num_features if num_features is not None else C.NUM_FEATURES)
    cat = list(cat_features if cat_features is not None else C.CAT_FEATURES)
    if family == "scaled":
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    elif family == "tree":
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    elif family == "discretized":
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("bins", KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile",
                                      subsample=None)),
        ])
    else:
        raise ValueError(family)
    return ColumnTransformer(
        [("num", num_pipe, num), ("cat", _cat_pipe(), cat)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


TRANSFORM_TABLE = [
    # transformation, LogReg, NaiveBayes (Gaussian | Bernoulli), kNN, Trees/Ensembles
    ("Numeric: median imputation", "yes", "yes | yes", "yes", "yes"),
    ("Numeric: standardisation", "yes (optimisation, penalty)", "yes | n/a", "yes (distances)", "no (split-invariant)"),
    ("Numeric: quantile discretisation", "no", "no | yes (binary features)", "no", "no"),
    ("Categorical: constant 'missing' level", "yes", "yes | yes", "yes", "yes"),
    ("Categorical: one-hot, handle_unknown=ignore", "yes", "yes | yes", "yes", "yes"),
    ("Class weighting / resampling (train only)", "class_weight", "n/a", "n/a (resampling only)", "class_weight / sample_weight"),
    ("Leaky feature 'duration'", "dropped", "dropped", "dropped", "dropped"),
]
