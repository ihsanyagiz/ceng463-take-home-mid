"""Model factory: rebuild any frozen candidate from (name, params).

Used by Q2/Q4 (after tuning), Q5 (frozen configs) and q5_final_test.
"""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from . import config as C
from .preprocessing import make_preprocessor


class SampleWeighted(BaseEstimator, ClassifierMixin):
    """Wraps a Pipeline whose last step is 'clf'; optionally fits with balanced
    sample weights computed on the TRAINING data passed to fit()."""

    def __init__(self, estimator=None, balanced=False):
        self.estimator = estimator
        self.balanced = balanced

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        kw = {"clf__sample_weight": compute_sample_weight("balanced", y)} if self.balanced else {}
        self.estimator_.fit(X, y, **kw)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def build(name: str, params: dict):
    p = dict(params)
    if name == "Dummy":
        from sklearn.dummy import DummyClassifier
        return Pipeline([("pre", make_preprocessor("tree")), ("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "LogReg":
        return Pipeline([("pre", make_preprocessor("scaled")),
                         ("clf", LogisticRegression(max_iter=3000, random_state=C.SEED, **p))])
    if name == "NaiveBayes":
        rep = p.pop("representation")
        if rep == "gaussian":
            return Pipeline([("pre", make_preprocessor("scaled")), ("clf", GaussianNB(**p))])
        n_bins = p.pop("n_bins")
        return Pipeline([("pre", make_preprocessor("discretized", n_bins=n_bins)), ("clf", BernoulliNB(**p))])
    if name == "kNN":
        return Pipeline([("pre", make_preprocessor("scaled")), ("clf", KNeighborsClassifier(n_jobs=C.N_JOBS, **p))])
    if name == "DecisionTree":
        return Pipeline([("pre", make_preprocessor("tree")), ("clf", DecisionTreeClassifier(random_state=C.SEED, **p))])
    if name == "RandomForest":
        return Pipeline([("pre", make_preprocessor("tree")),
                         ("clf", RandomForestClassifier(random_state=C.SEED, n_jobs=C.N_JOBS, **p))])
    if name in ("AdaBoost", "GradientBoosting"):
        balanced = p.pop("balanced_weights", False)
        if name == "AdaBoost":
            clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm="SAMME",
                                     random_state=C.SEED, **p)
        else:
            clf = GradientBoostingClassifier(random_state=C.SEED, max_depth=C.GB_MAX_DEPTH, **p)
        return SampleWeighted(Pipeline([("pre", make_preprocessor("tree")), ("clf", clf)]), balanced=balanced)
    raise ValueError(name)


def final_config_path(name):
    return C.MODELS / f"final_{name}.json"
