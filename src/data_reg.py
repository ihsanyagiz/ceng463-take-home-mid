"""California Housing: load + one-time 70/15/15 split (seed 42).

Target MedHouseVal is in units of $100,000 and is capped at 5.00001.
Test rows are only served to src.q5_final_test.
"""
import inspect

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from . import config as C

TARGET = "MedHouseVal"


def load_reg():
    return fetch_california_housing(as_frame=True, data_home=str(C.SKLEARN_DATA_HOME)).frame


def ensure_split():
    if (C.DATA_SPLITS / "reg_train_idx.csv").exists():
        return
    n = len(load_reg())
    idx = np.arange(n)
    tr, rest = train_test_split(idx, train_size=C.TRAIN_FRAC, random_state=C.SEED)
    va, te = train_test_split(rest, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC), random_state=C.SEED)
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        pd.Series(np.sort(ix), name="row").to_csv(C.DATA_SPLITS / f"reg_{name}_idx.csv", index=False)


def _idx(name):
    return pd.read_csv(C.DATA_SPLITS / f"reg_{name}_idx.csv")["row"].values


def load_reg_train_val():
    ensure_split()
    df = load_reg()
    tr, va = df.iloc[_idx("train")], df.iloc[_idx("val")]
    feats = [c for c in df.columns if c != TARGET]
    return tr[feats], tr[TARGET].values, va[feats], va[TARGET].values


def reg_split_sizes():
    ensure_split()
    return pd.DataFrame([{"partition": p, "n": len(_idx(p))} for p in ("train", "val", "test")])


def load_reg_test_partition():
    callers = [str(f.frame.f_globals.get("__file__", "")) for f in inspect.stack()[1:4]]
    if not any(c.replace("\\", "/").endswith("src/q5_final_test.py") for c in callers):
        raise PermissionError("Regression test partition may only be loaded by src.q5_final_test")
    ensure_split()
    df = load_reg()
    te = df.iloc[_idx("test")]
    feats = [c for c in df.columns if c != TARGET]
    return te[feats], te[TARGET].values
