"""Bank Marketing (UCI id=222): loading, row-wise cleaning, one-time split.

The split is created ONCE (fixed seed) before any fitting and stored as index
files. Train/validation rows are served by `load_train_val()`. The test rows can
only be obtained through `load_test_partition()`, which refuses to run unless
it is called from `src.q5_final_test`.
"""
import inspect

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw_bank() -> pd.DataFrame:
    """Return the raw table (cached in data/raw). 'unknown' values arrive as NaN."""
    if not C.BANK_RAW_CSV.exists():
        from ucimlrepo import fetch_ucirepo

        d = fetch_ucirepo(id=222)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
        df.to_csv(C.BANK_RAW_CSV, index=False)
    return pd.read_csv(C.BANK_RAW_CSV)


def clean_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, fit-free cleaning only (no statistics computed -> no leakage).

    * target y: 'yes' -> 1, 'no' -> 0
    * prev_contacted: indicator for pdays != -1 (pdays=-1 is a sentinel for
      "never contacted", not a real number of days)
    """
    df = df.copy()
    df[C.TARGET_CLS] = (df[C.TARGET_CLS] == "yes").astype(int)
    df["prev_contacted"] = (df["pdays"] != -1).astype(int)
    return df


def _make_split(df: pd.DataFrame):
    idx = np.arange(len(df))
    y = df[C.TARGET_CLS].values
    if C.CLS_SUBSAMPLE_N is not None and C.CLS_SUBSAMPLE_N < len(df):
        idx, _ = train_test_split(idx, train_size=C.CLS_SUBSAMPLE_N, stratify=y, random_state=C.SEED)
    y_sub = y[idx]
    train_idx, rest_idx = train_test_split(
        idx, train_size=C.TRAIN_FRAC, stratify=y_sub, random_state=C.SEED
    )
    rel_test = C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=rel_test, stratify=y[rest_idx], random_state=C.SEED
    )
    for name, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        pd.Series(np.sort(ix), name="row").to_csv(C.DATA_SPLITS / f"cls_{name}_idx.csv", index=False)


def ensure_split():
    if not (C.DATA_SPLITS / "cls_train_idx.csv").exists():
        _make_split(clean_bank(load_raw_bank()))


def _read_idx(name):
    return pd.read_csv(C.DATA_SPLITS / f"cls_{name}_idx.csv")["row"].values


def feature_columns(include_leaky=False):
    cols = C.NUM_FEATURES + C.CAT_FEATURES
    return cols + [C.LEAKY_FEATURE] if include_leaky else cols


def load_train_val(include_leaky=False):
    """Return X_train, y_train, X_val, y_val (DataFrames / arrays). Never test."""
    ensure_split()
    df = clean_bank(load_raw_bank())
    cols = feature_columns(include_leaky)
    tr, va = df.iloc[_read_idx("train")], df.iloc[_read_idx("val")]
    return tr[cols], tr[C.TARGET_CLS].values, va[cols], va[C.TARGET_CLS].values


def split_sizes():
    """Sizes and class proportions of the three partitions (labels of test are
    only counted, never used for modelling)."""
    ensure_split()
    df = clean_bank(load_raw_bank())
    rows = []
    for name in ("train", "val", "test"):
        y = df.iloc[_read_idx(name)][C.TARGET_CLS]
        rows.append({"partition": name, "n": len(y), "n_pos": int(y.sum()), "pos_rate": y.mean()})
    return pd.DataFrame(rows)


def load_test_partition(include_leaky=False):
    """ONLY for src.q5_final_test. Raises if called from anywhere else."""
    callers = [str(f.frame.f_globals.get("__file__", "")) for f in inspect.stack()[1:4]]
    if not any(c.replace("\\", "/").endswith("src/q5_final_test.py") for c in callers):
        raise PermissionError("Test partition may only be loaded by src.q5_final_test")
    ensure_split()
    df = clean_bank(load_raw_bank())
    cols = feature_columns(include_leaky)
    te = df.iloc[_read_idx("test")]
    return te[cols], te[C.TARGET_CLS].values
