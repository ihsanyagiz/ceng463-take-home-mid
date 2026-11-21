"""Shared helpers: metrics, timing, figure/table export, JSON I/O."""
import json
import time
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from . import config as C  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cls_metrics(y_true, y_pred, y_score=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    for k, name in enumerate(C.CLASS_NAMES):
        out[f"precision_{name}"] = p[k]
        out[f"recall_{name}"] = r[k]
        out[f"f1_{name}"] = f[k]
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out


def positive_score(model, X):
    """Probability (or decision score) of the positive class, if available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def macro_f1_binary_from_counts(tp, fp, fn, tn):
    """Vectorised macro-F1 for binary problems (arrays allowed)."""
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fn + fp) > 0)
    return (f1_pos + f1_neg) / 2


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@contextmanager
def timer():
    t = {"start": time.perf_counter()}
    yield t
    t["elapsed"] = time.perf_counter() - t["start"]


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _latex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#").replace("{", r"\{").replace("}", r"\}")
            .replace("$", r"\$").replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))


def save_table(df: pd.DataFrame, name, caption=None, float_fmt="%.4f", index=False, latex=True):
    """Save CSV to results/tables and a booktabs LaTeX table to report/tables."""
    df.to_csv(C.TABLES / f"{name}.csv", index=index)
    if not latex:
        return
    d = df.copy()
    if index:
        d = d.reset_index()
    d.columns = [_latex_escape(c) for c in d.columns]
    for col in d.columns:
        if d[col].dtype == object:
            d[col] = d[col].map(lambda v: _latex_escape(v) if isinstance(v, str) else v)
    body = d.to_latex(index=False, float_format=lambda v: float_fmt % v, na_rep="--",
                      escape=False, column_format="l" * len(d.columns))
    body = body.replace("\\toprule", "\\toprule").strip()
    cap = _latex_escape(caption or name)
    tex = ("\\begin{table}[htbp]\n\\centering\n\\small\n"
           f"\\caption{{{cap}}}\n\\label{{tab:{name}}}\n"
           "\\resizebox{\\linewidth}{!}{%\n" + body + "\n}\n\\end{table}\n")
    (C.REPORT_TABLES / f"{name}.tex").write_text(tex, encoding="utf-8")


def _to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_to_jsonable), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def strip_prefix(params, prefix="clf__"):
    return {k.replace(prefix, ""): v for k, v in params.items()}


def write_macros(macros: dict, name):
    """Write \\newcommand macros so the report never hand-types numbers."""
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (C.REPORT_TABLES / f"{name}_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
