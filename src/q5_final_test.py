"""Q5e - THE ONLY PLACE THE TEST SETS ARE USED.

Loads the frozen candidate configurations and selection rule written by
q5_robustness_selection (classification) and q3_regression (regression), refits
on train (or train+val, see --refit), and evaluates every frozen candidate ONCE.
A guard file prevents accidental re-runs; use --force to override (logged).

Run: python -m src.q5_final_test [--refit trainval|train] [--force]
"""
import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import config as C
from .data_cls import load_test_partition, load_train_val
from .data_reg import load_reg_test_partition, load_reg_train_val
from .models import build
from .q3_regression import reg_metrics
from .utils import cls_metrics, load_json, macro_f1_binary_from_counts, save_fig, save_json, save_table

GUARD = C.LOGS / "final_test_done.json"


def bootstrap_ci(y, p, B=C.BOOTSTRAP_B):
    rng = np.random.default_rng(C.SEED)
    ind = {"tp": (p == 1) & (y == 1), "fp": (p == 1) & (y == 0), "fn": (p == 0) & (y == 1), "tn": (p == 0) & (y == 0)}
    vals = []
    for s in range(0, B, 500):
        idx = rng.integers(0, len(y), (min(500, B - s), len(y)))
        c = {k: v[idx].sum(1).astype(float) for k, v in ind.items()}
        vals.append(macro_f1_binary_from_counts(c["tp"], c["fp"], c["fn"], c["tn"]))
    return np.percentile(np.concatenate(vals), [2.5, 97.5])


def build_reg(name, params):
    p = dict(params)
    reg = {"Dummy": lambda: DummyRegressor(), "OLS": lambda: LinearRegression(), "Ridge": lambda: Ridge(**p),
           "Lasso": lambda: Lasso(max_iter=50000, **p),
           "DecisionTree": lambda: DecisionTreeRegressor(random_state=C.SEED, **p),
           "RandomForest": lambda: RandomForestRegressor(random_state=C.SEED, n_jobs=C.N_JOBS, **p)}[name]()
    return Pipeline([("scale", StandardScaler()), ("reg", reg)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", choices=["train", "trainval"], default=C.FINAL_REFIT_ON)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if GUARD.exists() and not args.force:
        print(f"Final test evaluation already performed: {GUARD.read_text()}\nRefusing to run again (use --force).")
        raise SystemExit(1)

    rule = load_json(C.MODELS / "selection_rule.json")
    configs = rule["candidates"]
    h = hashlib.sha256(json.dumps(configs, sort_keys=True, default=str).encode()).hexdigest()
    assert h == rule["config_hash"], "Frozen configurations were modified after freezing!"

    # ---------------- classification ----------------
    Xtr, ytr, Xva, yva = load_train_val()
    if args.refit == "trainval":
        Xfit, yfit = pd.concat([Xtr, Xva]), np.concatenate([ytr, yva])
    else:
        Xfit, yfit = Xtr, ytr
    Xte, yte = load_test_partition()
    rows, preds = [], {}
    for name, params in configs.items():
        m = build(name, params).fit(Xfit, yfit)
        p = m.predict(Xte)
        preds[name] = p
        met = cls_metrics(yte, p, m.predict_proba(Xte)[:, 1])
        lo, hi = bootstrap_ci(yte, p)
        rows.append({"model": name, "selected": name == rule["selected_model"], **{f"test_{k}": v for k, v in met.items()},
                     "test_macro_f1_ci95_lo": lo, "test_macro_f1_ci95_hi": hi})
    res = pd.DataFrame(rows)
    save_table(res[["model", "selected", "test_accuracy", "test_macro_f1", "test_macro_f1_ci95_lo",
                    "test_macro_f1_ci95_hi", "test_roc_auc", "test_pr_auc"]], "q5_test_results",
               f"Q5e single final test evaluation of all frozen candidates (refit on {args.refit}); "
               f"95% CI = bootstrap over test examples")
    save_table(res[["model"] + [c for c in res.columns if c.startswith(("test_precision", "test_recall", "test_f1"))]],
               "q5_test_per_class", "Q5e per-class test metrics")
    res.to_csv(C.TABLES / "q5_test_results_full.csv", index=False)

    sel = rule["selected_model"]
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[sel], labels=[0, 1]), display_labels=C.CLASS_NAMES).plot(
        ax=ax, colorbar=False, values_format="d")
    ax.grid(False)
    ax.set_title(f"Q5e test confusion matrix: {sel}")
    save_fig(fig, "q5_test_confusion_selected")

    # validation vs test (generalisation gap) - validation values come from Q5 / refit on train
    # ---------------- regression (frozen Q3 models) ----------------
    Rtr, rtr, Rva, rva = load_reg_train_val()
    Rte, rte = load_reg_test_partition()
    if args.refit == "trainval":
        Rfit, rfit = pd.concat([Rtr, Rva]), np.concatenate([rtr, rva])
    else:
        Rfit, rfit = Rtr, rtr
    rrows = []
    for name in ["Dummy", "OLS", "Ridge", "Lasso", "DecisionTree", "RandomForest"]:
        cfg = load_json(C.MODELS / f"final_reg_{name}.json")
        m = build_reg(name, cfg["params"]).fit(Rfit, rfit)
        rrows.append({"model": name, **{f"test_{k}": v for k, v in reg_metrics(rte, m.predict(Rte)).items()}})
    save_table(pd.DataFrame(rrows), "q5_test_results_regression",
               f"Q3 frozen regression models on the regression test set (refit on {args.refit}; MAE/RMSE in $100k)")

    stamp = {"timestamp": datetime.now().isoformat(timespec="seconds"), "refit_on": args.refit,
             "forced": bool(args.force), "selected_model": sel, "config_hash": h}
    with open(C.LOGS / "final_test_runs.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(stamp) + "\n")
    save_json(stamp, GUARD)
    pd.set_option("display.width", 220)
    print(res[["model", "selected", "test_accuracy", "test_macro_f1", "test_macro_f1_ci95_lo", "test_macro_f1_ci95_hi",
               "test_roc_auc", "test_pr_auc"]].to_string(index=False))
    print(pd.DataFrame(rrows).to_string(index=False))
    print("SELECTED MODEL:", sel)


if __name__ == "__main__":
    main()
