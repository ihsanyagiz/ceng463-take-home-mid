"""Q2 - Dummy, Logistic Regression, Naive Bayes, kNN: tuning, metrics,
class-imbalance interventions, threshold sweep, validation error export.

Run: python -m src.q2_classifiers
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_curve,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier

from . import config as C
from .data_cls import load_train_val
from .models import build, final_config_path
from .preprocessing import make_preprocessor
from .utils import cls_metrics, save_fig, save_json, save_table, strip_prefix, timer

CV = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)


def grid_search(name, base_params, grid, Xtr, ytr):
    """GridSearchCV on train (5-fold stratified, Macro-F1). Returns (gs, cv table)."""
    gs = GridSearchCV(build(name, base_params), grid, scoring=C.SCORING, cv=CV, n_jobs=C.N_JOBS, refit=True)
    gs.fit(Xtr, ytr)
    tab = pd.DataFrame(gs.cv_results_)
    keep = [c for c in tab.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                                   "mean_fit_time", "rank_test_score"]
    tab = tab[keep].rename(columns=lambda c: c.replace("param_clf__", "").replace("param_pre__", ""))
    return gs, tab.sort_values("rank_test_score")


def tune_all(Xtr, ytr):
    results = {}

    # Logistic Regression: C x class_weight
    gs, tab = grid_search("LogReg", {}, C.LR_GRID, Xtr, ytr)
    tab.insert(0, "model", "LogReg")
    results["LogReg"] = (gs, strip_prefix(gs.best_params_), tab)

    # Naive Bayes: representation x smoothing (two representations, same CV)
    gs_g, tab_g = grid_search("NaiveBayes", {"representation": "gaussian"},
                              {"clf__var_smoothing": C.NB_GAUSS_VAR_SMOOTHING}, Xtr, ytr)
    tab_g.insert(0, "representation", "gaussian (scaled num + one-hot)")
    gs_b, tab_b = grid_search("NaiveBayes", {"representation": "bernoulli", "n_bins": 10},
                              {"clf__alpha": C.NB_BERN_ALPHA, "pre__num__bins__n_bins": C.NB_BERN_BINS}, Xtr, ytr)
    tab_b.insert(0, "representation", "bernoulli (quantile-binned num + one-hot)")
    tab_nb = pd.concat([tab_g, tab_b], ignore_index=True)
    tab_nb.insert(0, "model", "NaiveBayes")
    if gs_g.best_score_ >= gs_b.best_score_:
        nb_params = {"representation": "gaussian", **strip_prefix(gs_g.best_params_)}
        gs_nb = gs_g
    else:
        bp = gs_b.best_params_
        nb_params = {"representation": "bernoulli", "alpha": bp["clf__alpha"], "n_bins": bp["pre__num__bins__n_bins"]}
        gs_nb = gs_b
    results["NaiveBayes"] = (gs_nb, nb_params, tab_nb)

    # kNN: k x weights x p
    gs, tab = grid_search("kNN", {}, C.KNN_GRID, Xtr, ytr)
    tab.insert(0, "model", "kNN")
    results["kNN"] = (gs, strip_prefix(gs.best_params_), tab)

    # Dummy baseline (no hyper-parameters)
    gs, tab = grid_search("Dummy", {}, {"clf__strategy": ["most_frequent"]}, Xtr, ytr)
    tab.insert(0, "model", "Dummy")
    results["Dummy"] = (gs, {}, tab)
    return results


def evaluate(results, Xtr, ytr, Xva, yva):
    rows, per_class, scores, fitted = [], [], {}, {}
    for name, (gs, params, _) in results.items():
        model = build(name, params)
        with timer() as tf:
            model.fit(Xtr, ytr)
        with timer() as tp:
            pred = model.predict(Xva)
        score = model.predict_proba(Xva)[:, 1]
        m = cls_metrics(yva, pred, score)
        fitted[name] = model
        scores[name] = (pred, score)
        rows.append({"model": name, "best_params": str(params), "cv_macro_f1": gs.best_score_,
                     "cv_macro_f1_std": gs.cv_results_["std_test_score"][gs.best_index_],
                     **{f"val_{k}": v for k, v in m.items()}, "fit_time_s": tf["elapsed"],
                     "predict_time_s": tp["elapsed"]})
        for k, cname in enumerate(C.CLASS_NAMES):
            per_class.append({"model": name, "class": cname, "precision": m[f"precision_{cname}"],
                              "recall": m[f"recall_{cname}"], "f1": m[f"f1_{cname}"],
                              "support": int((yva == k).sum())})
        if name != "Dummy":
            save_json({"name": name, "params": params, "source": "Q2", "cv_macro_f1": gs.best_score_},
                      final_config_path(name))
    return pd.DataFrame(rows), pd.DataFrame(per_class), scores, fitted


def plot_confusions(scores, yva):
    names = list(scores)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.3))
    for ax, name in zip(axes, names):
        cm = confusion_matrix(yva, scores[name][0], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=C.CLASS_NAMES).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)
        ax.grid(False)
        f, a = plt.subplots(figsize=(3.6, 3.3))
        ConfusionMatrixDisplay(cm, display_labels=C.CLASS_NAMES).plot(ax=a, colorbar=False, values_format="d")
        a.grid(False)
        a.set_title(f"Q2 confusion matrix (validation): {name}")
        save_fig(f, f"q2_cm_{name}")
    fig.suptitle("Q2c confusion matrices on validation (rows = true, cols = predicted)")
    save_fig(fig, "q2_confusion_matrices")


def plot_roc_pr(scores, yva):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (_, s) in scores.items():
        fpr, tpr, _ = roc_curve(yva, s)
        a1.plot(fpr, tpr, label=name)
        p, r, _ = precision_recall_curve(yva, s)
        a2.plot(r, p, label=name)
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a1.set_title("ROC (validation)")
    a2.axhline(yva.mean(), color="k", ls="--", lw=0.8, label="prevalence")
    a2.set_xlabel("Recall (yes)")
    a2.set_ylabel("Precision (yes)")
    a2.set_title("Precision-Recall (validation)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    save_fig(fig, "q2_roc_pr")


def imbalance_study(lr_params, knn_params, Xtr, ytr, Xva, yva):
    """Controlled interventions: everything identical except the imbalance handling."""
    lr_base = {k: v for k, v in lr_params.items() if k != "class_weight"}
    pre = lambda: make_preprocessor("scaled")  # noqa: E731
    variants = {
        "LogReg | none": ImbPipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=3000, **lr_base))]),
        "LogReg | class_weight=balanced": ImbPipeline([("pre", pre()), ("clf", LogisticRegression(
            max_iter=3000, class_weight="balanced", **lr_base))]),
        "LogReg | RandomOverSampler (train only)": ImbPipeline([("pre", pre()), ("ros", RandomOverSampler(
            random_state=C.SEED)), ("clf", LogisticRegression(max_iter=3000, **lr_base))]),
        "LogReg | SMOTE (train only)": ImbPipeline([("pre", pre()), ("smote", SMOTE(random_state=C.SEED)),
                                                     ("clf", LogisticRegression(max_iter=3000, **lr_base))]),
        "kNN | none": ImbPipeline([("pre", pre()), ("clf", KNeighborsClassifier(**knn_params))]),
        "kNN | RandomOverSampler (train only)": ImbPipeline([("pre", pre()), ("ros", RandomOverSampler(
            random_state=C.SEED)), ("clf", KNeighborsClassifier(**knn_params))]),
    }
    rows = []
    for name, model in variants.items():
        cv = cross_validate(model, Xtr, ytr, cv=CV, scoring="f1_macro", n_jobs=C.N_JOBS)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        m = cls_metrics(yva, pred, model.predict_proba(Xva)[:, 1])
        tn, fp, fn, tp = confusion_matrix(yva, pred, labels=[0, 1]).ravel()
        rows.append({"variant": name, "cv_macro_f1": cv["test_score"].mean(), "cv_std": cv["test_score"].std(),
                     "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_recall_no": m["recall_no"],
                     "val_recall_yes": m["recall_yes"], "val_precision_yes": m["precision_yes"],
                     "val_roc_auc": m["roc_auc"], "FP": fp, "FN": fn,
                     "error_rate_no": fp / (fp + tn), "error_rate_yes": fn / (fn + tp)})
    tab = pd.DataFrame(rows)
    save_table(tab, "q2_imbalance", "Q2d controlled class-imbalance interventions (only the imbalance handling differs)")
    return tab


def threshold_sweep(name, params, Xtr, ytr, Xva, yva):
    """Threshold chosen on out-of-fold TRAIN probabilities, then applied to validation."""
    model = build(name, params)
    oof = cross_val_predict(model, Xtr, ytr, cv=CV, method="predict_proba", n_jobs=C.N_JOBS)[:, 1]
    model.fit(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    ths = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rec = []
    for t in ths:
        rec.append({"threshold": t,
                    "oof_macro_f1": f1_score(ytr, oof >= t, average="macro"),
                    "oof_f1_yes": f1_score(ytr, oof >= t),
                    "val_macro_f1": f1_score(yva, pva >= t, average="macro"),
                    "val_f1_yes": f1_score(yva, pva >= t)})
    tab = pd.DataFrame(rec)
    t_star = float(tab.loc[tab.oof_macro_f1.idxmax(), "threshold"])
    tab.to_csv(C.TABLES / "q2_threshold_sweep.csv", index=False)
    summ = pd.DataFrame([
        {"rule": "default 0.5", "threshold": 0.5,
         "val_macro_f1": f1_score(yva, pva >= 0.5, average="macro"), "val_f1_yes": f1_score(yva, pva >= 0.5)},
        {"rule": "max OOF Macro-F1 (train CV)", "threshold": t_star,
         "val_macro_f1": f1_score(yva, pva >= t_star, average="macro"), "val_f1_yes": f1_score(yva, pva >= t_star)},
    ])
    save_table(summ, f"q2_threshold_summary", f"Q2d decision threshold for {name} (chosen on OOF train predictions)")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(tab.threshold, tab.oof_macro_f1, label="OOF train Macro-F1")
    ax.plot(tab.threshold, tab.val_macro_f1, label="Validation Macro-F1")
    ax.plot(tab.threshold, tab.oof_f1_yes, ls="--", label="OOF train F1(yes)")
    ax.plot(tab.threshold, tab.val_f1_yes, ls="--", label="Validation F1(yes)")
    ax.axvline(t_star, color="k", lw=0.8, ls=":", label=f"chosen t={t_star:.2f}")
    ax.set_xlabel("Decision threshold on P(yes)")
    ax.set_ylabel("F1")
    ax.set_title(f"Q2d threshold sweep: {name} {params}", fontsize=8)
    ax.legend(fontsize=8)
    save_fig(fig, "q2_threshold_sweep")
    return summ


def error_analysis(name, model, Xva, yva, pred, score, n_each=10):
    df = Xva.copy()
    df["y_true"] = yva
    df["y_pred"] = pred
    df["p_yes"] = score
    df["error_type"] = np.where((yva == 1) & (pred == 0), "FN", np.where((yva == 0) & (pred == 1), "FP", "correct"))
    err = df[df.error_type != "correct"]
    fn = err[err.error_type == "FN"].sort_values("p_yes").head(n_each)
    fp = err[err.error_type == "FP"].sort_values("p_yes", ascending=False).head(n_each)
    sample = pd.concat([fn, fp])
    sample.index.name = "val_row_id"
    sample.to_csv(C.TABLES / "q2_errors.csv")
    show = sample[["age", "job", "balance", "contact", "month", "campaign", "pdays", "poutcome", "y_true", "p_yes",
                   "error_type"]].reset_index()
    save_table(show, "q2_errors", f"Q2e most confident validation errors of {name} (10 FN + 10 FP)", float_fmt="%.3f")
    err.to_csv(C.TABLES / "q2_errors_all.csv")

    groups = []
    df["p_bin"] = pd.cut(df.p_yes, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], include_lowest=True).astype(str)
    for col in ["poutcome", "contact", "month", "p_bin", "housing", "loan"]:
        g = df.assign(**{col: df[col].fillna("missing")}).groupby(col).agg(
            n=("y_true", "size"), pos_rate=("y_true", "mean"),
            n_FN=("error_type", lambda s: (s == "FN").sum()), n_FP=("error_type", lambda s: (s == "FP").sum()))
        g["error_rate"] = (g.n_FN + g.n_FP) / g.n
        g = g.reset_index().rename(columns={col: "level"})
        g.insert(0, "feature", col)
        groups.append(g)
    gtab = pd.concat(groups, ignore_index=True)
    save_table(gtab, "q2_error_groups", f"Q2e validation error rates of {name} grouped by input characteristics",
               float_fmt="%.3f")
    # numeric profile: errors vs correct
    prof = df.groupby("error_type")[["age", "balance", "campaign", "pdays", "previous", "p_yes"]].median().reset_index()
    save_table(prof, "q2_error_numeric_profile", "Q2e median numeric features by error type (validation)",
               float_fmt="%.2f")
    return sample


def main():
    Xtr, ytr, Xva, yva = load_train_val()
    results = tune_all(Xtr, ytr)
    cv_tab = pd.concat([r[2] for r in results.values()], ignore_index=True)
    cv_tab.to_csv(C.TABLES / "q2_cv_results.csv", index=False)
    for name in ["LogReg", "NaiveBayes", "kNN"]:
        save_table(results[name][2], f"q2_grid_{name}", f"Q2b CV grid for {name} (5-fold stratified, Macro-F1)")

    summary, per_class, scores, fitted = evaluate(results, Xtr, ytr, Xva, yva)
    save_table(summary.drop(columns="best_params"), "q2_summary", "Q2c validation metrics (positive class = yes)")
    save_table(summary[["model", "best_params"]], "q2_best_params", "Q2b selected hyper-parameters (by CV Macro-F1)")
    save_table(per_class, "q2_per_class", "Q2c per-class precision / recall / F1 (validation)")
    plot_confusions(scores, yva)
    plot_roc_pr({k: v for k, v in scores.items() if k != "Dummy"}, yva)

    imb = imbalance_study(results["LogReg"][1], results["kNN"][1], Xtr, ytr, Xva, yva)

    # finalized Q2 model = best CV Macro-F1 among non-dummy models (never validation/test)
    cand = summary[summary.model != "Dummy"].sort_values("cv_macro_f1", ascending=False)
    best = cand.iloc[0]["model"]
    save_json({"best_q2_model": best, "params": results[best][1], "selection": "max CV Macro-F1 on train"},
              C.MODELS / "q2_best.json")
    thr = threshold_sweep(best, results[best][1], Xtr, ytr, Xva, yva)
    error_analysis(best, fitted[best], Xva, yva, *scores[best])

    pd.set_option("display.width", 200)
    print(summary[["model", "cv_macro_f1", "val_accuracy", "val_macro_f1", "val_roc_auc", "val_pr_auc",
                   "fit_time_s", "predict_time_s"]].to_string(index=False))
    print(summary[["model", "best_params"]].to_string(index=False))
    print(imb[["variant", "cv_macro_f1", "val_macro_f1", "val_recall_yes", "val_accuracy"]].to_string(index=False))
    print(thr.to_string(index=False))
    print("best Q2 model:", best)


if __name__ == "__main__":
    main()
