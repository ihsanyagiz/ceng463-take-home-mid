"""Q4 - Decision tree, Random Forest, AdaBoost, Gradient Boosting on the Bank
Marketing task (same split/CV as Q2) + best Q2 model as reference.

Run: python -m src.q4_trees_ensembles
"""
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from . import config as C
from .data_cls import load_train_val
from .models import build, final_config_path
from .preprocessing import make_preprocessor
from .utils import cls_metrics, load_json, save_fig, save_json, save_table, strip_prefix, timer

CV = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)


def cv_table(gs):
    t = pd.DataFrame(gs.cv_results_)
    keep = [c for c in t.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                               "mean_train_score", "rank_test_score"]
    return t[keep].rename(columns=lambda c: c.replace("param_clf__", "")).sort_values("rank_test_score")


# --------------------------------------------------------------------------- #
# b) single decision tree
# --------------------------------------------------------------------------- #
def tune_tree(Xtr, ytr, Xva, yva):
    gs = GridSearchCV(build("DecisionTree", {}), C.DT_GRID, scoring=C.SCORING, cv=CV, n_jobs=C.N_JOBS,
                      return_train_score=True).fit(Xtr, ytr)
    tab = cv_table(gs)
    tab.to_csv(C.TABLES / "q4_dt_cv_results.csv", index=False)
    crit = tab.sort_values("mean_test_score", ascending=False).groupby("criterion").head(1)
    save_table(crit, "q4_dt_criterion", "Q4b best CV configuration per split criterion (5-fold, Macro-F1)")
    save_table(tab.head(10), "q4_dt_top10", "Q4b top-10 decision-tree configurations (5-fold CV Macro-F1)")
    params = strip_prefix(gs.best_params_)

    # entropy vs log_loss: sklearn implements both with the same impurity -> verify empirically
    m_ent = build("DecisionTree", {**params, "criterion": "entropy"}).fit(Xtr, ytr)
    m_log = build("DecisionTree", {**params, "criterion": "log_loss"}).fit(Xtr, ytr)
    same = bool((m_ent.predict(Xva) == m_log.predict(Xva)).all())
    save_json({"entropy_and_log_loss_identical_val_predictions": same}, C.RESULTS / "q4_logloss_check.json")

    # effect of each complexity control around the optimum (marginal curves)
    rows = []
    for p, values in (("max_depth", C.DT_GRID["clf__max_depth"]), ("min_samples_leaf", C.DT_GRID["clf__min_samples_leaf"]),
                      ("ccp_alpha", C.DT_GRID["clf__ccp_alpha"])):
        for v in values:
            m = build("DecisionTree", {**params, p: v}).fit(Xtr, ytr)
            rows.append({"param": p, "value": str(v), "train_macro_f1": f1_score(ytr, m.predict(Xtr), average="macro"),
                         "val_macro_f1": f1_score(yva, m.predict(Xva), average="macro"),
                         "n_leaves": m[-1].get_n_leaves(), "depth": m[-1].get_depth()})
    save_table(pd.DataFrame(rows), "q4_dt_complexity", "Q4b one-at-a-time complexity controls around the CV optimum")
    return gs, params


def small_tree(params, Xtr, ytr, Xva, yva):
    p = {**params, "max_depth": 3, "ccp_alpha": 0.0}
    model = build("DecisionTree", p).fit(Xtr, ytr)
    names = list(model[0].get_feature_names_out())
    tree = model[-1]
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(tree, feature_names=names, class_names=C.CLASS_NAMES, filled=True, impurity=True, proportion=False,
              rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Q4b small decision tree (max_depth=3, criterion={p['criterion']}, "
                 f"class_weight={p.get('class_weight')})")
    save_fig(fig, "q4_small_tree")
    rules = export_text(tree, feature_names=names, show_weights=True)
    (C.TABLES / "q4_small_tree_rules.txt").write_text(rules, encoding="utf-8")

    # decision paths for concrete validation rows (one predicted yes-correct, one no-correct, one error)
    Z = model[0].transform(Xva)
    pred = tree.predict(Z)
    cases = {"predicted yes, true yes": (1, 1), "predicted no, true no": (0, 0),
             "predicted no, true yes (error)": (0, 1), "predicted yes, true no (error)": (1, 0)}
    picks = {}
    for label, (pv, tv) in cases.items():
        hit = np.where((pred == pv) & (yva == tv))[0]
        if len(hit):
            picks[label] = hit[0]  # first such validation row (deterministic)
    node_ind = tree.decision_path(Z)
    leaf = tree.apply(Z)
    t = tree.tree_
    rows = []
    for label, i in picks.items():
        nodes = node_ind.indices[node_ind.indptr[i]: node_ind.indptr[i + 1]]
        steps = []
        for nd in nodes:
            if leaf[i] == nd:
                continue
            f, thr = t.feature[nd], t.threshold[nd]
            go_left = Z[i, f] <= thr
            steps.append(f"{names[f]} = {Z[i, f]:.3g} {'<=' if go_left else '>'} {thr:.3g}")
        val = t.value[leaf[i]][0]
        rows.append({"case": label, "val_row_position": int(i), "path": " AND ".join(steps),
                     "leaf_class_distribution(weighted)": np.round(val / val.sum(), 3).tolist(),
                     "predicted": C.CLASS_NAMES[pred[i]], "true": C.CLASS_NAMES[yva[i]]})
    dp = pd.DataFrame(rows)
    save_table(dp, "q4_decision_paths", "Q4b decision paths of the depth-3 tree for selected validation rows")
    return model


# --------------------------------------------------------------------------- #
# c) random forest
# --------------------------------------------------------------------------- #
def tune_rf(Xtr, ytr):
    base = Pipeline([("pre", make_preprocessor("tree")),
                     ("clf", RandomForestClassifier(C.RF_N_ESTIMATORS_GRID_FIXED, random_state=C.SEED, n_jobs=1))])
    gs = GridSearchCV(base, C.RF_GRID, scoring=C.SCORING, cv=CV, n_jobs=C.N_JOBS, return_train_score=True).fit(Xtr, ytr)
    tab = cv_table(gs)
    save_table(tab, "q4_rf_cv_results", f"Q4c Random Forest grid ({C.RF_N_ESTIMATORS_GRID_FIXED} trees, 5-fold CV)")
    params = {"n_estimators": C.RF_N_ESTIMATORS_GRID_FIXED, **strip_prefix(gs.best_params_)}
    return gs, params


def rf_curves(params, Xtr, ytr, Xva, yva):
    pre = make_preprocessor("tree").fit(Xtr)
    Ztr, Zva = pre.transform(Xtr), pre.transform(Xva)
    rows = []
    # number of trees (warm start adds trees to the same forest)
    rf = RandomForestClassifier(**{**params, "n_estimators": 1}, warm_start=True, random_state=C.SEED, n_jobs=C.N_JOBS)
    for n in C.RF_N_ESTIMATORS_CURVE:
        rf.set_params(n_estimators=n).fit(Ztr, ytr)
        rows.append({"param": "n_estimators", "value": n, "train_macro_f1": f1_score(ytr, rf.predict(Ztr), average="macro"),
                     "val_macro_f1": f1_score(yva, rf.predict(Zva), average="macro")})
    for p, values in (("max_depth", C.RF_DEPTH_CURVE), ("max_features", C.RF_MAX_FEATURES_CURVE)):
        for v in values:
            m = RandomForestClassifier(**{**params, "n_estimators": 60, p: v}, random_state=C.SEED,
                                       n_jobs=C.N_JOBS).fit(Ztr, ytr)
            rows.append({"param": p, "value": v, "train_macro_f1": f1_score(ytr, m.predict(Ztr), average="macro"),
                         "val_macro_f1": f1_score(yva, m.predict(Zva), average="macro")})
    tab = pd.DataFrame(rows)
    tab["value"] = tab["value"].astype(str)
    save_table(tab, "q4_rf_curves", "Q4c Random Forest: train vs. validation Macro-F1 (other params at CV optimum)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, p in zip(axes, ["n_estimators", "max_depth", "max_features"]):
        s = tab[tab.param == p]
        ax.plot(s.value, s.train_macro_f1, "o-", label="train")
        ax.plot(s.value, s.val_macro_f1, "o-", label="validation")
        ax.set_xlabel(p)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"RF: {p}")
        ax.legend()
    fig.suptitle("Q4c Random Forest complexity study")
    save_fig(fig, "q4_rf_curves")

    # tree correlation vs max_features: mean pairwise correlation of per-tree P(yes) on validation
    corr_rows = []
    for v in C.RF_MAX_FEATURES_CURVE:
        m = RandomForestClassifier(**{**params, "n_estimators": 60, "max_features": v}, random_state=C.SEED,
                                   n_jobs=C.N_JOBS).fit(Ztr, ytr)
        P = np.array([t.predict_proba(Zva)[:, 1] for t in m.estimators_])
        cm = np.corrcoef(P)
        iu = np.triu_indices_from(cm, 1)
        ind_f1 = np.mean([f1_score(yva, (p >= 0.5).astype(int), average="macro") for p in P])
        corr_rows.append({"max_features": str(v), "mean_pairwise_tree_corr": np.nanmean(cm[iu]),
                          "mean_single_tree_val_macro_f1": ind_f1,
                          "forest_val_macro_f1": f1_score(yva, m.predict(Zva), average="macro"),
                          "var_of_tree_probs": P.var(axis=0).mean()})
    ct = pd.DataFrame(corr_rows)
    save_table(ct, "q4_rf_tree_correlation", "Q4c tree correlation vs. max_features (60 trees, validation)")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ct.max_features, ct.mean_pairwise_tree_corr, "o-", label="mean pairwise tree corr.")
    ax.plot(ct.max_features, ct.mean_single_tree_val_macro_f1, "s--", label="single-tree Macro-F1")
    ax.plot(ct.max_features, ct.forest_val_macro_f1, "^-", label="forest Macro-F1")
    ax.set_xlabel("max_features")
    ax.set_title("Q4c randomised feature selection vs. tree correlation")
    ax.legend(fontsize=8)
    save_fig(fig, "q4_rf_tree_correlation")
    return tab, ct


# --------------------------------------------------------------------------- #
# d) boosting: staged cross-validation
# --------------------------------------------------------------------------- #
def _staged_fold(name, lr, balanced, max_est, tr_idx, va_idx, Xtr, ytr):
    m = build(name, {"n_estimators": max_est, "learning_rate": lr, "balanced_weights": balanced})
    m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
    pipe = m.estimator_
    Z = pipe[0].transform(Xtr.iloc[va_idx])
    scores = {}
    for k, pred in enumerate(pipe[-1].staged_predict(Z), start=1):
        if k in C.BOOST_STAGE_GRID:
            scores[k] = f1_score(ytr[va_idx], pred, average="macro")
    return lr, balanced, scores


def tune_boosting(name, lrs, max_est, Xtr, ytr):
    folds = list(CV.split(Xtr, ytr))
    jobs = list(itertools.product(lrs, C.BOOST_SAMPLE_WEIGHTING, range(len(folds))))
    out = Parallel(n_jobs=C.N_JOBS)(delayed(_staged_fold)(name, lr, bal, max_est, *folds[f], Xtr, ytr)
                                    for lr, bal, f in jobs)
    rec = [{"learning_rate": lr, "balanced_weights": bal, "n_estimators": k, "score": s}
           for lr, bal, sc in out for k, s in sc.items()]
    tab = (pd.DataFrame(rec).groupby(["learning_rate", "balanced_weights", "n_estimators"])["score"]
           .agg(mean_test_score="mean", std_test_score="std").reset_index()
           .sort_values("mean_test_score", ascending=False))
    save_table(tab.head(15), f"q4_{name}_cv_top", f"Q4d {name}: top-15 configurations (staged 5-fold CV Macro-F1)")
    tab.to_csv(C.TABLES / f"q4_{name}_cv_results.csv", index=False)
    b = tab.iloc[0]
    params = {"n_estimators": int(b.n_estimators), "learning_rate": float(b.learning_rate),
              "balanced_weights": bool(b.balanced_weights)}
    return tab, params, float(b.mean_test_score), float(b.std_test_score)


def staged_val_curves(ada_params, gb_params, ada_tab, gb_tab, Xtr, ytr, Xva, yva):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rows = []
    for j, (name, lrs, best, tab, max_est) in enumerate(
            (("AdaBoost", C.ADA_LEARNING_RATES, ada_params, ada_tab, C.ADA_MAX_ESTIMATORS),
             ("GradientBoosting", C.GB_LEARNING_RATES, gb_params, gb_tab, C.GB_MAX_ESTIMATORS))):
        ax = axes[0, j]
        for lr in lrs:
            m = build(name, {"n_estimators": max_est, "learning_rate": lr,
                             "balanced_weights": best["balanced_weights"]}).fit(Xtr, ytr)
            pipe = m.estimator_
            Ztr, Zva = pipe[0].transform(Xtr), pipe[0].transform(Xva)
            tr_s = [f1_score(ytr, p, average="macro") for p in pipe[-1].staged_predict(Ztr)]
            va_s = [f1_score(yva, p, average="macro") for p in pipe[-1].staged_predict(Zva)]
            n = np.arange(1, len(va_s) + 1)
            line, = ax.plot(n, va_s, label=f"lr={lr} val")
            ax.plot(n, tr_s, ls=":", color=line.get_color(), label=f"lr={lr} train")
            for k in C.BOOST_STAGE_GRID:
                if k <= len(va_s):
                    rows.append({"model": name, "learning_rate": lr, "n_estimators": k,
                                 "train_macro_f1": tr_s[k - 1], "val_macro_f1": va_s[k - 1]})
        ax.set_xlabel("number of estimators (boosting stage)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{name}: staged train/validation Macro-F1 (balanced={best['balanced_weights']})")
        ax.legend(fontsize=7, ncol=2)
        ax = axes[1, j]
        for (lr, bal), s in tab.groupby(["learning_rate", "balanced_weights"]):
            s = s.sort_values("n_estimators")
            ax.errorbar(s.n_estimators, s.mean_test_score, yerr=s.std_test_score, marker="o", ms=3, capsize=2,
                        label=f"lr={lr}, bal={bal}")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("CV Macro-F1 (mean ± sd)")
        ax.set_title(f"{name}: 5-fold CV grid")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Q4d boosting: performance vs. ensemble size and learning rate")
    save_fig(fig, "q4_boosting_staged")
    save_table(pd.DataFrame(rows), "q4_boosting_staged_val", "Q4d staged train/validation Macro-F1 (fit on train)")


# --------------------------------------------------------------------------- #
# e) comparison & importances
# --------------------------------------------------------------------------- #
def _aggregate_importance(names, imp):
    """Sum one-hot column importances back to the original feature."""
    agg = {}
    for n, v in zip(names, imp):
        orig = next((c for c in C.CAT_FEATURES if n.startswith(c + "_")), n)
        agg[orig] = agg.get(orig, 0.0) + v
    return pd.Series(agg)


def compare_and_importance(configs, cv_scores, Xtr, ytr, Xva, yva):
    rows, fitted = [], {}
    for name, params in configs.items():
        m = build(name, params)
        with timer() as tf:
            m.fit(Xtr, ytr)
        with timer() as tp:
            pred = m.predict(Xva)
        met = cls_metrics(yva, pred, m.predict_proba(Xva)[:, 1])
        fitted[name] = m
        rows.append({"model": name, "cv_macro_f1": cv_scores[name][0], "cv_std": cv_scores[name][1],
                     "val_macro_f1": met["macro_f1"], "val_accuracy": met["accuracy"], "val_roc_auc": met["roc_auc"],
                     "val_pr_auc": met["pr_auc"], "val_recall_yes": met["recall_yes"],
                     "fit_time_s": tf["elapsed"], "predict_time_s": tp["elapsed"],
                     "n_jobs": "all cores" if name == "RandomForest" else "1"})
    comp = pd.DataFrame(rows)
    save_table(comp, "q4_comparison", "Q4e tree/ensemble models vs. best Q2 model (validation; times on this machine)")

    imp_rows = []
    tree_models = [n for n in ("DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting") if n in fitted]
    for name in tree_models:
        m = fitted[name]
        pipe = m.estimator_ if hasattr(m, "estimator_") else m
        builtin = _aggregate_importance(pipe[0].get_feature_names_out(), pipe[-1].feature_importances_)
        perm = permutation_importance(m, Xva, yva, scoring="f1_macro", n_repeats=5, random_state=C.SEED,
                                      n_jobs=C.N_JOBS)
        for j, f in enumerate(Xva.columns):
            imp_rows.append({"model": name, "feature": f, "builtin_impurity": builtin.get(f, 0.0),
                             "perm_mean_drop_macro_f1": perm.importances_mean[j],
                             "perm_std": perm.importances_std[j]})
    imp = pd.DataFrame(imp_rows)
    imp["builtin_rank"] = imp.groupby("model")["builtin_impurity"].rank(ascending=False)
    imp["perm_rank"] = imp.groupby("model")["perm_mean_drop_macro_f1"].rank(ascending=False)
    imp.to_csv(C.TABLES / "q4_importances.csv", index=False)
    rank_corr = imp.groupby("model").apply(
        lambda d: d["builtin_rank"].corr(d["perm_rank"], method="spearman"), include_groups=False).rename(
        "spearman_rank_corr").reset_index()
    save_table(rank_corr, "q4_importance_rank_corr", "Q4e Spearman correlation of built-in vs. permutation importance ranks")
    for name in tree_models:
        save_table(imp[imp.model == name].sort_values("perm_mean_drop_macro_f1", ascending=False).drop(columns="model"),
                   f"q4_importance_{name}", f"Q4e {name}: built-in (impurity) vs. permutation importance on validation")

    fig, axes = plt.subplots(len(tree_models), 2, figsize=(12, 3.4 * len(tree_models)))
    for i, name in enumerate(tree_models):
        d = imp[imp.model == name].sort_values("builtin_impurity")
        axes[i, 0].barh(d.feature, d.builtin_impurity, color="#4C72B0")
        axes[i, 0].set_title(f"{name}: built-in impurity importance (train)")
        d = imp[imp.model == name].sort_values("perm_mean_drop_macro_f1")
        axes[i, 1].barh(d.feature, d.perm_mean_drop_macro_f1, xerr=d.perm_std, color="#DD8452")
        axes[i, 1].set_title(f"{name}: permutation importance (validation, ΔMacro-F1)")
        for a in axes[i]:
            a.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save_fig(fig, "q4_importances")
    return comp, imp, fitted


def importance_control(rf_params, Xtr, ytr, Xva, yva):
    """Control experiment: add pure-noise features (continuous + high-cardinality ID)
    and a near-duplicate of 'age' to see how importances behave."""
    rng = np.random.default_rng(C.SEED)

    def add(X):
        X = X.copy()
        X["NOISE_continuous"] = rng.normal(size=len(X))
        X["NOISE_id_highcard"] = rng.integers(0, 5000, len(X)).astype(float)
        X["age_near_duplicate"] = X["age"] + rng.normal(0, 0.5, len(X))
        return X

    Xtr2, Xva2 = add(Xtr), add(Xva)
    num = C.NUM_FEATURES + ["NOISE_continuous", "NOISE_id_highcard", "age_near_duplicate"]
    m = Pipeline([("pre", make_preprocessor("tree", num, C.CAT_FEATURES)),
                  ("clf", RandomForestClassifier(**rf_params, random_state=C.SEED, n_jobs=C.N_JOBS))]).fit(Xtr2, ytr)
    builtin = _aggregate_importance(m[0].get_feature_names_out(), m[-1].feature_importances_)
    perm = permutation_importance(m, Xva2, yva, scoring="f1_macro", n_repeats=5, random_state=C.SEED, n_jobs=C.N_JOBS)
    t = pd.DataFrame({"feature": Xva2.columns, "builtin_impurity": [builtin.get(f, 0) for f in Xva2.columns],
                      "perm_mean_drop_macro_f1": perm.importances_mean, "perm_std": perm.importances_std})
    t = t.sort_values("builtin_impurity", ascending=False)
    t["builtin_rank"] = np.arange(1, len(t) + 1)
    save_table(t, "q4_importance_control", "Q4e control: RF importances with added noise / near-duplicate features")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in ((axes[0], "builtin_impurity", "built-in impurity (train)"),
                           (axes[1], "perm_mean_drop_macro_f1", "permutation on validation")):
        d = t.sort_values(col)
        colors = ["red" if ("NOISE" in f or "duplicate" in f) else "#4C72B0" for f in d.feature]
        ax.barh(d.feature, d[col], color=colors)
        ax.set_title(f"RF {title}")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Q4e misleading importance control (red = added noise / duplicate)")
    save_fig(fig, "q4_importance_control")
    return t


def main():
    Xtr, ytr, Xva, yva = load_train_val()

    gs_dt, dt_params = tune_tree(Xtr, ytr, Xva, yva)
    small_tree(dt_params, Xtr, ytr, Xva, yva)
    gs_rf, rf_params = tune_rf(Xtr, ytr)
    rf_tab, corr_tab = rf_curves(rf_params, Xtr, ytr, Xva, yva)
    ada_tab, ada_params, ada_cv, ada_sd = tune_boosting("AdaBoost", C.ADA_LEARNING_RATES, C.ADA_MAX_ESTIMATORS, Xtr, ytr)
    gb_tab, gb_params, gb_cv, gb_sd = tune_boosting("GradientBoosting", C.GB_LEARNING_RATES, C.GB_MAX_ESTIMATORS, Xtr, ytr)
    staged_val_curves(ada_params, gb_params, ada_tab, gb_tab, Xtr, ytr, Xva, yva)

    q2 = load_json(C.MODELS / "q2_best.json")
    q2_cfg = load_json(final_config_path(q2["best_q2_model"]))
    q2_cv = pd.read_csv(C.TABLES / "q2_summary.csv").set_index("model")
    configs = {"DecisionTree": dt_params, "RandomForest": rf_params, "AdaBoost": ada_params,
               "GradientBoosting": gb_params, q2["best_q2_model"]: q2_cfg["params"]}
    cv_scores = {"DecisionTree": (gs_dt.best_score_, gs_dt.cv_results_["std_test_score"][gs_dt.best_index_]),
                 "RandomForest": (gs_rf.best_score_, gs_rf.cv_results_["std_test_score"][gs_rf.best_index_]),
                 "AdaBoost": (ada_cv, ada_sd), "GradientBoosting": (gb_cv, gb_sd),
                 q2["best_q2_model"]: (q2_cv.loc[q2["best_q2_model"], "cv_macro_f1"],
                                       q2_cv.loc[q2["best_q2_model"], "cv_macro_f1_std"])}
    for name in ("DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting"):
        save_json({"name": name, "params": configs[name], "source": "Q4", "cv_macro_f1": cv_scores[name][0]},
                  final_config_path(name))
    save_table(pd.DataFrame([{"model": k, "params": str(v)} for k, v in configs.items()]), "q4_best_params",
               "Q4 selected hyper-parameters (by CV Macro-F1 on train)")
    comp, imp, _ = compare_and_importance(configs, cv_scores, Xtr, ytr, Xva, yva)
    ctrl = importance_control(rf_params, Xtr, ytr, Xva, yva)

    pd.set_option("display.width", 220)
    print(comp.to_string(index=False))
    print(configs)
    print(corr_tab.to_string(index=False))
    print(ctrl.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
