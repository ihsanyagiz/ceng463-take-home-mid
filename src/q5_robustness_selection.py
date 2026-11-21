"""Q5 a-d - Repeated CV of frozen candidates, robustness study, paired bootstrap,
multi-criteria selection + sensitivity analysis. NO test-set access here.

Run: python -m src.q5_robustness_selection
"""
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from . import config as C
from .data_cls import load_train_val
from .models import build, final_config_path
from .utils import load_json, macro_f1_binary_from_counts, save_fig, save_json, save_table, timer

CANDIDATES = ["LogReg", "NaiveBayes", "kNN", "DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting"]
CONTINUOUS = ["age", "balance", "day_of_week", "campaign", "pdays", "previous"]  # binary flag left untouched


def load_configs():
    return {n: load_json(final_config_path(n))["params"] for n in CANDIDATES}


# --------------------------------------------------------------------------- #
# a) repeated CV
# --------------------------------------------------------------------------- #
def repeated_cv(configs, Xtr, ytr):
    rcv = RepeatedStratifiedKFold(n_splits=C.CV_FOLDS, n_repeats=C.REPEATED_CV_REPEATS, random_state=C.SEED)
    rows, folds = [], []
    for name, p in configs.items():
        r = cross_validate(build(name, p), Xtr, ytr, cv=rcv, scoring="f1_macro",
                           n_jobs=1 if name == "RandomForest" else C.N_JOBS)
        s = r["test_score"]
        rows.append({"model": name, "cv_mean_macro_f1": s.mean(), "cv_std_macro_f1": s.std(ddof=1),
                     "cv_min": s.min(), "cv_max": s.max(), "n_folds": len(s)})
        folds += [{"model": name, "fold": i, "macro_f1": v} for i, v in enumerate(s)]
    tab = pd.DataFrame(rows).sort_values("cv_mean_macro_f1", ascending=False)
    fold_df = pd.DataFrame(folds)
    fold_df.to_csv(C.TABLES / "q5_repeated_cv_folds.csv", index=False)
    save_table(tab, "q5_repeated_cv", f"Q5a repeated stratified CV ({C.CV_FOLDS}x{C.REPEATED_CV_REPEATS}) Macro-F1 on train")
    fig, ax = plt.subplots(figsize=(9, 4))
    order = list(tab.model)
    ax.boxplot([fold_df[fold_df.model == m].macro_f1 for m in order], tick_labels=order)
    ax.set_ylabel("Macro-F1 per fold")
    ax.set_title(f"Q5a fold-wise Macro-F1 ({C.CV_FOLDS}-fold x {C.REPEATED_CV_REPEATS} repeats, frozen configs)")
    ax.tick_params(axis="x", labelsize=8)
    save_fig(fig, "q5_repeated_cv_box")
    return tab, fold_df


def fit_all(configs, Xtr, ytr, Xva):
    fitted, cost = {}, []
    for name, p in configs.items():
        m = build(name, p)
        with timer() as tf:
            m.fit(Xtr, ytr)
        with timer() as tp:
            m.predict(Xva)
        fitted[name] = m
        cost.append({"model": name, "fit_time_s": tf["elapsed"],
                     "predict_ms_per_1000": 1000 * tp["elapsed"] / len(Xva) * 1000})
    return fitted, pd.DataFrame(cost)


# --------------------------------------------------------------------------- #
# b) robustness on validation
# --------------------------------------------------------------------------- #
def corrupt(X, scenario, severity, rep, train_sd):
    rng = np.random.default_rng(C.SEED + 1000 * rep + int(severity * 100))
    X = X.copy()
    if severity == 0:
        return X
    if scenario == "gaussian_noise":
        for c in CONTINUOUS:
            X[c] = X[c].astype(float) + rng.normal(0, severity * train_sd[c], len(X))
    elif scenario == "missing_features":
        cols = C.NUM_FEATURES + C.CAT_FEATURES
        mask = rng.random((len(X), len(cols))) < severity
        for j, c in enumerate(cols):
            X[c] = X[c].astype(float if c in C.NUM_FEATURES else object)
            X.loc[mask[:, j], c] = np.nan
    return X


def robustness(fitted, Xtr, Xva, yva):
    train_sd = Xtr[CONTINUOUS].std()
    rows = []
    for scenario, sevs in (("gaussian_noise", C.NOISE_SEVERITIES), ("missing_features", C.MISSING_RATES)):
        for sev in sevs:
            for rep in range(C.ROBUSTNESS_REPEATS if sev > 0 else 1):
                Xc = corrupt(Xva, scenario, sev, rep, train_sd)  # identical corrupted copy for every model
                for name, m in fitted.items():
                    rows.append({"scenario": scenario, "severity": sev, "repeat": rep, "model": name,
                                 "macro_f1": f1_score(yva, m.predict(Xc), average="macro")})
    raw = pd.DataFrame(rows)
    raw.to_csv(C.TABLES / "q5_robustness_raw.csv", index=False)
    summ = raw.groupby(["scenario", "model", "severity"])["macro_f1"].agg(["mean", "std"]).reset_index()
    wide = summ.pivot_table(index=["scenario", "model"], columns="severity", values="mean").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    save_table(wide, "q5_robustness", "Q5b validation Macro-F1 under corruption (mean of 5 repeats); "
               "noise severity = SD multiple of training SD; missing = cell rate")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (scenario, xl) in zip(axes, (("gaussian_noise", "noise SD (x training SD), continuous features"),
                                         ("missing_features", "fraction of feature values set missing"))):
        s = summ[summ.scenario == scenario]
        for name in fitted:
            d = s[s.model == name]
            ax.errorbar(d.severity, d["mean"], yerr=d["std"].fillna(0), marker="o", ms=3, capsize=2, label=name)
        ax.set_xlabel(xl)
        ax.set_ylabel("validation Macro-F1")
        ax.set_title(scenario)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Q5b robustness: Macro-F1 vs. corruption severity (same corrupted data for all models)")
    save_fig(fig, "q5_robustness")

    # robustness score per model: Macro-F1 retained at max severity, averaged over scenarios
    ret = []
    for name in fitted:
        r = []
        for scenario, sevs in (("gaussian_noise", C.NOISE_SEVERITIES), ("missing_features", C.MISSING_RATES)):
            s = summ[(summ.scenario == scenario) & (summ.model == name)].set_index("severity")["mean"]
            r.append(s[max(sevs)] / s[0.0])
            ret.append({"model": name, "scenario": scenario, "clean": s[0.0], "max_severity": s[max(sevs)],
                        "retention": s[max(sevs)] / s[0.0], "mean_over_severities": s.mean()})
    ret = pd.DataFrame(ret)
    save_table(ret, "q5_robustness_retention", "Q5b Macro-F1 retention at maximum severity (max / clean); "
               "informative only, not used as MCDA criterion")

    # MCDA robustness criterion: absolute Macro-F1 at the highest Gaussian-noise severity.
    # Also reported: normalised AUC of each degradation curve (trapezoidal area / severity range
    # = average Macro-F1 over the severity range).
    crit = []
    for name in fitted:
        rec = {"model": name}
        for scenario, sevs in (("gaussian_noise", C.NOISE_SEVERITIES), ("missing_features", C.MISSING_RATES)):
            s = summ[(summ.scenario == scenario) & (summ.model == name)].set_index("severity")["mean"].sort_index()
            x = s.index.to_numpy(float)
            rec[f"auc_{scenario}"] = np.trapezoid(s.to_numpy(), x) / (x.max() - x.min())
            if scenario == "gaussian_noise":
                rec["macro_f1_max_noise"] = s[max(sevs)]
        crit.append(rec)
    rob = pd.DataFrame(crit).set_index("model")[["macro_f1_max_noise", "auc_gaussian_noise", "auc_missing_features"]]
    save_table(rob.reset_index(), "q5_robustness_criteria",
               f"Q5b robustness criteria: validation Macro-F1 at the highest noise severity "
               f"(sigma={max(C.NOISE_SEVERITIES)}) and normalised AUC of the degradation curves")
    return summ, rob


# --------------------------------------------------------------------------- #
# c) paired bootstrap
# --------------------------------------------------------------------------- #
def paired_bootstrap(name_a, name_b, fitted, Xva, yva):
    pa, pb = fitted[name_a].predict(Xva), fitted[name_b].predict(Xva)
    y = yva.astype(int)
    n = len(y)
    rng = np.random.default_rng(C.SEED)
    deltas = np.empty(C.BOOTSTRAP_B)
    fa = fb = None
    ind = {}
    for tag, p in (("a", pa), ("b", pb)):
        ind[tag] = {"tp": ((p == 1) & (y == 1)), "fp": ((p == 1) & (y == 0)), "fn": ((p == 0) & (y == 1)),
                    "tn": ((p == 0) & (y == 0))}
    chunk = 500
    for start in range(0, C.BOOTSTRAP_B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, C.BOOTSTRAP_B - start), n))  # sampling unit: validation example
        res = {}
        for tag in ("a", "b"):
            c = {k: v[idx].sum(1).astype(float) for k, v in ind[tag].items()}
            res[tag] = macro_f1_binary_from_counts(c["tp"], c["fp"], c["fn"], c["tn"])
        deltas[start:start + len(idx)] = res["a"] - res["b"]
    obs = f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    out = {"model_A": name_a, "model_B": name_b, "statistic": "Macro-F1(A) - Macro-F1(B) on validation",
           "sampling_unit": "validation example (paired: same resampled indices for A and B)",
           "B": C.BOOTSTRAP_B, "seed": C.SEED, "n_validation": n, "observed_delta": obs,
           "ci95_percentile": [lo, hi], "bootstrap_mean": deltas.mean(), "bootstrap_sd": deltas.std(ddof=1),
           "frac_delta_gt_0": float((deltas > 0).mean()),
           "macro_f1_A": f1_score(y, pa, average="macro"), "macro_f1_B": f1_score(y, pb, average="macro"),
           "n_disagreements": int((pa != pb).sum())}
    save_json(out, C.RESULTS / "q5_bootstrap.json")
    save_table(pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in out.items()
                              if k not in ("statistic", "sampling_unit")}]).T.reset_index()
               .rename(columns={"index": "quantity", 0: "value"}), "q5_bootstrap",
               "Q5c paired bootstrap of the Macro-F1 difference (validation)")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(deltas, bins=80, color="#4C72B0")
    ax.axvline(obs, color="k", label=f"observed Δ={obs:.4f}")
    ax.axvline(lo, color="r", ls="--", label=f"95% CI [{lo:.4f}, {hi:.4f}]")
    ax.axvline(hi, color="r", ls="--")
    ax.axvline(0, color="grey", ls=":")
    ax.set_xlabel(f"Macro-F1({name_a}) - Macro-F1({name_b})")
    ax.set_ylabel("bootstrap count")
    ax.set_title(f"Q5c paired bootstrap, B={C.BOOTSTRAP_B}, unit = validation example")
    ax.legend(fontsize=8)
    save_fig(fig, "q5_bootstrap")
    return out


# --------------------------------------------------------------------------- #
# d) multi-criteria decision analysis
# --------------------------------------------------------------------------- #
CRITERIA = {  # criterion: (column, higher_is_better, log10-scale before min-max)
    "performance": ("cv_mean_macro_f1", True, False),
    "stability": ("cv_std_macro_f1", False, False),
    "robustness": ("macro_f1_max_noise", True, False),
    "interpretability": ("interpretability_score", True, False),
    # costs span orders of magnitude -> log10 so one slow model does not compress all others to ~1
    "train_cost": ("fit_time_s", False, True),
    "inference_cost": ("predict_ms_per_1000", False, True),
    "implementation": ("implementation_simplicity", True, False),
}


def mcda(cv_tab, rob, cost):
    raw = cv_tab.set_index("model")[["cv_mean_macro_f1", "cv_std_macro_f1"]].join(rob).join(cost.set_index("model"))
    raw["interpretability_score"] = pd.Series(C.INTERPRETABILITY_SCORE)
    raw["implementation_simplicity"] = pd.Series(C.IMPLEMENTATION_SIMPLICITY)
    norm = pd.DataFrame(index=raw.index)
    for crit, (col, hib, use_log) in CRITERIA.items():
        v = raw[col].astype(float)
        if use_log:
            v = np.log10(v)
        rng = v.max() - v.min()
        z = (v - v.min()) / rng if rng > 0 else v * 0 + 1.0
        norm[crit] = z if hib else 1 - z
    W = pd.Series(C.MCDA_BASE_WEIGHTS)
    assert abs(W.sum() - 1) < 1e-9
    base_score = norm[W.index] @ W
    table = raw.join(norm.add_prefix("norm_")).assign(weighted_score=base_score).sort_values("weighted_score",
                                                                                                ascending=False)
    save_table(table.reset_index().rename(columns={"index": "model"}), "q5_mcda",
               "Q5d multi-criteria table: raw values, min-max normalised scores (1 = best) and weighted score")
    save_table(W.rename("weight").reset_index().rename(columns={"index": "criterion"}), "q5_mcda_weights",
               "Q5d base criterion weights")

    schemes = {"base": W}
    schemes["equal"] = pd.Series(1 / len(W), index=W.index)
    for focus in ["performance", "robustness", "interpretability"]:
        w = pd.Series(0.4 / (len(W) - 1), index=W.index)
        w[focus] = 0.6
        schemes[f"{focus}-heavy (0.6)"] = w
    w = pd.Series(0.0, index=W.index)
    w[["train_cost", "inference_cost"]] = 0.3
    w[["performance", "robustness"]] = 0.2
    schemes["cost-heavy"] = w
    schemes["performance only"] = pd.Series({k: (1.0 if k == "performance" else 0.0) for k in W.index})
    for drop in W.index:  # leave-one-criterion-out
        w = W.copy()
        w[drop] = 0
        schemes[f"base without {drop}"] = w / w.sum()
    ranks, winners = {}, []
    for sname, w in schemes.items():
        s = norm[w.index] @ w
        ranks[sname] = s.rank(ascending=False, method="min")
        winners.append({"scheme": sname, "winner": s.idxmax(), "winner_score": s.max(),
                        "runner_up": s.drop(s.idxmax()).idxmax(), **{f"w_{k}": v for k, v in w.items()}})
    rank_df = pd.DataFrame(ranks)
    win_df = pd.DataFrame(winners)
    save_table(win_df[["scheme", "winner", "winner_score", "runner_up"]], "q5_sensitivity_schemes",
               "Q5d sensitivity: winner under alternative weight schemes")
    win_df.to_csv(C.TABLES / "q5_sensitivity_schemes_full.csv", index=False)

    # Two Dirichlet analyses: uniform over the simplex (alpha = 1) and centred on the base
    # weights (alpha = 20 * w, i.e. mean = w with moderate spread).
    freqs = {}
    for label, alpha in (("uniform_alpha1", np.ones(len(W))),
                         (f"centered_alpha{C.DIRICHLET_CENTER_CONC:g}w", C.DIRICHLET_CENTER_CONC * W.values)):
        rng = np.random.default_rng(C.SEED)
        D = rng.dirichlet(alpha, C.N_DIRICHLET)
        S = norm[W.index].values @ D.T  # models x draws
        freqs[f"win_freq_{label}"] = pd.Series(np.bincount(S.argmax(0), minlength=len(norm)),
                                               index=norm.index) / C.N_DIRICHLET
    freq = pd.DataFrame(freqs)
    freq = freq.sort_values(freq.columns[1], ascending=False).reset_index().rename(columns={"index": "model"})
    save_table(freq, "q5_dirichlet_wins",
               f"Q5d winner frequency over {C.N_DIRICHLET} Dirichlet weight vectors: uniform (alpha=1) and "
               f"centred on the base weights (alpha={C.DIRICHLET_CENTER_CONC:g}*w)")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(rank_df.loc[table.index], annot=True, fmt=".0f", cmap="viridis_r", cbar_kws={"label": "rank"}, ax=ax)
    ax.set_title("Q5d model rank under each weighting scheme (1 = best)")
    ax.tick_params(axis="x", labelsize=7)
    save_fig(fig, "q5_sensitivity_heatmap")
    return table, win_df, freq, base_score.idxmax()


def main():
    Xtr, ytr, Xva, yva = load_train_val()
    configs = load_configs()
    cv_tab, _ = repeated_cv(configs, Xtr, ytr)
    fitted, cost = fit_all(configs, Xtr, ytr, Xva)
    save_table(cost, "q5_cost", "Q5d training time (s, full train) and inference time (ms per 1000 rows, validation)")
    summ, rob = robustness(fitted, Xtr, Xva, yva)
    top2 = list(cv_tab.model[:2])  # two best by repeated-CV mean (not by validation/test)
    boot = paired_bootstrap(top2[0], top2[1], fitted, Xva, yva)
    table, win_df, freq, winner = mcda(cv_tab, rob, cost)

    # freeze configs + selection rule (hash allows verifying nothing changed before the test run)
    frozen = {"timestamp": datetime.now().isoformat(timespec="seconds"), "candidates": configs,
              "selection_rule": "highest weighted MCDA score using config.MCDA_BASE_WEIGHTS (criteria computed "
                                "from train CV and validation only)",
              "selected_model": winner, "base_weights": C.MCDA_BASE_WEIGHTS, "refit_on": C.FINAL_REFIT_ON}
    frozen["config_hash"] = hashlib.sha256(json.dumps(configs, sort_keys=True, default=str).encode()).hexdigest()
    save_json(frozen, C.MODELS / "selection_rule.json")

    pd.set_option("display.width", 220)
    print(cv_tab.to_string(index=False))
    print(cost.to_string(index=False))
    print(rob.to_string())
    print({k: boot[k] for k in ("model_A", "model_B", "observed_delta", "ci95_percentile", "frac_delta_gt_0")})
    print(table[["weighted_score"]].to_string())
    print(win_df[["scheme", "winner"]].to_string(index=False))
    print(freq.to_string(index=False))
    print("SELECTED (base weights):", winner)


if __name__ == "__main__":
    main()
