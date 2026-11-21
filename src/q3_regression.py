"""Q3 - Regression on California Housing: audit, baselines, linear/regularised,
non-linear models, complexity & learning curves, Ridge/Lasso paths, residuals.

Target unit: $100,000 (MedHouseVal). Run: python -m src.q3_regression
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import config as C
from .data_reg import TARGET, load_reg, load_reg_train_val, reg_split_sizes
from .utils import save_fig, save_json, save_table, timer, write_macros

CV = KFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
RMSE = "neg_root_mean_squared_error"


def reg_metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p)}


# --------------------------------------------------------------------------- #
# a) audit (train rows only for distribution/correlation statistics)
# --------------------------------------------------------------------------- #
def audit(Xtr, ytr):
    df_all = load_reg()
    aud = []
    tr = Xtr.assign(**{TARGET: ytr})
    for col in tr.columns:
        s = tr[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        aud.append({"feature": col, "n_missing_all_rows": int(df_all[col].isna().sum()), "mean": s.mean(),
                    "std": s.std(), "min": s.min(), "median": s.median(), "max": s.max(), "skew": s.skew(),
                    "n_iqr_outliers_train": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
    aud = pd.DataFrame(aud)
    save_table(aud, "q3_audit", "Q3a California Housing audit (statistics on training rows)", float_fmt="%.3f")
    cap = pd.DataFrame([{"rows_at_or_above_5.0_train": int((ytr >= 5.0).sum()), "pct_train": 100 * (ytr >= 5.0).mean(),
                         "n_duplicates_all_rows": int(df_all.duplicated().sum())}])
    save_table(cap, "q3_target_cap", "Q3a capped target values (MedHouseVal >= 5.0) and duplicates", float_fmt="%.2f")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(ytr, bins=80, color="#4C72B0")
    ax.axvline(5.0, color="red", ls="--", label="cap = 5.0")
    ax.set_xlabel("MedHouseVal ($100k)")
    ax.set_ylabel("count")
    ax.set_title("Q3a target distribution (train)")
    ax.legend()
    save_fig(fig, "q3_target_hist")

    corr = tr.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Q3a Pearson correlation (train)")
    save_fig(fig, "q3_corr_heatmap")
    pairs = [{"feature_1": a, "feature_2": b, "r": corr.loc[a, b]} for i, a in enumerate(corr.columns)
             for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) > 0.5]
    pairs = pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)
    save_table(pairs, "q3_high_corr_pairs", "Q3a predictor pairs with |r| > 0.5 (train); |r| > 0.8 is highly correlated",
               float_fmt="%.3f")
    return aud, pairs


# --------------------------------------------------------------------------- #
# b) models
# --------------------------------------------------------------------------- #
def scaled(reg):
    return Pipeline([("scale", StandardScaler()), ("reg", reg)])


def fit_models(Xtr, ytr, Xva, yva):
    tuned = {}
    gs_r = GridSearchCV(scaled(Ridge()), {"reg__alpha": C.REG_ALPHAS}, scoring=RMSE, cv=CV, n_jobs=C.N_JOBS).fit(Xtr, ytr)
    gs_l = GridSearchCV(scaled(Lasso(max_iter=50000)), {"reg__alpha": C.REG_LASSO_ALPHAS}, scoring=RMSE, cv=CV,
                        n_jobs=C.N_JOBS).fit(Xtr, ytr)
    gs_dt = GridSearchCV(scaled(DecisionTreeRegressor(random_state=C.SEED)), C.REG_DT_GRID, scoring=RMSE, cv=CV,
                         n_jobs=C.N_JOBS).fit(Xtr, ytr)
    gs_rf = GridSearchCV(scaled(RandomForestRegressor(C.REG_RF_N_ESTIMATORS, random_state=C.SEED, n_jobs=1)),
                         C.REG_RF_GRID, scoring=RMSE, cv=CV, n_jobs=C.N_JOBS).fit(Xtr, ytr)
    for name, gs in (("Ridge", gs_r), ("Lasso", gs_l), ("DecisionTree", gs_dt), ("RandomForest", gs_rf)):
        tuned[name] = gs
        t = pd.DataFrame(gs.cv_results_)
        t = t[[c for c in t.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "rank_test_score"]]
        t["mean_cv_rmse"] = -t.pop("mean_test_score")
        t.rename(columns=lambda c: c.replace("param_reg__", "")).sort_values("rank_test_score").to_csv(
            C.TABLES / f"q3_cv_{name}.csv", index=False)

    models = {
        "Dummy(mean)": (scaled(DummyRegressor()), {}),
        "OLS": (scaled(LinearRegression()), {}),
        "Ridge": (scaled(Ridge(alpha=gs_r.best_params_["reg__alpha"])), {"alpha": gs_r.best_params_["reg__alpha"]}),
        "Lasso": (scaled(Lasso(alpha=gs_l.best_params_["reg__alpha"], max_iter=50000)),
                  {"alpha": gs_l.best_params_["reg__alpha"]}),
        "DecisionTree": (scaled(DecisionTreeRegressor(random_state=C.SEED, **{k.replace("reg__", ""): v for k, v in
                                                                               gs_dt.best_params_.items()})),
                         {k.replace("reg__", ""): v for k, v in gs_dt.best_params_.items()}),
        "RandomForest": (scaled(RandomForestRegressor(C.REG_RF_N_ESTIMATORS, random_state=C.SEED, n_jobs=C.N_JOBS,
                                                      **{k.replace("reg__", ""): v for k, v in
                                                         gs_rf.best_params_.items()})),
                         {"n_estimators": C.REG_RF_N_ESTIMATORS,
                          **{k.replace("reg__", ""): v for k, v in gs_rf.best_params_.items()}}),
    }
    rows, fitted, preds = [], {}, {}
    for name, (m, params) in models.items():
        params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}
        cv_rmse = (-tuned[name].best_score_) if name in tuned else -cross_val_score(
            m, Xtr, ytr, scoring=RMSE, cv=CV).mean()
        with timer() as t:
            m.fit(Xtr, ytr)
        ptr, pva = m.predict(Xtr), m.predict(Xva)
        fitted[name], preds[name] = m, pva
        rows.append({"model": name, "params": str(params), "cv_RMSE": cv_rmse,
                     **{f"train_{k}": v for k, v in reg_metrics(ytr, ptr).items()},
                     **{f"val_{k}": v for k, v in reg_metrics(yva, pva).items()}, "fit_time_s": t["elapsed"]})
        save_json({"name": name, "params": params, "source": "Q3"}, C.MODELS / f"final_reg_{name.replace('(mean)', '')}.json")
    res = pd.DataFrame(rows)
    save_table(res.drop(columns="params"), "q3_model_comparison",
               "Q3b/e regression results. MAE and RMSE in units of $100,000; R2 unitless")
    save_table(res[["model", "params"]], "q3_best_params", "Q3b selected hyper-parameters (5-fold CV RMSE on train)")
    return res, fitted, preds, gs_r, gs_l


# --------------------------------------------------------------------------- #
# c) complexity and learning curves
# --------------------------------------------------------------------------- #
def complexity_curves(Xtr, ytr):
    poly = Pipeline([("poly", PolynomialFeatures(include_bias=False)), ("scale", StandardScaler()),
                     ("reg", Ridge(alpha=1e-3))])
    tr_s, va_s = validation_curve(poly, Xtr, ytr, param_name="poly__degree", param_range=C.REG_POLY_DEGREES,
                                  cv=CV, scoring=RMSE, n_jobs=C.N_JOBS)
    dt = DecisionTreeRegressor(random_state=C.SEED)
    tr_d, va_d = validation_curve(dt, Xtr, ytr, param_name="max_depth", param_range=C.REG_TREE_DEPTHS, cv=CV,
                                  scoring=RMSE, n_jobs=C.N_JOBS)
    rows = []
    for kind, rng, a, b in (("poly_degree (Ridge alpha=1e-3)", C.REG_POLY_DEGREES, tr_s, va_s),
                            ("tree_max_depth", C.REG_TREE_DEPTHS, tr_d, va_d)):
        for v, t_, w_ in zip(rng, -a, -b):
            rows.append({"curve": kind, "complexity": v, "train_rmse_mean": t_.mean(), "train_rmse_std": t_.std(),
                         "cv_val_rmse_mean": w_.mean(), "cv_val_rmse_std": w_.std()})
    tab = pd.DataFrame(rows)
    save_table(tab, "q3_complexity_curves", "Q3c complexity curves: train vs. CV-validation RMSE ($100k)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind in zip(axes, tab.curve.unique()):
        s = tab[tab.curve == kind]
        for col, lab in (("train_rmse", "train"), ("cv_val_rmse", "CV validation")):
            ax.errorbar(s.complexity, s[f"{col}_mean"], yerr=s[f"{col}_std"], marker="o", ms=3, capsize=2, label=lab)
        ax.set_xlabel(kind)
        ax.set_ylabel("RMSE ($100k)")
        ax.legend()
    axes[0].set_yscale("log")
    axes[0].set_title("Polynomial degree (log y-axis)")
    axes[1].set_title("Decision-tree depth")
    fig.suptitle("Q3c under-/over-fitting: training vs. validation error")
    save_fig(fig, "q3_complexity_curves")

    # learning curve for two models of different capacity
    sizes = np.linspace(0.05, 1.0, 10)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    lrows = []
    for name, est in (("OLS", scaled(LinearRegression())),
                      ("DecisionTree depth=None", DecisionTreeRegressor(random_state=C.SEED))):
        n, tr, va = learning_curve(est, Xtr, ytr, train_sizes=sizes, cv=CV, scoring=RMSE, n_jobs=C.N_JOBS,
                                   shuffle=True, random_state=C.SEED)
        ax.plot(n, -tr.mean(1), "o--", ms=3, label=f"{name} train")
        ax.plot(n, -va.mean(1), "o-", ms=3, label=f"{name} CV val")
        for i in range(len(n)):
            lrows.append({"model": name, "n_train": n[i], "train_rmse": -tr[i].mean(), "cv_val_rmse": -va[i].mean()})
    ax.set_xlabel("training examples")
    ax.set_ylabel("RMSE ($100k)")
    ax.set_title("Q3c learning curves")
    ax.legend(fontsize=8)
    save_fig(fig, "q3_learning_curves")
    pd.DataFrame(lrows).to_csv(C.TABLES / "q3_learning_curves.csv", index=False)
    return tab


# --------------------------------------------------------------------------- #
# d) Ridge / Lasso regularisation paths
# --------------------------------------------------------------------------- #
def regularisation(Xtr, ytr, gs_r, gs_l):
    feats = list(Xtr.columns)
    Z = StandardScaler().fit_transform(Xtr)
    paths = {"Ridge": [], "Lasso": []}
    for a in C.REG_ALPHAS:
        paths["Ridge"].append(Ridge(alpha=a).fit(Z, ytr).coef_)
    for a in C.REG_LASSO_ALPHAS:
        paths["Lasso"].append(Lasso(alpha=a, max_iter=50000).fit(Z, ytr).coef_)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (name, alphas, best) in zip(axes, (("Ridge", C.REG_ALPHAS, gs_r.best_params_["reg__alpha"]),
                                               ("Lasso", C.REG_LASSO_ALPHAS, gs_l.best_params_["reg__alpha"]))):
        P = np.array(paths[name])
        for j, f in enumerate(feats):
            ls = "-" if f in ("AveRooms", "AveBedrms", "Latitude", "Longitude") else ":"
            ax.plot(alphas, P[:, j], ls, label=f)
        ax.axvline(best, color="k", lw=0.8, ls="--", label=f"CV best alpha={best:.3g}")
        ax.set_xscale("log")
        ax.set_xlabel("alpha (log)")
        ax.set_ylabel("standardised coefficient")
        ax.set_title(f"{name} coefficient path")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Q3d coefficient paths (solid = correlated pairs AveRooms/AveBedrms, Latitude/Longitude)")
    save_fig(fig, "q3_coef_paths")

    for name, gs in (("Ridge", gs_r), ("Lasso", gs_l)):
        t = pd.DataFrame(gs.cv_results_)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.errorbar(t.param_reg__alpha.astype(float), -t.mean_test_score, yerr=t.std_test_score, marker="o", ms=3)
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_ylabel("CV RMSE ($100k)")
        ax.set_title(f"Q3d {name}: CV RMSE vs alpha")
        save_fig(fig, f"q3_cv_alpha_{name}")

    ols = LinearRegression().fit(Z, ytr).coef_
    ridge = Ridge(alpha=gs_r.best_params_["reg__alpha"]).fit(Z, ytr).coef_
    lasso = Lasso(alpha=gs_l.best_params_["reg__alpha"], max_iter=50000).fit(Z, ytr).coef_
    # stronger Lasso for a sparsity illustration (alpha=0.05)
    lasso_strong = Lasso(alpha=0.05, max_iter=50000).fit(Z, ytr).coef_
    coef = pd.DataFrame({"feature": feats, "OLS": ols, "Ridge(best)": ridge, "Lasso(best)": lasso,
                         "Lasso(alpha=0.05)": lasso_strong})
    save_table(coef, "q3_coefficients", f"Q3d standardised coefficients. Ridge alpha={gs_r.best_params_['reg__alpha']:.4g}, "
               f"Lasso alpha={gs_l.best_params_['reg__alpha']:.4g}")
    nz = pd.DataFrame([{"model": c, "n_zero_coefs": int((np.abs(coef[c]) < 1e-10).sum()),
                        "L1_norm": np.abs(coef[c]).sum(), "L2_norm": np.sqrt((coef[c] ** 2).sum())}
                       for c in coef.columns[1:]])
    save_table(nz, "q3_sparsity", "Q3d sparsity and coefficient norms")

    # instability of correlated pairs: coefficients across 20 bootstrap resamples of train
    rng = np.random.default_rng(C.SEED)
    boot = []
    for b in range(20):
        ix = rng.integers(0, len(Z), len(Z))
        for name, est in (("OLS", LinearRegression()), ("Ridge", Ridge(alpha=gs_r.best_params_["reg__alpha"])),
                          ("Lasso", Lasso(alpha=gs_l.best_params_["reg__alpha"], max_iter=50000))):
            c = est.fit(Z[ix], ytr[ix]).coef_
            boot.append({"model": name, "b": b, **dict(zip(feats, c))})
    boot = pd.DataFrame(boot)
    st = boot.groupby("model")[["AveRooms", "AveBedrms", "Latitude", "Longitude"]].agg(["mean", "std"])
    st.columns = [f"{a}_{b}" for a, b in st.columns]
    save_table(st.reset_index(), "q3_corr_coef_stability",
               "Q3d bootstrap (20 resamples of train) mean/std of coefficients of correlated predictors", float_fmt="%.3f")
    return coef, nz


# --------------------------------------------------------------------------- #
# e) residual analysis
# --------------------------------------------------------------------------- #
def residuals(fitted, preds, Xva, yva, res):
    nonlin = res[res.model.isin(["DecisionTree", "RandomForest"])].sort_values("cv_RMSE").iloc[0]["model"]
    show = ["Ridge", nonlin]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for j, name in enumerate(show):
        p = preds[name]
        r = yva - p
        ax = axes[0, j]
        ax.scatter(yva, p, s=4, alpha=0.3)
        ax.plot([0, 5.2], [0, 5.2], "r--", lw=1)
        ax.set_xlabel("actual ($100k)")
        ax.set_ylabel("predicted ($100k)")
        ax.set_title(f"{name}: predicted vs. actual (validation)")
        ax = axes[1, j]
        ax.scatter(p, r, s=4, alpha=0.3)
        ax.axhline(0, color="r", lw=1)
        ax.set_xlabel("predicted ($100k)")
        ax.set_ylabel("residual = actual - predicted")
        ax.set_title(f"{name}: residuals vs. predicted (validation)")
    fig.suptitle("Q3e diagnostic plots")
    save_fig(fig, "q3_residual_plots")

    p = preds[nonlin]
    d = Xva.copy()
    d["actual"] = yva
    d["predicted"] = p
    d["residual"] = yva - p
    d["abs_residual"] = d.residual.abs()
    d["ridge_pred"] = preds["Ridge"]
    top = d.sort_values("abs_residual", ascending=False).head(10)
    top.index.name = "row_id"
    save_table(top.reset_index().drop(columns="abs_residual"), "q3_top_residuals",
               f"Q3e ten largest validation residuals of {nonlin} ($100k)", float_fmt="%.3f")
    # residual summary by actual-value band (incl. capped rows)
    d["actual_band"] = pd.cut(d.actual, [0, 1, 2, 3, 4, 4.99, 5.1]).astype(str)
    band = d.groupby("actual_band").agg(n=("residual", "size"), mean_residual=("residual", "mean"),
                                        mae=("abs_residual", "mean")).reset_index()
    save_table(band, "q3_residual_by_band", f"Q3e residuals of {nonlin} by actual-value band (5.0 = capped)",
               float_fmt="%.3f")
    return nonlin


def main():
    Xtr, ytr, Xva, yva = load_reg_train_val()
    sizes = reg_split_sizes()
    save_table(sizes, "q3_split_sizes", "Q3a partition sizes (random 70/15/15, seed 42)")
    audit(Xtr, ytr)
    res, fitted, preds, gs_r, gs_l = fit_models(Xtr, ytr, Xva, yva)
    complexity_curves(Xtr, ytr)
    coef, nz = regularisation(Xtr, ytr, gs_r, gs_l)
    nonlin = residuals(fitted, preds, Xva, yva, res)
    n = dict(zip(sizes.partition, sizes.n))
    write_macros({"RegNTrain": n["train"], "RegNVal": n["val"], "RegNTest": n["test"]}, "q3")
    pd.set_option("display.width", 200)
    print(res.drop(columns="params").to_string(index=False))
    print(res[["model", "params"]].to_string(index=False))
    print(nz.to_string(index=False))
    print("residual analysis model:", nonlin)


if __name__ == "__main__":
    main()
