"""Q1 - Data audit, partitioning, leakage-safe preprocessing, leakage demonstration.

Run: python -m src.q1_data_leakage
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from . import config as C
from .data_cls import clean_bank, ensure_split, load_raw_bank, load_train_val, split_sizes
from .preprocessing import TRANSFORM_TABLE, make_preprocessor
from .utils import cls_metrics, save_fig, save_json, save_table, write_macros

CV = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)


# --------------------------------------------------------------------------- #
# a) audit
# --------------------------------------------------------------------------- #
def audit(df_raw: pd.DataFrame):
    df = df_raw.copy()
    rows = []
    for col in df.columns:
        s = df[col]
        r = {"feature": col, "dtype": str(s.dtype), "n_missing_unknown": int(s.isna().sum()),
             "pct_missing_unknown": 100 * s.isna().mean(), "n_unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            r.update({"min": s.min(), "median": s.median(), "max": s.max(),
                      "n_iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
        else:
            r.update({"min": np.nan, "median": np.nan, "max": np.nan, "n_iqr_outliers": np.nan})
        rows.append(r)
    aud = pd.DataFrame(rows)
    notes = {
        "duration": "LEAKAGE: known only after the call (outcome-dependent); excluded",
        "pdays": "sentinel -1 = never contacted; flag prev_contacted added",
        "poutcome": "mostly missing = no previous campaign; kept as 'missing' level",
        "contact": "'unknown' kept as 'missing' level",
        "balance": "negative values and heavy right tail",
        "campaign": "heavy right tail (extreme call counts)",
        "previous": "heavy right tail; 0 for never-contacted",
        "y": "target (yes=1); imbalanced",
        "day_of_week": "actually day of month (1-31) in UCI file",
        "default": "very rare 'yes' level",
    }
    aud["note"] = aud["feature"].map(notes).fillna("")
    save_table(aud, "q1_audit", "Q1a dataset audit (Bank Marketing, all rows)", float_fmt="%.2f")

    anomalies = pd.DataFrame([
        {"check": "rows", "count": len(df)},
        {"check": "exact duplicate rows", "count": int(df.duplicated().sum())},
        {"check": "duplicate rows ignoring duration", "count": int(df.drop(columns="duration").duplicated().sum())},
        {"check": "pdays == -1 (never contacted)", "count": int((df["pdays"] == -1).sum())},
        {"check": "balance < 0", "count": int((df["balance"] < 0).sum())},
        {"check": "campaign > 20", "count": int((df["campaign"] > 20).sum())},
        {"check": "previous > 50", "count": int((df["previous"] > 50).sum())},
        {"check": "duration == 0", "count": int((df["duration"] == 0).sum())},
        {"check": "rows with >=1 missing/unknown", "count": int(df.isna().any(axis=1).sum())},
    ])
    save_table(anomalies, "q1_anomalies", "Q1a duplicate and anomaly checks", float_fmt="%.0f")

    y = (df["y"] == "yes").astype(int)
    cls = pd.DataFrame({"class": C.CLASS_NAMES, "count": [int((y == 0).sum()), int((y == 1).sum())]})
    cls["proportion"] = cls["count"] / cls["count"].sum()
    save_table(cls, "q1_class_dist", "Q1a class distribution (all rows)")

    # duration vs target (illustrates why it leaks)
    dur = df.groupby("y")["duration"].describe()[["mean", "50%", "std"]].reset_index()
    dur.columns = ["y", "duration_mean", "duration_median", "duration_std"]
    save_table(dur, "q1_duration_by_target", "Q1a call duration by target class", float_fmt="%.1f")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, col in zip(axes, ["balance", "campaign", "pdays", "duration"]):
        ax.hist(df[col], bins=60, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count (log)")
    fig.suptitle("Q1a heavy-tailed / sentinel-valued numeric features (all rows)")
    save_fig(fig, "q1_numeric_distributions")
    return aud, anomalies, cls


# --------------------------------------------------------------------------- #
# b) stratified vs random splitting
# --------------------------------------------------------------------------- #
def three_way(idx, y, stratify, seed):
    a, rest = train_test_split(idx, train_size=C.TRAIN_FRAC, stratify=y[idx] if stratify else None, random_state=seed)
    rel = C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC)
    b, c = train_test_split(rest, test_size=rel, stratify=y[rest] if stratify else None, random_state=seed)
    return {"train": a, "val": b, "test": c}


def split_comparison(df, n_seeds=100):
    y = df[C.TARGET_CLS].values
    p_full = y.mean()
    recs = []
    settings = [("full", np.arange(len(y)))]
    # small-sample case: 200 rows drawn at random once (seed 42)
    small = np.random.default_rng(C.SEED).choice(len(y), 200, replace=False)
    settings.append(("small_n200", small))
    for setting, idx in settings:
        p_ref = y[idx].mean()
        for seed in range(n_seeds):
            for strat in (True, False):
                parts = three_way(idx, y, strat, seed)
                for pname, pidx in parts.items():
                    p = y[pidx].mean()
                    recs.append({"setting": setting, "method": "stratified" if strat else "random", "seed": seed,
                                 "partition": pname, "n": len(pidx), "n_pos": int(y[pidx].sum()), "pos_rate": p,
                                 "abs_dev_pp": 100 * abs(p - p_ref)})
    res = pd.DataFrame(recs)
    res.to_csv(C.TABLES / "q1_split_seeds_raw.csv", index=False)
    summ = (res.groupby(["setting", "method", "partition"])
            .agg(n=("n", "first"), mean_abs_dev_pp=("abs_dev_pp", "mean"), max_abs_dev_pp=("abs_dev_pp", "max"),
                 min_n_pos=("n_pos", "min"), frac_seeds_lt3_pos=("n_pos", lambda s: (s < 3).mean()))
            .reset_index())
    save_table(summ, "q1_split_comparison",
               f"Q1b absolute class-proportion deviation (percentage points) over {n_seeds} seeds; "
               f"full-data positive rate = {p_full:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, setting in zip(axes, ["full", "small_n200"]):
        sub = res[res.setting == setting]
        labels, data = [], []
        for pname in ["train", "val", "test"]:
            for m in ["stratified", "random"]:
                labels.append(f"{pname}\n{m[:5]}")
                data.append(sub[(sub.partition == pname) & (sub.method == m)]["abs_dev_pp"].values)
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_ylabel("|positive rate - reference| (pp)")
        ax.set_title(f"{setting} ({n_seeds} seeds)")
    fig.suptitle("Q1b stratified vs. random 70/15/15 partitioning")
    save_fig(fig, "q1_split_deviation")
    return summ


# --------------------------------------------------------------------------- #
# d) leakage demonstration
# --------------------------------------------------------------------------- #
def lr():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


def score_val(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr)
    m = cls_metrics(yva, model.predict(Xva), model.predict_proba(Xva)[:, 1])
    return m["macro_f1"], m["roc_auc"]


def cv_scores(model, X, y):
    r = cross_validate(model, X, y, cv=CV, scoring=["f1_macro", "roc_auc"], n_jobs=C.N_JOBS)
    return r["test_f1_macro"].mean(), r["test_f1_macro"].std(), r["test_roc_auc"].mean()


def leakage_demo():
    Xtr, ytr, Xva, yva = load_train_val(include_leaky=True)
    rows = []

    def add(name, valid, cv, val, desc):
        rows.append({"procedure": name, "valid": valid, "cv_macro_f1": cv[0], "cv_macro_f1_std": cv[1],
                     "cv_roc_auc": cv[2], "val_macro_f1": val[0], "val_roc_auc": val[1], "description": desc})

    # VALID: duration dropped, all preprocessing inside the pipeline
    num, cat = C.NUM_FEATURES, C.CAT_FEATURES
    valid = Pipeline([("pre", make_preprocessor("scaled", num, cat)), ("clf", lr())])
    add("VALID", True, cv_scores(valid, Xtr[num + cat], ytr), score_val(valid, Xtr[num + cat], ytr, Xva[num + cat], yva),
        "duration dropped; impute/scale/encode fit inside training folds")

    # INVALID-1: target leakage through post-outcome feature `duration`
    inv1 = Pipeline([("pre", make_preprocessor("scaled", num + [C.LEAKY_FEATURE], cat)), ("clf", lr())])
    add("INVALID_1_duration", False, cv_scores(inv1, Xtr, ytr), score_val(inv1, Xtr, ytr, Xva, yva),
        "INVALID: uses duration, which is only known after the outcome")

    # INVALID-2: preprocessing + SelectKBest fit on ALL development rows (train+val)
    # before CV / validation. Test rows are NOT used even here.
    Xall = pd.concat([Xtr[num + cat], Xva[num + cat]])
    yall = np.concatenate([ytr, yva])
    pre = Pipeline([("pre", make_preprocessor("scaled", num, cat)), ("kbest", SelectKBest(f_classif, k=20))])
    Zall = pre.fit_transform(Xall, yall)  # <-- INVALID: sees validation rows and fold-validation labels
    Ztr, Zva = Zall[: len(Xtr)], Zall[len(Xtr):]
    add("INVALID_2_global_scale_kbest", False, cv_scores(lr(), Ztr, ytr), score_val(lr(), Ztr, ytr, Zva, yva),
        "INVALID: scaling + SelectKBest(k=20) fit on train+val before CV/validation")
    # valid counterpart of INVALID-2 (same steps inside the pipeline)
    v2 = Pipeline([("pre", make_preprocessor("scaled", num, cat)), ("kbest", SelectKBest(f_classif, k=20)), ("clf", lr())])
    add("VALID_scale_kbest_in_pipeline", True, cv_scores(v2, Xtr[num + cat], ytr),
        score_val(v2, Xtr[num + cat], ytr, Xva[num + cat], yva), "same steps as INVALID_2 but fit inside folds")

    # INVALID-3: target-derived feature. Mean target per high-cardinality group
    # (job x month x day) computed on ALL development rows incl. the row itself.
    key_tr = Xtr["job"].astype(str) + "|" + Xtr["month"] + "|" + Xtr["day_of_week"].astype(str)
    key_va = Xva["job"].astype(str) + "|" + Xva["month"] + "|" + Xva["day_of_week"].astype(str)
    te_map = pd.Series(yall).groupby(pd.concat([key_tr, key_va]).values).mean()  # INVALID
    Xtr3, Xva3 = Xtr[num + cat].copy(), Xva[num + cat].copy()
    Xtr3["INVALID_te_group"] = key_tr.map(te_map).values
    Xva3["INVALID_te_group"] = key_va.map(te_map).values
    inv3 = Pipeline([("pre", make_preprocessor("scaled", num + ["INVALID_te_group"], cat)), ("clf", lr())])
    add("INVALID_3_target_encoding_all_rows", False, cv_scores(inv3, Xtr3, ytr), score_val(inv3, Xtr3, ytr, Xva3, yva),
        "INVALID: group mean of y computed on train+val (incl. own row)")

    # valid counterpart: sklearn TargetEncoder (cross-fitted) inside the pipeline
    Xtr4, Xva4 = Xtr[num + cat].copy(), Xva[num + cat].copy()
    Xtr4["group_key"], Xva4["group_key"] = key_tr.values, key_va.values
    from sklearn.compose import ColumnTransformer
    pre4 = ColumnTransformer([("base", make_preprocessor("scaled", num, cat), num + cat),
                              ("te", TargetEncoder(random_state=C.SEED), ["group_key"])])
    v4 = Pipeline([("pre", pre4), ("clf", lr())])
    add("VALID_target_encoding_in_pipeline", True, cv_scores(v4, Xtr4, ytr), score_val(v4, Xtr4, ytr, Xva4, yva),
        "TargetEncoder (internal cross-fitting) inside training folds")

    res = pd.DataFrame(rows)
    save_table(res.drop(columns="description"), "q1_leakage_demo",
               "Q1d leakage demonstration (LogReg C=1, class_weight=balanced). INVALID rows are deliberately wrong.")
    res.to_csv(C.TABLES / "q1_leakage_demo_with_desc.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    xs = np.arange(len(res))
    ax.bar(xs - 0.2, res.cv_macro_f1, 0.4, yerr=res.cv_macro_f1_std, label="CV Macro-F1 (train)", color="#4C72B0")
    ax.bar(xs + 0.2, res.val_macro_f1, 0.4, label="Validation Macro-F1", color="#DD8452")
    ax.set_xticks(xs)
    ax.set_xticklabels([p.replace("_", "\n", 2) for p in res.procedure], fontsize=7)
    for x, v in zip(xs, res.valid):
        if not v:
            ax.text(x, 0.02, "INVALID", ha="center", color="red", fontsize=8, fontweight="bold")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Q1d valid vs. deliberately INVALID procedures")
    save_fig(fig, "q1_leakage_demo")
    return res


# --------------------------------------------------------------------------- #
# e) protocol diagram
# --------------------------------------------------------------------------- #
def protocol_diagram(sizes):
    n = dict(zip(sizes.partition, sizes.n))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="black", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2, txt=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, txt, fontsize=7, ha="center")

    box(0.1, 2.5, 1.9, 1.0, f"Raw Bank Marketing\n{sum(n.values())} rows\nrow-wise cleaning only\n(drop duration)", "#EEEEEE")
    box(2.6, 2.5, 1.7, 1.0, "Stratified split\nseed = 42\n70 / 15 / 15", "#DDDDFF")
    arrow(2.0, 3.0, 2.6, 3.0)
    box(5.0, 4.4, 2.0, 0.9, f"TRAIN (n={n['train']})", "#CCE5CC")
    box(5.0, 2.55, 2.0, 0.9, f"VALIDATION (n={n['val']})", "#FFF2CC")
    box(5.0, 0.6, 2.0, 0.9, f"TEST (n={n['test']})\nlocked: q5_final_test only", "#F4CCCC")
    for yy in (4.85, 3.0, 1.05):
        arrow(4.3, 3.0, 5.0, yy)
    box(7.6, 4.2, 2.1, 1.3, "5-fold stratified CV\nPipeline(impute, scale,\nencode, [resample], model)\nGridSearch, Macro-F1", "#CCE5CC")
    arrow(7.0, 4.85, 7.6, 4.85)
    box(7.6, 2.4, 2.1, 1.2, "Refit best config on train\nevaluate on validation:\nerror analysis, robustness,\nbootstrap, MCDA", "#FFF2CC")
    arrow(8.65, 4.2, 8.65, 3.6, "best params")
    arrow(7.0, 3.0, 7.6, 3.0)
    box(10.1, 2.4, 1.8, 1.2, "FREEZE configs +\nselection rule\n(results/models/)", "#DDDDFF")
    arrow(9.7, 3.0, 10.1, 3.0)
    box(10.1, 0.5, 1.8, 1.2, "Refit on train+val\nONE evaluation on test\n(guarded, logged)", "#F4CCCC")
    arrow(11.0, 2.4, 11.0, 1.7)
    arrow(7.0, 1.05, 10.1, 1.05, "used once")
    ax.set_title("Q1e experimental protocol (classification). Regression (Q3) uses the same 70/15/15, seed 42 scheme.")
    save_fig(fig, "q1_protocol")


def main():
    df_raw = load_raw_bank()
    ensure_split()
    audit(df_raw)
    summ = split_comparison(clean_bank(df_raw))
    sizes = split_sizes()
    save_table(sizes, "q1_split_sizes", "Q1b exact partition sizes and positive rates (stratified, seed 42)")
    save_table(pd.DataFrame(TRANSFORM_TABLE, columns=["transformation", "LogReg", "NaiveBayes (Gauss | Bern)",
                                                      "kNN", "Trees/Ensembles"]),
               "q1_transform_matrix", "Q1c transformation x model family")
    res = leakage_demo()
    protocol_diagram(sizes)
    n = dict(zip(sizes.partition, sizes.n))
    write_macros({"NTotal": len(df_raw), "NTrain": n["train"], "NVal": n["val"], "NTest": n["test"],
                  "PosRate": f"{(df_raw['y'] == 'yes').mean():.4f}"}, "q1")
    save_json({"split_sizes": sizes.to_dict("records")}, C.RESULTS / "q1_summary.json")
    print(sizes.to_string(index=False))
    print(summ[summ.setting == "small_n200"].to_string(index=False))
    print(res[["procedure", "cv_macro_f1", "cv_roc_auc", "val_macro_f1", "val_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
