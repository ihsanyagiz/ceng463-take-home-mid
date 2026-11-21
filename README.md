# CENG 463 Take-Home Midterm: reproducible ML pipeline

Classification: **UCI Bank Marketing** (`bank-full`, 45,211 rows, binary `y`), used in Q1, Q2, Q4 and Q5.
Regression: **California Housing** (scikit-learn), used in Q3.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
(Linux/macOS: `.venv/bin/python`.) Data are downloaded automatically on first run into `data/raw/`
(the Bank data through `ucimlrepo`, so internet access is needed once).

## Run everything
```powershell
.venv\Scripts\python run_all.py --clean
```
This runs `environment → Q1 → Q2 → Q3 → Q4 → Q5 (selection) → Q5 final test → copy figures to report/`.
Per-step runtimes go to `results/logs/runtime.json`. The final test evaluation is **guarded**. If
`results/logs/final_test_done.json` exists it refuses to run again; `--clean` (which deletes `results/`) or
`--force-final` (logged in `results/logs/final_test_runs.log`) are the only ways to repeat it.

Individual steps (each runs on its own):
```powershell
.venv\Scripts\python -m src.q1_data_leakage
.venv\Scripts\python -m src.q2_classifiers
.venv\Scripts\python -m src.q3_regression
.venv\Scripts\python -m src.q4_trees_ensembles       # needs Q2 outputs
.venv\Scripts\python -m src.q5_robustness_selection  # needs Q2 + Q4 outputs, no test access
.venv\Scripts\python -m src.q5_final_test            # ONLY place the test sets are read
```

## Protocol
- `SEED = 42`. The stratified 70/15/15 split is created once, before any fitting, and stored in `data/splits/`.
- Test rows can only be loaded by `src/q5_final_test.py`; the loader raises `PermissionError` anywhere else.
- All preprocessing (imputation, scaling, encoding, discretisation, resampling, feature selection) lives in
  sklearn/imblearn `Pipeline`s, so it is fit on training folds only.
- Tuning uses 5-fold stratified CV on train, with Macro-F1 as the selection metric (RMSE for regression). Validation is used
  for analysis (errors, robustness, bootstrap, MCDA). Configurations are frozen in `results/models/` before the
  test run.

## Outputs
| Path | Content |
|------|---------|
| `results/tables/*.csv` | every number used in the report |
| `results/figures/*.png, *.pdf` | figures (300 dpi) |
| `results/models/*.json` | tuned/frozen configurations, `selection_rule.json` |
| `results/environment.txt` | hardware/software summary |
| `report/tables/*.tex` | LaTeX tables auto-generated from the CSVs |
| `report/main.tex` | report source (interpretation placeholders marked `\studenttodo`) |

## Report
```powershell
cd report
latexmk -pdf main.tex
```
(Needs MiKTeX/TeX Live with `todonotes`, `booktabs`, `subcaption`; or upload `report/` to Overleaf.)

## Submission archive
```powershell
.venv\Scripts\python make_submission.py --student-id 300201071
```
This builds `CENG463_Midterm_300201071.zip` without `.venv` or raw data.
