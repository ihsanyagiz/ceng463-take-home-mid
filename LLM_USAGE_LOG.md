# LLM Usage Log (for the report's "LLM Usage Disclosure" section)

**Tool:** Claude Code (Anthropic, model Claude Opus 5.5), used as a coding assistant in the Claude desktop app.
**Scope (permitted by the exam):** coding, debugging and environment setup only.
**Not done by the assistant:** scientific interpretation/discussion in the report. Every interpretation
paragraph is a `\studenttodo{...}` placeholder for the student to write. No number was hand-typed: all tables
are generated from `results/` by the code.

## Prompt given
The student supplied `AGENT_PROMPT.md` (in the workspace) together with the assignment text
`CENG463_TakeHome.md`. In short, that prompt asks for: the project structure, the datasets (UCI Bank Marketing,
California Housing), the global protocol (seed 42, stratified 70/15/15, 5-fold CV, Macro-F1), the per-question
experiments Q1a-Q5e, a LaTeX report with interpretation placeholders, a README, a packaging script, and this log.

## Tasks performed by the assistant
| # | Task | Details |
|---|------|---------|
| 1 | Environment setup | Found a broken global Python (NumPy 2.4 vs. compiled SciPy/matplotlib). Created a project `.venv` and pinned versions in `requirements.txt`. Found no LaTeX distribution. |
| 2 | Data download | Bank Marketing via `ucimlrepo` (id 222) → `data/raw/`; California Housing via scikit-learn. |
| 3 | Code: config / data / preprocessing | `src/config.py`, `src/data_cls.py`, `src/data_reg.py` (one-time split, test loader restricted to `q5_final_test.py`), `src/preprocessing.py`, `src/models.py`, `src/utils.py`. |
| 4 | Code: Q1-Q5 scripts | `src/q1_data_leakage.py` … `src/q5_final_test.py`, `run_all.py`, `src/environment.py`, `src/export_report_assets.py`, `make_submission.py`. |
| 5 | Runtime check | Timed GB/RF/kNN on the full data to decide whether subsampling was needed (it was not). Boosting is tuned with staged CV to save time. |
| 6 | Debugging | Fixed the test-loader guard so it works with `python -m`; replaced `GridSearchCV` with an empty grid by `cross_val_score`; matplotlib deprecation fix; converted numpy scalars in saved params. |
| 7 | Grid adjustment | Widened LR `C`, kNN `k` and NB bins grids after the first run put optima on grid edges (decided on CV results only, never test). |
| 8 | LaTeX skeleton | `report/main.tex`, `report/sections/*.tex` with factual method descriptions, auto-generated `\input` tables/figures and `\studenttodo` placeholders. |
| 9 | README | Reproduction instructions. |

## Design choices made in code (the student should review and justify them)
- Categorical missing values (`"unknown"`) are kept as an explicit `"missing"` level instead of most-frequent imputation.
- Interpretability / implementation-simplicity ordinal scores and MCDA base weights in `src/config.py` are **placeholders** that the student must review and justify.
- Final test refit uses train+validation (`FINAL_REFIT_ON` in `src/config.py`; `--refit train` is available).
