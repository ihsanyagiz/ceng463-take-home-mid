# LLM Usage Log (for the report's "LLM Usage Disclosure" section)

**Tool:** Claude Code (Anthropic, model Claude Opus 5.5), used as a coding assistant in the Claude desktop app.
**Scope (permitted by the exam):** coding, debugging and environment setup only.
**Not done by the assistant:** scientific interpretation/discussion in the report. Every interpretation
paragraph was left as a `\studenttodo{...}` placeholder; the student wrote all 23 paragraphs and the assistant
only pasted them verbatim (task 14). No number was hand-typed: all tables are generated from `results/` by the code.

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
| 10 | MCDA revision (student's specification) | On the student's instruction: robustness criterion = absolute Macro-F1 at the highest noise severity (degradation-curve AUC also reported); log10-scaled costs before min-max; base weights set by the student (0.40/0.20/0.10/0.15/0.05/0.05/0.05); centred Dirichlet ($\alpha = 20w$) added next to the uniform one. Re-ran Q5 selection only. |
| 11 | kNN grid (student's instruction) | Added $k = 1$ to `KNN_GRID` (best $k = 3$ was on the grid edge); re-ran Q2 kNN tuning and Q5 selection so the frozen configs and `selection_rule.json` were current before the test. |
| 12 | Git / GitHub | Initialised the repository and pushed to the student's GitHub repo; one commit per logical change; the final test kept in its own commit after the configs were frozen. |
| 13 | Final test (Q5e) | On the student's instruction, ran `src/q5_final_test.py` once (not forced; config hash matched the frozen `selection_rule.json`) and exported the generated tables/figures. |
| 14 | Report text insertion | Replaced each `\studenttodo{...}` block with the student's own paragraph verbatim (from the student's answer file); removed the `\listoftodos` and `todonotes` scaffolding; filled in name and ID. No content was rewritten. |
| 15 | PDF build | No local LaTeX: downloaded the portable Tectonic 0.17.0 release (outside the repo) and compiled `report/main.pdf` (BibTeX included). Fixed overfull lines only (layout): `\emergencystretch`, `\footnotesize` for the two verbatim blocks, breakable `\nolinkurl` for long paths/names. Log checked for undefined references/citations and missing figures/tables (none). |
| 16 | Submission archive | Built `CENG463_Midterm_<StudentID>.zip` with `make_submission.py` and checked its contents (`.venv`, `data/raw` excluded). |

## Design choices made in code (the student should review and justify them)
- Categorical missing values (`"unknown"`) are kept as an explicit `"missing"` level instead of most-frequent imputation.
- Interpretability / implementation-simplicity ordinal scores in `src/config.py` were proposed as initial values; the student reviewed and kept them. The MCDA criteria and base weights were set by the student (task 10).
- Final test refit uses train+validation (`FINAL_REFIT_ON` in `src/config.py`; `--refit train` is available).
