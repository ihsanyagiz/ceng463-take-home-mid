# CENG 463 – Machine Learning
## Take-Home Midterm Examination

| | |
|---|---|
| **Deadline** | December 1, 2026, 23:59 |
| **Weight** | 20% |
| **Type** | Individual |
| **Instructor** | Prof. Dr. Aytuğ Onan |

---

## Scope and General Instructions

**All five questions are compulsory.** This examination assesses the full machine-learning workflow covered in the course: data preparation and leakage control; classification; numerical regression; decision trees and ensemble learning; and reliable evaluation and model selection. The emphasis is on **understanding, reproducibility, justified design choices, and evidence-based analysis**, not only high test scores.

Students may reuse the **same classification dataset and fixed split** in Questions 1, 2, 4, and 5 to build a coherent study. Question 3 requires a numerical regression target and may use a separate dataset. Choose datasets of manageable size; computational efficiency is part of the engineering challenge. All mandatory models must be trained and assessed; a strong explanation is not a substitute for the requested experiments.

## Submission Format and Deliverables

Submit a single archive named **`CENG463_Midterm_<StudentID>.zip`**, containing:

- A **LaTeX source report** and its compiled PDF, organized according to the required report structure below.
- Executable **Jupyter notebooks or Python scripts**, a **`requirements.txt`** or environment file, and instructions to reproduce the results.
- Saved tables and figures, a compact summary of the hardware/software environment, and a public or access-granted **GitHub/Colab repository link** in the report.

**Submission method:** Submit the archive **exclusively through the designated assignment on Microsoft Teams** before the deadline. Email submissions will not be accepted.

## Academic Integrity and Experimental Protocol

- Work individually and acknowledge all data, code, and external references. Report actual experimental results only; fabricated observations or invented metrics are prohibited.
- Establish **train/validation/test partitions before fitting transformations**. Use a fixed random seed and report exact partition sizes. Fit imputation, scaling, encoding, feature selection, and text vectorization **only on training data** (within the corresponding training fold in cross-validation).
- Use validation data or cross-validation to choose hyperparameters. Keep the test set untouched during development; evaluate all **finalized** candidate models on it once, at the end. **Never select the winning configuration based on test metrics.**
- State the models' search spaces, selection metric, preprocessing pipeline, and computational resources. If a task is computationally demanding, reduce dataset size **consistently and transparently**, not the number of required comparisons.
- **LLMs may be used solely for coding, debugging, and environment setup assistance.** Do not use them to fabricate results or write the report's scientific interpretation. Describe permitted assistance and the corresponding prompts/tools in an **LLM Usage Disclosure** section. LLM-based modeling is **not** a requirement of this exam.

---

## Question 1 – Data Preparation, Leakage, and Experimental Design (20 points)

**Objective.** Demonstrate that a trustworthy machine-learning result begins with a correct data pipeline and evaluation protocol.

**a)** Select a real, labeled **classification** dataset with at least two classes. Describe its domain, target, feature types, sample size, and class distribution. Identify missing values, duplicates, anomalous values, and potentially problematic predictors.

**b)** Create justified training/validation/test partitions (or training/test plus cross-validation). Compare **stratified** and ordinary random sampling by quantifying class-proportion deviations, including a discussion of when either approach may fail.

**c)** Construct a leakage-safe preprocessing pipeline. For numeric variables, address missingness and scaling; for categorical variables, address encoding and unknown categories. If using text, implement training-only tokenization/vectorization. Explain which transformations are relevant for each model family.

**d)** Design and run a **controlled leakage demonstration**: compare a valid pipeline to one deliberately invalid procedure (e.g., scaling/feature selection on all rows or using a target-derived feature). Label the invalid procedure explicitly and use it **only** to demonstrate the problem. Report its effect on cross-validation or validation results; explain why any apparent improvement is untrustworthy.

**e)** Produce a **dataset audit table** and an **experimental-protocol diagram**. Explain at least three failure modes that could invalidate reported results even if the code executes successfully.

---

## Question 2 – Classification Algorithms and Error Analysis (20 points)

**Objective.** Compare fundamental classifiers and understand how representation, assumptions, and decision thresholds influence outcomes.

**a)** On the Question 1 dataset, train **at least three** classifiers: **Logistic Regression**, **Naive Bayes** (appropriate variant for the feature representation), and **k-Nearest Neighbors**. Include a **majority-class dummy baseline**. If the dataset has no naturally suitable representation for a model, justify and implement a compatible representation.

**b)** Tune **at least two hyperparameters per non-dummy model** where meaningful (e.g., regularization strength and class weighting for logistic regression; smoothing and representation for Naive Bayes; neighborhood size and distance weighting for k-NN). For a model with only one numerical parameter, the second comparison may be a controlled preprocessing/representation choice. Use the same validation protocol for all candidates.

**c)** Report **Accuracy, Macro-F1, per-class Precision/Recall/F1, and confusion matrices**. For binary data, add **ROC-AUC and PR-AUC** when probability or score outputs are available; for multiclass data, report a justified multiclass ROC-AUC formulation if applicable. Identify and explain which measure is most useful for the dataset.

**d)** Evaluate the impact of **class imbalance** using class-wise errors and at least one controlled intervention (e.g., class weights or training-only resampling). Discuss whether aggregate accuracy can conceal important failures.

**e)** Study **at least ten validation examples** misclassified by a finalized candidate: categorize error patterns, connect them to input characteristics, and propose concrete improvements without inspecting the held-out test labels.

---

## Question 3 – Numerical Regression, Regularization, and Generalization (20 points)

**Objective.** Explain how model complexity and regularization affect the prediction of continuous outcomes.

**a)** Select a real dataset with a **continuous response**. Audit missing data, examine target distribution, and identify high-correlation predictors and possible outliers. Establish leakage-safe partitions and preprocessing.

**b)** Fit **Dummy Regressor**, **Ordinary Least Squares**, **Ridge**, **Lasso**, and **at least one non-linear regression model** (e.g., decision-tree regressor or random-forest regressor). Apply appropriate scaling inside model pipelines.

**c)** Investigate **underfitting and overfitting** by varying model complexity (e.g., polynomial degree, decision-tree depth, or number of features), and draw training-versus-validation learning or complexity curves. Explain the patterns using the bias–variance framework.

**d)** Tune regularization strength for Ridge and Lasso using cross-validation. Discuss the differences in coefficient shrinkage and sparsity; explain the effect of highly correlated predictors.

**e)** Report **MAE, RMSE, and R²**, with consistent units and a careful interpretation. Plot **predictions versus actual values** and **residuals versus predictions**. Investigate **at least five large validation residuals**, and explain what the residual patterns suggest.

---

## Question 4 – Decision Trees, Bagging, and Boosting (20 points)

**Objective.** Analyze tree induction, ensemble diversity, and the trade-off between interpretability and predictive power.

**a)** Using the classification task of Questions 1–2, train a **Decision Tree**, **Random Forest**, **AdaBoost**, and **Gradient Boosting** classifier. Include the strongest non-tree classifier from Question 2 as a reference. Use a consistent training and validation protocol.

**b)** For the single tree, compare **Gini impurity versus entropy/log loss** (where supported). Tune at least **max depth**, **minimum samples per leaf**, and **one further complexity control**. Visualize a small tree and interpret at least three decision paths.

**c)** For Random Forest, study **number of trees**, **maximum depth**, and **maximum features**. Report training and validation behavior and explain how bagging and randomized feature selection affect variance and tree correlation.

**d)** For the two boosting methods, examine **number of estimators** and **learning rate**; explain the differences between sequential error correction and independent bagged learners. Provide at least one plot showing performance against an ensemble complexity parameter.

**e)** Compare **Macro-F1, Accuracy, training time, inference time, and feature importance**. Use **permutation importance on held-out validation data** to cross-check built-in tree importance. Explain at least two cases in which feature importance can be misleading. No test results may influence model tuning.

---

## Question 5 – Robustness, Statistical Comparison, and Model Selection (20 points)

**Objective.** Make a defensible final model choice based on more than a single headline metric.

**a)** Choose **at least four finalized classification candidates** from Questions 2 and 4, including at least one linear/probabilistic method and one ensemble. Compare their cross-validation scores and variation across folds (mean and standard deviation).

**b)** Conduct a **controlled robustness study** on validation data: evaluate performance under **at least three corruption severities** in one realistic scenario (e.g., increasing numeric-feature noise, missing-feature rates, or text perturbations). Ensure the same perturbations and evaluation examples are used for all models. **Plot Macro-F1 against severity.**

**c)** Estimate uncertainty in a **difference between two models** using a **paired bootstrap confidence interval** on validation/out-of-fold predictions; discuss what the interval permits and does not permit you to conclude. State the sampling unit and number of resamples.

**d)** Build a **multi-criteria selection table** comparing predictive performance, robustness, interpretability, training/inference cost, and implementation complexity. Justify weights and perform a **sensitivity analysis** showing whether the chosen model changes when weights vary.

**e)** **Freeze** hyperparameters and model-selection rules. Perform **one final evaluation** of all frozen candidates on the untouched test set. Report final metrics, identify the selected model, and discuss the limitations and threats to validity of the full study.

---

## Required Report Structure

1. Introduction, research questions, and datasets
2. Data audit, preprocessing, and leakage controls
3. Methods and experimental design (Questions 1–5)
4. Validation scheme, tuning grids, random seeds, and computing environment
5. Results (properly labeled tables and plots) and interpretation
6. Error analysis, robustness, uncertainty, and limitations
7. Final model-selection rationale and test-set evaluation
8. LLM Usage Disclosure
9. References and reproducibility appendix (commands, links, configurations)

## Grading Rubric

| Criterion | Weight |
|---|---|
| Question 1 – Data preparation and leakage | 20% |
| Question 2 – Classification and error analysis | 20% |
| Question 3 – Regression and regularization | 20% |
| Question 4 – Trees and ensembles | 20% |
| Question 5 – Robustness and model selection | 20% |

**Assessment within each question:** experimental correctness and methodological rigor (40%), depth of evidence-based analysis (30%), reproducibility and code quality (20%), and clarity of reporting (10%). Academic-integrity violations are handled under course/university rules.

---

*End of Examination. All five questions are required.*
