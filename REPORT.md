# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

> STUDENT: the content under each heading is either (a) factual content taken
> from the code and the training run, or (b) a `NOTES FOR YOU` block with
> points to explain **in your own words**. Per the AI policy, the analysis and
> conclusions must be yours: write the prose yourself and delete every
> `NOTES FOR YOU` block (and this note) before submitting.

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

Facts:
- IBM Telco Customer Churn (sample dataset published by IBM), downloaded from
  `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`,
  committed as `data/telco_churn.csv`.
- 7,043 rows x 21 columns. One row = one customer of a fictional US telecom
  provider at a single snapshot in time.
- Columns: `customerID`; demographics (`gender`, `SeniorCitizen`, `Partner`,
  `Dependents`); account (`tenure` in months, `Contract`, `PaperlessBilling`,
  `PaymentMethod`, `MonthlyCharges`, `TotalCharges`); services (`PhoneService`,
  `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
  `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`); target `Churn` (Yes/No).
- Class balance: 1,869 churners (26.5%) vs 5,174 non-churners (73.5%).
- Data-quality issue: 11 rows have a blank `TotalCharges`; all 11 have `tenure = 0`.

> NOTES FOR YOU (delete before submitting):
> - Why you picked this dataset (a decision with a clear cost asymmetry; enough categorical structure that feature engineering matters).
> - What one row represents, and that there is no date column (this matters for the split).

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

Facts (as implemented):
- Decision supported: which customers a retention team should contact with a
  retention offer. Contact if predicted P(churn) >= threshold.
- Cost model (constants at the top of `train.py`, all ASSUMED):
  - `OFFER_COST = $50` per contacted customer
  - `HORIZON_MONTHS = 12`: a churner is worth `MonthlyCharges x 12` of lost revenue
  - `SAVE_RATE = 0.30`: the offer retains 30% of contacted would-be churners
  - Per customer: missed churner (FN) = MonthlyCharges x 12; contacted churner (TP)
    = $50 + 0.7 x MonthlyCharges x 12; false alarm (FP) = $50; TN = $0.
- Threshold: the value on a 0.01 grid that minimises total cost on the
  **validation** set (chosen: 0.28 for all three models). Not 0.5.
- Split: stratified random 60/20/20 (train 4,225 / validation 1,409 / test 1,409), seed 42.
  Churn rate is 26.5% in each split.
- Also reported: ROC-AUC, PR-AUC, precision, recall, F1.

> NOTES FOR YOU (delete before submitting):
> - Why a cost-based threshold rather than 0.5 or accuracy: errors have very different costs (a missed churner ~ $900 vs a $50 offer), so the cheapest threshold is well below 0.5.
> - Why a time-based split is impossible here (single snapshot, no dates) and why a stratified random split is acceptable; why `tenure` is a customer attribute (a feature) and not a time index.
> - Why validation is used for choosing C and the threshold, and test is touched only once for final numbers.
> - The cost constants are assumptions; say how the decision would change if they were different (try the inputs in the dashboard's "Threshold & cost" tab).

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

Facts (code in `pipeline.py` and `train.py`):
1. `TotalCharges` blanks (11 rows, all tenure 0) set to `MonthlyCharges` (treated as first billing month).
2. `Churn` mapped to 1/0.
3. "No internet service" collapsed to "No" in the 6 add-on columns; "No phone service" collapsed to "No" in `MultipleLines`.
4. `log_tenure = log(1 + tenure)` replaces raw `tenure`.
5. `n_addons` = number of the 6 internet add-ons the customer has (0-6).
6. `charge_diff = MonthlyCharges - TotalCharges / max(tenure, 1)` (current price vs historical average).
7. `fiber_x_monthly` = 1 if fiber optic internet AND month-to-month contract.
8. Dropped: `customerID` (identifier), `TotalCharges` (~ MonthlyCharges x tenure), raw `tenure`, `gender`
   (churn 26.9% female vs 26.2% male), and the individual dummies of `OnlineBackup`, `DeviceProtection`,
   `StreamingTV`, `StreamingMovies` (they enter only through `n_addons`; keeping them would make `n_addons` exactly collinear).
9. Final model inputs: 6 numeric (`SeniorCitizen`, `log_tenure`, `MonthlyCharges`, `charge_diff`, `n_addons`,
   `fiber_x_monthly`) + 10 categorical (`Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`,
   `OnlineSecurity`, `TechSupport`, `Contract`, `PaperlessBilling`, `PaymentMethod`) = 20 model columns after encoding.
10. `ColumnTransformer` fitted on the training split only: `StandardScaler` on numerics,
    `OneHotEncoder(drop="first", handle_unknown="ignore")` on categoricals. Validation and test are only transformed.
11. Observation from the run: `MonthlyCharges` is almost a linear combination of the service columns
    (smallest eigenvalue of the scaled design matrix ~0.0014), so the coefficients of `MonthlyCharges`,
    `PhoneService` and `InternetService` are not individually reliable.

> NOTES FOR YOU (delete before submitting):
> - Justify each step in your own words, especially the TotalCharges choice (why not drop the rows or set 0: setting 0 would fake a huge price increase in `charge_diff`).
> - Why fit the preprocessor on train only (leakage), and why feature engineering before the split is safe here (row-wise, learns nothing).
> - Why `drop="first"` (collinearity with the intercept; reference levels: Month-to-month, DSL, Bank transfer) and why scaling matters for L2 and for gradient descent.
> - Honest finding: `charge_diff` (coef -0.015) and `fiber_x_monthly` (+0.043) ended up contributing very little. Explain why you kept them or whether you would drop them.
> - The collinearity of MonthlyCharges with the services (fiber has a negative coefficient while MonthlyCharges is strongly positive: they partly cancel).

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

Facts:
- Model: L2-regularised logistic regression (binary target).
- C chosen on validation log loss from {0.01, 0.1, 1, 10}: 0.4104, 0.3985, 0.3962, 0.3959 -> **C = 10**.
- The same objective in all three: mean BCE + ||w||^2 / (2 x C x n_train), bias not penalised
  (equivalent to sklearn's `C x sum(logloss) + 0.5 ||w||^2`). Float64 everywhere.
  1. scikit-learn `LogisticRegression(C=10, solver="lbfgs")`.
  2. Manual PyTorch: raw tensors, numerically stable BCE written by hand, `loss.backward()`,
     update inside `torch.no_grad()`, grads zeroed; full-batch gradient descent, lr = 2.0, 30,000 epochs.
  3. Standard PyTorch: `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss`, `torch.optim.LBFGS` (strong Wolfe line search).
- Final penalised training loss: 0.40810605 for all three.

Test-set results (1,409 customers; threshold chosen on validation):

| Model | ROC-AUC | PR-AUC | Threshold | Precision | Recall | F1 | Contacted | Business cost | Cost at 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.8399 | 0.6558 | 0.28 | 0.5214 | 0.7834 | 0.6261 | 562 | $284,573 | $298,190 |
| Manual PyTorch | 0.8399 | 0.6558 | 0.28 | 0.5214 | 0.7834 | 0.6261 | 562 | $284,573 | $298,190 |
| Standard PyTorch | 0.8399 | 0.6558 | 0.28 | 0.5214 | 0.7834 | 0.6261 | 562 | $284,573 | $298,190 |
| Baseline: contact nobody | 0.5 | 0.2654 | - | 0 | 0 | 0 | 0 | $337,971 | - |
| Baseline: contact everyone | 0.5 | 0.2654 | - | 0.2654 | 1.0 | 0.4195 | 1,409 | $307,030 | - |

Agreement (max absolute difference on test):

| Pair | Coefficients | Predicted probabilities |
|---|---|---|
| sklearn vs manual PyTorch | 1.4e-06 | 3.9e-07 |
| sklearn vs standard PyTorch | 1.3e-06 | 5.9e-07 |
| manual vs standard PyTorch | 4.7e-07 | 2.6e-07 |

> NOTES FOR YOU (delete before submitting):
> - Why the three agree (same convex objective with a unique optimum, same data, matched L2 penalty, float64, trained to convergence).
> - What went wrong first: with all one-hot levels kept and all six add-on dummies plus `n_addons`, the manual gradient-descent model was still far from the optimum after 20,000 steps (coefficients off by ~2, probabilities by ~0.018) while LBFGS had converged. The cause was exact collinearity (an ill-conditioned problem). Removing it, and raising the learning rate to 2.0 (below the stability limit 2/L, L ~ 0.92), fixed it.
> - The model saves ~$22,500 (7%) on the test set vs the best naive baseline (contact everyone), and ~$13,600 vs using 0.5 as the threshold. Say what that means for the business.
> - C = 10 is the edge of the grid (weakest regularisation): with 4,225 rows and 20 features, regularisation barely helps here.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

> NOTES FOR YOU (delete before submitting): write real limitations with a concrete fix for each, e.g.
> - Costs are assumed (offer cost, 12-month horizon, 30% save rate); with data from a real retention campaign you would estimate them, or model uplift (who is *persuadable*) rather than churn risk.
> - No time dimension: a random split cannot test whether the model holds up on future customers; with snapshot dates you would use a time-based split.
> - Sample US dataset of unknown age; may not transfer to a real (e.g. Spanish) operator.
> - Correlated features (MonthlyCharges vs services) make individual coefficients hard to interpret.
> - Linear decision boundary; only one hand-made interaction. Could try more interactions or a tree model as a comparison.
> - One threshold for all customers, although the cost of a miss depends on MonthlyCharges: you could contact when the expected saving p x 0.3 x 12 x MonthlyCharges > $50 instead.
> - The threshold was chosen on 1,409 validation rows; the cost curve is fairly flat near the optimum (see dashboard), so the exact value is noisy.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

> STUDENT: draft below, edit so it is accurate.

Claude (Anthropic) was used to generate the code for data loading, the
preprocessing pipeline, the three model implementations, saving artifacts and
the Gradio dashboard, following decisions made by the student (dataset choice,
business framing, cost-based threshold, split strategy, features to build). It
also filled in the factual parts of this report (dataset facts, list of
processing steps, results tables from the training run). The analysis,
interpretation and conclusions in this report were written by the student.
