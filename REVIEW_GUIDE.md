# Review guide (Assignment 1)

One-page walkthrough of what was built on branch `assignment-1`, so you can
check it, change anything you disagree with, and defend every decision.
Delete this file before submitting if you prefer (it is not required).

## How to run

```bash
uv sync
uv run python main.py train   # ~40 s; prints metrics, writes artifacts/
uv run python main.py app     # Gradio on http://127.0.0.1:7860 (open it yourself)
```

## Files

| File | What it does |
|---|---|
| `data/telco_churn.csv` | IBM Telco churn CSV, committed so training never needs the network |
| `pipeline.py` | Shared by train and app: `load_raw` (TotalCharges fix), `engineer_features`, `business_cost`, the `nn.Module` |
| `train.py` | Cost constants at the top; split, preprocessing, 3 models, thresholds, metrics, saves `artifacts/` |
| `app.py` | Gradio dashboard; loads `artifacts/` only, re-scores the saved test set with the saved models (inference, no training) and asserts it matches training's predictions |
| `artifacts/` | Committed: preprocessor + sklearn model (joblib), `torch_manual.pt` (w, b), `torch_nn.pt` (state_dict), `model_meta.json` (features, thresholds, costs, C), `metrics.json/.csv/.md`, `test_predictions.csv`, `test_set.csv`, `distributions.csv` |
| `REPORT.md` | Facts and tables filled in; `NOTES FOR YOU` blocks list what you must write yourself |
| `train.ipynb`, `app.ipynb` | Deleted: `main.py` refuses to run when both `X.py` and `X.ipynb` exist |

## Design decisions (one line each)

Framing and evaluation
- Contact if P(churn) >= threshold; threshold minimises total $ cost on **validation**, because errors are asymmetric (miss ~ $900, offer $50).
- Cost constants (`OFFER_COST=50`, `HORIZON_MONTHS=12`, `SAVE_RATE=0.30`) are assumptions, editable at the top of `train.py` and live in the dashboard.
- Cost is per customer (uses that customer's `MonthlyCharges`), so high-value misses cost more.
- Stratified random 60/20/20 split, seed 42: no date column, so no time split; stratified keeps 26.5% churn in every set.
- `tenure` is a feature (customer age with the company at snapshot time), not a time index.
- C and threshold chosen on validation; test is used once for the final table.
- Baselines: contact nobody (= predict "no churn") and contact everyone, both costed the same way.

Preprocessing and features
- Blank `TotalCharges` (11 rows, tenure 0) -> `MonthlyCharges`: they are new customers, not random missing; 0 would fake a price rise in `charge_diff`.
- Drop `customerID` (identifier), `gender` (no signal: 26.9% vs 26.2%, and a fairness concern for offers), `TotalCharges` (~ MonthlyCharges x tenure), raw `tenure` (replaced by its log).
- "No internet service"/"No phone service" -> "No": the information is already in `InternetService`/`PhoneService`.
- `log_tenure`: churn risk drops fastest in the first months; log makes that closer to linear.
- `n_addons`: engagement/switching cost; the 4 non-protective add-ons enter only through it, `OnlineSecurity` and `TechSupport` are kept as dummies (strongest single add-on effects: among internet users churn is ~15% with vs ~42% without; streaming barely differs, 30% vs 34%). Keeping all 6 dummies + the count is exactly collinear.
- `charge_diff`: price-change signal (current vs historical average monthly charge).
- `fiber_x_monthly`: interaction of the two biggest risk factors (a linear model cannot learn it alone).
- Engineering runs before the split: it is row-wise and learns nothing, so no leakage. Scaler/encoder are fitted on train only.
- `StandardScaler` on numerics (fair L2 penalty, fast gradient descent); `OneHotEncoder(drop="first", handle_unknown="ignore")` (no collinearity with the intercept; unseen category -> reference level).

Models
- Same objective everywhere: mean BCE + ||w||^2/(2*C*n_train), bias unpenalised = sklearn's `C*sum(logloss) + 0.5||w||^2`.
- C grid {0.01, 0.1, 1, 10} by validation log loss -> C = 10 (edge of grid; regularisation barely matters here).
- float64 in torch so the three can match to ~1e-6.
- Manual PyTorch: plain full-batch GD, lr 2.0 (stable because lr < 2/L, L ~ 0.92), 30,000 epochs, stable BCE `max(z,0) - z*y + log(1+exp(-|z|))`.
- Standard PyTorch: `nn.Linear` + `BCEWithLogitsLoss` + `LBFGS` (same quasi-Newton family as sklearn's solver), L2 added to the loss by hand (not `weight_decay`, which would also penalise the bias).
- Initial weights zero (problem is convex, so the start does not change the optimum).

Dashboard
- Tabs: model comparison (metrics + agreement tables, calibration plot, P(churn) histograms by true class), distributions (dropdown, coloured by churn, incl. gender and the 4 add-ons that were not used directly), threshold & cost (slider, editable costs, confusion matrix, cost curve with validation-chosen threshold and test-curve minimum marked), coefficients (top 15, three models side by side).
- `server_name="127.0.0.1", server_port=7860, inbrowser=False` so headless CI does not block.

## Headline results (test set, 1,409 customers)

| Model | ROC-AUC | PR-AUC | Thr. | Precision | Recall | F1 | Cost |
|---|---|---|---|---|---|---|---|
| sklearn / manual torch / nn torch (identical) | 0.8399 | 0.6558 | 0.28 | 0.521 | 0.783 | 0.626 | $284,573 |
| Contact nobody | - | - | - | - | 0 | 0 | $337,971 |
| Contact everyone | - | - | - | 0.265 | 1.0 | 0.420 | $307,030 |

Saving vs best baseline: ~$22.5k (7.3%); vs a 0.5 threshold: ~$13.6k. Max coefficient difference between implementations 1.4e-6, max probability difference 5.9e-7.

## What did not work / what I am unsure about (honest)

- First version kept all one-hot levels + all 6 add-on dummies + `n_addons`: exact collinearity made plain GD crawl (coef diff ~2, prob diff ~0.018 after 20k steps, AUC still equal). Fixed by removing the collinearity and raising lr to 2.0. Good story for the report.
- `MonthlyCharges` is still nearly a linear function of the services (eigenvalue ~0.0014), so its coefficient (+1.23) and fiber's (-0.20, counter-intuitive sign) should not be read on their own.
- `charge_diff` and `fiber_x_monthly` add almost nothing (coefs -0.015, +0.043). Kept because they were justified a priori; you may choose to drop them and say so.
- C = 10 sits at the edge of the grid; larger C would be marginally better on validation but differences are tiny (log loss 0.3962 vs 0.3959 for C 1 vs 10).
- The optimal threshold under assumed costs is 0.28 and the cost curve is flat near it; with other cost assumptions it moves a lot (try the dashboard inputs).
- The dashboard's cost-curve minimum is computed on test (for exploration only); the official threshold is the validation one.
- Artifacts were generated on Windows; CI regenerates them on Linux. Results should be identical up to float noise.
