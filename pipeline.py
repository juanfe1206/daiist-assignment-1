"""
Shared code used by BOTH train.py and app.py.

Keeping these pieces in one place guarantees that the app transforms data and
computes the business cost exactly the way training did (no train/serve skew):

- load_raw()            : read the committed CSV and fix data-quality issues
- engineer_features()   : the feature engineering (pure function of one row,
                          so it is safe to apply before the split: nothing is
                          learned from the data here)
- business_cost()       : the cost model behind the decision threshold
- LogisticRegressionNN  : the nn.Module used by the "standard PyTorch" model
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
ARTIFACTS = ROOT / "artifacts"

# The 6 internet add-on services. For customers without internet these say
# "No internet service", which carries no extra information beyond
# InternetService == "No", so we collapse it to "No" (avoids a redundant dummy
# column per service that would be perfectly collinear with InternetService_No).
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns fed to the ColumnTransformer (after engineer_features()).
NUMERIC_FEATURES = [
    "SeniorCitizen",     # already 0/1
    "log_tenure",        # log1p(tenure): churn risk falls fastest in the first months
    "MonthlyCharges",    # current price level
    "charge_diff",       # MonthlyCharges - historical average charge (price-change signal)
    "n_addons",          # number of add-on services (engagement / switching cost)
    "fiber_x_monthly",   # interaction: fiber optic AND month-to-month contract
]
# Of the 6 add-ons only the two "protective" services (OnlineSecurity,
# TechSupport) are kept as individual dummies; the other four only enter via
# n_addons. Keeping all six dummies AND n_addons would be perfectly collinear
# (n_addons is exactly their sum), which makes the coefficients unidentifiable
# and plain gradient descent converge very slowly.
CATEGORICAL_FEATURES = [
    "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "TechSupport",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and fix the one known data-quality issue."""
    df = pd.read_csv(path)

    # TotalCharges is read as text because 11 rows contain a blank string " ".
    # All 11 have tenure == 0: brand-new customers who have not been billed yet.
    # They are NOT missing at random, so we do not drop or mean-impute them.
    # Choice: TotalCharges = MonthlyCharges (i.e. treat them as being in their
    # first billing month, consistent with tenure being clipped to 1 below).
    # This gives charge_diff = 0 for them ("no price change observed"), whereas
    # setting TotalCharges = 0 would fake a large price increase.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = total.fillna(df["MonthlyCharges"])

    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering. Learns nothing from the data, so applying
    it to the full dataset before splitting cannot leak information."""
    out = df.copy()

    # Collapse "No internet service" / "No phone service" into "No" (see above).
    for col in ADDON_COLS:
        out[col] = out[col].replace("No internet service", "No")
    out["MultipleLines"] = out["MultipleLines"].replace("No phone service", "No")

    # Tenure: churn hazard is highest in the first months and then flattens,
    # which a log transform makes closer to linear in the logit.
    out["log_tenure"] = np.log1p(out["tenure"])

    # Engagement: how many add-ons the customer pays for (0-6).
    out["n_addons"] = (out[ADDON_COLS] == "Yes").sum(axis=1)

    # Price-change signal: average historical monthly charge vs the current one.
    # Positive charge_diff = the customer pays more now than they used to.
    avg_charge = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["charge_diff"] = out["MonthlyCharges"] - avg_charge

    # Interaction: fiber optic and month-to-month are the two strongest risk
    # factors; a linear model can only express "both together is worse than the
    # sum" through an explicit product term.
    out["fiber_x_monthly"] = (
        (out["InternetService"] == "Fiber optic") & (out["Contract"] == "Month-to-month")
    ).astype(int)

    # Dropped on purpose:
    # - customerID: an identifier, no predictive meaning (would only memorise rows)
    # - TotalCharges: ~ MonthlyCharges * tenure, so it is nearly collinear with
    #   the two; its useful extra information is kept in charge_diff
    # - tenure (raw): replaced by log_tenure
    # - gender: churn rates are almost identical by gender, and targeting a
    #   commercial offer on gender is a fairness concern we do not need
    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def business_cost(y_true, contact, monthly_charges, offer_cost, horizon_months, save_rate):
    """Total cost (in $) of a contact decision on a set of customers.

    Per customer:
      - not contacted & churns (FN): lose MonthlyCharges * horizon_months
      - contacted & churns     (TP): pay the offer, and still lose the revenue
                                     with probability (1 - save_rate)
      - contacted & stays      (FP): pay the offer for nothing
      - not contacted & stays  (TN): 0
    """
    y_true = np.asarray(y_true).astype(bool)
    contact = np.asarray(contact).astype(bool)
    lost = np.asarray(monthly_charges, dtype=float) * horizon_months
    cost = np.where(contact, offer_cost, 0.0)
    cost += np.where(y_true & ~contact, lost, 0.0)
    cost += np.where(y_true & contact, (1 - save_rate) * lost, 0.0)
    return float(cost.sum())


class LogisticRegressionNN(nn.Module):
    """Logistic regression as a PyTorch module: one linear layer.
    forward() returns logits; the sigmoid lives in the loss (BCEWithLogitsLoss)
    for numerical stability, and is applied explicitly at prediction time."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
