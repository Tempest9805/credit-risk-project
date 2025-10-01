# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# notebooks/04_model_pd.py
# Train a Probability of Default (PD) model (logistic regression) + calibration
# Outputs:
#  - models/pd_logistic.joblib
#  - dashboards/pd_predictions.csv
#  - dashboards/pd_model_metrics.json
#  - dashboards/pd_calibration.csv
#
# Run cell-by-cell in VSCode/Jupyter or `python notebooks/04_model_pd.py`

# %%
#%pip install scikit-learn

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
import joblib
from IPython.display import display

# %% CONFIG PATHS
BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))
DASHBOARD_FOLDER = os.path.join(BASE, 'dashboards')
MODELS_FOLDER = os.path.join(BASE, 'models')
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

SCORECARD_CSV = os.path.join(DASHBOARD_FOLDER, "customer_scorecard_features.csv")
OUT_PRED = os.path.join(DASHBOARD_FOLDER, "pd_predictions.csv")
OUT_METRICS = os.path.join(DASHBOARD_FOLDER, "pd_model_metrics.json")
OUT_CAL = os.path.join(DASHBOARD_FOLDER, "pd_calibration.csv")
MODEL_FILE = os.path.join(MODELS_FOLDER, "pd_logistic.joblib")

RANDOM_STATE = 42

print("BASE:", BASE)
print("Loading scorecard from:", SCORECARD_CSV)

# %%
# --- LOAD DATA ---
if not Path(SCORECARD_CSV).exists():
    raise FileNotFoundError(f"No scorecard CSV found at {SCORECARD_CSV}. Run 03_aggregates_export.py first.")

df = pd.read_csv(SCORECARD_CSV)
print("Loaded scorecard rows:", len(df))
display(df.head(3))

# %% PREP: select features & target

TARGET_COL = 'default_flag'
ID_COL = 'person_id'

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {SCORECARD_CSV}")


feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]

X = df[feature_cols].copy()
X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
y = df[TARGET_COL].astype(int).values
ids = df[ID_COL].values

print("Using features:", feature_cols)
print("Target distribution (counts):")
print(pd.Series(y).value_counts())

# %% SPLIT train/test
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train/test sizes:", X_train.shape, X_test.shape)

# %% BUILD pipeline (scaler + logistic)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        solver='liblinear', 
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    ))
])

# Quick cross-val AUC on train
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print("CV ROC-AUC (train):", auc_scores, "mean:", auc_scores.mean())

# %% FIT + CALIBRATION
# Fit base pipeline
pipe.fit(X_train, y_train)


calibrated_clf = CalibratedClassifierCV(estimator=pipe, cv=5, method="sigmoid")
calibrated_clf.fit(X_train, y_train)

# %% EVAL on test
y_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print("\n--- Test Metrics ---")
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))
print("Brier score:", round(brier, 4))

# basic classification at 0.5
y_pred = (y_pred_proba >= 0.5).astype(int)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# %% CALIBRATION CURVE
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

cal_df = pd.DataFrame({
    "prob_pred_mean": prob_pred,
    "prob_true": prob_true
})
cal_df.to_csv(OUT_CAL, index=False)
print("Saved calibration curve ->", OUT_CAL)

# %% SAVE MODEL + METRICS
joblib.dump(calibrated_clf, MODEL_FILE)
print("Saved calibrated model ->", MODEL_FILE)

metrics = {
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "brier_score": float(brier),
    "cv_auc_mean": float(auc_scores.mean())
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics ->", OUT_METRICS)

# %% EXPORT PREDICTIONS (all rows)
all_proba = calibrated_clf.predict_proba(X)[:, 1]
df_pred = pd.DataFrame({
    ID_COL: ids,
    "true_default": y,
    "predicted_pd": all_proba
})


df_pred["pd_bucket"] = pd.qcut(df_pred["predicted_pd"], 10, labels=[f"Q{i}" for i in range(1, 11)])
df_pred.to_csv(OUT_PRED, index=False)
print("Saved predictions ->", OUT_PRED)

display(df_pred.head(10))
print("04_model_pd.py finished successfully 🚀")


