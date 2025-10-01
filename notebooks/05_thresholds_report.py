# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# notebooks/05_thresholds_report.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# %%
BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))
DASH = os.path.join(BASE, 'dashboards')
PRED_CSV = os.path.join(DASH, 'pd_predictions.csv')
OUT_THRESH = os.path.join(DASH, 'pd_thresholds.csv')
OUT_PLOT1 = os.path.join(DASH, 'pd_thresholds_plot.png')
OUT_PLOT2 = os.path.join(DASH, 'pd_decision_metrics.png')

# %%
os.makedirs(DASH, exist_ok=True)

# %%
df = pd.read_csv(PRED_CSV)
if 'true_default' not in df.columns:
    raise SystemExit("pd_predictions.csv needs 'true_default' column to compute metrics.")

# %%
y_true = df['true_default'].values
y_score = df['predicted_pd'].values

# %%
thresholds = np.linspace(0.01, 0.99, 99)
rows = []
for t in thresholds:
    y_pred = (y_score >= t).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    flagged_count = int(y_pred.sum())
    defaults_in_flagged = int(((y_true==1) & (y_pred==1)).sum())
    default_rate_flagged = (defaults_in_flagged / max(1, flagged_count)) * 100
    rows.append({
        'threshold': round(float(t),3),
        'precision': round(float(prec),4),
        'recall': round(float(rec),4),
        'f1': round(float(f1),4),
        'flagged_count': flagged_count,
        'defaults_in_flagged': defaults_in_flagged,
        'default_rate_flagged_pct': round(default_rate_flagged,3)
    })

# %%
th_df = pd.DataFrame(rows)
th_df.to_csv(OUT_THRESH, index=False)
print("Saved thresholds table ->", OUT_THRESH)

# %%
# Plot precision/recall/f1 vs threshold
plt.figure(figsize=(8,5))
plt.plot(th_df['threshold'], th_df['precision'], label='precision')
plt.plot(th_df['threshold'], th_df['recall'], label='recall')
plt.plot(th_df['threshold'], th_df['f1'], label='f1')
plt.xlabel('threshold')
plt.ylabel('score')
plt.title('Precision / Recall / F1 vs threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT1, dpi=150)
print("Saved plot ->", OUT_PLOT1)

# %%
# Plot flagged_count and default_rate_flagged_pct on twin axes
fig, ax1 = plt.subplots(figsize=(8,4))
ax2 = ax1.twinx()
ax1.plot(th_df['threshold'], th_df['flagged_count'], label='flagged_count', color='C0')
ax2.plot(th_df['threshold'], th_df['default_rate_flagged_pct'], label='default_rate_pct', color='C1')
ax1.set_xlabel('threshold')
ax1.set_ylabel('flagged_count')
ax2.set_ylabel('default_rate_flagged_pct')
ax1.grid(alpha=0.2)
plt.title('Flagged volume and default rate by threshold')
fig.tight_layout()
plt.savefig(OUT_PLOT2, dpi=150)
print("Saved plot ->", OUT_PLOT2)

# %%
print("Top thresholds preview:")
print(th_df.sort_values('f1', ascending=False).head(10).to_string(index=False))
