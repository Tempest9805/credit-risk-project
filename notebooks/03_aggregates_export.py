# -*- coding: utf-8 -*-
# %%
# notebooks/03_aggregates_export.py
# Genera agregados y features para dashboards y modelado
# Outputs (dashboards/):
#  - defaults_by_age_decile.csv
#  - defaults_by_income_decile.csv
#  - defaults_by_utilization_bucket.csv
#  - customer_scorecard_features.csv
#
# Ejecuta celda por celda en VSCode/Jupyter o ejecuta `python notebooks/03_aggregates_export.py`

# %%
import os
import json
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from IPython.display import display

# %% CONFIG
BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_FOLDER = os.path.join(BASE, 'data')
DASHBOARD_FOLDER = os.path.join(BASE, 'dashboards')
DB_PATH = os.path.join(DATA_FOLDER, 'credit.db')
SCHEMA_PATH = os.path.join(DATA_FOLDER, 'schema.json')
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

OUT_AGE = os.path.join(DASHBOARD_FOLDER, "defaults_by_age_decile.csv")
OUT_INCOME = os.path.join(DASHBOARD_FOLDER, "defaults_by_income_decile.csv")
OUT_UTIL = os.path.join(DASHBOARD_FOLDER, "defaults_by_utilization_bucket.csv")
OUT_SCORE = os.path.join(DASHBOARD_FOLDER, "customer_scorecard_features.csv")

print("BASE:", BASE)
print("DB path:", DB_PATH)
print("Dashboards out:", DASHBOARD_FOLDER)

# %% HELPERS
def load_clean_table(engine):
    for table_name in ("credit_table_clean", "credit_table"):
        try:
            df = pd.read_sql_table(table_name, con=engine)
            print(f"Loaded table from DB: {table_name} ({df.shape[0]} rows).")
            return df
        except Exception:
            continue
    csvp = os.path.join(DATA_FOLDER, 'raw', 'credit.csv')
    if os.path.exists(csvp):
        df = pd.read_csv(csvp)
        print(f"Loaded CSV fallback: {csvp} ({df.shape[0]} rows).")
        return df
    raise FileNotFoundError("No data found in DB (credit_table_clean/credit_table) or data/raw/credit.csv")

def safe_qcut(series, q=10, labels=None):
    
    try:
        if series.dropna().nunique() < 2:
            labels = labels or [0]
            return pd.Series([0]*len(series), index=series.index), [series.min(), series.max()]
        buckets = pd.qcut(series, q=q, labels=labels, duplicates='drop')
        return buckets, None
    except Exception:
        try:
            buckets = pd.cut(series, bins=q, labels=labels)
            return buckets, None
        except Exception:
            return pd.Series([np.nan]*len(series), index=series.index), None

# %% LOAD DATA
engine = create_engine(f"sqlite:///{DB_PATH}")
df = load_clean_table(engine)

if 'person_id' not in df.columns:
    df = df.reset_index().rename(columns={'index': 'person_id'})
if 'SeriousDlqin2yrs' in df.columns and 'default_flag' not in df.columns:
    df = df.rename(columns={'SeriousDlqin2yrs': 'default_flag'})

# %% BASIC FEATURE ENGINEERING (row-level)
df = df.copy()
late_cols = [c for c in ['NumberOfTime30-59DaysPastDueNotWorse',
                            'NumberOfTime60-89DaysPastDueNotWorse',
                            'NumberOfTimes90DaysLate'] if c in df.columns]
df['late_count'] = df[late_cols].sum(axis=1).astype(int) if late_cols else 0
df['open_lines'] = df['NumberOfOpenCreditLinesAndLoans'].astype(int) if 'NumberOfOpenCreditLinesAndLoans' in df.columns else 0
df['DebtRatio'] = df['DebtRatio'] if 'DebtRatio' in df.columns else 0.0
df['debt_ratio'] = pd.to_numeric(df['DebtRatio'], errors='coerce').fillna(0.0)
if 'MonthlyIncome' in df.columns:
    df['MonthlyIncome'] = pd.to_numeric(df['MonthlyIncome'], errors='coerce')
    df['income_flag'] = (df['MonthlyIncome'] > 0).astype(int)
else:
    df['MonthlyIncome'] = np.nan
    df['income_flag'] = 0
df['dependents'] = pd.to_numeric(df['NumberOfDependents'], errors='coerce').fillna(0).astype(int) if 'NumberOfDependents' in df.columns else 0
df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median() if 'age' in df.columns else np.nan).astype(int) if 'age' in df.columns else np.nan
df['default_flag'] = pd.to_numeric(df['default_flag'], errors='coerce').fillna(0).astype(int) if 'default_flag' in df.columns else 0

# %% AGG: Age buckets (deciles)
age_series = df['age'].dropna()
if not age_series.empty:
    age_labels = [f"Q{i+1}" for i in range(10)]
    age_buckets, _ = safe_qcut(df['age'], q=10, labels=age_labels)
    df['age_decile'] = age_buckets
else:
    df['age_decile'] = np.nan

agg_age = (
    df
    .groupby('age_decile', dropna=False)
    .agg(
        count=('person_id', 'count'),
        defaults=('default_flag', 'sum'),
        default_rate=('default_flag', lambda x: round(x.sum()/max(1, x.count())*100, 3)),
        median_income=('MonthlyIncome', 'median'),
        mean_debt_ratio=('debt_ratio', 'mean')
    )
    .reset_index()
    .sort_values(by='age_decile')
)
agg_age.to_csv(OUT_AGE, index=False)
print("Saved:", OUT_AGE)

# %% AGG: Income deciles -> default rates
if 'MonthlyIncome' in df.columns and df['MonthlyIncome'].notna().sum() > 0:
    income_labels = [f"Q{i+1}" for i in range(10)]
    income_buckets, _ = safe_qcut(df['MonthlyIncome'], q=10, labels=income_labels)
    df['income_decile'] = income_buckets
else:
    df['income_decile'] = np.nan

agg_income = (
    df
    .groupby('income_decile', dropna=False)
    .agg(
        count=('person_id', 'count'),
        defaults=('default_flag', 'sum'),
        default_rate=('default_flag', lambda x: round(x.sum()/max(1, x.count())*100, 3)),
        median_debt_ratio=('debt_ratio', 'median'),
        median_income=('MonthlyIncome', 'median')
    )
    .reset_index()
    .sort_values(by='income_decile')
)
agg_income.to_csv(OUT_INCOME, index=False)
print("Saved:", OUT_INCOME)

# %% AGG: Utilization buckets (equal-width bins, 10)
if 'RevolvingUtilizationOfUnsecuredLines' in df.columns:
    util_col = pd.to_numeric(df['RevolvingUtilizationOfUnsecuredLines'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['util'] = util_col
    df['util_bucket'] = pd.cut(df['util'], bins=10, labels=[f"B{i+1}" for i in range(10)], include_lowest=True)
    agg_util = (
        df
        .groupby('util_bucket', dropna=False)
        .agg(
            count=('person_id', 'count'),
            defaults=('default_flag', 'sum'),
            default_rate=('default_flag', lambda x: round(x.sum()/max(1, x.count())*100, 3)),
            median_income=('MonthlyIncome', 'median'),
            mean_debt_ratio=('debt_ratio', 'mean')
        )
        .reset_index()
    )
    agg_util.to_csv(OUT_UTIL, index=False)
    print("Saved:", OUT_UTIL)
else:
    print("No RevolvingUtilizationOfUnsecuredLines column — skipping util buckets.")
    df['util'] = np.nan
    df['util_bucket'] = np.nan

# %% SCORECARD FEATURES (person-level)
score_df = df[['person_id']].copy()
score_df = score_df.drop_duplicates(subset=['person_id']).set_index('person_id')

score_df['age'] = df.drop_duplicates('person_id').set_index('person_id')['age']
score_df['late_count'] = df.groupby('person_id')['late_count'].first()
score_df['open_lines'] = df.groupby('person_id')['open_lines'].first()
score_df['debt_ratio'] = df.groupby('person_id')['debt_ratio'].first()
score_df['monthly_income'] = df.groupby('person_id')['MonthlyIncome'].first()
score_df['income_flag'] = df.groupby('person_id')['income_flag'].first()
score_df['dependents'] = df.groupby('person_id')['dependents'].first()
score_df['age_decile'] = df.groupby('person_id')['age_decile'].first()
score_df['income_decile'] = df.groupby('person_id')['income_decile'].first()
score_df['util_bucket'] = df.groupby('person_id')['util_bucket'].first()
score_df['default_flag'] = df.groupby('person_id')['default_flag'].first()

score_df['high_util'] = (score_df['debt_ratio'] > score_df['debt_ratio'].median()).astype(int)
score_df['many_lates'] = (score_df['late_count'] >= 2).astype(int)

score_df.reset_index().to_csv(OUT_SCORE, index=False)
print("Saved:", OUT_SCORE)

# %% NOTES & CLEANUP
summary = {
    "rows": int(df.shape[0]),
    "unique_persons": int(score_df.shape[0]),
    "defaults_by_age_decile": OUT_AGE,
    "defaults_by_income_decile": OUT_INCOME,
    "defaults_by_utilization_bucket": OUT_UTIL,
    "customer_scorecard_features": OUT_SCORE
}
with open(os.path.join(DASHBOARD_FOLDER, 'aggregates_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

try:
    engine.dispose()
except Exception:
    pass

print("Done. Summary:", summary)
display(pd.read_csv(OUT_AGE).head(6))
display(pd.read_csv(OUT_INCOME).head(6))
try:
    display(pd.read_csv(OUT_SCORE).head(6))
except Exception:
    pass

