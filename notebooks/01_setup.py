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
# #%pip install pandas
# #%pip install sqlalchemy

import os
import pandas as pd
from sqlalchemy import create_engine
import sys
import json
#############################################################################end cell###############################################
# %%
BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))
RAW = os.path.join(BASE, 'data', 'raw', 'credit.csv')
DB_PATH = os.path.join(BASE, 'data', 'credit.db')

print("BASE:", BASE)
print("RAW (string):", RAW)
print("DB_PATH (string):", DB_PATH)
#############################################################################end cell###############################################

# %%
# --- Comprobación: existe el CSV?---
if not os.path.exists(RAW):
    raise FileNotFoundError(f"No encontré {RAW}. Mueve el csv a data/raw/credit.csv o actualiza la ruta.")
print("CSV encontrado ✅ :", RAW)
#############################################################################end cell###############################################

# %%
# --- Leer CSV ---
df = pd.read_csv(RAW)
print("Shape:", df.shape)
print("First columns:", df.columns.tolist()[:12])

try:
    from IPython.display import display
    display(df.head())
except Exception:
    print(df.head().to_string())
#############################################################################end cell###############################################

# %%
# --- Crear person_id si no existe y renombrar columna target ---
if 'person_id' not in df.columns:
    df = df.reset_index().rename(columns={'index': 'person_id'})
    print("Added person_id from index (0-based).")

# Renombrar SeriousDlqin2yrs -> default_flag 
if 'SeriousDlqin2yrs' in df.columns:
    df = df.rename(columns={'SeriousDlqin2yrs': 'default_flag'})
    print("Renamed SeriousDlqin2yrs -> default_flag")

print("Columns now:", df.columns.tolist())
#############################################################################end cell###############################################

# %%
# --- Asegurar carpeta data y crear la base SQLite ---
data_folder = os.path.join(BASE, 'data')
os.makedirs(data_folder, exist_ok=True)

# engine string para sqlite (usa DB_PATH)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
#############################################################################end cell###############################################

# %%
# --- Volcar DataFrame a SQLite (chunksize para seguridad) ---
df.to_sql(
    'credit_table',
    engine,
    if_exists='replace',
    index=False,
    method='multi',
    chunksize=max(1, 999 // max(1, len(df.columns)))
)
print("Saved credit_table to SQLite at:", DB_PATH)
#############################################################################end cell###############################################

# %%
# --- Sanity checks: conteo y tipos ---
from sqlalchemy import text

with engine.connect() as conn:
    rows = conn.execute(text("SELECT COUNT(*) FROM credit_table")).scalar_one()
print("Rows in DB table:", rows)

# Mostrar dtypes en pandas
print("\nDtypes (pandas):")
print(df.dtypes)

# Mostrar esquema sqlite
with engine.connect() as conn:
    res = conn.execute(text("PRAGMA table_info(credit_table);")).all()
print("\nSQLite schema (PRAGMA table_info):")
for r in res:
    print(r)
#############################################################################end cell###############################################


# %%
# --- Guardar metadatos (schema) tipos de datos ---
meta = {col: str(dtype) for col, dtype in df.dtypes.items()}
meta_path = os.path.join(data_folder, 'schema.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("Saved schema metadata to:", meta_path)
#############################################################################end cell###############################################

# %%
