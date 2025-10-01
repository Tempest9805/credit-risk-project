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
# notebooks/02_dq_checks.py - Data Quality Checks Optimizado
import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# %%
# --- CONFIGURACIÓN DE PATHS ---
BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_FOLDER = os.path.join(BASE, 'data')
DASHBOARD_FOLDER = os.path.join(BASE, 'dashboards')
DB_PATH = os.path.join(DATA_FOLDER, 'credit.db')
SCHEMA_PATH = os.path.join(DATA_FOLDER, 'schema.json')

OUT_SUMMARY = os.path.join(DASHBOARD_FOLDER, "dq_metrics.csv")
OUT_ISSUES = os.path.join(DASHBOARD_FOLDER, "dq_issues_sample.csv")
OUT_NOTES = os.path.join(DASHBOARD_FOLDER, "dq_notes.txt")

os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

print("Project BASE:", BASE)
print("DB path:", DB_PATH)
print("Schema path:", SCHEMA_PATH)
print("Dashboards output folder:", DASHBOARD_FOLDER)

# %%
# --- FUNCIONES PRINCIPALES ---
def load_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    try:
        df = pd.read_sql_table("credit_table", engine)
        print("✅ Datos cargados desde SQLite")
    except:
        csv_path = os.path.join(DATA_FOLDER, 'raw', 'credit.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("✅ Datos cargados desde CSV (fallback)")
        else:
            raise FileNotFoundError("No se encontraron datos en DB ni CSV")
    
    # Normalizaciones básicas
    if 'person_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'person_id'})
    if 'SeriousDlqin2yrs' in df.columns:
        df = df.rename(columns={'SeriousDlqin2yrs': 'default_flag'})
    
    return df, engine

def load_schema():
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, 'r') as f:
            return json.load(f)
    return {}

def count_rule_violations(df, col, rule):

    if col not in df.columns: return None
    
    series, mask = df[col], df[col].notna()
    
    if "allowed" in rule:
        return int((mask & ~series.isin(rule["allowed"])).sum())
    
    violations = 0
    if rule.get("min"): violations += (mask & (series < rule["min"])).sum()
    if rule.get("max"): violations += (mask & (series > rule["max"])).sum()
    
    return int(violations)

# %%
# --- REGLAS DE NEGOCIO ---
RULES = {
    "age": {"min": 18, "max": 120},
    "MonthlyIncome": {"min": 0},
    "DebtRatio": {"min": 0},
    "default_flag": {"allowed": [0, 1]},
}

# %%
# --- CARGA INICIAL ---
df, engine = load_data()
schema = load_schema()
print(f"📊 Dataset inicial: {len(df)} filas, {len(df.columns)} columnas")

# %%
# --- ANÁLISIS PRE-LIMPIEZA ---
def calculate_metrics(df):

    n_rows = len(df)
    
    missing = df.isna().sum().to_frame('missing_count')
    missing['missing_pct'] = (missing['missing_count'] / n_rows * 100).round(3)
    
    violations = []
    for col, rule in RULES.items():
        viol = count_rule_violations(df, col, rule)
        if viol is not None and viol > 0:
            violations.append({'column': col, 'violations': viol})
    
    # Duplicados
    dups = df.duplicated('person_id').sum() if 'person_id' in df.columns else 0
    
    return {
        'n_rows': n_rows,
        'missing': missing.reset_index().rename(columns={'index': 'column'}),
        'violations': pd.DataFrame(violations) if violations else pd.DataFrame(),
        'duplicates': dups,
        'dtypes': df.dtypes.astype(str)
    }

pre_metrics = calculate_metrics(df)
print("📈 Métricas pre-limpieza calculadas")

# %%
# --- LIMPIEZA DE DATOS ---
def clean_data(df):
    df_clean = df.copy()
    
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Unnamed: 0'])
        print("✅ Columna 'Unnamed: 0' eliminada")
    
    imputation_config = {
        'MonthlyIncome': ('MonthlyIncome_missing', 'median'),
        'NumberOfDependents': ('NumberOfDependents_missing', 'median'),
    }
    
    for col, (flag_col, method) in imputation_config.items():
        if col in df_clean.columns:
            missing_mask = df_clean[col].isna()
            if missing_mask.any():
                df_clean[flag_col] = missing_mask.astype(int)
                impute_value = df_clean[col].median() if method == 'median' else df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(impute_value)
                print(f"💰 {col}: {missing_mask.sum()} valores imputados")
    
    if 'age' in df_clean.columns:
        invalid_age = (df_clean['age'] < 18) | (df_clean['age'] > 120)
        if invalid_age.any():
            df_clean.loc[invalid_age, 'age'] = np.nan
            df_clean['age_missing'] = df_clean['age'].isna().astype(int)
            median_age = df_clean['age'].median()
            df_clean['age'] = df_clean['age'].fillna(median_age).astype('int64')
            print(f"👶 Age: {invalid_age.sum()} valores corregidos")
    
    return df_clean

df_clean = clean_data(df)
print("🧹 Limpieza completada")

# %%
# --- MÉTRICAS POST-LIMPIEZA ---
post_metrics = calculate_metrics(df_clean)

new_schema = {col: str(dtype) for col, dtype in df_clean.dtypes.items()}
with open(SCHEMA_PATH, 'w') as f:
    json.dump(new_schema, f, indent=2)

print("📊 Métricas post-limpieza calculadas")

# %%
# --- DETECCIÓN DE ISSUES ---
def detect_issues(df):
    issues = pd.DataFrame(index=df.index)
    
    issues['missing_any'] = df.isna().any(axis=1)
    
    for col, rule in RULES.items():
        if col in df.columns:
            issue_name = f'issue_{col}'
            if 'allowed' in rule:
                issues[issue_name] = ~df[col].isin(rule['allowed']) & df[col].notna()
            else:
                violation = pd.Series(False, index=df.index)
                if rule.get('min'): violation |= (df[col] < rule['min'])
                if rule.get('max'): violation |= (df[col] > rule['max'])
                issues[issue_name] = violation & df[col].notna()
    
    if 'person_id' in df.columns:
        issues['dup_person_id'] = df.duplicated('person_id')

    issues['dq_issues_list'] = issues.apply(
        lambda row: ';'.join([col for col, val in row.items() if val]), 
        axis=1
    )
    
    return issues

issues = detect_issues(df_clean)
df_with_issues = df_clean.copy()
df_with_issues['dq_issues_list'] = issues['dq_issues_list']

has_issues = issues['dq_issues_list'] != ''
issues_count = has_issues.sum()
issues_sample = df_with_issues[has_issues].sample(
    n=min(50, issues_count), 
    random_state=42
) if issues_count > 0 else pd.DataFrame()

print(f"🔍 Issues detectados: {issues_count} filas")

# %%
# --- GUARDAR RESULTADOS ---
def save_results(df, metrics, issues_sample):
    """Guardar todos los outputs"""
    # 1. Guardar tabla limpia
    df.to_sql('credit_table_clean', engine, if_exists='replace', index=False)
    print("✅ Tabla limpia guardada en SQLite")
    
    # 2. Métricas consolidadas - CORREGIDO
    metrics_df = metrics['missing'].merge(
        pd.DataFrame({'column': metrics['dtypes'].index, 'dtype': metrics['dtypes'].values}),
        on='column'
    )
    
    # Manejar violaciones correctamente
    if not metrics['violations'].empty:
        metrics_df = metrics_df.merge(metrics['violations'], on='column', how='left')
        metrics_df['rule_violations'] = metrics_df['violations'].fillna(0).astype(int)
        metrics_df = metrics_df.drop(columns=['violations'])
    else:
        metrics_df['rule_violations'] = 0
    
    metrics_df.to_csv(OUT_SUMMARY, index=False)
    print(f"✅ Métricas guardadas: {OUT_SUMMARY}")
    
    # 3. Sample de issues
    issues_sample.to_csv(OUT_ISSUES, index=False)
    print(f"✅ Issues sample guardado: {OUT_ISSUES}")
    
    # 4. Notas automáticas
    notes = [
        "ANÁLISIS DE CALIDAD DE DATOS - POST LIMPIEZA",
        "=" * 50,
        f"Dataset: {len(df)} filas, {len(df.columns)} columnas",
        f"Filas con issues: {issues_count} ({issues_count/len(df)*100:.1f}%)",
        f"Duplicados: {metrics['duplicates']}",
        "",
        "IMPUTACIONES:"
    ]
    
    # Agregar info de imputaciones
    for col in ['MonthlyIncome_missing', 'NumberOfDependents_missing', 'age_missing']:
        if col in df.columns:
            count = int(df[col].sum())
            notes.append(f"• {col.replace('_missing', '')}: {count} valores imputados")
    
    # Issues remanentes
    if issues_count > 0:
        notes.extend(["", "ISSUES REMANENTES:", "Revisar dq_issues_sample.csv para detalles"])
    
    # Violaciones de reglas post-limpieza - CORREGIDO
    current_issues = []
    if not metrics['violations'].empty:
        for _, row in metrics['violations'].iterrows():
            current_issues.append(f"{row['column']}: {int(row['violations'])} violaciones")
    
    if current_issues:
        notes.extend(["", "VIOLACIONES DE REGLAS:", *[f"• {issue}" for issue in current_issues]])
    
    with open(OUT_NOTES, 'w') as f:
        f.write('\n'.join(notes))
    
    print(f"✅ Notas guardadas: {OUT_NOTES}")

save_results(df_clean, post_metrics, issues_sample)

# %%
# --- RESUMEN FINAL ---
print("\n" + "="*50)
print(" ANÁLISIS DE CALIDAD COMPLETADO")
print("="*50)
print(f" Dataset final: {len(df_clean)} filas × {len(df_clean.columns)} columnas")
print(f" Filas con issues: {issues_count} ({issues_count/len(df_clean)*100:.1f}%)")
print(f" Duplicados: {post_metrics['duplicates']}")
print(f" Tabla 'credit_table_clean' guardada en SQLite")

if issues_count == 0:
    print("✅ ¡Calidad de datos óptima!")
else:
    print(f"ℹ️  Revisar {issues_count} filas con issues en dq_issues_sample.csv")


# %%
# --- VERIFICACIÓN DE IMPUTACIONES ---
print("🔍 VERIFICACIÓN DE IMPUTACIONES:")
imputation_cols = ['MonthlyIncome_missing', 'NumberOfDependents_missing', 'age_missing']
for col in imputation_cols:
    if col in df_clean.columns:
        count = int(df_clean[col].sum())
        pct = (count / len(df_clean) * 100)
        print(f"   {col}: {count} valores imputados ({pct:.2f}%)")

# Verificar que no hay valores fuera de rango en age
if 'age' in df_clean.columns:
    valid_age = ((df_clean['age'] >= 18) & (df_clean['age'] <= 120)).all()
    print(f"   Age en rango válido [18,120]: {valid_age}")

# Verificar tipos de datos
print("\n📊 TIPOS DE DATOS FINALES:")
print(df_clean.dtypes.value_counts())

