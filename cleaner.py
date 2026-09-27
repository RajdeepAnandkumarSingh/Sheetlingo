import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

def clean_and_profile_dataframe(df: pd.DataFrame, score_cutoff: int = 80) -> pd.DataFrame:
    """
    Normalizes column headers, performs fuzzy typo standardization on categorical columns,
    and removes exact duplicate records.
    """
    df = df.copy()
    
    # 1. Clean Column Names (strip whitespace, normalize spaces)
    df.columns = [str(col).strip() for col in df.columns]
    
    # 2. Fuzzy Typo Standardization on Categorical Columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        
        val_counts = df[col].dropna().value_counts().to_dict()
        unique_vals = list(val_counts.keys())
        
        replacement_map = {}
        for val in unique_vals:
            if not isinstance(val, str) or len(val.strip()) < 3:
                continue
            
            matches = process.extract(val, unique_vals, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
            candidates = [m[0] for m in matches]
            best_canonical = sorted(candidates, key=lambda x: (-val_counts.get(x, 0), -len(x)))[0]
            
            if best_canonical != val:
                replacement_map[val] = best_canonical
                
        if replacement_map:
            df[col] = df[col].replace(replacement_map)

    # 3. Drop duplicate rows
    df = df.drop_duplicates()
    return df

def detect_anomalies(df: pd.DataFrame, target_col: str, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detects statistical outliers in a numerical target column using Isolation Forest.
    """
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        df_copy = df.copy()
        clean_target = df_copy[[target_col]].fillna(df_copy[target_col].median())
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(clean_target)
        df_copy['is_anomaly'] = preds
        anomalies = df_copy[df_copy['is_anomaly'] == -1].copy()
        return anomalies
    return pd.DataFrame()

def impute_missing_values(df: pd.DataFrame, method: str = 'knn') -> pd.DataFrame:
    """
    Imputes missing values using KNNImputer or MICE (IterativeImputer).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if len(num_cols) > 0:
        if method == 'mice':
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df[num_cols] = imputer.fit_transform(df[num_cols])
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df[num_cols] = imputer.fit_transform(df[num_cols])
        else:
            for col in num_cols:
                df[col] = df[col].fillna(df[col].median())
                
    for col in cat_cols:
        mode_val = df[col].mode()
        fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)
        
    return df

def encode_categorical_features(df: pd.DataFrame, method: str = 'onehot') -> pd.DataFrame:
    """
    Encodes categorical features for machine learning readiness using One-Hot or Label Encoding.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        return df
        
    if method == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    elif method == 'label':
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

def generate_data_profile(df: pd.DataFrame) -> dict:
    """
    Returns advanced statistical profiling metrics including skewness, kurtosis, and data hygiene score.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    null_counts = df.isnull().sum().to_dict()
    null_percentages = {col: round((count / max(total_rows, 1)) * 100, 2) for col, count in null_counts.items()}
    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Skewness calculation for numeric columns
    num_df = df.select_dtypes(include=[np.number])
    skewness = num_df.skew().round(2).to_dict() if not num_df.empty else {}
    
    # Data Hygiene Score (0 - 100%)
    total_cells = total_rows * max(total_cols, 1)
    total_nulls = sum(null_counts.values())
    hygiene_score = round(max(0, 100 - (total_nulls / max(total_cells, 1) * 100)), 1)
    
    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_types": col_types,
        "null_counts": null_counts,
        "null_percentages": null_percentages,
        "total_missing_cells": total_nulls,
        "duplicate_rows": int(df.duplicated().sum()),
        "skewness": skewness,
        "data_hygiene_score": hygiene_score
    }
