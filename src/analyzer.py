import pandas as pd
import numpy as np

def analyze_dataset(df: pd.DataFrame) -> dict:
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    
    # Missing values analysis
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    # Duplicate analysis
    duplicate_rows = int(df.duplicated().sum())
    
    # Data type summary
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Outlier detection (IQR method for numeric columns)
    outliers_by_col = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        if outlier_count > 0:
            outliers_by_col[col] = outlier_count
            
    # Quality score calculation (0 - 100%)
    issue_weight = total_missing + (duplicate_rows * total_cols)
    quality_score = max(0, round(100 - (issue_weight / (total_cells + 1e-5) * 100), 1))
    
    return {
        "rows": total_rows,
        "cols": total_cols,
        "missing_by_col": missing_by_col,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "dtypes": dtypes,
        "outliers_by_col": outliers_by_col,
        "quality_score": quality_score
    }