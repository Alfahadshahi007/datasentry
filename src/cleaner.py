import pandas as pd
from sklearn.linear_model import LinearRegression

GENDER_MAP = {
    "f": "Female",
    "fem": "Female",
    "female": "Female",
    "females": "Female",
    "frmakes": "Female",
    "m": "Male",
    "male": "Male",
    "males": "Male",
}

WORD_NUMBERS = {
    "thirty": 30,
    "twenty": 20,
    "forty": 40,
    "fifty": 50,
}


def clean_dataset(
    df: pd.DataFrame,
    drop_dups: bool = False,
    standardize_text: bool = False,
    drop_sparse_cols: bool = True,
    sparsity_threshold: float = 0.60,
    missing_method: str = "None",
) -> tuple[pd.DataFrame, list[str]]:
    clean_df = df.copy()
    logs: list[str] = []

    # 1. Remove duplicate rows
    if drop_dups:
        before = len(clean_df)
        clean_df = clean_df.drop_duplicates()
        removed = before - len(clean_df)
        if removed > 0:
            logs.append(f"Purged {removed} duplicate record(s).")

    # 2. Standardize text formatting + canonicalize domain aliases (e.g., Gender)
    if standardize_text:
        text_cols = clean_df.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in text_cols:
            clean_df[col] = clean_df[col].apply(
                lambda x: x.strip().title() if isinstance(x, str) else x
            )
            if "gender" in col.lower():
                clean_df[col] = clean_df[col].apply(
                    lambda x: GENDER_MAP.get(x.strip().lower(), x) if isinstance(x, str) else x
                )
        if text_cols:
            logs.append(
                f"Standardized casing, whitespace, and canonical categories across {len(text_cols)} text column(s)."
            )

    # 3. Coerce numeric columns polluted by text tokens (e.g., 'thirty', 'not_available')
    for col in clean_df.columns:
        if clean_df[col].dtype in ["object", "string"]:
            lowered = clean_df[col].astype(str).str.strip().str.lower()
            mapped = lowered.replace(WORD_NUMBERS)
            numeric_candidate = pd.to_numeric(mapped, errors="coerce")
            non_null = clean_df[col].notna().sum()
            if non_null > 0 and (numeric_candidate.notna().sum() / non_null) >= 0.20:
                clean_df[col] = numeric_candidate
                logs.append(f"Coerced '{col}' from mixed text tokens into numeric dtype.")

    # 3b. Drop columns exceeding the missing-value sparsity threshold
    if drop_sparse_cols and len(clean_df) > 0:
        cols_to_drop = []
        for col in clean_df.columns:
            missing_ratio = clean_df[col].isna().sum() / len(clean_df)
            if missing_ratio > sparsity_threshold:
                cols_to_drop.append((col, round(missing_ratio * 100, 1)))
        if cols_to_drop:
            clean_df = clean_df.drop(columns=[c for c, _ in cols_to_drop])
            for col, pct in cols_to_drop:
                logs.append(
                    f"Dropped column '{col}' ({pct}% missing — exceeded {int(sparsity_threshold * 100)}% sparsity threshold)."
                )

    # 4. Handle missing numerical values
    num_cols = clean_df.select_dtypes(include=["number"]).columns.tolist()

    if missing_method == "Column Median":
        for col in num_cols:
            missing_count = int(clean_df[col].isnull().sum())
            if missing_count > 0:
                median_val = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(median_val)
                logs.append(f"Imputed {missing_count} gap(s) in '{col}' using median ({median_val}).")

    elif missing_method == "Linear Interpolation (Between Rows)":
        for col in num_cols:
            missing_count = int(clean_df[col].isnull().sum())
            if missing_count > 0:
                clean_df[col] = clean_df[col].interpolate(method="linear").bfill().ffill().round(2)
                logs.append(f"Interpolated {missing_count} gap(s) in '{col}' between adjacent rows.")

    elif missing_method == "Linear Regression Prediction":
        base_df = clean_df.copy()
        for target_col in num_cols:
            missing_count = int(base_df[target_col].isnull().sum())
            if missing_count == 0:
                continue

            feature_cols = [c for c in num_cols if c != target_col and base_df[c].isnull().sum() == 0]
            known = base_df[base_df[target_col].notnull()]
            missing = base_df[base_df[target_col].isnull()]

            if feature_cols and len(known) >= 2:
                model = LinearRegression()
                model.fit(known[feature_cols], known[target_col])
                preds = model.predict(missing[feature_cols])
                clean_df.loc[clean_df[target_col].isnull(), target_col] = preds.round(2)
                logs.append(
                    f"Predicted {missing_count} value(s) in '{target_col}' via Linear Regression using {feature_cols}."
                )
            else:
                median_val = clean_df[target_col].median()
                clean_df[target_col] = clean_df[target_col].fillna(median_val)
                logs.append(
                    f"Filled '{target_col}' with median ({median_val}) — insufficient complete predictors."
                )

    return clean_df, logs