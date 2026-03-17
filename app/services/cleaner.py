# ==============================================================================
# SmartClean — Data Cleaning Service
# ==============================================================================

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── NA variant detection ───────────────────────────────────────────────────────

NA_VARIANTS = {"na", "n/a", "nan", "null", "none", "--", "-", "n.a.", "n.a", "?", "missing", ""}
EXTRA_NA    = ["na", "n/a", "NA", "N/A", "null", "NULL", "none", "None", "NONE",
               "--", "-", "?", "missing", "MISSING", "n.a.", "N.A."]

# Regex to detect identifier columns by name
_ID_NAME_RE = re.compile(
    r"(^id$|_id$|^pid$|^idx$|^index$|^key$|^pk$|^ref$|_ref$|^uuid$|_uuid$)",
    re.IGNORECASE,
)


def is_id_column(series: pd.Series, col_name: str) -> bool:
    """Return True if the column looks like a unique identifier that should not be modified."""
    if _ID_NAME_RE.search(col_name):
        return True
    non_null = series.dropna()
    if len(non_null) < 10:
        return False
    return (non_null.nunique() / len(non_null)) >= 0.95


# ── NA normalisation ───────────────────────────────────────────────────────────

def normalize_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Replace all textual NA variants with real NaN, then attempt numeric coercion
    for columns that are mostly numeric.
    Returns (cleaned_df, list_of_change_messages).
    """
    changes: list[str] = []
    df = df.copy()

    # Unify StringDtype → object
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)

    for col in df.columns:
        # Step 1 — replace NA variants
        str_vals = df[col].apply(lambda v: "" if pd.isna(v) else str(v).strip().lower())
        mask = str_vals.isin(NA_VARIANTS)
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            changes.append(f"'{col}': {n} non-standard value(s) (na, n/a, --…) → NaN")

        # Step 2 — numeric coercion when ≥50% of column is numeric
        if df[col].dtype == object:
            already_nan = df[col].isna()
            coerced     = pd.to_numeric(df[col], errors="coerce")
            new_nan     = coerced.isna()
            invalid     = (~already_nan) & new_nan
            valid_num   = (~already_nan) & (~new_nan)
            non_null    = int((~already_nan).sum())
            mostly_num  = non_null > 0 and int(valid_num.sum()) / non_null >= 0.5

            if mostly_num:
                n_invalid = int(invalid.sum())
                if n_invalid:
                    ex = df.loc[invalid, col].iloc[0]
                    changes.append(f"'{col}': {n_invalid} invalid value(s) (e.g. '{ex}') → NaN")
                df[col] = coerced

    # Step 3 — auto-convert to integer where ≥80% of values have no decimals
    for col in df.select_dtypes(include=[np.number]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        pct_integer = (non_null % 1 == 0).sum() / len(non_null)
        if pct_integer >= 0.8:
            df[col] = df[col].round().astype("Int64")
            changes.append(f"'{col}': converted to integer (Int64)")

    return df, changes


# ── Main cleaning pipeline ─────────────────────────────────────────────────────

def process_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Full cleaning pipeline.
    config keys: duplicates, missing_values, outliers, normalize.
    Returns (cleaned_df, list_of_change_messages).
    """
    changes: list[str] = []

    # 1. Normalize NA values
    df, na_changes = normalize_missing(df)
    changes.extend(na_changes)

    # 2. Remove duplicates
    if config.get("duplicates"):
        before = len(df)
        df     = df.drop_duplicates()
        n      = before - len(df)
        if n:
            changes.append(f"{n} duplicate row(s) removed")

    # 3. Auto-detect categorical columns
    for col in df.select_dtypes(include=[np.number]).columns:
        n_unique     = df[col].nunique()
        n_total      = len(df[col].dropna())
        unique_ratio = n_unique / n_total if n_total else 1
        if not is_id_column(df[col], col) and n_unique <= 10 and unique_ratio <= 0.3:
            df[col] = df[col].astype("category")
            changes.append(f"'{col}': converted to categorical ({n_unique} unique values)")

    # 4. Fix Y/N columns — invalid values → NaN
    for col in df.columns:
        if df[col].dtype == object:
            vals_upper = df[col].dropna().str.upper().unique()
            if any(v in ("Y", "N") for v in vals_upper):
                invalid_mask = df[col].notna() & ~df[col].str.upper().isin(["Y", "N"])
                n_inv = int(invalid_mask.sum())
                if n_inv:
                    df.loc[invalid_mask, col] = np.nan
                    changes.append(f"'{col}': {n_inv} invalid Y/N value(s) → NaN")

    # 5. Fill missing values (median / mode)
    if config.get("missing_values"):
        filled = 0
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            if is_id_column(df[col], col):
                changes.append(f"'{col}': {n_missing} missing value(s) — ID column, not modified")
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null  = pd.to_numeric(df[col].dropna(), errors="coerce").dropna()
                fill_val  = float(df[col].median())
                if len(non_null) > 0 and (non_null % 1 == 0).all():
                    df[col] = df[col].fillna(int(round(fill_val)))
                else:
                    df[col] = df[col].fillna(round(fill_val, 1))
            else:
                modes  = df[col].dropna().mode()
                df[col] = df[col].fillna(modes.iloc[0] if not modes.empty else "N/A")
            filled += n_missing
        if filled:
            changes.append(f"{filled} missing value(s) filled (median / mode)")

    # 6. Remove outliers (IQR method)
    if config.get("outliers"):
        before  = len(df)
        skipped = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dropna().count() < 30:
                skipped.append(col)
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr     = q3 - q1
            if iqr == 0:
                continue
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
        n = before - len(df)
        if n:
            changes.append(f"{n} outlier(s) removed (IQR method)")
        if skipped:
            changes.append(f"Outlier detection skipped for {len(skipped)} column(s) (<30 rows)")

    # 7. Min-Max normalisation (excludes ID columns)
    if config.get("normalize"):
        num_cols          = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [c for c in num_cols if not is_id_column(df[c], c)]
        if cols_to_normalize:
            df[cols_to_normalize] = MinMaxScaler().fit_transform(df[cols_to_normalize])
            changes.append(
                f"Min-Max normalisation applied ({len(cols_to_normalize)} column(s), ID columns excluded)"
            )

    if not changes:
        changes.append("No changes applied (data was already clean)")

    return df, changes
