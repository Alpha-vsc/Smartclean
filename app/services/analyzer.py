# ==============================================================================
# SmartClean — Data Analysis Service
# ==============================================================================

import pandas as pd


def analyze_dataframe(df: pd.DataFrame, filename: str) -> dict:
    """
    Build a full statistical profile of *df*.
    Returns a JSON-serialisable dict.
    """
    col_stats = []
    for col in df.columns:
        s    = df[col]
        stat = {
            "name":        col,
            "dtype":       str(s.dtype),
            "total":       len(s),
            "missing":     int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 1),
            "unique":      int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            nn = s.dropna()
            if len(nn):
                stat.update(
                    {
                        "min":    round(float(nn.min()),    4),
                        "max":    round(float(nn.max()),    4),
                        "mean":   round(float(nn.mean()),   4),
                        "median": round(float(nn.median()), 4),
                        "std":    round(float(nn.std()),    4),
                    }
                )
        else:
            top = s.dropna().value_counts().head(3).to_dict()
            stat["top_values"] = {str(k): int(v) for k, v in top.items()}
        col_stats.append(stat)

    return {
        "filename":      filename,
        "rows":          len(df),
        "cols":          len(df.columns),
        "total_missing": int(df.isna().sum().sum()),
        "duplicates":    int(df.duplicated().sum()),
        "columns":       col_stats,
    }
