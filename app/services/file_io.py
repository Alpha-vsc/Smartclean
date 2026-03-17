# ==============================================================================
# SmartClean — File I/O Service
# ==============================================================================

import io
import os
import xml.etree.ElementTree as ET

import pandas as pd

from .cleaner import EXTRA_NA

SUPPORTED_FORMATS = {"csv", "xlsx", "xls", "json", "xml", "tsv", "parquet"}
EXPORT_FORMATS    = {"csv", "xlsx", "json", "xml", "tsv", "parquet"}


# ── Reading ────────────────────────────────────────────────────────────────────

def read_file(source, extension: str) -> pd.DataFrame:
    """
    Read *source* (file-like object or path) as a DataFrame.
    *extension* must be lowercase (e.g. 'csv', 'xlsx').
    """
    ext = extension.lower()

    if ext == "csv":
        return pd.read_csv(source, na_values=EXTRA_NA, keep_default_na=True)

    if ext in ("xlsx", "xls"):
        return pd.read_excel(source, na_values=EXTRA_NA, keep_default_na=True)

    if ext == "json":
        try:
            return pd.read_json(source)
        except ValueError:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_json(source, orient="records")

    if ext == "xml":
        try:
            return pd.read_xml(source)
        except Exception:
            # Fallback: parse manually and retry with explicit xpath
            path = source if isinstance(source, (str, os.PathLike)) else _write_temp(source)
            tree = ET.parse(path)
            root = tree.getroot()
            children = list(root)
            if children:
                return pd.read_xml(path, xpath=f"./{children[0].tag}")
            raise ValueError("Unsupported XML structure")

    if ext == "tsv":
        return pd.read_csv(source, sep="\t")

    if ext == "parquet":
        return pd.read_parquet(source)

    raise ValueError(f"Unsupported format: {ext}")


def _write_temp(file_obj) -> str:
    """Write a FileStorage / file-like object to a temp path and return it."""
    import tempfile
    suffix = ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_obj.read())
        return f.name


# ── Writing ────────────────────────────────────────────────────────────────────

def write_file(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Write *df* to *path* in the requested format."""
    fmt = fmt.lower()
    # Normalise pandas 2.x StringDtype → object for compatibility
    for col in df.columns:
        if str(df[col].dtype) in ("string", "StringDtype") or hasattr(df[col].dtype, "na_value"):
            df[col] = df[col].astype(object)

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "xlsx":
        df.to_excel(path, index=False, engine="openpyxl")
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    elif fmt == "xml":
        df.to_xml(path, index=False)
    elif fmt == "tsv":
        df.to_csv(path, sep="\t", index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
