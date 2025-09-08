# tools/preprocessing_tools.py
"""
Preprocessing tools (CrewAI BaseTool implementations).

Exposed tools:
    • DiscretizeFeatureTool
        Discretizes a numeric column using equal-width bins, equal-frequency bins,
        or custom edges; optionally replaces the original column.
    • OneHotEncodeFeatureTool
        One-hot encodes a categorical column or all categorical columns using `pandas.get_dummies()`.
    • NullCleanerColumnTool
        Cleans NaNs in a single column by dropping rows or imputing values.
    • DropColumnTool
        Drops one or more columns from the dataset.

Usage requirements:
    
    - Inputs are validated with Pydantic; tools return concise, human-readable text.
    - Each tool reads from and writes to a CSV dataset at `dataset_path`.
    - Side effects (if any) are documented per tool."""

from __future__ import annotations

from typing import List, Optional, Type, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from crewai.tools import BaseTool

from tfg_ml.context import CTX

dataset_path="data/dataset.csv"




def _safe_new_col_name(df: pd.DataFrame, base: str) -> str:
    """
    Return a unique column name based on `base` that does not collide
    with existing columns in `df`.
    """
    name, i = base, 1
    while name in df.columns:
        name = f"{base}_{i}"
        i += 1
    return name


def _bool_col_summary(df: pd.DataFrame, cols: List[str]) -> str:
    """
    Produce a compact summary for boolean/dummy columns.

    For each column, reports number of ones/zeros/NaNs and mean.
    """
    parts = []
    for c in cols:
        s = df[c]
        ones = int((s == 1).sum())
        zeros = int((s == 0).sum())
        na = int(s.isna().sum())
        parts.append(f"{c}: ones={ones}, zeros={zeros}, NaN={na}, mean={float(s.mean()):.4f}")
    return "\n".join(parts)


# ---------------------------------------------------------------------
# DiscretizeFeature
# ---------------------------------------------------------------------
class DiscretizeFeatureInput(BaseModel):
    """
    Structured inputs for DiscretizeFeatureTool.

    Parameters
    ----------
    column : str
        Source numeric column to discretize.
    strategy : {'equal_width','equal_freq','custom_edges'}
        Binning strategy. For 'equal_width' provide `bins`; for 'equal_freq' provide `q`;
        for 'custom_edges' provide `edges`.
    bins : int | None
        Number of equal-width bins (>= 2) when strategy='equal_width'.
    q : int | None
        Number of quantile bins (>= 2) when strategy='equal_freq'.
    edges : list[float] | None
        Strictly increasing bin edges (len >= 2) when strategy='custom_edges'.
    labels : list[str] | None
        Optional labels for bins (must match effective number of bins).
    right : bool
        Whether bins include the rightmost edge (passed to `pd.cut`).
    include_lowest : bool
        Include the lowest value in the first interval (passed to `pd.cut`).
    drop_original : bool
        If True, drop the original column after creating the binned column.
    new_column : str | None
        Name of the new binned column; auto-suffixed if it collides.
    """
    column: str
    strategy: Literal["equal_width", "equal_freq", "custom_edges"] = "equal_width"
    bins: Optional[int] = None
    q: Optional[int] = None
    edges: Optional[List[float]] = None
    labels: Optional[List[str]] = None
    right: bool = True
    include_lowest: bool = True
    drop_original: bool = True
    new_column: Optional[str] = None
    if_exists: Literal["reuse", "overwrite", "suffix", "fail"] = "reuse" 
    @field_validator("bins")
    @classmethod
    def _check_bins(cls, v, info):
        if info.data.get("strategy") == "equal_width" and (v is None or v < 2):
            raise ValueError("'bins' must be >= 2 for equal_width.")
        return v

    @field_validator("q")
    @classmethod
    def _check_q(cls, v, info):
        if info.data.get("strategy") == "equal_freq" and (v is None or v < 2):
            raise ValueError("'q' must be >= 2 for equal_freq.")
        return v

    @field_validator("edges")
    @classmethod
    def _check_edges(cls, v, info):
        if info.data.get("strategy") == "custom_edges":
            if not v or len(v) < 2 or any(v[i] >= v[i + 1] for i in range(len(v) - 1)):
                raise ValueError("Provide strictly increasing edges (len>=2).")
        return v


class DiscretizeFeatureTool(BaseTool):
    """
    Discretize a numeric column via equal-width, equal-frequency, or custom edges.

    Side effects:
        - Adds the new (categorical) binned column.
    """
    name: str = "discretize_feature"
    description: str = "Discretize a numeric column (cut/qcut or custom edges)."
    args_schema: Type[BaseModel] = DiscretizeFeatureInput

    def _run(self, **kwargs) -> str:
        """
        Execute discretization according to validated `kwargs`.
        """
        df = pd.read_csv(dataset_path)
        
        column = kwargs["column"]

        

        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            return f"Column '{column}' must be numeric."

        strategy = kwargs.get("strategy", "equal_width")
        labels = kwargs.get("labels")
        right = kwargs.get("right", True)
        include_lowest = kwargs.get("include_lowest", True)
        new_column = kwargs.get("new_column") or f"{column}_binned"
        if_exists = kwargs.get("if_exists", "reuse")

        if strategy == "equal_width":
            cut = pd.cut(series, bins=kwargs["bins"], labels=labels, right=right, include_lowest=include_lowest)
        elif strategy == "equal_freq":
            cut = pd.qcut(series, q=kwargs["q"], labels=labels, duplicates="drop")
            if labels and len(labels) != cut.cat.categories.size:
                return "Provided labels length doesn't match effective quantile bins."
        else:  # custom_edges
            cut = pd.cut(series, bins=kwargs["edges"], labels=labels, right=right, include_lowest=include_lowest)

        if new_column in df.columns:
            if if_exists == "reuse":
                desc = df[new_column].describe().to_string()
                return f"'{new_column}' already exists. Reusing.\n{desc}"
            elif if_exists == "overwrite":
                df[new_column] = cut
            elif if_exists == "suffix":
                new_column = _safe_new_col_name(df, new_column)
                df[new_column] = cut
            elif if_exists == "fail":
                return (f"Column '{new_column}' already exists. "
                        f"Use if_exists='overwrite' or 'suffix' to proceed.")
        else:
            df[new_column] = cut
        
        
        
        df.to_csv(dataset_path, index=False)
        desc = df[new_column].describe().to_string()
        
        return f"Created '{new_column}'.\n{desc}"


# ---------------------------------------------------------------------
# OneHotEncodeFeature
# ---------------------------------------------------------------------
class OneHotEncodeInput(BaseModel):
    """
    Inputs for OneHotEncodeFeatureTool.

    You can specify exactly ONE of:
      - column: a single column
      - columns: a list of columns
      - all_categoricals=True: auto-detect all object/category columns
    """
    # modos de selección
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    all_categoricals: bool = False

    # opciones de codificación
    prefix: Optional[str] = None              # si no se indica, usa el nombre de cada columna
    drop_first: bool = False                  # deja todas las categorías por defecto
    dtype: Optional[str] = "uint8"
    if_exists: Literal["reuse", "overwrite", "suffix", "fail"] = "reuse"


class OneHotEncodeFeatureTool(BaseTool):
    """
    One-hot encode one, many or all categorical columns using pandas.get_dummies().

    Side effects:
      - Adds/updates dummy columns.
      - Saves to dataset_path.
      - Returns an exact, human-readable summary of changes.
    """
    name: str = "one_hot_encode_feature"
    description: str = "One-hot encode one, many, or all categorical columns."
    args_schema: Type[BaseModel] = OneHotEncodeInput

    def _run(
        self,
        column: Optional[str] = None,
        columns: Optional[List[str]] = None,
        all_categoricals: bool = False,
        prefix: Optional[str] = None,
        drop_first: bool = False,
        dtype: Optional[str] = "uint8",
        if_exists: str = "reuse",
    ) -> str:
        df = pd.read_csv(dataset_path)

        selected: List[str] = []

        if all_categoricals:
            selected = df.select_dtypes(include=["object", "category"]).columns.tolist()
        elif columns is not None:
            selected = [c for c in columns]
        elif column is not None:
            selected = [column]
        else:
            return (
                "Please specify one of: 'column', 'columns', or set 'all_categoricals=True'."
            )

        missing = [c for c in selected if c not in df.columns]
        selected = [c for c in selected if c in df.columns]
        if not selected:
            return f"No valid columns to encode. Missing: {missing}" if missing else "No columns provided."

        created_global: List[str] = []
        overwritten_global: List[str] = []
        reused_global: List[str] = []
        per_col_created: dict = {}
        per_col_overwritten: dict = {}
        per_col_reused: dict = {}

        for col in selected:
            pref = (prefix or col)

            # generar dummies de esa columna
            dummies = pd.get_dummies(df[col], prefix=pref, drop_first=drop_first, dtype=dtype)
            dummy_cols = list(dummies.columns)

            created: List[str] = []
            overwritten: List[str] = []
            reused: List[str] = []

            for c in dummy_cols:
                if c in df.columns:
                    if if_exists == "reuse":
                        reused.append(c)  # no tocar
                    elif if_exists == "overwrite":
                        df[c] = dummies[c]
                        overwritten.append(c)
                    elif if_exists == "suffix":
                        new_c = _safe_new_col_name(df, c)
                        df[new_c] = dummies[c]
                        created.append(new_c)
                    elif if_exists == "fail":
                        return (f"Column '{c}' already exists. "
                                f"Use if_exists='overwrite' or 'suffix' to proceed.")
                else:
                    df[c] = dummies[c]
                    created.append(c)

            # acumular
            created_global.extend(created)
            overwritten_global.extend(overwritten)
            reused_global.extend(reused)
            if created:    per_col_created[col] = created
            if overwritten: per_col_overwritten[col] = overwritten
            if reused:     per_col_reused[col] = reused

        df.to_csv(dataset_path, index=False)

        parts: List[str] = []
        parts.append("[one_hot_encode_feature]")
        parts.append(f"Status: success")
        parts.append(f"Columns encoded: {', '.join(selected)}")
        if missing:
            parts.append(f"Missing (skipped): {', '.join(missing)}")
        parts.append(f"Options → drop_first={drop_first}, dtype={dtype}, if_exists={if_exists}, prefix={'<per-column>' if prefix is None else prefix}")

        if created_global:
            parts.append("Created: " + ", ".join(created_global))
        if overwritten_global:
            parts.append("Overwritten: " + ", ".join(overwritten_global))
        if reused_global:
            parts.append("Reused (kept existing): " + ", ".join(reused_global))
        if not (created_global or overwritten_global or reused_global):
            parts.append("No changes.")

        if per_col_created:
            for k, v in per_col_created.items():
                parts.append(f"Created for '{k}': " + ", ".join(v))
        if per_col_overwritten:
            for k, v in per_col_overwritten.items():
                parts.append(f"Overwritten for '{k}': " + ", ".join(v))
        if per_col_reused:
            for k, v in per_col_reused.items():
                parts.append(f"Reused for '{k}': " + ", ".join(v))

        return "\n".join(parts)

class NullCleanerColumnInput(BaseModel):
    """
    Clean NaN values in a single column.

    Parameters
    ----------
    column : str
        Column name to clean.
    mode : {'drop_rows','impute'}
        - 'drop_rows': drop rows with NaN in this column.
        - 'impute': fill NaNs with a value depending on the strategy.
    numeric_strategy : {'median','mean','zero'}
        Strategy for imputing numeric columns (only if mode='impute').
    categorical_strategy : {'most_frequent','constant'}
        Strategy for imputing categorical columns (only if mode='impute').
    constant_value : str
        Value to use when categorical_strategy='constant'.
    persist : bool
        If True, save the cleaned dataset back to dataset_path.
    """
    column: str
    mode: Literal["drop_rows", "impute"] = "impute"
    numeric_strategy: Literal["median", "mean", "zero"] = "median"
    categorical_strategy: Literal["most_frequent", "constant"] = "most_frequent"
    constant_value: str = "Unknown"


class NullCleanerColumnTool(BaseTool):
    """
    Clean NaNs in a single column (drop rows or impute).
    """
    name: str = "null_cleaner_column"
    description: str = "Clean NaNs in a single column (drop rows or impute)."
    args_schema: Type[BaseModel] = NullCleanerColumnInput

    def _run(
        self,
        column: str,
        mode: str = "impute",
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        constant_value: str = "Unknown",
    ) -> str:

        df = pd.read_csv(dataset_path)
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        n_rows_before = len(df)
        na_before = int(df[column].isna().sum())

        if mode == "drop_rows":
            df = df.dropna(subset=[column])
            action_desc = f"Dropped {na_before} rows with NaN in column '{column}'."

        elif mode == "impute":
            if pd.api.types.is_numeric_dtype(df[column]):
                if numeric_strategy == "median":
                    fill_val = df[column].median()
                elif numeric_strategy == "mean":
                    fill_val = df[column].mean()
                else:  # zero
                    fill_val = 0
                df[column] = df[column].fillna(fill_val)
                action_desc = f"Imputed NaN in '{column}' with {numeric_strategy} ({fill_val})."

            else:  # categorical
                if categorical_strategy == "most_frequent":
                    mode_val = df[column].mode(dropna=True)
                    fill_val = mode_val.iloc[0] if not mode_val.empty else constant_value
                else:  # constant
                    fill_val = constant_value
                df[column] = df[column].fillna(fill_val)
                action_desc = f"Imputed NaN in '{column}' with {categorical_strategy} ('{fill_val}')."
        else:
            return "Invalid mode. Use 'drop_rows' or 'impute'."

        n_rows_after = len(df)
        na_after = int(df[column].isna().sum())

        
        df.to_csv(dataset_path, index=False)

        return "\n".join([
            f"NullCleanerColumn(mode={mode}, column={column})",
            action_desc,
            f"Rows: {n_rows_before} -> {n_rows_after}",
            f"NaNs in '{column}': {na_before} -> {na_after}",
            f"Saved to: {dataset_path}",
        ])
    

class DropColumnInput(BaseModel):
    """
    Drop one or more columns from the dataset.

    Parameters
    ----------
    columns : List[str]
        List of column names to drop.
    """
    columns: List[str]


class DropColumnTool(BaseTool):
    """
    Drop one or more columns from the dataset.
    Changes are always persisted to dataset_path.
    """
    name: str = "drop_column"
    description: str = "Remove one or more columns from the dataset."
    args_schema: type = DropColumnInput

    def _run(self, columns: List[str]) -> str:
        df = pd.read_csv(dataset_path)
        

        n_rows, n_cols_before = df.shape
        df = df.drop(columns=columns, errors="ignore")
        n_rows_after, n_cols_after = df.shape

        df.to_csv(dataset_path, index=False)

        return "\n".join([
            f"Dropped columns: {columns}",
            f"Shape: {n_rows}x{n_cols_before} -> {n_rows_after}x{n_cols_after}",
            f"Saved to: {dataset_path}",
        ])
    
