# tools/preprocessing_tools.py
"""
Preprocessing tools (CrewAI BaseTool implementations).

Exposed tools:
    • DiscretizeFeatureTool
        Discretizes a numeric column using equal-width bins, equal-frequency bins,
        or custom edges; optionally replaces the original column.
    • OneHotEncodeFeatureTool
        One-hot encodes a categorical column using `pandas.get_dummies()`.

Usage requirements:
    - Attach a pandas DataFrame before running:
        tool.dataset = df
    - Inputs are validated with Pydantic; tools return concise, human-readable text.
    - Decisions are logged via `CTX.add_decision(...)` for traceability.
"""

from __future__ import annotations

from typing import List, Optional, Type, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from crewai.tools import BaseTool

from tfg_ml.context import CTX


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _get_df(tool: BaseTool) -> pd.DataFrame:
    """
    Retrieve the DataFrame attached to a tool.

    Raises:
        ValueError: If no DataFrame is attached.
    """
    df = getattr(tool, "dataset", None)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("No dataset assigned to tool. Set `tool.dataset = df` before running.")
    return df


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
        - Optionally drops the original numeric column.
    """
    name: str = "discretize_feature"
    description: str = "Discretize a numeric column (cut/qcut or custom edges)."
    args_schema: Type[BaseModel] = DiscretizeFeatureInput
    dataset: Optional[pd.DataFrame] = None  # set by the orchestrator/coordinator

    def _run(self, **kwargs) -> str:
        """
        Execute discretization according to validated `kwargs`.
        """
        df = _get_df(self)
        column = kwargs["column"]

        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            return f"Column '{column}' must be numeric."

        strategy = kwargs.get("strategy", "equal_width")
        labels = kwargs.get("labels")
        right = kwargs.get("right", True)
        include_lowest = kwargs.get("include_lowest", True)
        drop_original = kwargs.get("drop_original", True)
        new_column = kwargs.get("new_column") or f"{column}_binned"

        if strategy == "equal_width":
            cut = pd.cut(series, bins=kwargs["bins"], labels=labels, right=right, include_lowest=include_lowest)
        elif strategy == "equal_freq":
            cut = pd.qcut(series, q=kwargs["q"], labels=labels, duplicates="drop")
            if labels and len(labels) != cut.cat.categories.size:
                return "Provided labels length doesn't match effective quantile bins."
        else:  # custom_edges
            cut = pd.cut(series, bins=kwargs["edges"], labels=labels, right=right, include_lowest=include_lowest)

        if new_column in df.columns:
            new_column = _safe_new_col_name(df, new_column)

        df[new_column] = cut
        if drop_original:
            df.drop(columns=[column], inplace=True)

        desc = df[new_column].describe().to_string()
        CTX.add_decision("preprocessing", f"Discretize '{column}' -> '{new_column}'")
        return f"Created '{new_column}'.\n{desc}"


# ---------------------------------------------------------------------
# OneHotEncodeFeature
# ---------------------------------------------------------------------
class OneHotEncodeInput(BaseModel):
    """
    Structured inputs for OneHotEncodeFeatureTool.

    Parameters
    ----------
    column : str
        Categorical column to one-hot encode.
    prefix : str | None
        Prefix for new dummy columns (defaults to source column name).
    drop_first : bool
        If True, drop the first level to avoid multicollinearity.
    dtype : str | None
        Dtype for the dummy columns (e.g., 'uint8').
    drop_original : bool
        If True, drop the original categorical column.
    """
    column: str
    prefix: Optional[str] = None
    drop_first: bool = True
    dtype: Optional[str] = "uint8"
    drop_original: bool = True


class OneHotEncodeFeatureTool(BaseTool):
    """
    One-hot encode a categorical column using `pandas.get_dummies()`.

    Side effects:
        - Adds the generated dummy columns (with collision-safe names).
        - Optionally drops the original column.
        - Returns a compact boolean/dummy summary for the new columns.
    """
    name: str = "one_hot_encode_feature"
    description: str = "One-hot encode a categorical column using pandas.get_dummies()."
    args_schema: Type[BaseModel] = OneHotEncodeInput
    dataset: Optional[pd.DataFrame] = None  # set by the orchestrator/coordinator

    def _run(
        self,
        column: str,
        prefix: Optional[str] = None,
        drop_first: bool = True,
        dtype: Optional[str] = "uint8",
        drop_original: bool = True,
    ) -> str:
        df = _get_df(self)

        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        pref = prefix or column
        dummies = pd.get_dummies(df[column], prefix=pref, drop_first=drop_first, dtype=dtype)

        # Ensure no name collisions with existing columns
        new_cols: List[str] = []
        for c in dummies.columns:
            new_cols.append(c if c not in df.columns else _safe_new_col_name(df, c))
        dummies.columns = new_cols

        df[new_cols] = dummies
        if drop_original:
            df.drop(columns=[column], inplace=True)

        CTX.add_decision("preprocessing", f"OneHot '{column}' -> {len(new_cols)} columnas")
        return "One-hot encoding created columns:\n" + ", ".join(new_cols) + "\n\n" + _bool_col_summary(df, new_cols)
