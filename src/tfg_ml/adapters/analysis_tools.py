# tools/analysis_tools.py
"""
Analysis tools (CrewAI BaseTool implementations) for EDA-style actions.

Exposed tools:
    - DescribeFeatureTool: Summarize a single column with `pandas.Series.describe()`
      and (for categorical/object) show the top 10 values including NaNs.
    - ComputeStatisticTool: Compute a single statistic on a column, optionally
      grouped by another column.

Usage notes:
    • A pandas DataFrame must be attached prior to execution:
        tool.dataset = df
    • Both tools rely on structured Pydantic input schemas.
    • Context logging is recorded via `CTX.add_decision(...)`.

Dependencies:
    - pandas, numpy, pydantic, crewai
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from tfg_ml.context import CTX

path="data/dataset.csv"


# ---------------------------------------------------------------------
# DescribeFeature
# ---------------------------------------------------------------------
class DescribeFeatureInput(BaseModel):
    """Structured inputs for DescribeFeatureTool."""
    column: str = Field(..., description="Column name to describe.")
    percentiles: Optional[List[float]] = Field(
        None,
        description="Percentiles in [0, 1] passed to pandas describe().",
    )
    include: Optional[Union[str, List[str]]] = Field(
        None,
        description="Included dtypes for DataFrame.describe(); ignored for Series.",
    )
    


class DescribeFeatureTool(BaseTool):
    """
    Describe a single column using `pandas.Series.describe()`.

    For object/categorical columns, also includes a 'Top 10 values (including NaN)'
    frequency table.

    Important:
        Ensure `self.dataset` is set to a pandas DataFrame before invoking.
    """
    name: str = "describe_feature"
    description: str = "Describe a specific column using pandas describe()."
    args_schema: Type[BaseModel] = DescribeFeatureInput

    def _run(
        self,
        column: str,
        percentiles: Optional[List[float]] = None,
        include: Optional[Union[str, List[str]]] = None,  # note: ignored for Series
    ) -> str:
        df = pd.read_csv(path)
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        ser = df[column]
        desc = ser.describe(percentiles=percentiles) if percentiles else ser.describe()

        lines: List[str] = [f"Description for '{column}':", desc.to_string()]

        # Show top categories for non-numeric columns
        if is_object_dtype(ser) or is_categorical_dtype(ser):
            vc = ser.value_counts(dropna=False).head(10)
            lines += ["", "Top 10 values (including NaN):", vc.to_string()]

        out = "\n".join(lines)
        return out


# ---------------------------------------------------------------------
# ComputeStatistic
# ---------------------------------------------------------------------
SUPPORTED_STATS = {
    "mean", "median", "mode", "std", "var", "min", "max", "sum",
    "count", "nunique", "skew", "kurtosis",
}


class ComputeStatisticInput(BaseModel):
    """Structured inputs for ComputeStatisticTool."""
    column: str = Field(..., description="Column to compute on.")
    stat: str = Field(..., description=f"One of: {sorted(SUPPORTED_STATS)}")
    dropna: bool = Field(True, description="Drop NaNs before computing.")
    groupby: Optional[str] = Field(None, description="Optional grouping column.")


class ComputeStatisticTool(BaseTool):
    """
    Compute a single statistic on a column, optionally grouped by another column.

    Supported statistics:
        mean, median, mode, std, var, min, max, sum, count, nunique, skew, kurtosis

    Important:
        Ensure `self.dataset` is set to a pandas DataFrame before invoking.
    """
    name: str = "compute_statistic"
    description: str = "Compute a statistic over a column, optionally grouped."
    args_schema: Type[BaseModel] = ComputeStatisticInput

    def _run(
        self,
        column: str,
        stat: str,
        dropna: bool = True,
        groupby: Optional[str] = None,
    ) -> str:
        df = pd.read_csv(path)

        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        stat = stat.lower()
        if stat not in SUPPORTED_STATS:
            return f"Unsupported stat '{stat}'. Supported: {sorted(SUPPORTED_STATS)}"

        def compute_on_series(s: pd.Series) -> Any:
            if dropna:
                s = s.dropna()

            # Numeric-ish ops: keep behavior consistent with numpy/pandas defaults.
            if stat == "mean":
                return float(np.mean(s))
            if stat == "median":
                return float(np.median(s))
            if stat == "mode":
                return pd.Series(s).mode(dropna=False).tolist()
            if stat == "std":
                return float(np.std(s, ddof=1))
            if stat == "var":
                return float(np.var(s, ddof=1))
            if stat == "min":
                return s.min()
            if stat == "max":
                return s.max()
            if stat == "sum":
                return s.sum()
            if stat == "count":
                return int(s.count())
            if stat == "nunique":
                return int(s.nunique(dropna=not dropna))
            if stat == "skew":
                return float(pd.Series(s).skew())
            if stat == "kurtosis":
                return float(pd.Series(s).kurtosis())

            # Should never reach here due to SUPPORTED_STATS check.
            return None

        # Grouped computation
        if groupby:
            if groupby not in df.columns:
                return f"Group-by column '{groupby}' not found. Available: {list(df.columns)}"

            parts = [f"Per-group {stat} for '{column}' grouped by '{groupby}':"]
            for g, sub in df.groupby(groupby, dropna=False):
                parts.append(f"{groupby}={g!r}: {stat}({column}) = {compute_on_series(sub[column])}")
            return "\n".join(parts)

        # Ungrouped computation
        result_str = f"{stat}({column}) = {compute_on_series(df[column])}"
        return result_str
