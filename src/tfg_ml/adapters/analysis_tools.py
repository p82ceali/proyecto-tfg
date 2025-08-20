# tools/analysis_tools.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from tfg_ml.context import CTX

def _get_df(tool: BaseTool) -> pd.DataFrame:
    df = getattr(tool, "dataset", None)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("No dataset assigned to tool. Set `tool.dataset = df` before running.")
    return df

class DescribeFeatureInput(BaseModel):
    column: str = Field(..., description="Column name to describe.")
    percentiles: Optional[List[float]] = Field(None, description="List of percentiles in [0,1].")
    include: Optional[Union[str, List[str]]] = Field(None, description="Include param for DataFrame.describe (if needed).")

class DescribeFeatureTool(BaseTool):
    name: str = "describe_feature"
    description: str = "Describe a specific column using pandas describe()."
    args_schema: Type[BaseModel] = DescribeFeatureInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, column: str, percentiles: Optional[List[float]] = None, include: Optional[Union[str, List[str]]] = None) -> str:
        df = _get_df(self)
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        ser = df[column]
        desc = ser.describe(percentiles=percentiles) if percentiles else ser.describe()
        lines = [f"Description for '{column}':", desc.to_string()]
        if ser.dtype == "object" or pd.api.types.is_categorical_dtype(ser):
            vc = ser.value_counts(dropna=False).head(10)
            lines += ["", "Top 10 values (incl. NaN):", vc.to_string()]
        out="\n".join(lines)
        CTX.add_decision("analysis", f"describe({column}) ejecutado")
        return out

SUPPORTED_STATS = {"mean","median","mode","std","var","min","max","sum","count","nunique","skew","kurtosis"}

class ComputeStatisticInput(BaseModel):
    column: str = Field(..., description="Column to compute on.")
    stat: str = Field(..., description=f"One of: {sorted(SUPPORTED_STATS)}")
    dropna: bool = Field(True)
    groupby: Optional[str] = Field(None)

class ComputeStatisticTool(BaseTool):
    name: str = "compute_statistic"
    description: str = "Compute a statistic over a column, optionally grouped."
    args_schema: Type[BaseModel] = ComputeStatisticInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, column: str, stat: str, dropna: bool = True, groupby: Optional[str] = None) -> str:
        df = _get_df(self)
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        stat = stat.lower()
        if stat not in SUPPORTED_STATS:
            return f"Unsupported stat '{stat}'. Supported: {sorted(SUPPORTED_STATS)}"

        def compute_on_series(s: pd.Series):
            if dropna:
                s = s.dropna()
            if stat == "mean": return float(np.mean(s))
            if stat == "median": return float(np.median(s))
            if stat == "mode": return pd.Series(s).mode(dropna=False).tolist()
            if stat == "std": return float(np.std(s, ddof=1))
            if stat == "var": return float(np.var(s, ddof=1))
            if stat == "min": return s.min()
            if stat == "max": return s.max()
            if stat == "sum": return s.sum()
            if stat == "count": return int(s.count())
            if stat == "nunique": return int(s.nunique(dropna=not dropna))
            if stat == "skew": return float(pd.Series(s).skew())
            if stat == "kurtosis": return float(pd.Series(s).kurtosis())

        if groupby:
            if groupby not in df.columns:
                return f"Group-by column '{groupby}' not found. Available: {list(df.columns)}"
            parts = [f"Per-group {stat} for '{column}' grouped by '{groupby}':"]
            for g, sub in df.groupby(groupby, dropna=False):
                parts.append(f"{groupby}={g!r}: {stat}({column}) = {compute_on_series(sub[column])}")
            return "\n".join(parts)

        msg= f"{stat}({column}) = {compute_on_series(df[column])}"
        CTX.add_decision("analysis", f"stat {stat} sobre {column}" + (f" por {groupby}" if groupby else ""))
        return msg
