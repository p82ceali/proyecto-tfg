# tools/feature_selection_tools.py
"""
Feature selection tools (CrewAI BaseTool implementations).

Exposed tools:
    • SelectKBestTool
        Selects the top-k numeric features using a scikit-learn scoring function.
    • VarianceThresholdTool
        Filters numeric features below a variance threshold.
    • RFImportanceSelectTool
        Ranks features by RandomForest importance (classification/regression heuristic).
    • CorrelationFilterTool
        Selects numeric features whose absolute Pearson correlation with a numeric
        target is above a threshold.

Usage requirements:
    - A pandas DataFrame must be attached before running:
        tool.dataset = df
    - Each tool receives structured Pydantic inputs (no ad-hoc JSON strings).
    - Decisions are logged via `CTX.add_decision(...)` for traceability.
"""

from __future__ import annotations

from typing import Optional, Type, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from tfg_ml.context import CTX


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _get_df(tool: BaseTool) -> pd.DataFrame:
    """
    Retrieve the DataFrame attached to a tool.

    Raises:
        ValueError: If no DataFrame has been attached.
    """
    df = getattr(tool, "dataset", None)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("No dataset assigned to tool. Set `tool.dataset = df`.")
    return df


# ---------------------------------------------------------------------
# SelectKBest
# ---------------------------------------------------------------------
SUPPORTED_SCORING = {
    "f_classif",
    "f_regression",
    "mutual_info_classif",
    "mutual_info_regression",
}


class SelectKBestInput(BaseModel):
    """Inputs for SelectKBestTool."""
    target: str
    k: int = Field(..., ge=1, description="Number of features to select (top-k).")
    scoring: str = Field(..., description=f"One of: {sorted(list(SUPPORTED_SCORING))}")


class SelectKBestTool(BaseTool):
    """
    Select top-k numeric features using a scikit-learn score function.

    Notes:
        - Only numeric predictors are considered. Non-numeric columns are ignored.
        - Ensure the chosen scoring matches your problem type (e.g., f_classif for
          classification with categorical/label targets; f_regression for numeric targets).
    """
    name: str = "select_kbest"
    description: str = "Select top-k features by a sklearn score function."
    args_schema: Type[BaseModel] = SelectKBestInput
    dataset: Optional[pd.DataFrame] = None  # set by the orchestrator/coordinator

    def _run(self, target: str, k: int, scoring: str) -> str:
        df = _get_df(self)

        if target not in df.columns:
            return f"Target '{target}' not found. Available: {list(df.columns)}"
        if scoring not in SUPPORTED_SCORING:
            return f"Unsupported scoring '{scoring}'. Supported: {sorted(list(SUPPORTED_SCORING))}"

        from sklearn.feature_selection import (
            SelectKBest,
            f_classif,
            f_regression,
            mutual_info_classif,
            mutual_info_regression,
        )

        X = df.drop(columns=[target])
        y = df[target]

        # Use only numeric predictors
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            return "No numeric columns available for SelectKBest."

        mapping = {
            "f_classif": f_classif,
            "f_regression": f_regression,
            "mutual_info_classif": mutual_info_classif,
            "mutual_info_regression": mutual_info_regression,
        }
        selector = SelectKBest(score_func=mapping[scoring], k=min(k, X_num.shape[1]))
        selector.fit(X_num, y)

        selected = list(X_num.columns[selector.get_support()])
        removed = [c for c in X.columns if c not in selected]
        dropped = [c for c in X.columns if c not in X_num.columns]

        lines = [
            f"SelectKBest(scoring={scoring}, k={k})",
            f"Selected: {selected}",
        ]
        if dropped:
            lines.append(f"Ignored non-numeric columns: {dropped}")
        if removed:
            lines.append(f"Removed: {removed}")

        out = "\n".join(lines)
        CTX.add_decision("feature_selection", f"SelectKBest({scoring}, k={k}) -> {selected}")
        return out


# ---------------------------------------------------------------------
# VarianceThreshold
# ---------------------------------------------------------------------
class VarianceThresholdInput(BaseModel):
    """Inputs for VarianceThresholdTool."""
    threshold: float = Field(0.0, ge=0.0, description="Minimum variance required to keep a feature.")


class VarianceThresholdTool(BaseTool):
    """
    Filter numeric features by variance threshold.

    Notes:
        - Operates only on numeric columns.
    """
    name: str = "variance_threshold"
    description: str = "Filter numeric features by variance threshold."
    args_schema: Type[BaseModel] = VarianceThresholdInput
    dataset: Optional[pd.DataFrame] = None  # set by the orchestrator/coordinator

    def _run(self, threshold: float = 0.0) -> str:
        from sklearn.feature_selection import VarianceThreshold

        df = _get_df(self)
        X_num = df.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            return "No numeric columns to filter."

        vt = VarianceThreshold(threshold=threshold).fit(X_num)
        selected = list(X_num.columns[vt.get_support()])
        removed = [c for c in X_num.columns if c not in selected]

        msg = (
            f"VarianceThreshold(threshold={threshold})\n"
            f"Selected: {selected}\n"
            f"Removed: {removed}"
        )
        CTX.add_decision("feature_selection", f"VarianceThreshold(th={threshold}) -> {len(selected)} keep")
        return msg


# ---------------------------------------------------------------------
# RandomForest Importances
# ---------------------------------------------------------------------
class RFImportanceInput(BaseModel):
    """Inputs for RFImportanceSelectTool."""
    target: str
    n_estimators: int = Field(200, ge=10, le=2000)
    max_depth: Optional[int] = Field(None, ge=1)
    top: Optional[int] = Field(None, ge=1)


class RFImportanceSelectTool(BaseTool):
    """
    Rank features by RandomForest importance.

    Heuristic:
        - If the target appears low-cardinality or boolean/integer, use a classifier;
          otherwise, use a regressor.
    """
    name: str = "rf_importance_select"
    description: str = "Rank features by RandomForest importance; classification or regression heuristic."
    args_schema: Type[BaseModel] = RFImportanceInput
    dataset: Optional[pd.DataFrame] = None  # set by the orchestrator/coordinator

    def _run(
        self,
        target: str,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        top: Optional[int] = None,
    ) -> str:
        df = _get_df(self)

        if target not in df.columns:
            return f"Target '{target}' not found. Available: {list(df.columns)}"

        X = df.drop(columns=[target])
        y = df[target]

        # Numeric predictors only; fill NaNs for tree models
        X_num = X.select_dtypes(include=[np.number]).fillna(0.0)
        if X_num.shape[1] == 0:
            return "No numeric columns to compute importances."

        is_classification = (
            pd.api.types.is_integer_dtype(y)
            or pd.api.types.is_bool_dtype(y)
            or y.nunique(dropna=False) <= 20
        )

        if is_classification:
            from sklearn.ensemble import RandomForestClassifier as RF
        else:
            from sklearn.ensemble import RandomForestRegressor as RF

        model = RF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        ).fit(X_num, y)

        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        ranked = [(X_num.columns[i], float(importances[i])) for i in order]
        if top is not None:
            ranked = ranked[:top]

        lines = [
            f"RandomForest {'Classifier' if is_classification else 'Regressor'} "
            f"(n_estimators={n_estimators}, max_depth={max_depth})",
            "Importances (desc):",
            *[f"- {name}: {imp:.6f}" for name, imp in ranked],
        ]

        dropped = [c for c in X.columns if c not in X_num.columns]
        if dropped:
            lines.append(f"Ignored non-numeric columns: {dropped}")

        out = "\n".join(lines)
        CTX.add_decision("feature_selection", f"RF rank top={top or len(ranked)}")
        return out


# ---------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------
class CorrelationFilterInput(BaseModel):
    """Inputs for CorrelationFilterTool."""
    target: str
    threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum |Pearson r| to keep a feature.")


class CorrelationFilterTool(BaseTool):
    """
    Select numeric features with absolute Pearson correlation ≥ threshold
    relative to a numeric target.
    """
    name: str = "correlation_filter"
    description: str = "Select numeric features with |corr| >= threshold vs numeric target."
    args_schema: Type[BaseModel] = CorrelationFilterInput
    dataset: Optional[pd.DataFrame] = None  # set by the orchestrator/coordinator

    def _run(self, target: str, threshold: float = 0.1) -> str:
        df = _get_df(self)

        if target not in df.columns:
            return f"Target '{target}' not found. Available: {list(df.columns)}"
        if not pd.api.types.is_numeric_dtype(df[target]):
            return "Target must be numeric for Pearson correlation."

        X_num = df.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            return "No numeric columns."

        corr = X_num.corr(numeric_only=True)
        if target not in corr.index:
            return "Could not compute correlation with target."

        abs_corr = corr[target].abs().sort_values(ascending=False)
        selected = abs_corr[abs_corr >= threshold].index.tolist()
        removed = [c for c in X_num.columns if c not in selected]

        msg = (
            f"CorrelationFilter(threshold={threshold})\n"
            f"Selected: {selected}\n"
            f"Removed: {removed}"
        )
        CTX.add_decision("feature_selection", f"CorrFilter(|r|>={threshold}) -> {len(selected)}")
        return msg
