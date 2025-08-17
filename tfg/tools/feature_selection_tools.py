# tools/feature_selection_tools.py
from __future__ import annotations
from typing import Optional, Type, List
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

def _get_df(tool: BaseTool) -> pd.DataFrame:
    df = getattr(tool, "dataset", None)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("No dataset assigned to tool. Set `tool.dataset = df`.")
    return df

# -------- SelectKBest --------
SUPPORTED_SCORING = {"f_classif","f_regression","mutual_info_classif","mutual_info_regression"}

class SelectKBestInput(BaseModel):
    target: str
    k: int = Field(..., ge=1)
    scoring: str = Field(..., description=f"One of: {sorted(list(SUPPORTED_SCORING))}")

class SelectKBestTool(BaseTool):
    name: str = "select_kbest"
    description: str = "Select top-k features by a sklearn score function."
    args_schema: Type[BaseModel] = SelectKBestInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, target: str, k: int, scoring: str) -> str:
        df = _get_df(self)
        if target not in df.columns:
            return f"Target '{target}' not found. Available: {list(df.columns)}"
        if scoring not in SUPPORTED_SCORING:
            return f"Unsupported scoring '{scoring}'. Supported: {sorted(list(SUPPORTED_SCORING))}"

        from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
        X = df.drop(columns=[target]); y = df[target]
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            return "No numeric columns available for SelectKBest."

        mapping = {"f_classif": f_classif, "f_regression": f_regression,
                   "mutual_info_classif": mutual_info_classif, "mutual_info_regression": mutual_info_regression}
        selector = SelectKBest(score_func=mapping[scoring], k=min(k, X_num.shape[1]))
        selector.fit(X_num, y)
        selected = list(X_num.columns[selector.get_support()])
        removed = [c for c in X.columns if c not in selected]
        dropped = [c for c in X.columns if c not in X_num.columns]
        lines = [f"SelectKBest(scoring={scoring}, k={k})", f"Selected: {selected}"]
        if dropped: lines.append(f"Ignored non-numeric columns: {dropped}")
        if removed: lines.append(f"Removed: {removed}")
        return "\n".join(lines)

# -------- VarianceThreshold --------
class VarianceThresholdInput(BaseModel):
    threshold: float = Field(0.0, ge=0.0)

class VarianceThresholdTool(BaseTool):
    name: str = "variance_threshold"
    description: str = "Filter numeric features by variance threshold."
    args_schema: Type[BaseModel] = VarianceThresholdInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, threshold: float = 0.0) -> str:
        from sklearn.feature_selection import VarianceThreshold
        df = _get_df(self)
        X_num = df.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            return "No numeric columns to filter."
        vt = VarianceThreshold(threshold=threshold).fit(X_num)
        selected = list(X_num.columns[vt.get_support()])
        removed = [c for c in X_num.columns if c not in selected]
        return f"VarianceThreshold(threshold={threshold})\nSelected: {selected}\nRemoved: {removed}"

# -------- RF Importances --------
class RFImportanceInput(BaseModel):
    target: str
    n_estimators: int = Field(200, ge=10, le=2000)
    max_depth: Optional[int] = Field(None, ge=1)
    top: Optional[int] = Field(None, ge=1)

class RFImportanceSelectTool(BaseTool):
    name: str = "rf_importance_select"
    description: str = "Rank features by RandomForest importance; classification or regression heuristic."
    args_schema: Type[BaseModel] = RFImportanceInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, target: str, n_estimators: int = 200, max_depth: Optional[int] = None, top: Optional[int] = None) -> str:
        df = _get_df(self)
        if target not in df.columns:
            return f"Target '{target}' not found. Available: {list(df.columns)}"
        X = df.drop(columns=[target]); y = df[target]
        X_num = X.select_dtypes(include=[np.number]).fillna(0.0)
        if X_num.shape[1] == 0:
            return "No numeric columns to compute importances."
        is_classification = (pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y) or y.nunique(dropna=False) <= 20)
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier as RF
        else:
            from sklearn.ensemble import RandomForestRegressor as RF
        model = RF(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42).fit(X_num, y)
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        ranked = [(X_num.columns[i], float(importances[i])) for i in order]
        if top is not None: ranked = ranked[:top]
        lines = [f"RandomForest {'Classifier' if is_classification else 'Regressor'} (n_estimators={n_estimators}, max_depth={max_depth})", "Importances (desc):"]
        lines += [f"- {name}: {imp:.6f}" for name, imp in ranked]
        dropped = [c for c in X.columns if c not in X_num.columns]
        if dropped: lines.append(f"Ignored non-numeric columns: {dropped}")
        return "\n".join(lines)

# -------- Correlation filter --------
class CorrelationFilterInput(BaseModel):
    target: str
    threshold: float = Field(0.1, ge=0.0, le=1.0)

class CorrelationFilterTool(BaseTool):
    name: str = "correlation_filter"
    description: str = "Select numeric features with |corr| >= threshold vs numeric target."
    args_schema: Type[BaseModel] = CorrelationFilterInput
    dataset: Type[pd.DataFrame] = None

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
        return f"CorrelationFilter(threshold={threshold})\nSelected: {selected}\nRemoved: {removed}"
