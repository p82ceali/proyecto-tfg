# feature_selection_agent_structured_tools.py
from __future__ import annotations

from typing import List, Optional, Any, Dict, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

from dotenv import load_dotenv
import os


load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)

# =====================================================================
# Helpers
# =====================================================================

def _get_dataset(context: Dict[str, Any]) -> pd.DataFrame:
    df = context.get("dataset")
    if df is None:
        raise ValueError(
            "No dataset in tool context. Pass it with crew.kickoff(inputs={'dataset': df, 'user_request': '...'})"
        )
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame under context['dataset'].")
    return df


# =====================================================================
# Feature Selection Structured Tools
#   - All follow: args_schema + `dataset` attribute
#   - The dataset is injected externally: tool.dataset = df
# =====================================================================

# ---------------------------
# SelectKBestTool
# ---------------------------
SUPPORTED_SCORING = {"f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression"}

class SelectKBestInput(BaseModel):
    target: str = Field(..., description="Target column name")
    k: int = Field(..., ge=1, description="Number of features to select")
    scoring: str = Field(..., description=f"Scoring function. One of: {sorted(list(SUPPORTED_SCORING))}")

class SelectKBestTool(BaseTool):
    name: str = "select_kbest"
    description: str = (
        "Select the top-k features according to a sklearn scoring function. "
        "Requires: target (str), k (int), scoring (str)."
    )
    args_schema: Type[BaseModel] = SelectKBestInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, target: str, k: int, scoring: str) -> str:
        try:
            # df = _get_dataset(self.context)
            df = getattr(self, "dataset", None)
            if df is None:
                return "No dataset assigned to tool."
            if target not in df.columns:
                return f"Target '{target}' not found. Available: {list(df.columns)}"
            if scoring not in SUPPORTED_SCORING:
                return f"Unsupported scoring '{scoring}'. Supported: {sorted(list(SUPPORTED_SCORING))}"

            from sklearn.feature_selection import (
                SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
            )

            X = df.drop(columns=[target])
            y = df[target]
            X_num = X.select_dtypes(include=[np.number])  # typical SelectKBest expects numeric
            dropped = [c for c in X.columns if c not in X_num.columns]
            if X_num.shape[1] == 0:
                return "No numeric columns available for SelectKBest."

            mapping = {
                "f_classif": f_classif,
                "f_regression": f_regression,
                "mutual_info_classif": mutual_info_classif,
                "mutual_info_regression": mutual_info_regression,
            }
            score_func = mapping[scoring]
            selector = SelectKBest(score_func=score_func, k=min(k, X_num.shape[1]))
            selector.fit(X_num, y)
            mask = selector.get_support()
            selected = list(X_num.columns[mask])
            removed = [c for c in X.columns if c not in selected]
            lines = [
                f"SelectKBest(scoring={scoring}, k={k})",
                f"Selected: {selected}",
            ]
            if dropped:
                lines.append(f"Ignored non-numeric columns: {dropped}")
            if removed:
                lines.append(f"Removed: {removed}")
            return "\n".join(lines)
        except Exception as e:
            return f"SelectKBestTool failed: {type(e).__name__}: {e}"


# ---------------------------
# VarianceThresholdTool
# ---------------------------
class VarianceThresholdInput(BaseModel):
    threshold: float = Field(0.0, ge=0.0, description="Minimum variance threshold")

class VarianceThresholdTool(BaseTool):
    name: str = "variance_threshold"
    description: str = (
        "Filter numeric columns whose variance is below a threshold. "
        "Optional: threshold (float, default 0.0)."
    )
    args_schema: Type[BaseModel] = VarianceThresholdInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, threshold: float = 0.0) -> str:
        try:
            df = getattr(self, "dataset", None)
            if df is None:
                return "No dataset assigned to tool."
            X_num = df.select_dtypes(include=[np.number])
            if X_num.shape[1] == 0:
                return "No numeric columns to filter."
            from sklearn.feature_selection import VarianceThreshold
            vt = VarianceThreshold(threshold=threshold)
            vt.fit(X_num)
            selected = list(X_num.columns[vt.get_support()])
            removed = [c for c in X_num.columns if c not in selected]
            return (
                f"VarianceThreshold(threshold={threshold})\n"
                f"Selected: {selected}\nRemoved: {removed}"
            )
        except Exception as e:
            return f"VarianceThresholdTool failed: {type(e).__name__}: {e}"


# ---------------------------
# RFImportanceSelectTool
# ---------------------------
class RFImportanceInput(BaseModel):
    target: str = Field(..., description="Target column name")
    n_estimators: int = Field(200, ge=10, le=2000)
    max_depth: Optional[int] = Field(None, ge=1)
    top: Optional[int] = Field(None, ge=1, description="If provided, return the top-N by importance")

class RFImportanceSelectTool(BaseTool):
    name: str = "rf_importance_select"
    description: str = (
        "Compute feature importances with RandomForest (classifier or regressor depending on target) and list the most relevant. "
        "Requires: target. Optional: n_estimators, max_depth, top."
    )
    args_schema: Type[BaseModel] = RFImportanceInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, target: str, n_estimators: int = 200, max_depth: Optional[int] = None, top: Optional[int] = None) -> str:
        try:
            df = getattr(self, "dataset", None)
            if df is None:
                return "No dataset assigned to tool."
            if target not in df.columns:
                return f"Target '{target}' not found. Available: {list(df.columns)}"
            X = df.drop(columns=[target])
            y = df[target]

            # Use numeric only; one-hot encoding should be done earlier if needed
            X_num = X.select_dtypes(include=[np.number]).fillna(0.0)
            dropped = [c for c in X.columns if c not in X_num.columns]
            if X_num.shape[1] == 0:
                return "No numeric columns to compute importances."

            # Simple heuristic for problem type
            is_classification = False
            if pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y) or y.nunique(dropna=False) <= 20:
                is_classification = True

            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)

            model.fit(X_num, y)
            importances = model.feature_importances_
            order = np.argsort(importances)[::-1]
            ranked = [(X_num.columns[i], float(importances[i])) for i in order]
            if top is not None:
                ranked = ranked[:top]

            lines = [
                f"RandomForest {'Classifier' if is_classification else 'Regressor'} (n_estimators={n_estimators}, max_depth={max_depth})",
                "Importances (desc):",
            ]
            for name, imp in ranked:
                lines.append(f"- {name}: {imp:.6f}")
            if dropped:
                lines.append(f"Ignored non-numeric columns: {dropped}")
            return "\n".join(lines)
        except Exception as e:
            return f"RFImportanceSelectTool failed: {type(e).__name__}: {e}"


# ---------------------------
# CorrelationFilterTool
# ---------------------------
class CorrelationFilterInput(BaseModel):
    target: str = Field(..., description="Target column name (numeric)")
    threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum absolute correlation threshold")

class CorrelationFilterTool(BaseTool):
    name: str = "correlation_filter"
    description: str = (
        "Select numeric columns with absolute correlation >= threshold against a numeric target. "
        "Requires: target. Optional: threshold."
    )
    args_schema: Type[BaseModel] = CorrelationFilterInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, target: str, threshold: float = 0.1) -> str:
        try:
            df = getattr(self, "dataset", None)
            if df is None:
                return "No dataset assigned to tool."
            if target not in df.columns:
                return f"Target '{target}' not found. Available: {list(df.columns)}"
            if not pd.api.types.is_numeric_dtype(df[target]):
                return "Target must be numeric for Pearson correlation."

            X_num = df.select_dtypes(include=[np.number])
            if X_num.shape[1] == 0:
                return "No numeric columns."

            corr = X_num.corr(numeric_only=True)
            if target not in corr.index:
                return "Could not compute correlation with target (not numeric or not in X_num)."

            abs_corr = corr[target].abs().sort_values(ascending=False)
            selected = abs_corr[abs_corr >= threshold].index.tolist()
            removed = [c for c in X_num.columns if c not in selected]
            return (
                f"CorrelationFilter(threshold={threshold})\n"
                f"Selected: {selected}\nRemoved: {removed}"
            )
        except Exception as e:
            return f"CorrelationFilterTool failed: {type(e).__name__}: {e}"


# =====================================================================
# Agent + Task wiring
# =====================================================================

def build_agent() -> Agent:
    return Agent(
        role="Feature Selection Agent",
        goal=(
            "Understand the user's request and call only the necessary feature-selection tool. "
            "When calling a tool, fill the structured parameters according to the tool's schema. "
            "Ask a brief clarifying question if the request lacks required parameters (e.g., missing 'target' or 'k')."
        ),
        backstory=(
            "You are precise and concise. You don't run unnecessary tools. "
            "You prefer structured inputs and return readable results (selected vs removed features)."
        ),
        tools=[
            SelectKBestTool(),
            VarianceThresholdTool(),
            RFImportanceSelectTool(),
            CorrelationFilterTool(),
        ],
        verbose=True,
        llm=llm,
    )


def build_task(agent: Agent) -> Task:
    return Task(
        description=(
            "User request: {user_request}\n\n"
            "Use ONLY the necessary tool to answer. "
            "If a tool needs parameters, pass them using the tool's structured fields (do NOT serialize JSON strings). "
            "If the request is underspecified (e.g., missing target), ask one short follow-up question and stop."
        ),
        expected_output=(
            "A concise answer that states: method applied, parameters used, and selected/removed features."
        ),
        agent=agent,
    )


# =====================================================================
# Local demo (optional): assign dataset by attribute, just like the analysis agent example
# =====================================================================

def demo():
    # Example dataset
    df = pd.DataFrame({
        "x1": [1,2,3,4,5,6,7,8,9,10],
        "x2": [0,1,0,1,0,1,0,1,0,1],
        "noise": [0,0,0,0,0,0,0,0,0,0],
        "y":  [0,0,0,1,1,1,1,1,0,0],
    })

    # Create tools and assign dataset via attribute
    t1 = SelectKBestTool(); t2 = VarianceThresholdTool(); t3 = RFImportanceSelectTool(); t4 = CorrelationFilterTool()
    for t in (t1, t2, t3, t4):
        t.dataset = df

    # Build the agent with dataset-aware tools
    agent = Agent(
        role="Feature Selection Agent",
        goal="Answer only with the necessary selection method.",
        backstory="A smart selector that already knows the dataset.",
        tools=[t1, t2, t3, t4],
        llm=llm,
        verbose=True,
    )

    task = Task(
        description="User request: {user_request}",
        expected_output="Selected/removed features with method and params.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    examples = [
        "Select k best features: target=y, k=2, scoring=f_classif",
        "Variance threshold 0.0",
        "RF importance top=3 target=y n_estimators=100",
        "Correlation filter target=y threshold=0.2",
        "SelectKBest without target",  # should ask for clarification
    ]

    for req in examples:
        print("\n" + "="*80)
        print("USER:", req)
        result = crew.kickoff(inputs={"user_request": req})
        print("ASSISTANT:\n", result)


if __name__ == "__main__":
    demo()
