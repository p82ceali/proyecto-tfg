# preprocessing_agent_tools.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

from dotenv import load_dotenv
import os

# Establece tu API key
load_dotenv()

# Crea el modelo base para los agentes
llm = LLM(model="gemini/gemini-2.0-flash-lite",
          api_key=os.getenv("GOOGLE_API_KEY"),
          custom_llm_provider="gemini"
          ) # OpenAI(temperature=0.3)


# ---------------------------
# Helpers
# ---------------------------

def _get_df(tool: BaseTool) -> pd.DataFrame:
    df = getattr(tool, "dataset", None)
    if df is None:
        raise ValueError("No dataset assigned to tool. Set `tool.dataset = your_dataframe` before running.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`dataset` must be a pandas DataFrame.")
    return df

def _safe_new_col_name(df: pd.DataFrame, base: str) -> str:
    name = base
    i = 1
    while name in df.columns:
        name = f"{base}_{i}"
        i += 1
    return name

def _describe_series(ser: pd.Series, name: str) -> str:
    lines = [f"Description for '{name}':"]
    try:
        desc = ser.describe()
        lines.append(desc.to_string())
    except Exception:
        # fallback
        lines.append(str(pd.Series(ser).describe()))
    # For categorical or object, show top values
    if ser.dtype == "object" or pd.api.types.is_categorical_dtype(ser):
        vc = ser.value_counts(dropna=False).head(10)
        lines.append("\nTop 10 values (incl. NaN):")
        lines.append(vc.to_string())
    return "\n".join(lines)

def _bool_col_summary(df: pd.DataFrame, cols: List[str]) -> str:
    parts = []
    for c in cols:
        s = df[c]
        ones = int((s == 1).sum())
        zeros = int((s == 0).sum())
        na = int(s.isna().sum())
        parts.append(f"{c}: ones={ones}, zeros={zeros}, NaN={na}, mean={float(s.mean()):.4f}")
    return "\n".join(parts)


# ---------------------------
# DiscretizeFeatureTool
# ---------------------------

class DiscretizeFeatureInput(BaseModel):
    column: str = Field(..., description="Column to discretize (numeric).")
    strategy: Literal["equal_width", "equal_freq", "custom_edges"] = Field(
        "equal_width",
        description="Binning strategy: 'equal_width' (pd.cut), 'equal_freq' (pd.qcut), or 'custom_edges' (pd.cut with explicit edges)."
    )
    bins: Optional[int] = Field(None, description="Number of bins for 'equal_width'. Must be >= 2.")
    q: Optional[int] = Field(None, description="Number of quantile bins for 'equal_freq'. Must be >= 2.")
    edges: Optional[List[float]] = Field(None, description="Explicit bin edges for 'custom_edges'. Must be sorted.")
    labels: Optional[List[str]] = Field(None, description="Optional labels for bins. Length must match number of bins.")
    right: bool = Field(True, description="Whether bins include the rightmost edge.")
    include_lowest: bool = Field(True, description="Whether the first interval should be left-inclusive.")
    drop_original: bool = Field(True, description="If true, drops the original column after creating the binned column.")
    new_column: Optional[str] = Field(None, description="Optional name for the new binned column. Defaults to '<column>_binned'.")

    @field_validator("bins")
    @classmethod
    def _check_bins(cls, v, info):
        if info.data.get("strategy") == "equal_width":
            if v is None or v < 2:
                raise ValueError("For strategy 'equal_width', 'bins' must be an integer >= 2.")
        return v

    @field_validator("q")
    @classmethod
    def _check_q(cls, v, info):
        if info.data.get("strategy") == "equal_freq":
            if v is None or v < 2:
                raise ValueError("For strategy 'equal_freq', 'q' must be an integer >= 2.")
        return v

    @field_validator("edges")
    @classmethod
    def _check_edges(cls, v, info):
        if info.data.get("strategy") == "custom_edges":
            if not v or len(v) < 2:
                raise ValueError("For 'custom_edges', provide at least two sorted edge values.")
            if any(v[i] >= v[i+1] for i in range(len(v)-1)):
                raise ValueError("Edges must be strictly increasing.")
        return v


class DiscretizeFeatureTool(BaseTool):
    name: str = "discretize_feature"
    description: str = (
        "Discretize a numeric column into bins. Supports equal-width (pd.cut), "
        "equal-frequency (pd.qcut), or custom edges. Prints a description of the new binned column."
    )
    args_schema: Type[DiscretizeFeatureInput] = DiscretizeFeatureInput
    # IMPORTANT: dataset will be attached externally, e.g., `tool.dataset = df`
    dataset: Type[pd.DataFrame] = None

    def _run(
        self,
        column: str,
        strategy: str = "equal_width",
        bins: Optional[int] = None,
        q: Optional[int] = None,
        edges: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        right: bool = True,
        include_lowest: bool = True,
        drop_original: bool = True,
        new_column: Optional[str] = None,
    ) -> str:
        try:
            df = _get_df(self)
            if column not in df.columns:
                return f"Column '{column}' not found. Available: {list(df.columns)}"

            series = df[column]
            if not pd.api.types.is_numeric_dtype(series):
                return f"Column '{column}' must be numeric to discretize."

            # Determine number of bins for label validation
            if strategy == "equal_width":
                n_bins = bins
                cut = pd.cut(
                    series,
                    bins=bins,
                    labels=labels if labels else None,
                    right=right,
                    include_lowest=include_lowest
                )
            elif strategy == "equal_freq":
                n_bins = q
                cut = pd.qcut(
                    series,
                    q=q,
                    labels=labels if labels else None,
                    duplicates="drop"
                )
                # pd.qcut may reduce bins if duplicates, adjust n_bins
                if labels and len(labels) != cut.cat.categories.size:
                    return "Provided labels length doesn't match effective quantile bins after duplicate handling."
            elif strategy == "custom_edges":
                n_bins = len(edges) - 1 if edges else None
                cut = pd.cut(
                    series,
                    bins=edges,
                    labels=labels if labels else None,
                    right=right,
                    include_lowest=include_lowest
                )
            else:
                return f"Unknown strategy '{strategy}'."

            if labels and n_bins is not None and len(labels) != n_bins:
                return f"'labels' length ({len(labels)}) must equal number of bins ({n_bins})."

            new_col = new_column or f"{column}_binned"
            if new_col in df.columns:
                new_col = _safe_new_col_name(df, new_col)

            df[new_col] = cut

            if drop_original:
                df.drop(columns=[column], inplace=True)

            # Describe the new binned column
            return _describe_series(df[new_col], new_col)

        except Exception as e:
            return f"DiscretizeFeatureTool failed: {type(e).__name__}: {e}"


# ---------------------------
# OneHotEncodeFeatureTool
# ---------------------------

class OneHotEncodeInput(BaseModel):
    column: str = Field(..., description="Categorical (or castable) column to one-hot encode.")
    prefix: Optional[str] = Field(None, description="Prefix for new columns. Defaults to the column name.")
    drop_first: bool = Field(True, description="Whether to drop the first level to avoid multicollinearity.")
    dtype: Optional[str] = Field("uint8", description="Dtype for new columns (e.g., 'uint8').")
    drop_original: bool = Field(True, description="If true, drops the original column after encoding.")

class OneHotEncodeFeatureTool(BaseTool):
    name: str = "one_hot_encode_feature"
    description: str = (
        "Apply one-hot encoding to a single categorical column using pandas.get_dummies(). "
        "Prints descriptions of the created dummy columns."
    )
    args_schema: Type[OneHotEncodeInput] = OneHotEncodeInput
    # dataset will be attached externally
    dataset: Type[pd.DataFrame] = None

    def _run(
        self,
        column: str,
        prefix: Optional[str] = None,
        drop_first: bool = True,
        dtype: Optional[str] = "uint8",
        drop_original: bool = True,
    ) -> str:
        try:
            df = _get_df(self)
            if column not in df.columns:
                return f"Column '{column}' not found. Available: {list(df.columns)}"

            pref = prefix or column
            dummies = pd.get_dummies(df[column], prefix=pref, drop_first=drop_first, dtype=dtype)

            # Ensure unique column names (just in case)
            new_cols = []
            for c in dummies.columns:
                nc = c if c not in df.columns else _safe_new_col_name(df, c)
                new_cols.append(nc)
            dummies.columns = new_cols

            df[new_cols] = dummies

            if drop_original:
                df.drop(columns=[column], inplace=True)

            # Describe the new dummy columns
            desc_lines = ["One-hot encoding created columns:", ", ".join(new_cols), ""]
            desc_lines.append("Column summaries (proportion of 1s etc.):")
            desc_lines.append(_bool_col_summary(df, new_cols))
            return "\n".join(desc_lines)

        except Exception as e:
            return f"OneHotEncodeFeatureTool failed: {type(e).__name__}: {e}"


# ---------------------------
# Agent + Task
# ---------------------------

def build_preprocessing_agent(tools: List[BaseTool]) -> Agent:
    return Agent(
        role="Data Preprocessing Agent",
        goal=(
            "Understand the user's preprocessing request and call only the necessary tool with correct parameters. "
            "After each operation, print a concise description of the affected (new or modified) columns."
        ),
        backstory=(
            "You perform preprocessing steps like discretization and one-hot encoding. "
            "You never run tools unnecessarily. You ask a brief clarifying question if required parameters are missing."
        ),
        tools=tools,
        verbose=True,
        llm=llm
    )

def build_task(agent: Agent) -> Task:
    return Task(
        description=(
            "User request: {user_request}\n\n"
            "Choose and run only the relevant preprocessing tool. "
            "Fill the structured parameters according to the tool's schema. "
            "If the request is underspecified (e.g., missing 'column' or binning details), ask ONE short follow-up question and stop."
        ),
        expected_output=(
            "A concise report of the preprocessing step performed and a description of the affected features."
        ),
        agent=agent,
    )


# ---------------------------
# Demo
# ---------------------------

def demo():
    # Sample dataset
    df = pd.DataFrame({
        "age": [22, 25, 25, 29, 31, np.nan, 40, 52, 37, 28],
        "dept": ["A", "B", "B", "A", "A", "B", "A", "C", "C", "B"],
        "city": ["Madrid", "Seville", "Madrid", "Madrid", "Valencia", "Seville", "Bilbao", "Bilbao", "Madrid", "Valencia"]
    })
    df = pd.read_csv('iris.csv')


    # Create tools and attach dataset
    discretize_tool = DiscretizeFeatureTool()
    onehot_tool = OneHotEncodeFeatureTool()
    for t in (discretize_tool, onehot_tool):
        t.dataset = df  # attach dataset to tool instances

    agent = build_preprocessing_agent([discretize_tool, onehot_tool])
    task = build_task(agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    examples = [
        # Discretization (equal-width) with custom name
        "Discretize the 'age' column into 3 equal-width bins and name it 'age_bin'; drop the original.",
        # Discretization (equal-frequency)
        "Bin the age column into 4 equal-frequency bins (quantiles). Keep the original column.",
        # One-hot encode categorical with all levels
        "One-hot encode the dept column without dropping the first level.",
        # One-hot encode with prefix
        "Apply one hot encoding to the city feature with prefix 'cty' and drop the original column.",
        # Intentionally underspecified to see follow-up
        "Discretize a feature.",
    ]
    examples = [
        "Discretize the 'petal.length' column into 5 equal-frequency bins. keep the original column.",
        "One-hot encode the new discretized petal.length' column",
    ]

    for req in examples:
        print("\n" + "=" * 100)
        print("USER:", req)
        result = crew.kickoff(inputs={"user_request": req})
        print("ASSISTANT:\n", result)

    print("\nFinal DataFrame columns:", df.columns.tolist())
    print(df.head())


if __name__ == "__main__":
    demo()
