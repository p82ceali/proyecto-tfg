# eda_agent_structured_tools.py
from __future__ import annotations

from typing import List, Optional, Union, Any, Dict, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

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
# Helper utilities
# ---------------------------

def _get_dataset(context: Dict[str, Any]) -> pd.DataFrame:
    df = context.get("dataset")
    if df is None:
        raise ValueError(
            "No dataset in tool context. Pass it with crew.kickoff(inputs={'dataset': df, 'user_request': '...'})"
        )
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame under context['dataset'].")
    return df


# ---------------------------
# DescribeFeatureTool
# ---------------------------

class DescribeFeatureInput(BaseModel):
    column: str = Field(..., description="Column name to describe.")
    percentiles: Optional[List[float]] = Field(
        None, description="Optional list of percentile floats in [0,1], e.g. [0.05,0.5,0.95]."
    )
    include: Optional[Union[str, List[str]]] = Field(
        None, description="Optional 'include' arg (usually for DataFrame.describe). Not used for Series."
    )

class DescribeFeatureTool(BaseTool):
    # NOTE: type annotations are REQUIRED with Pydantic v2
    name: str = "describe_feature"
    description: str = (
        "Describe a specific column of the dataset using pandas describe(). "
        "Requires: column (str). Optional: percentiles (list[float]), include."
    )
    args_schema: Type[BaseModel] = DescribeFeatureInput  # structured params
    dataset: Type[pd.DataFrame] = None

    # When args_schema is provided, _run receives kwargs matching the schema.
    def _run(self, column: str, percentiles: Optional[List[float]] = None, include: Optional[Union[str, List[str]]] = None) -> str:
        try:
            #df = _get_dataset(self.context)
            df = getattr(self, "dataset", None)
            if df is None:
                return "No dataset assigned to tool."
            if column not in df.columns:
                return f"Column '{column}' not found. Available: {list(df.columns)}"

            ser = df[column]
            desc = ser.describe(percentiles=percentiles) if percentiles else ser.describe()

            lines = [f"Description for '{column}':", desc.to_string()]

            # If categorical-like, show top values
            if ser.dtype == "object" or pd.api.types.is_categorical_dtype(ser):
                vc = ser.value_counts(dropna=False).head(10)
                lines.append("\nTop 10 values (incl. NaN):")
                lines.append(vc.to_string())

            return "\n".join(lines)
        except Exception as e:
            return f"DescribeFeatureTool failed: {type(e).__name__}: {e}"


# ---------------------------
# ComputeStatisticTool
# ---------------------------

SUPPORTED_STATS = {
    "mean", "median", "mode", "std", "var", "min", "max", "sum", "count", "nunique", "skew", "kurtosis"
}

class ComputeStatisticInput(BaseModel):
    column: str = Field(..., description="Column to compute the statistic on.")
    stat: str = Field(..., description=f"Statistic to compute. One of: {sorted(SUPPORTED_STATS)}")
    dropna: bool = Field(True, description="Whether to drop NaNs before computing.")
    groupby: Optional[str] = Field(None, description="Optional group-by column. If provided, compute per group.")

class ComputeStatisticTool(BaseTool):
    name: str = "compute_statistic"
    description: str = (
        "Compute a statistic for a specific column, optionally grouped. "
        "Required: column (str), stat (str). Optional: dropna (bool, default True), groupby (str). "
        f"Supported stats: {sorted(SUPPORTED_STATS)}"
    )
    args_schema: Type[BaseModel] = ComputeStatisticInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, column: str, stat: str, dropna: bool = True, groupby: Optional[str] = None) -> str:
        try:
            #df = _get_dataset(self.context)
            df = getattr(self, "dataset", None)
            if df is None:
                return "No dataset assigned to tool."

            if column not in df.columns:
                return f"Column '{column}' not found. Available: {list(df.columns)}"
            stat = stat.lower()
            if stat not in SUPPORTED_STATS:
                return f"Unsupported stat '{stat}'. Supported: {sorted(SUPPORTED_STATS)}"

            def compute_on_series(s: pd.Series) -> Any:
                if dropna:
                    s = s.dropna()

                if stat == "mean":
                    return float(np.mean(s))
                if stat == "median":
                    return float(np.median(s))
                if stat == "mode":
                    # Return list of modes (can be multiple)
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
                raise ValueError(f"Unhandled stat: {stat}")

            if groupby:
                if groupby not in df.columns:
                    return f"Group-by column '{groupby}' not found. Available: {list(df.columns)}"
                parts = []
                for g, sub in df.groupby(groupby, dropna=False):
                    val = compute_on_series(sub[column])
                    parts.append(f"{groupby}={g!r}: {stat}({column}) = {val}")
                header = f"Per-group {stat} for '{column}' grouped by '{groupby}':"
                return "\n".join([header] + parts)

            # No groupby → single value
            val = compute_on_series(df[column])
            return f"{stat}({column}) = {val}"

        except Exception as e:
            return f"ComputeStatisticTool failed: {type(e).__name__}: {e}"


# ---------------------------
# Agent + Task wiring
# ---------------------------

def build_agent():
    return Agent(
        role="Exploratory Data Analyst",
        goal=(
            "Understand the user's request and call only the necessary tools. "
            "When calling a tool, fill the structured parameters according to the tool's schema. "
            "Ask a brief clarifying question if the request lacks required parameters (e.g., missing 'column')."
        ),
        backstory=(
            "You are precise and concise. You don't run unnecessary tools. "
            "You prefer structured inputs and return readable results."
        ),
        tools=[DescribeFeatureTool(), ComputeStatisticTool()],
        verbose=True,
        llm=llm
    )

def build_task(agent: Agent) -> Task:
    return Task(
        description=(
            "User request: {user_request}\n\n"
            "Use ONLY the necessary tools to answer. "
            "If a tool needs parameters, pass them using the tool's structured fields "
            "(do NOT serialize JSON strings). "
            "If the request is underspecified (e.g., missing column), ask one short follow-up question and stop."
        ),
        expected_output=(
            "A concise answer with any computed values. "
            "If clarification was required, a single crisp question."
        ),
        agent=agent,
    )

def demo():
    # Example dataset
    df = pd.DataFrame({
        "age": [22, 25, 25, 29, 31, np.nan, 40],
        "salary": [38000, 42000, 42000, 52000, 61000, 73000, 61000],
        "dept": ["A", "B", "B", "A", "A", "B", "A"]
    })
    df = pd.read_csv('iris.csv')

    #print(df.columns, flush=True)

    # Create tools and attach dataset
    describe_tool = DescribeFeatureTool()
    compute_tool = ComputeStatisticTool()
    for t in (describe_tool, compute_tool):
        t.dataset = df

    # Build agent with dataset-aware tools
    agent = Agent(
        role="EDA Agent",
        goal="Answer only with relevant statistics.",
        backstory="A smart analyst that knows the dataset already.",
        tools=[describe_tool, compute_tool],
        llm=llm,
        verbose=True
    )

    task = Task(
        description="User request: {user_request}",
        expected_output="Answer to the statistical question.",
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    examples = [
        "Describe the salary column with 5%, 50% and 95% percentiles.",
        "What is the median of the age column (ignore nulls)?",
        "Give me the mean salary per dept.",
        "Compute the mode of age, grouped by dept.",
        "Compute the mean.",  # should ask for the column
    ]

    examples = [
        "What is the mean of the petal.width?",
        "Describe the sepal.length column with 5%, 50% and 95% percentiles.",
        "What is the median of the petal.length column (ignore nulls)?",
        "How many different values are there in variety?"
        #"Give me the mean salary per dept.",
        #"Compute the mode of age, grouped by dept.",
        #"Compute the mean.",  # should ask for the column
    ]

    for req in examples:
        print("\n" + "="*80)
        print("USER:", req)
        result = crew.kickoff(inputs={"user_request": req})
        print("ASSISTANT:\n", result)

if __name__ == "__main__":
    demo()
