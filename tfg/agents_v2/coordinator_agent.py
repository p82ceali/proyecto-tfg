# coordinator_agent.py (updated to include Instance Selection Agent)
from __future__ import annotations
from typing import Type, Optional

import os
import pandas as pd
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

# === Relative imports (keep your current package layout) ===
from . import analysis_agent as analysis
from . import preprocessing_agent as preprocessing
from . import feature_selection_agent as fsel
from . import instance_selection_agent as isel  # <--- NEW

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------
load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _attach_dataset_to_agent_tools(agent: Agent, df: pd.DataFrame) -> None:
    """Attach the same pandas DataFrame instance to every tool on an agent."""
    for t in getattr(agent, "tools", []):
        t.dataset = df


# ---------------------------------------------------------------------
# Structured input shared by delegates
# ---------------------------------------------------------------------
class DelegateInput(BaseModel):
    user_request: str = Field(..., description="Raw user request to forward to the chosen sub‑agent.")


# ---------------------------------------------------------------------
# Delegates
# ---------------------------------------------------------------------
class DelegateToAnalysisTool(BaseTool):
    name: str = "delegate_to_analysis"
    description: str = (
        "Delegate to the Analysis Agent (describe feature, mean/median/mode, grouped stats, etc.). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, user_request: str) -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to analysis delegate. Attach a DataFrame via tool.dataset = df."
        sub_agent = analysis.build_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to ANALYSIS', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = analysis.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request})
        return "🔀 Route: ANALYSIS\n" + str(result)


class DelegateToPreprocessingTool(BaseTool):
    name: str = "delegate_to_preprocessing"
    description: str = (
        "Delegate to the Preprocessing Agent (discretization/binning, one‑hot encoding, etc.). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    dataset: Type[pd.DataFrame] = None
    autosave_path: Optional[str] = "pipeline_data/dataset.csv"

    def _run(self, user_request: str) -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to preprocessing delegate. Attach a DataFrame via tool.dataset = df."
        sub_agent = preprocessing.build_preprocessing_agent(
            tools=[preprocessing.DiscretizeFeatureTool(), preprocessing.OneHotEncodeFeatureTool()]
        )
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to PREPROCESSING', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = preprocessing.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request})

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                self.dataset.to_csv(self.autosave_path, index=False)
                result = str(result) + f"\n\n💾 Saved changes to {self.autosave_path}"
            except Exception as se:
                result = str(result) + f"\n\n⚠️ Failed to save dataset: {se}"

        return "🔀 Route: PREPROCESSING\n" + str(result)


class DelegateToFeatureSelectionTool(BaseTool):
    name: str = "delegate_to_feature_selection"
    description: str = (
        "Delegate to the Feature Selection Agent (k‑best, variance/correlation filters, RF importances, etc.). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    dataset: Type[pd.DataFrame] = None
    autosave_path: Optional[str] = "pipeline_data/dataset.csv"

    def _run(self, user_request: str) -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to feature‑selection delegate. Attach a DataFrame via tool.dataset = df."
        sub_agent = fsel.build_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to FEATURE_SELECTION', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = fsel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request})

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                self.dataset.to_csv(self.autosave_path, index=False)
                result = str(result) + f"\n\n💾 Saved changes to {self.autosave_path}"
            except Exception as se:
                result = str(result) + f"\n\n⚠️ Failed to save dataset: {se}"

        return "🔀 Route: FEATURE_SELECTION\n" + str(result)


class DelegateToInstanceSelectionTool(BaseTool):  # <--- NEW
    name: str = "delegate_to_instance_selection"
    description: str = (
        "Delegate to the Instance Selection Agent (stratified/random/class‑balanced/clustered sampling, "
        "or dataset splits train/val/test, time‑series split). Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    dataset: Type[pd.DataFrame] = None
    autosave_path: Optional[str] = "pipeline_data/dataset.csv"

    def _run(self, user_request: str) -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to instance‑selection delegate. Attach a DataFrame via tool.dataset = df."
        # Expecting the structured multi‑tools implementation with build_agent/build_task
        sub_agent = isel.build_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to INSTANCE_SELECTION', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = isel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request})

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                # instance selection/splitting typically creates new CSVs; also persist the working df
                self.dataset.to_csv(self.autosave_path, index=False)
                result = str(result) + f"\n\n💾 Saved working dataset to {self.autosave_path}"
            except Exception as se:
                result = str(result) + f"\n\n⚠️ Failed to save dataset: {se}"

        return "🔀 Route: INSTANCE_SELECTION\n" + str(result)


# ---------------------------------------------------------------------
# Coordinator agent + task
# ---------------------------------------------------------------------

def build_coordinator_agent(df: pd.DataFrame) -> Agent:
    """Create a coordinator with delegate tools and attach the dataset to all delegates."""
    to_analysis = DelegateToAnalysisTool()
    to_pre = DelegateToPreprocessingTool()
    to_fs = DelegateToFeatureSelectionTool()
    to_is = DelegateToInstanceSelectionTool()  # <--- NEW

    for t in (to_analysis, to_pre, to_fs, to_is):
        t.dataset = df

    return Agent(
        role="Coordinator",
        goal=(
            "Read the user's request and decide whether it is an ANALYSIS request (describe/compute stats), "
            "a PREPROCESSING request (discretize, one‑hot, etc.), a FEATURE SELECTION request (k‑best/filters/importances), "
            "or an INSTANCE SELECTION/SPLITTING request (stratified/random/class‑balanced/clustered, train/val/test, time‑series). "
            "Call exactly ONE delegate tool with the original user_request and return its result verbatim."
        ),
        backstory=(
            "You are a precise orchestrator. You never compute yourself; you delegate to the most relevant specialist "
            "and deliver their answer."
        ),
        tools=[to_analysis, to_pre, to_fs, to_is],
        verbose=True,
        llm=llm,
    )


def build_coordinator_task(agent: Agent) -> Task:
    return Task(
        description=(
            "User request: {user_request}\n\n"
            "Decide:\n"
            "- If DESCRIPTIVE ANALYSIS or stats: call delegate_to_analysis.\n"
            "- If PREPROCESSING (discretization/binning, one‑hot, imputation/encoding handled elsewhere): call delegate_to_preprocessing.\n"
            "- If FEATURE SELECTION (k‑best, variance/correlation filters, RF importances, etc.): call delegate_to_feature_selection.\n"
            "- If INSTANCE SELECTION / SPLITTING (stratified, random, class‑balanced, clustered sampling; train/val/test; time‑series): "
            "  call delegate_to_instance_selection.\n\n"
            "Call exactly ONE delegate tool and return the sub‑agent's result. If the request is ambiguous, ask a single short clarifying question."
        ),
        expected_output="Either the sub‑agent's final answer or one short clarifying question.",
        agent=agent,
    )


# ---------------------------------------------------------------------
# Optional demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "x1": [1,2,3,4,5,6,7,8,9,10],
        "x2": [0,1,0,1,0,1,0,1,0,1],
        "y":  [0,0,0,1,1,1,1,1,0,0],
        "ds": pd.date_range("2024-01-01", periods=10, freq="D")
    })
    agent = build_coordinator_agent(df)
    task = build_coordinator_task(agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    examples = [
        "Compute mean of x1",
        "One‑hot encode x2",
        "Select k best features target=y k=1 scoring=f_classif",
        "Stratified sample target=y sample_size=0.6",
        "Time‑series split time_column=ds test_size=0.2 val_size=0.2",
    ]

    for req in examples:
        print("\n" + "="*100)
        print("USER:", req)
        print(crew.kickoff(inputs={"user_request": req}))
