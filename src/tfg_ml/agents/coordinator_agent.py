# agents/coordinator_agent.py
"""
Coordinator agent that routes user requests to specialized sub-agents.

Responsibilities
----------------
- Detect small talk / off-topic and answer directly (no tools).
- For ML-related intents, choose exactly one specialized delegate:
  * Analysis (descriptive statistics, grouped stats, etc.)
  * Preprocessing (binning, one-hot encoding, …)
  * Feature selection (k-best, filters, importances)
  * Instance selection / splitting (sampling and splits)
  * Model training (model choice, training, metrics, artifacts)
- Forward the user's request and recent chat context to the chosen delegate.
- Return the delegate's answer (this file does **not** post-process it).

Notes
-----
- The dataset (a pandas DataFrame) is attached to each delegate tool via the
  `_attach_dataset_to_agent_tools` helper.
"""

from __future__ import annotations

from typing import Optional, List, Type
import os

import pandas as pd
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv

# Local delegates (keep your current package layout)
from . import analysis_agent as analysis
from . import preprocessing_agent as preprocessing
from . import feature_selection_agent as fsel
from . import instance_selection_agent as isel
from . import model_training_agent as mtrain

# ---------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------
load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
    # temperature=0.2
)

dataset_path="data/dataset.csv"

# ---------------------------------------------------------------------
# Structured input shared by delegates
# ---------------------------------------------------------------------
class DelegateInput(BaseModel):
    """
    Payload forwarded from the Coordinator to a single sub-agent.
    """
    user_request: str = Field(
        ...,
        description="Raw user request to forward to the chosen sub-agent.",
    )
    chat_context: Optional[str] = Field(
        "",
        description="Window of recent chat history.",
    )


# ---------------------------------------------------------------------
# Delegates
# ---------------------------------------------------------------------
class DelegateToAnalysisTool(BaseTool):
    """
    Delegate to the Analysis Agent (describe feature, mean/median/mode,
    grouped stats, etc.).
    """
    name: str = "delegate_to_analysis"
    description: str = (
        "Delegate to the Analysis Agent (describe feature, mean/median/mode, grouped stats, etc.). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    #dataset: Optional[pd.DataFrame] = None

    def _run(self, user_request: str, chat_context: str = "", ) -> str:
       
        sub_agent = analysis.build_agent()
        sub_task = analysis.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})
        return str(result)


class DelegateToPreprocessingTool(BaseTool):
    """
    Delegate to the Preprocessing Agent (discretization/binning, one-hot encoding, etc.).
    Optionally autosaves the working dataset.
    """
    name: str = "delegate_to_preprocessing"
    description: str = (
        "Delegate to the Preprocessing Agent (discretization/binning, one-hot encoding, etc.). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    autosave_path: Optional[str] = "data/dataset.csv"
    

    def _run(self, user_request: str, chat_context: str = "") -> str:
        dataset=pd.read_csv(dataset_path)
        if not isinstance(dataset, pd.DataFrame):
            return "No dataset found."
        sub_agent = preprocessing.build_preprocessing_agent()
        sub_task = preprocessing.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                dataset.to_csv(self.autosave_path, index=False)
                result = str(result)
            except Exception as se:
                result = f"\n\n⚠️ Failed to save dataset: {se}"

        return str(result)


class DelegateToFeatureSelectionTool(BaseTool):
    """Delegate to the Feature Selection Agent (k-best, filters, importances)."""
    name: str = "delegate_to_feature_selection"
    description: str = (
        "Delegate to the Feature Selection Agent (k-best, variance/correlation filters, RF importances, etc.). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    autosave_path: Optional[str] = "pipeline_data/dataset.csv"
    def _run(self, user_request: str, chat_context: str = "") -> str:
        dataset=pd.read_csv(dataset_path)
        if not isinstance(dataset, pd.DataFrame):
            return "No dataset found."
        sub_agent = fsel.build_agent()
        sub_task = fsel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                dataset.to_csv(self.autosave_path, index=False)
                result = str(result) 
            except Exception as se:
                result = f"\n\n⚠️ Failed to save dataset: {se}"

        return str(result)


class DelegateToInstanceSelectionTool(BaseTool):
    """
    Delegate to the Instance Selection Agent (sampling and dataset splitting):
    stratified/random/class-balanced/clustered sampling, train/val/test split,
    or time-series split.
    """
    name: str = "delegate_to_instance_selection"
    description: str = (
        "Delegate to the Instance Selection Agent (stratified/random/class-balanced/clustered sampling, "
        "or dataset splits train/val/test, time-series split). Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    autosave_path: Optional[str] = "data/dataset.csv"
    def _run(self, user_request: str, chat_context: str = "") -> str:
        dataset=pd.read_csv(dataset_path)

        if not isinstance(dataset, pd.DataFrame):
            return "No dataset found."

        sub_agent = isel.build_agent()
        sub_task = isel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                dataset.to_csv(self.autosave_path, index=False)
                result = str(result)
            except Exception as se:
                result = f"\n\n⚠️ Failed to save dataset: {se}" 

        return str(result)


# ------------------------- Model Training delegate -------------------------
class DelegateToModelTrainingTool(BaseTool):
    """Delegate to the Model Training Agent (model selection, training, metrics, artifacts)."""
    name: str = "delegate_to_model_training"
    description: str = (
        "Delegate to the Model Training Agent (model selection, training, metrics, and artifact persistence). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    artifacts_dir: Optional[str] = "pipeline_data/artifacts"
    def _run(self, user_request: str, chat_context: str = "") -> str:

        sub_agent = mtrain.build_model_training_agent()
        sub_task = mtrain.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})
        return str(result)


# ---------------------------------------------------------------------
# Coordinator agent + task
# ---------------------------------------------------------------------
def build_coordinator_agent() -> Agent:
    """
    Create the Coordinator and attach the dataset to all delegate tools.

    Args:
        df: In-memory dataset shared with the delegate tools.

    Returns:
        Agent: Configured Coordinator agent.
    """
    delegates: List[BaseTool] = [
        DelegateToAnalysisTool(),
        DelegateToPreprocessingTool(),
        DelegateToFeatureSelectionTool(),
        DelegateToInstanceSelectionTool(),
        DelegateToModelTrainingTool(),  # included
    ]

    return Agent(
        role="Coordinator",
        goal=(
            "Read the conversation context to know the recent interactions."
            "Read the user's request. If the message is SMALL TALK / OFF-TOPIC "
            "(e.g., greetings like 'hola/hello', thanks, casual chit-chat, jokes, meta-questions about the assistant, "
            "or any general non-ML request), reply directly using the LLM in the SAME language as the user — "
            "do NOT call any tool and do NOT delegate. "
            "Otherwise, decide whether it is an ANALYSIS, PREPROCESSING, FEATURE SELECTION, "
            "INSTANCE SELECTION/SPLITTING, or MODEL TRAINING request. "
            "Call exactly ONE delegate tool with the original user_request and return its result verbatim. "
            "If the request is ambiguous (and NOT small talk), ask ONE short clarifying question and stop."
        ),
        backstory=(
            "You are a precise orchestrator. First you detect small talk/off-topic and answer it yourself "
            "(no tools, no delegation, brief and friendly, same language). "
            "For ML-related intents, you never compute yourself; you delegate to the single most relevant specialist "
            "and deliver their answer verbatim. You never call more than one tool."
        ),
        tools=delegates,
        verbose=True,
        llm=llm,
        max_iter=3,
        max_execution_time=30,
    )


def build_coordinator_task(agent: Agent) -> Task:
    """
    Define the high-level instruction for the Coordinator, including routing policy.

    Returns:
        Task: A CrewAI Task that injects context/inputs and constrains the output.
    """
    return Task(
        description=(
            "CONVERSATION CONTEXT:\n{chat_context}\n\n"
            "User request: {user_request}\n\n"
            "Decision policy:\n"
            "FIRST — If the message is SMALL TALK / OFF-TOPIC (e.g., greetings like 'hola', 'hello', 'buenas', "
            "'thanks', casual chit-chat, jokes, meta-questions about the assistant, or any general non-ML request):\n"
            "- Reply directly using the LLM in the SAME language as the user.\n"
            "- DO NOT call any tool.\n"
            "- DO NOT delegate to any sub-agent.\n"
            "- Keep it brief and friendly.\n\n"
            "ELSE — Choose exactly ONE delegate tool and return the sub-agent's result:\n"
            "- If DESCRIPTIVE ANALYSIS or stats: call delegate_to_analysis.\n"
            "- If PREPROCESSING (discretization/binning, one-hot): call delegate_to_preprocessing.\n"
            "- If FEATURE SELECTION (k-best, filters, importances): call delegate_to_feature_selection.\n"
            "- If INSTANCE SELECTION / SPLITTING (sampling/splits): call delegate_to_instance_selection.\n"
            "- If MODEL TRAINING (choose/train/evaluate/persist): call delegate_to_model_training.\n\n"
            "If the request is ambiguous (and NOT small talk), ask ONE short clarifying question and stop."
        ),
        expected_output=(
            "Either a direct short reply (no tools, no delegation), the sub-agent's final answer, "
            "or one short clarifying question."
        ),
        agent=agent,
    )
