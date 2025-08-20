# coordinator_agent.py (updated to include Model Training Agent) 
from __future__ import annotations
from typing import Type, Optional, List

import os
import pandas as pd
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv


# === Relative imports (keep your current package layout) ===
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
    chat_context: Optional[str] = Field("", description="Window of chat history.")


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

    def _run(self, user_request: str, chat_context: str = "") -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to analysis delegate. Attach a DataFrame via tool.dataset = df."
        sub_agent = analysis.build_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to ANALYSIS', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = analysis.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={
            "user_request": user_request,
            "chat_context": chat_context, 
        })
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

    def _run(self, user_request: str, chat_context: str = "") -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to preprocessing delegate. Attach a DataFrame via tool.dataset = df."
        sub_agent = preprocessing.build_preprocessing_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to PREPROCESSING', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = preprocessing.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={
            "user_request": user_request,
            "chat_context": chat_context, 
        })

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

    def _run(self, user_request: str, chat_context: str = "") -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to feature‑selection delegate. Attach a DataFrame via tool.dataset = df."
        sub_agent = fsel.build_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to FEATURE_SELECTION', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = fsel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={
            "user_request": user_request,
            "chat_context": chat_context,
        })

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                self.dataset.to_csv(self.autosave_path, index=False)
                result = str(result) + f"\n\n💾 Saved changes to {self.autosave_path}"
            except Exception as se:
                result = str(result) + f"\n\n⚠️ Failed to save dataset: {se}"

        return "🔀 Route: FEATURE_SELECTION\n" + str(result)


class DelegateToInstanceSelectionTool(BaseTool):
    name: str = "delegate_to_instance_selection"
    description: str = (
        "Delegate to the Instance Selection Agent (stratified/random/class‑balanced/clustered sampling, "
        "or dataset splits train/val/test, time‑series split). Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    dataset: Type[pd.DataFrame] = None
    autosave_path: Optional[str] = "pipeline_data/dataset.csv"

    def _run(self, user_request: str, chat_context: str = "") -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to instance‑selection delegate. Attach a DataFrame via tool.dataset = df."
        sub_agent = isel.build_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to INSTANCE_SELECTION', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = isel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={
            "user_request": user_request,
            "chat_context": chat_context, 
        })

        if self.autosave_path:
            try:
                os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
                self.dataset.to_csv(self.autosave_path, index=False)
                result = str(result) + f"\n\n💾 Saved working dataset to {self.autosave_path}"
            except Exception as se:
                result = str(result) + f"\n\n⚠️ Failed to save dataset: {se}"

        return "🔀 Route: INSTAN CE_SELECTION\n" + str(result)


# -------------------------Model Training delegate -------------------------
class DelegateToModelTrainingTool(BaseTool):
    name: str = "delegate_to_model_training"
    description: str = (
        "Delegate to the Model Training Agent (model selection, training, metrics, and artifact persistence). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    dataset: Type[pd.DataFrame] = None
    artifacts_dir: Optional[str] = "pipeline_data/artifacts"

    def _run(self, user_request: str, chat_context: str = "") -> str:
        if not hasattr(self, "dataset") or not isinstance(self.dataset, pd.DataFrame):
            return "No dataset attached to model‑training delegate. Attach a DataFrame via tool.dataset = df."

        sub_agent = mtrain.build_model_training_agent()
        print('\n', '*'*30, '\n', 'COORDINATOR: delegating to MODEL_TRAINING', '\n', '*'*30)
        _attach_dataset_to_agent_tools(sub_agent, self.dataset)
        sub_task = mtrain.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={
            "user_request": user_request,
            "chat_context": chat_context,  
        })
        return "🔀 Route: MODEL_TRAINING\n" + str(result)


# ---------------------------------------------------------------------
# Coordinator agent + task
# ---------------------------------------------------------------------
def build_coordinator_agent(df: pd.DataFrame) -> Agent:
    """Create a coordinator with delegate tools and attach the dataset to all delegates."""
    delegates: List[BaseTool] = [
        DelegateToAnalysisTool(),
        DelegateToPreprocessingTool(),
        DelegateToFeatureSelectionTool(),
        DelegateToInstanceSelectionTool(),
        DelegateToModelTrainingTool(),  # <-- NEW
    ]
    for t in delegates:
        t.dataset = df

    return Agent(
        role="Coordinator",
        goal = (
            "Read the user's request. If the message is SMALL TALK / OFF-TOPIC "
            "(e.g., greetings like 'hola/hello', thanks, casual chit-chat, jokes, meta-questions about the assistant, "
            "or any general non-ML request), reply directly using the LLM in the SAME language as the user — "
            "do NOT call any tool and do NOT delegate. "
            "Otherwise, decide whether it is an ANALYSIS, PREPROCESSING, FEATURE SELECTION, "
            "INSTANCE SELECTION/SPLITTING, or MODEL TRAINING request. "
            "Call exactly ONE delegate tool with the original user_request and return its result verbatim. "
            "If the request is ambiguous (and NOT small talk), ask ONE short clarifying question and stop."
        ),

        backstory = (
        "You are a precise orchestrator. First you detect small talk/off-topic and answer it yourself "
        "(no tools, no delegation, brief and friendly, same language). "
        "For ML-related intents, you never compute yourself; you delegate to the single most relevant specialist "
        "and deliver their answer verbatim. You never call more than one tool."
     ),
        tools=delegates,
        verbose=True,
        llm=llm,
    )


def build_coordinator_task(agent: Agent) -> Task:
    return Task(
        description=(
            "CONVERSATION CONTEXT (last turns):\n{chat_context}\n\n"
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
        expected_output="Either a direct short reply (no tools, no delegation), the sub-agent's final answer, or one short clarifying question.",
        agent=agent,
    )


