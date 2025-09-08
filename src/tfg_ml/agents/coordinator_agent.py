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

"""

from __future__ import annotations

import hashlib
from threading import Lock
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

class _OncePerInputMixin:
    """
    Prevents re-running the same _run(...) with the same (user_request, chat_context)
    within the lifetime of this tool instance.
    """
    _lock: Lock = Lock()
    _last_fp: Optional[str] = None
    _last_result: Optional[str] = None

    def _fingerprint(self, user_request: str, chat_context: str) -> str:
        raw = (user_request or "").strip() + "||" + (chat_context or "").strip()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_duplicate(self, fp: str) -> bool:
        with self._lock:
            if self._last_fp == fp:
                return True
            self._last_fp = fp
            return False

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
class DelegateToAnalysisTool(_OncePerInputMixin, BaseTool):
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

    def _run(self, user_request: str, chat_context: str = "", ) -> str:
       
        fp = self._fingerprint(user_request, chat_context)
        if self._is_duplicate(fp):
            return self._last_result
        sub_agent = analysis.build_agent()
        sub_task = analysis.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})
        result_str = str(result)
        self._last_result = result_str 

        return result_str


class DelegateToPreprocessingTool(_OncePerInputMixin, BaseTool):
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
        fp = self._fingerprint(user_request, chat_context)
        if self._is_duplicate(fp):
            return self._last_result

        sub_agent = preprocessing.build_preprocessing_agent()
        sub_task = preprocessing.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})
        result_str = str(result)
        self._last_result = result_str 

        return result_str


class DelegateToFeatureSelectionTool(_OncePerInputMixin, BaseTool):
    """Delegate to the Feature Selection Agent (k-best, filters, importances)."""
    name: str = "delegate_to_feature_selection"
    description: str = (
        "Delegate to the Feature Selection Agent (k-best, variance/correlation filters, RF importances, etc.). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    autosave_path: Optional[str] = "data/dataset.csv"
    def _run(self, user_request: str, chat_context: str = "") -> str:
        fp = self._fingerprint(user_request, chat_context)
        if self._is_duplicate(fp):
            return self._last_result
        
        sub_agent = fsel.build_agent()
        sub_task = fsel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})

        result_str = str(result)
        self._last_result = result_str 

        return result_str


class DelegateToInstanceSelectionTool(_OncePerInputMixin, BaseTool):
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
        fp = self._fingerprint(user_request, chat_context)
        if self._is_duplicate(fp):
            return self._last_result
        
        sub_agent = isel.build_agent()
        sub_task = isel.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})

        

        result_str = str(result)
        self._last_result = result_str 

        return result_str


# ------------------------- Model Training delegate -------------------------
class DelegateToModelTrainingTool(_OncePerInputMixin, BaseTool):
    """Delegate to the Model Training Agent (model selection, training, metrics, artifacts)."""
    name: str = "delegate_to_model_training"
    description: str = (
        "Delegate to the Model Training Agent (model selection, training, metrics, and artifact persistence). "
        "Input must include {'user_request': <text>}."
    )
    args_schema: Type[BaseModel] = DelegateInput
    artifacts_dir: Optional[str] = "pipeline_data/artifacts"
    def _run(self, user_request: str, chat_context: str = "") -> str:
        fp = self._fingerprint(user_request, chat_context)
        if self._is_duplicate(fp):
            return self._last_result
        sub_agent = mtrain.build_model_training_agent()
        sub_task = mtrain.build_task(sub_agent)
        sub_crew = Crew(agents=[sub_agent], tasks=[sub_task], verbose=False)
        result = sub_crew.kickoff(inputs={"user_request": user_request, "chat_context": chat_context})
        result_str = str(result)
        self._last_result = result_str 

        return result_str


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
        DelegateToModelTrainingTool(), 
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
            "Call at most ONE delegate tool exactly once with the original user_request. After receiving its result, STOP and return it verbatim. Do NOT retry or call any other tool."
            "If the request is ambiguous (and NOT small talk), ask ONE short clarifying question and stop."
        ),
        backstory=(
            "You are a precise orchestrator. First you detect small talk/off-topic and answer it yourself "
            "(no tools, no delegation, brief and friendly, same language). "
            "For ML-related intents, you never compute yourself; you delegate to the single most relevant specialist "
            "and deliver their answer verbatim. You never call more than one tool." \
            "You never call more than one tool, never repeat the same tool, "
            "and once you have a delegate’s answer, you return it as Final Answer and terminate."
        ),
        tools=delegates,
        verbose=True,
        llm=llm,
        max_execution_time=30,
        allow_delegation=False,
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
            "**After calling a delegate and receiving its response, STOP immediately. "
            "Do not call any delegate again. Do not retry.**"
        ),
        expected_output=(
            "Either a direct short reply (no tools, no delegation), the sub-agent's final answer, "
            "or one short clarifying question."
        ),
        agent=agent,
    )
