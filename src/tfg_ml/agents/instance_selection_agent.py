# instance_selection_agent_structured_multi_tools.py
from __future__ import annotations
from tfg_ml.adapters.instance_selection_tools import (
    StratifiedSampleTool, RandomSampleTool, ClassBalancedSampleTool, ClusteredSampleTool,
    TrainValTestSplitTool, TimeSeriesSplitTool
)

from crewai import Agent, Task, LLM

from dotenv import load_dotenv
import os

# =====================================================================
# LLM Setup
# =====================================================================
load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)



# =====================================================================
# Agent + Task wiring
# =====================================================================

def build_agent() -> Agent:
    return Agent(
        role="Instance Selection & Splitting Expert",
        goal=(
            "Understand the user's request and call only the necessary tool: stratified/random/class-balanced/clustered "
            "sampling or dataset splitting (train/val/test, time series). Ask a short question if required params are missing."
        ),
        backstory=(
            "You are a data sampling specialist who minimizes bias and preserves representativeness. "
            "You never run unnecessary tools and you always report dataset sizes and save locations."
        ),
        tools=[
            StratifiedSampleTool(),
            RandomSampleTool(),
            ClassBalancedSampleTool(),
            ClusteredSampleTool(),
            TrainValTestSplitTool(),
            TimeSeriesSplitTool(),
        ],
        verbose=True,
        llm=llm,
    )


def build_task(agent: Agent) -> Task:
    return Task(
        description=(
            "CONVERSATION CONTEXT (last turns):\n{chat_context}\n\n"
            "User request: {user_request}\n\n"
            "Use ONLY the necessary tool to answer. If a required parameter (e.g., target, time_column) is missing, "
            "ask one brief clarifying question and stop."
        ),
        expected_output=(
            "A concise answer stating: method applied, parameters used, size(s) obtained, and save location(s)."
        ),
        agent=agent,
    )
