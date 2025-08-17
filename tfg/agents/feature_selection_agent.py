# feature_selection_agent_structured_tools.py
from crewai import Agent, Task, LLM
from tools.feature_selection_tools import (
    SelectKBestTool, VarianceThresholdTool,
    RFImportanceSelectTool, CorrelationFilterTool
)

from crewai import Agent, Task, LLM

from dotenv import load_dotenv
import os


load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)


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


