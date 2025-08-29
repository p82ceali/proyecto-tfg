# agents/preprocessing_agent.py
"""
Data preprocessing agent (CrewAI).

Responsibilities:
    - Execute preprocessing steps such as discretization/binning and one-hot encoding.
    - Use only the necessary tool with correctly structured parameters.
    - Report succinctly which columns were created or modified.

Environment:
    Requires `GOOGLE_API_KEY` (e.g., via `.env`) for the configured LLM.

Notes:
    Tools expose structured parameters (Pydantic schemas). Pass arguments via
    those fields—avoid ad-hoc JSON strings.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from crewai import Agent, Task, LLM

from tfg_ml.adapters.preprocessing_tools import (
    DiscretizeFeatureTool,
    OneHotEncodeFeatureTool,
)

# ---------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------
load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
    # If supported by your CrewAI version, consider a lower temperature:
    # temperature=0.2
)


# ---------------------------------------------------------------------
# Agent & Task builders
# ---------------------------------------------------------------------
def build_preprocessing_agent() -> Agent:
    """
    Create the Data Preprocessing agent.

    Behavior:
        - Interprets the user's request.
        - Invokes only the relevant preprocessing tool(s).
        - Supplies parameters using each tool's structured schema.
        - After each operation, briefly describes the affected columns
          (new or modified).
        - If required parameters are missing, asks one brief clarifying
          question and stops.

    Returns:
        Agent: Configured CrewAI Agent with preprocessing tools attached.
    """
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
        tools=[DiscretizeFeatureTool(), OneHotEncodeFeatureTool()],
        verbose=True,
        llm=llm,
    )


def build_task(agent: Agent) -> Task:
    """
    Define the task prompt/instructions for the preprocessing agent.

    The task injects recent chat context and the current user request, then
    instructs the agent to:
        - Use ONLY the necessary preprocessing tool(s).
        - Pass parameters via structured fields (not JSON strings).
        - Ask exactly one short follow-up question if the request is underspecified
          (e.g., missing `column` or binning details).

    Args:
        agent: The agent that will execute this task.

    Returns:
        Task: Configured CrewAI Task with a concise expected output contract.
    """
    return Task(
        description=(
            "CONVERSATION CONTEXT (last turns):\n{chat_context}\n\n"
            "User request: {user_request}\n\n"
            "Choose and run only the relevant preprocessing tool. "
            "Fill the structured parameters according to the tool's schema "
            "(do NOT serialize JSON strings). "
            "If the request is underspecified (e.g., missing 'column' or binning details), "
            "ask ONE short follow-up question and stop."
        ),
        expected_output=(
            "A concise report of the preprocessing step(s) performed, including the method(s) used, "
            "parameters applied, and a brief description of the affected/new columns."
        ),
        agent=agent,
    )
