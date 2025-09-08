# agents/feature_selection_agent_structured_tools.py
"""
Feature Selection agent using structured tools.

This module defines:
- A lightweight LLM configuration (Gemini via CrewAI).
- `build_agent()`: an Agent preloaded with feature-selection tools.
- `build_task(agent)`: a Task instructing the agent to use structured parameters
  and request clarification when inputs are underspecified.

Environment:
    Requires `GOOGLE_API_KEY` (e.g., via `.env`).

Notes:
    Tools declare structured parameters; the agent must pass arguments via each
    tool's schema (avoid ad-hoc JSON strings).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from crewai import Agent, Task, LLM

from tfg_ml.adapters.feature_selection_tools import (
    SelectKBestTool,
    VarianceThresholdTool,
    RFImportanceSelectTool,
    CorrelationFilterTool,
)

load_dotenv()


llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)


def build_agent() -> Agent:
    """
    Create the Feature Selection agent.

    Behavior:
        - Interprets the user's request.
        - Invokes only the necessary tool for feature selection.
        - Supplies parameters using each tool's structured schema.
        - If a required parameter is missing (e.g., `target`, `k`), asks one
          brief clarifying question and stops.

    Returns:
        Agent: Configured CrewAI Agent with feature-selection tools attached.
    """
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
        max_iter=3,
        max_execution_time=30,
    )


def build_task(agent: Agent) -> Task:
    """
    Define the task prompt/instructions for the feature-selection agent.

    The task injects recent chat context and the current user request, then
    instructs the agent to:
        - Use ONLY the necessary tool.
        - Pass parameters via structured fields (not JSON strings).
        - Ask exactly one short follow-up question if the request is underspecified.

    Args:
        agent: The agent that will execute this task.

    Returns:
        Task: Configured CrewAI Task with clear expected output.
    """
    return Task(
        description=(
            "CONVERSATION CONTEXT (last turns):\n{chat_context}\n\n"
            "User request: {user_request}\n\n"
            "Use ONLY the necessary tool to answer. "
            "If a tool needs parameters, pass them using the tool's structured fields (do NOT serialize JSON strings). "
            "If the request is underspecified (e.g., missing target), ask one short follow-up question and stop."
        ),
        expected_output=(
            "A concise answer stating the method applied, parameters used, and the selected vs. removed features."
        ),
        agent=agent,
    )
