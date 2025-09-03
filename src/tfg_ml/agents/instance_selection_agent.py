# agents/instance_selection_agent.py
"""
Instance selection & dataset splitting agent using structured tools.

This module defines:
- A lightweight LLM configuration (Gemini via CrewAI).
- `build_agent()`: an Agent with sampling/splitting tools (stratified, random,
  class-balanced, clustered sampling; train/val/test and time-series splits).
- `build_task(agent)`: a Task instructing the agent to use only the necessary
  tool with structured parameters and to request clarification when inputs
  are underspecified.

Environment:
    Requires `GOOGLE_API_KEY` (e.g., via `.env`).

Notes:
    Tools declare structured parameters; the agent must pass arguments through
    each tool's schema (avoid ad-hoc JSON strings).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from crewai import Agent, Task, LLM

from tfg_ml.adapters.instance_selection_tools import (
    StratifiedSampleTool,
    RandomSampleTool,
    ClassBalancedSampleTool,
    ClusteredSampleTool,
    TrainValTestSplitTool,
    TimeSeriesSplitTool,
)

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# Base LLM used by this agent
# If supported by your CrewAI version, you may pass model kwargs (e.g., temperature).
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)


def build_agent() -> Agent:
    """
    Create the Instance Selection & Splitting agent.

    Behavior:
        - Interprets the user's request.
        - Invokes only the necessary tool among sampling/splitting options.
        - Supplies parameters using each tool's structured schema.
        - If a required parameter is missing (e.g., `target`, `time_column`),
          asks one brief clarifying question and stops.

    Returns:
        Agent: Configured CrewAI Agent with sampling/splitting tools attached.
    """
    return Agent(
        role="Instance Selection & Splitting Expert",
        goal=(
            "Understand the user's request and call only the necessary tool: stratified/random/class-balanced/"
            "clustered sampling or dataset splitting (train/val/test, time series). "
            "Ask a short question if required parameters are missing."
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
        max_iter=3,
        max_execution_time=30,
    )


def build_task(agent: Agent) -> Task:
    """
    Define the task prompt/instructions for the instance selection / splitting agent.

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
            "Use ONLY the necessary tool to answer. If a required parameter (e.g., target, time_column) is missing, "
            "ask one brief clarifying question and stop. Pass parameters using the tool's structured fields "
            "(do NOT serialize JSON strings)."
        ),
        expected_output=(
            "A concise answer stating the method applied, parameters used, resulting size(s), and save location(s) if applicable."
        ),
        agent=agent,
    )
