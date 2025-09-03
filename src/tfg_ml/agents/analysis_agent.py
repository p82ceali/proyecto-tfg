# agents/analysis_agent.py
"""
Analysis agent for exploratory data analysis (EDA).

This module defines:
- A lightweight LLM configuration (Gemini via CrewAI).
- `build_agent()`: an Agent configured with EDA tools.
- `build_task(agent)`: a Task that instructs the agent to use tools with
  structured parameters and to request clarification when inputs are underspecified.

Environment:
    Requires `GOOGLE_API_KEY` to be present (e.g., via `.env`).

Notes:
    Tools must declare structured parameters. The agent should pass arguments
    via the tool schema (not ad-hoc JSON strings).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from crewai import Agent, Task, LLM


# EDA tools (must provide structured parameters)
from tfg_ml.adapters.analysis_tools import (
    DescribeFeatureTool,
    ComputeStatisticTool,
)

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# Base LLM used by this agent
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)




def build_agent() -> Agent:
    """
    Create the Exploratory Data Analyst agent.

    Behavior:
        - Interprets the user's request.
        - Invokes only the necessary tools.
        - Supplies parameters using each tool's structured schema.
        - If a required parameter is missing (e.g., `column`), asks a brief,
          single clarifying question and stops.

    Returns:
        Agent: Configured CrewAI Agent with EDA tools attached.
    """
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
        llm=llm,
        max_iter=3,
        max_execution_time=30,
    )


def build_task(agent: Agent) -> Task:
    """
    Define the task prompt/instructions for the analysis agent.

    The task injects recent chat context and the current user request, then
    instructs the agent to:
        - Use ONLY the necessary tools.
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
            "Use ONLY the necessary tools to answer. "
            "If a tool needs parameters, pass them using the tool's structured fields "
            "(do NOT serialize JSON strings). "
            "If the request is underspecified (e.g., missing column), ask one short follow-up question and stop."
        ),
        expected_output=(
            "A concise, readable answer in the user's language with any computed values. "
            "If clarification was required, respond with a single, crisp question."
        ),
        agent=agent,
    )
