# agents/model_training_agent.py
"""
Model Training agent (CrewAI).

Responsibilities:
    - Select and train a classical ML model using the attached dataset.
    - Persist artifacts when requested by the tool.
    - Report key configuration and metrics concisely.

Environment:
    Requires `GOOGLE_API_KEY` (e.g., via `.env`) for the configured LLM.

Notes:
    The agent invokes a single structured tool (`ModelTrainingTool`) and must
    pass parameters via the tool's schema (avoid ad-hoc JSON strings).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from crewai import Agent, Task, LLM

from tfg_ml.adapters.model_training_tools import ModelTrainingTool

# ---------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------
load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
    # If supported in your CrewAI version, consider a lower temperature for stability:
    # temperature=0.2
)


# ---------------------------------------------------------------------
# Agent & Task builders
# ---------------------------------------------------------------------
def build_model_training_agent() -> Agent:
    """
    Create the Model Training agent.

    Behavior:
        - Interprets the user's request.
        - Chooses and runs only the model-training tool.
        - Supplies parameters using the tool's structured schema.
        - Avoids unnecessary executions; validates essential inputs.

    Returns:
        Agent: Configured CrewAI Agent with the model-training tool attached.
    """
    return Agent(
        role="Model Training Agent",
        goal=(
            "Select and train the requested model using the attached dataset, persist artifacts, "
            "and report key metrics."
        ),
        backstory=(
            "Specialist in classical ML model training and evaluation. "
            "Understands the problem type, validates parameters, and avoids unnecessary tool runs."
        ),
        tools=[ModelTrainingTool()],
        verbose=True,
        llm=llm,
    )


def build_task(agent: Agent) -> Task:
    """
    Define the task prompt/instructions for the model-training agent.

    The task injects recent chat context and the current user request, then
    instructs the agent to:
        - Use ONLY the model-training tool.
        - Pass parameters via structured fields (not JSON strings).
        - Ask exactly one short follow-up question if essential information is missing
          (e.g., `target`, `problem_type`).

    Args:
        agent: The agent that will execute this task.

    Returns:
        Task: Configured CrewAI Task with a concise expected output contract.
    """
    return Task(
        description=(
            "CONVERSATION CONTEXT (last turns):\n{chat_context}\n\n"
            "User request: {user_request}\n\n"
            "Choose and execute ONLY the relevant training tool (model_training). "
            "Fill in structured parameters according to the tool's Pydantic schema "
            "(do NOT serialize JSON strings). "
            "If essential info is missing (e.g., target or problem_type), ask ONE brief follow-up question and stop."
        ),
        expected_output=(
            "A concise report including: chosen/ trained model, problem type, target, train/test split strategy, "
            "and key evaluation metrics. Mention artifact paths if any were persisted."
        ),
        agent=agent,
    )
