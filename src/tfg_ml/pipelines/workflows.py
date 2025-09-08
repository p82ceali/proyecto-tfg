# tfg_ml/pipelines/workflows.py
"""
Crew manager: orchestrates a single end-to-end interaction by routing the
user request to the Coordinator agent (which delegates to specialized agents).

Flow:
    1) Load the active dataset from disk.
    2) Build the Coordinator (dataset is attached to delegate tools).
    3) Run the Crew with chat context and the user's request.
    4) Persist the interaction in the global context (CTX).
    5) Return a conversational Markdown answer.

Environment:
    Assumes an uploaded dataset at 'data/dataset.csv' (set by the UI).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from crewai import Crew

from tfg_ml.agents.coordinator_agent import build_coordinator_agent, build_coordinator_task
from tfg_ml.context import CTX

DATASET_PATH = Path("data/dataset.csv")





def run_pipeline(pregunta: str) -> str:
    """
    Orchestrate one pipeline run for the given user request.

    The Coordinator:
      - Decides whether to answer small talk directly or delegate to exactly
        one specialized agent (analysis, preprocessing, feature selection,
        instance selection/splitting, model training).
      - Reformulates the delegate's raw output into conversational Markdown.

    Args:
        pregunta: Raw user request (any language).

    Returns:
        str: Conversational Markdown answer produced by the Coordinator.
    """
    

   
    chat_context = CTX.as_prompt()
    coordinator = build_coordinator_agent()
    task = build_coordinator_task(coordinator)

    crew = Crew(agents=[coordinator], tasks=[task], verbose=True)
    result = crew.kickoff(
        inputs={
            "user_request": pregunta,
            "chat_context": chat_context,
        }
    )

    respuesta = str(result)
    CTX.add(pregunta, respuesta)

    return respuesta
