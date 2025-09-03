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

# Default dataset location written by the UI
DATASET_PATH = Path("data/dataset.csv")


def _load_active_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the active dataset expected by the Coordinator.

    Args:
        path: Path to the CSV file persisted by the UI.

    Returns:
        pandas.DataFrame: The in-memory dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        pd.errors.EmptyDataError / ParserError: If the CSV is unreadable.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Upload a CSV from the UI before running the pipeline."
        )
    return pd.read_csv(path)


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
    

    # 2) Build Coordinator and its task (attach dataset to delegate tools)
    chat_context = CTX.as_prompt()
    coordinator = build_coordinator_agent()
    task = build_coordinator_task(coordinator)

    # 3) Run Crew with inputs
    crew = Crew(agents=[coordinator], tasks=[task], verbose=True)
    result = crew.kickoff(
        inputs={
            "user_request": pregunta,
            "chat_context": chat_context,
        }
    )

    # 4) Persist interaction in global context
    respuesta = str(result)
    CTX.add(pregunta, respuesta)

    # 5) Return formatted answer
    return respuesta
