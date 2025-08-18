# model_training_agent.py
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Literal, Tuple

import os
import json
import joblib
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field, field_validator

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

from dotenv import load_dotenv

from tools.model_training_tools import ModelTrainingTool

# --------------------------------------------------------------------------------------
# Unified LLM config (aligned with your new repository style)
# --------------------------------------------------------------------------------------
load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)


# --------------------------------------------------------------------------------------
# Agent + Task builders
# --------------------------------------------------------------------------------------

def build_model_training_agent() -> Agent:
    return Agent(
        role="Model Training Agent",
        goal=(
            "Select and train the requested model using the attached dataset, persist artifacts, and report key metrics."
        ),
        backstory=(
            "Specialist in classical ML model training and evaluation. "
            "Understands the problem type, validates parameters, and avoids unnecessary tool runs."
        ),
        tools=[
            ModelTrainingTool()],
        verbose=True,
        llm=llm,
    )


def build_task(agent: Agent) -> Task:
    return Task(
        description=(
            "CONVERSATION CONTEXT (last turns):\n{chat_context}\n\n"
            "User request: {user_request}"
            "Choose and execute ONLY the relevant training tool (model_training). "
            "Fill in structured parameters according to the Pydantic schema. "
            "If essential info is missing (target or problem_type), ask ONE brief follow-up question and stop."
        ),
        expected_output=(
            "A short report including: trained model, problem type, target, train/test split, and key metrics."
        ),
        agent=agent,
    )


