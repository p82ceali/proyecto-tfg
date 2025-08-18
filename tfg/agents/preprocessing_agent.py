from crewai import Agent, Task, LLM
from tools.preprocessing_tools import DiscretizeFeatureTool, OneHotEncodeFeatureTool
from crewai import Agent, Task, LLM


from dotenv import load_dotenv
import os

# Establece tu API key
load_dotenv()

# Crea el modelo base para los agentes
llm = LLM(model="gemini/gemini-2.0-flash-lite",
          api_key=os.getenv("GOOGLE_API_KEY"),
          custom_llm_provider="gemini"
          ) # OpenAI(temperature=0.3)


def build_preprocessing_agent() -> Agent:
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
        llm=llm
    )

def build_task(agent: Agent) -> Task:
    return Task(
        description=(
            "CONVERSATION CONTEXT (last turns):\n{chat_context}\n\n"
            "User request: {user_request}\n\n"
            "Choose and run only the relevant preprocessing tool. "
            "Fill the structured parameters according to the tool's schema. "
            "If the request is underspecified (e.g., missing 'column' or binning details), ask ONE short follow-up question and stop."
        ),
        expected_output=(
            "A concise report of the preprocessing step performed and a description of the affected features."
        ),
        agent=agent,
    )


