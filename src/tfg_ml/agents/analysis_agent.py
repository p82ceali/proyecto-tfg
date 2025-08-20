# agents/analysis_agent.py
from crewai import Agent, Task, LLM
from tfg_ml.adapters.analysis_tools import DescribeFeatureTool, ComputeStatisticTool 



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



def build_agent():
    
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
        llm=llm
    )

def build_task(agent: Agent) -> Task:
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
            "A concise answer with any computed values. "
            "If clarification was required, a single crisp question."
        ),
        agent=agent,
    )
