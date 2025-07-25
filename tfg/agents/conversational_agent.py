# agents/conversational_agent.py
from crewai import Agent, LLM
import os

class ConversationalAgent:
    def create_agent(self):
        return Agent(
            role="ML Conversational Assistant",
            goal=(
                "Engage in a friendly and professional conversation with the user, "
                "understand their needs related to the ML pipeline, and coordinate with the project manager agent "
                "to execute the appropriate tasks."
            ),
            backstory=(
                "You are the friendly interface of a Machine Learning project assistant. "
                "You talk to the user in natural language, clarify their goals, "
                "and collaborate with the project coordinator agent to get things done. "
                "Your mission is to make the interaction smooth, informative, and collaborative."
            ),
            llm=LLM(
                model="gemini/gemini-2.5-flash-lite",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            allow_delegation=True,  # Can delegate to the CoordinatorAgent
            verbose=True
        )
