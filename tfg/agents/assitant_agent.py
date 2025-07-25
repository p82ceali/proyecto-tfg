# agents/assistant_agent.py
from crewai import Agent, LLM
from tools.context_summary_tool import SummarizeContextTool
from tools.next_task_decision_tool import NextTaskDecisionTool
import os

class AssistantAgent:
    def create_agent(self):
        return Agent(
            role="Technical Assistant",
            goal="Support the Coordinator by providing context summaries and deciding the next ML tasks when needed.",
            backstory="You are an assistant agent that provides context analysis and helps decide the next steps in the ML pipeline.",
            tools=[SummarizeContextTool(), NextTaskDecisionTool()],
            llm=LLM(
                model="gemini/gemini-2.5-flash-lite",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            allow_delegation=False,
            verbose=True
        )
