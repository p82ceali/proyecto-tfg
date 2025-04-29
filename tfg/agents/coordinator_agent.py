from crewai import Agent, LLM
import os

class SuperAgent:
    def create_agent(self):
        return Agent(
            role="Machine Learning Crew Coordinator",
            goal="""Coordinate and assign the appropriate tasks based on user requests.""",
            backstory="""
                You are an expert ML project coordinator. 
                Your job is to listen carefully to the user's request and decide which task to execute:
                - eda
                - data_cleaning
                - instance_selection
                - feature_selection
                - model_training

                Always reason carefully and output only the best next task.
                """,
            llm=LLM(
                model="gemini/gemini-2.0-flash-exp",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            allow_delegation=True,
            verbose=True
        )