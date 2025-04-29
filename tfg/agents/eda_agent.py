from crewai import Agent, LLM
from tools.eda_tool import EDATool
from crewai_tools import DirectoryReadTool
import os

class EDAAgent:
    def create_agent(self):
        # Herramienta para realizar EDA
        eda_tool = EDATool()

        # Crear y devolver el agente
        return Agent(
            role="Senior Data Analyst",
            goal=(
                "[For technical users] Generate automated exploratory data analysis (EDA) reports "
                "describing dataset structure and quality. Provide insights into missing values, outliers, "
                "and key statistical summaries."
            ),
            backstory=(
                "You are a data analysis expert with 10+ years of experience exploring complex datasets. "
                "You have consulted for Fortune 500 companies, specializing in identifying data quality issues. "
                "Your strength lies in translating technical data into non-technical insights, helping teams "
                "make data-driven decisions."
            ),
            tools=[eda_tool],
            llm=LLM(
                model="gemini/gemini-2.0-flash-exp",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            verbose=True
        )