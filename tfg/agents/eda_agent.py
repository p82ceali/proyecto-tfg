from crewai import Agent, LLM
from tools.eda_tool import EDATool
from crewai_tools import DirectoryReadTool
from shared_context import SharedContext  # 👈 Añadimos esto
import os

shared_context = SharedContext()  # 👈 Instanciamos contexto compartido

def post_eda_step(step_details):
    """
    Callback para actualizar el contexto compartido tras el EDA.
    Guarda historial de ejecución con información básica.
    """
    try:
        file_path = shared_context.get_current_file()
        if file_path:
            shared_context.update_history(
                step_name="eda",
                notes="EDA completed successfully"
            )
    except Exception as e:
        print(f"[EDA step_callback error] {e}")
class EDAAgent:
    def create_agent(self):
        eda_tool = EDATool()
        data_read = DirectoryReadTool(directory='pipeline_data')

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
            tools=[eda_tool, data_read],
            llm=LLM(
                model="gemini/gemini-2.0-flash-lite",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            step_callback=post_eda_step,
            reasoning=True,
            max_reasoning_attempts=3,
            respect_context_window=True,
            max_iter=15,
            verbose=True
        )
