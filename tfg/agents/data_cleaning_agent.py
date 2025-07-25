from crewai import Agent, LLM
from tools.data_cleaning_tool import DataCleaningTool
import pandas as pd
from crewai_tools import DirectoryReadTool
import os
from shared_context import SharedContext

shared_context = SharedContext()

def on_step(step_details):
    # step_output puede ser AgentAction o ToolResult
    # Si ha ejecutado la herramienta de limpieza, podemos actualizar context
    try:
        cleaned = "pipeline_data/dataset.csv"
        if os.path.exists(cleaned):
            df = pd.read_csv(cleaned)
            shared_context.set_columns(df.columns.tolist())
            shared_context.set_current_file(cleaned)
            shared_context.update_history("data_cleaning", notes="Columns updated after cleaning")
    except Exception as e:
        print(f"[step_callback error] {e}")
class DataCleaningAgent:
    def create_agent(self):
        data_cleaning_tool = DataCleaningTool()
        data_read = DirectoryReadTool(directory='pipeline_data')

        return Agent(
            role="ml data quality engineer",
            goal=""" [For ML Engineers] Ensure high-quality datasets for production ML pipelines by performing advanced data cleaning tasks, 
                    including handling missing values with imputation techniques, removing duplicates, encoding categorical variables, 
                    and validating data consistency. Provide detailed explanations for each step to ensure transparency.""",
            backstory="""You are a highly experienced Machine Learning Engineer specializing in data preprocessing. 
                         You have successfully handled petabyte-scale datasets, designing data pipelines that reduced data processing 
                         times by 50% for Fortune 100 companies. Your expertise includes advanced null value imputation, outlier detection, 
                         deduplication, and feature engineering. You assist ML developers by transforming raw, noisy data into high-quality, 
                         production-ready datasets optimized for training and deployment.""",
            tools=[data_cleaning_tool, data_read],
            llm=LLM(
                model="gemini/gemini-2.5-flash-lite",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            step_callback=on_step, 
            reasoning=True,
            max_reasoning_attempts=3,
            respect_context_window=True,
            max_iter=15,
            verbose=True

        )
