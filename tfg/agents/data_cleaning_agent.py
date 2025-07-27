from crewai import Agent, LLM
from tools.data_cleaning_tool import DataCleaningTool
import pandas as pd
from crewai_tools import DirectoryReadTool
import os
from shared_context import SharedContext

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
            reasoning=True,
            max_reasoning_attempts=3,
            max_iter=15,
            verbose=True

        )
