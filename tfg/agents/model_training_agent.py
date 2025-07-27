from crewai import Agent, LLM
import pandas as pd
from tools.model_training_tool import ModelTrainingTool
from crewai_tools import DirectoryReadTool
from shared_context import SharedContext  # 👈 Añadido
import os


class ModelTrainingAgent:
    
    def create_agent(self):
        training_tool = ModelTrainingTool()
        data_read = DirectoryReadTool(directory='pipeline_data')
        
        return Agent(
            role="ml model trainer",
            goal="""[For ML DevOps] Train and optimize production-ready machine learning models with CI/CD integration, 
                    performance monitoring, and scalable deployment strategies. Ensure models are robust, efficient, and 
                    capable of handling real-world data scenarios. Provide detailed explanations of the training process, 
                    hyperparameter tuning, and model evaluation metrics.""",
            backstory="""You are a highly experienced MLOps engineer specializing in scalable and automated model training. 
                         As the MLOps Lead at Google Cloud AI (2021-Present), you built AutoML systems processing over 50,000 
                         inference requests daily. You have deep expertise in CI/CD pipelines for ML, ensuring seamless deployment 
                         and monitoring of machine learning models in production. Your mission is to assist ML developers in training, 
                         fine-tuning, and deploying models for real-world applications with maximum efficiency and reliability.""",
            tools=[training_tool, data_read],
            llm=LLM(
                model="gemini/gemini-2.5-flash-lite",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            reasoning=True,
            max_reasoning_attempts=3,
            respect_context_window=True,
            max_iter=15,
            verbose=True
        )
