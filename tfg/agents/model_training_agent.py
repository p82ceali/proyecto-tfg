from crewai import Agent, LLM
from tools.model_training_tool import ModelTrainingTool
from crewai_tools import DirectoryReadTool
import os

docs_tool = DirectoryReadTool(directory='instance_selection_data')


class ModelTrainingAgent:
    def create_agent(self):
        training_tool = ModelTrainingTool()
        data_read = DirectoryReadTool(directory='processed_data')

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
                model="gemini/gemini-2.0-flash-exp",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            allow_delegation=False,
            verbose=True
        )