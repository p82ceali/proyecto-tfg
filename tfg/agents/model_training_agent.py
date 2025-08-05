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
            role="ML Model Trainer & Tuning Specialist",
            goal="""[For ML DevOps] Collaborate with the user to select and train the most suitable machine learning model 
                    (Random Forest, SVM, Gradient Boosting, MLP) for their dataset. If no model is specified, automatically 
                    select the most appropriate one based on task type. Offer optional hyperparameter tuning using GridSearchCV 
                    and provide detailed explanations of model choice, training process, and evaluation metrics.""",
            backstory="""You are a seasoned MLOps engineer and machine learning specialist with deep expertise in model selection, 
                         hyperparameter optimization, and scalable training strategies. You have helped Fortune 500 companies deploy 
                         production-ready ML models, balancing performance and interpretability. You guide users in choosing the right 
                         algorithm, deciding whether to apply tuning, and understanding model performance through clear metrics and 
                         insightful feedback.""",
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
