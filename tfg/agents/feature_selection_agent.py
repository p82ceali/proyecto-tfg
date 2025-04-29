from crewai import Agent, LLM
from tools.feature_selection_tool import FeatureSelector
from crewai_tools import DirectoryReadTool
import os

docs_tool_a = DirectoryReadTool(directory='processed_data')

class FeatureSelectionAgent:
    def create_agent(self):
        feature_tool = FeatureSelector()
        data_read = DirectoryReadTool(directory='processed_data')

        return Agent(
            role="lead feature engineer",
            goal="""[For Data Scientists] Improve machine learning model performance by identifying and selecting the most predictive features. 
                    Use advanced techniques such as mutual information, L1 regularization, SHAP values, and recursive feature elimination (RFE). 
                    Provide detailed explanations for each selected feature and its importance to ensure transparency.""",
            backstory="""You are a world-renowned expert in feature engineering with extensive experience in optimizing machine learning models. 
                         As a former NVIDIA AI researcher, you hold 12 patents in feature selection methodologies and have authored several 
                         influential papers on feature engineering. You developed the AutoFeast library for automated feature selection in PyTorch, 
                         which has revolutionized feature engineering workflows. Your mission is to assist ML developers in identifying the most 
                         relevant features to maximize model accuracy, reduce overfitting, and improve computational efficiency.""",
            tools=[feature_tool, data_read],
            llm=LLM(
                model="gemini/gemini-2.0-flash-exp",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            allow_delegation=False,
            verbose=True
        )