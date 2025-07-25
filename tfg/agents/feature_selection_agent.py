# 📁 feature_selection_agent.py

from crewai import Agent, LLM
from tools.feature_selection_tool import FeatureSelector
from crewai_tools import DirectoryReadTool
from shared_context import SharedContext
import os

shared_context = SharedContext()
def post_feature_selection_step(step_details):
    """
    Callback para actualizar el contexto tras seleccionar características.
    Actualiza columnas en el contexto si el archivo resultante existe.
    """
    try:
        file_path = "pipeline_data/dataset.csv"
        if os.path.exists(file_path):
            import pandas as pd
            df = pd.read_csv(file_path)
            shared_context.set_columns(df.columns.tolist())
            shared_context.set_current_file(file_path)
            shared_context.update_history("feature_selection", notes="Updated dataset after feature selection")
    except Exception as e:
        print(f"[Feature Selection step_callback error] {e}")
class FeatureSelectionAgent:
    def create_agent(self):
        feature_tool = FeatureSelector()
        data_read = DirectoryReadTool(directory='pipeline_data')

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
                model="gemini/gemini-2.5-flash-lite",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            step_callback=post_feature_selection_step,  # Callback para actualizar contexto tras selección de características
            reasoning=True,
            max_reasoning_attempts=3,
            respect_context_window=True,
            max_iter=15,
            verbose=True
        )
