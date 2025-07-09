from crewai import Agent, LLM
from tools.instance_selection_tool import InstanceSelectionTool
from crewai_tools import DirectoryReadTool
from shared_context import SharedContext  # 👈 Añadido
import os

shared_context = SharedContext()  # 👈 Instancia compartida
def post_instance_selection_step(step_details):
    """
    Callback que actualiza el contexto después de reducir el dataset.
    Cambia el archivo actual a la nueva ruta y actualiza las columnas.
    """
    try:
        reduced_path = "pipeline_data/dataset.csv"
        if os.path.exists(reduced_path):
            import pandas as pd
            df = pd.read_csv(reduced_path)
            shared_context.set_current_file(reduced_path)
            shared_context.set_columns(df.columns.tolist())
            shared_context.update_history("instance_selection", notes="Reduced dataset to subset")
    except Exception as e:
        print(f"[Instance Selection step_callback error] {e}")
class InstanceSelectionAgent:
    def create_agent(self):
        instance_tool = InstanceSelectionTool()
        data_read = DirectoryReadTool(directory='pipeline_data')

        return Agent(
            role="data sampling architect",
            goal="""[For ML Engineers] Optimize dataset size while preserving statistical integrity using advanced sampling techniques, 
                    including stratified sampling, adaptive sampling, and clustering-based sampling. Ensure that the selected subset 
                    maintains representativeness and minimizes bias. Provide detailed explanations for the sampling strategy used.""",
            backstory="""You are a leading expert in data sampling strategies for machine learning. 
                         As the author of "Intelligent Sampling for Deep Learning" (O'Reilly, 2023), you have pioneered innovative techniques 
                         in data reduction. You designed Uber’s ML sampling framework, enabling real-time predictions while minimizing 
                         computational costs. Your mission is to assist ML developers in selecting representative subsets of data that 
                         preserve statistical integrity and maximize model performance while reducing computational overhead.""",
            tools=[instance_tool, data_read],
            llm=LLM(
                model="gemini/gemini-2.0-flash-exp",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
            ),
            step_callback=post_instance_selection_step,  # Callback para actualizar contexto tras selección de instancias
            reasoning=True,
            max_reasoning_attempts=3,
            respect_context_window=True,
            max_iter=15,
            verbose=True
        )
