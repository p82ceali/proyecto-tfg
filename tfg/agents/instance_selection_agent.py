from crewai import Agent, LLM
from tools.instance_selection_tool import InstanceSelectionTool
from crewai_tools import DirectoryReadTool
import os

docs_tool_a = DirectoryReadTool(directory='features_data')

class InstanceSelectionAgent:
    def create_agent(self):
        instance_tool = InstanceSelectionTool()
        data_read = DirectoryReadTool(directory='raw_data')

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
            allow_delegation=False,
            verbose=True
        )