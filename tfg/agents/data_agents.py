from agents.data_cleaning_agent import DataCleaningAgent
from agents.eda_agent import EDAAgent
from agents.feature_selection_agent import FeatureSelectionAgent
from agents.instance_selection_agent import InstanceSelectionAgent
from agents.model_training_agent import ModelTrainingAgent
from agents.coordinator_agent import CoordinatorAgent

class DataAgents:

    def coordinator_agent(self):
        return CoordinatorAgent().create_agent()
    
    def data_cleaning_agent(self):
        return DataCleaningAgent().create_agent()
    
    def eda_agent(self): 
        return EDAAgent().create_agent() 
    
    def feature_selection_agent(self):
        return FeatureSelectionAgent().create_agent()

    def instance_selection_agent(self):
        return InstanceSelectionAgent().create_agent()

    def model_training_agent(self):
        return ModelTrainingAgent().create_agent()
    
