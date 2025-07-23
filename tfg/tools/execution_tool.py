# 📁 tools/execution_tool.py
from crewai.tools.base_tool import BaseTool
from crewai import Crew, Task, Process
from pydantic import PrivateAttr

class ExecutionTool(BaseTool):
    name: str = "execution_tool"
    description: str = "Executes the appropriate pipeline task using its corresponding agent when all parameters are available."
    _agents: object = PrivateAttr()
    _chat_interface: object = PrivateAttr()

    def __init__(self, agents, chat_interface):
        super().__init__()
        self._agents = agents
        self._chat_interface = chat_interface

    def _run(self, task_name: str, inputs: dict) -> str:
        task_map = {
            "eda": {
                "agent": self._agents.eda_agent(),
                "description": "Perform exploratory data analysis on the dataset.",
                "expected_output": "An EDA report including summaries and visualizations."
            },
            "data_cleaning": {
                "agent": self._agents.data_cleaning_agent(),
                "description": "Clean the dataset (handle missing values, duplicates, categorical encoding).",
                "expected_output": "A cleaned dataset ready for further processing."
            },
            "instance_selection": {
                "agent": self._agents.instance_selection_agent(),
                "description": "Select representative instances from the dataset.",
                "expected_output": "A reduced dataset with representative samples."
            },
            "feature_selection": {
                "agent": self._agents.feature_selection_agent(),
                "description": "Select the most informative features for the target variable.",
                "expected_output": "A dataset with only the selected features."
            },
            "model_training": {
                "agent": self._agents.model_training_agent(),
                "description": "Train a model to predict the target variable.",
                "expected_output": "A trained model and evaluation metrics."
            },
        }
        if task_name not in task_map:
            return f"Unknown task: {task_name}"

        task_data = task_map[task_name]
        task = Task(
            description=task_data["description"],
            agent=task_data["agent"],
            expected_output=task_data["expected_output"],
            callback=lambda output: self._chat_interface.send(output.raw, user=output.agent, respond=False),
            final_answer_required=True
        )
        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        crew.kickoff(inputs=inputs)
        return f"Task '{task_name}' executed successfully."
