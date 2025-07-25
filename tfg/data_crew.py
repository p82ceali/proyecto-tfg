# data_crew.py
from crewai import Crew, Process, Task, LLM
from agents.data_agents import DataAgents
from agents.coordinator_agent import CoordinatorAgent
from agents.conversational_agent import ConversationalAgent
from agents.assitant_agent import AssistantAgent
from crewai.tasks.task_output import TaskOutput
import panel as pn
import os

class MLDataCrew:
    def __init__(self, target_variable: str, n_estimators: int, max_depth: int, chat_interface):
        self.agents = DataAgents()
        self.manager = CoordinatorAgent().create_agent()
        self.assistant = AssistantAgent().create_agent()  # Nuevo agente auxiliar
        self.conversational_agent = ConversationalAgent().create_agent()
        self.target_variable = target_variable
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.chat_interface = chat_interface

        # Planning LLM explícito
        self.planning_llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            custom_llm_provider="gemini"
        )

    def print_output(self, output: TaskOutput):
        if pn.state.curdoc:
            pn.state.curdoc.add_next_tick_callback(
                lambda: self.chat_interface.send(output.raw, user=output.agent, respond=False)
            )
        else:
            self.chat_interface.send(output.raw, user=output.agent, respond=False)

    # Subordinate tasks
    def eda_task(self):
        return Task(
            description="Perform exploratory data analysis (EDA) on the dataset.",
            agent=self.agents.eda_agent(),
            expected_output="EDA report with statistical summaries and visualizations.",
            callback=self.print_output,
            final_answer_required=False,
        )

    def data_cleaning_task(self):
        return Task(
            description="Clean the dataset by handling missing values, removing duplicates, and encoding categorical variables.",
            agent=self.agents.data_cleaning_agent(),
            expected_output="Cleaned dataset ready for modeling.",
            callback=self.print_output,
            final_answer_required=False,
        )

    def instance_selection_task(self):
        return Task(
            description="Select representative instances to reduce dataset size while maintaining diversity and informativeness.",
            agent=self.agents.instance_selection_agent(),
            expected_output="Reduced dataset with representative instances.",
            callback=self.print_output,
            final_answer_required=False,
        )

    def feature_selection_task(self):
        return Task(
            description=f"Select the most informative features for predicting '{self.target_variable}'.",
            agent=self.agents.feature_selection_agent(),
            expected_output="Dataset with selected features.",
            callback=self.print_output,
            final_answer_required=False,
        )

    def model_training_task(self):
        return Task(
            description=f"Train a Random Forest model to predict '{self.target_variable}'. Use n_estimators={self.n_estimators} and max_depth={self.max_depth}.",
            agent=self.agents.model_training_agent(),
            expected_output="Trained model and evaluation metrics (RMSE, R2 score).",
            callback=self.print_output,
            final_answer_required=True,  # End of pipeline
        )

    # Conversational step (single-shot)
    def interpret_user_intent(self, user_message: str):
        task = Task(
            description=f"""
            Interpret the user's request: "{user_message}".
            Respond clearly explaining what steps will be taken on the dataset.
            """,
            agent=self.conversational_agent,
            expected_output="Friendly and clear explanation for the user.",
            callback=self.print_output,
            final_answer_required=True
        )
        crew = Crew(
            agents=[self.conversational_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        return crew.kickoff(inputs={})

    # Technical pipeline execution
    def execute_pipeline(self, inputs: dict):
        all_tasks = [
            self.eda_task(),
            self.data_cleaning_task(),
            self.instance_selection_task(),
            self.feature_selection_task(),
            self.model_training_task(),
        ]
        crew = Crew(
            agents=[self.assistant] + [task.agent for task in all_tasks],
            tasks=all_tasks,
            manager_agent=self.manager,
            process=Process.hierarchical,
            planning=True,
            planning_llm=self.planning_llm,
            max_iterations=5,
            max_rpm=20,
            verbose=True,
        )
        return crew.kickoff(inputs=inputs)

    # Full flow
    def kickoff(self, user_message: str, inputs: dict) -> str:
        self.interpret_user_intent(user_message)
        return self.execute_pipeline(inputs)
