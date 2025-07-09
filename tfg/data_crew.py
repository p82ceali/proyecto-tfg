# ==========================
# 📁 data_crew.py
# ==========================
from crewai import Crew, Task, Process
from agents.data_agents import DataAgents
from crewai.tasks.task_output import TaskOutput
import panel as pn

class MLDataCrew:
    def __init__(self, target_variable: str, n_estimators: int, max_depth: int, chat_interface):
        self.agents = DataAgents()
        self.target_variable = target_variable
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.chat_interface = chat_interface
        self.coordinator = self.agents.coordinator_agent()

    def print_output(self, output: TaskOutput):
        if pn.state.curdoc:
            pn.state.curdoc.add_next_tick_callback(lambda: self.chat_interface.send(output.raw, user=output.agent, respond=False))
        else:
            self.chat_interface.send(output.raw, user=output.agent, respond=False)

    def eda_task(self):
        return Task(
            description="Perform exploratory data analysis (EDA) on the dataset.",
            agent=self.agents.eda_agent(),
            expected_output="An EDA report including statistical summaries and visualizations.",
            callback=self.print_output,
            final_answer_required=True,
        )

    def data_cleaning_task(self):
        return Task(
            description="Clean the dataset by handling missing values, removing duplicates, and converting categorical variables into dummy variables.",
            agent=self.agents.data_cleaning_agent(),
            expected_output="A cleaned dataset ready for feature selection and modeling.",
            callback=self.print_output,
            final_answer_required=True,
        )

    def instance_selection_task(self):
        return Task(
            description="Select representative instances from the dataset to reduce its size while maintaining its diversity and informativeness.",
            agent=self.agents.instance_selection_agent(),
            expected_output="A reduced dataset with representative instances.",
            callback=self.print_output,
            final_answer_required=True,
        )

    def feature_selection_task(self):
        return Task(
            description=f"Select the most informative features for predicting '{self.target_variable}' using mutual information regression.",
            agent=self.agents.feature_selection_agent(),
            expected_output="A dataset with only the selected features.",
            callback=self.print_output,
            final_answer_required=True,
        )

    def model_training_task(self):
        return Task(
            description=f"Train a Random Forest model to predict '{self.target_variable}'. Use n_estimators={self.n_estimators} and max_depth={self.max_depth}.",
            agent=self.agents.model_training_agent(),
            expected_output="A trained model and evaluation metrics (RMSE, R2 score).",
            callback=self.print_output,
            final_answer_required=True,
        )

    def interpret_user_request_task(self, message: str):
        return Task(
            description=f"""
                Given the following user request:
                \"{message}\"

                Decide the most appropriate task to execute based on the available tasks:
                - eda
                - data_cleaning
                - instance_selection
                - feature_selection
                - model_training

                Only respond with the task name (one of: eda, data_cleaning, instance_selection, feature_selection, model_training).
                Do NOT explain your choice.
                """,
            agent=self.coordinator,
            expected_output="A single word: eda, data_cleaning, instance_selection feature_selection, or model_training.",
            final_answer_required=True,
        )

    def decide_task_from_message(self, message: str) -> str:
        task = self.interpret_user_request_task(message)
        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()

        if hasattr(result, "raw"):
            task_name = result.raw.strip().lower()
        else:
            raise ValueError(f"Unexpected CrewAI result type: {type(result)}")

        return task_name



    def run_task(self, task_name: str, inputs: dict = None) -> str:
        task_map = {
            "eda": self.eda_task(),
            "data_cleaning": self.data_cleaning_task(),
            "instance_selection": self.instance_selection_task(),
            "feature_selection": self.feature_selection_task(),
            "model_training": self.model_training_task(),
        }
        if task_name not in task_map:
            raise ValueError(f"Unknown task: {task_name}")

        task = task_map[task_name]

        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        return crew.kickoff(inputs=inputs)


    def kickoff(self, inputs: dict) -> str:
        all_tasks = [
            self.eda_task(),
            self.data_cleaning_task(),
            self.feature_selection_task(),
            self.model_training_task()
        ]

        crew = Crew(
            agents=[task.agent for task in all_tasks],
            tasks=all_tasks,
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff(inputs=inputs)
    
    def explain_decision(self, task_decision: str) -> str:
        prompt = (
            f"You are the Machine Learning Crew Coordinator.\n"
            f"You have analyzed the user's request and decided that the best next task is: '{task_decision}'.\n"
            f"Please explain this decision to the user in a friendly, collaborative, and professional tone, "
            f"as if you were two teammates working together. Keep it clear and motivating.\n\n"
            f"Respond directly as if you were talking to the user."
        )
        
        try:
            response = self.coordinator.llm.call(prompt)
            return response if response else "He decidido el siguiente paso basándome en tu solicitud."
        except Exception as e:
            return f"He decidido que la siguiente tarea es '{task_decision}' (nota: no pude generar explicación automática por un error)."

    def suggest_next_task_based_on_result(self, last_task: str) -> str:
   
        prompt = (
            f"You are the Machine Learning Crew Coordinator.\n"
            f"You just successfully completed the task '{last_task}'.\n"
            f"Based on a typical machine learning project workflow and the current state of the project, "
            f"suggest to the user what the next logical task should be.\n"
            f"Explain your suggestion briefly and in a motivating, collaborative tone.\n"
            f"Invite the user to proceed if they agree.\n\n"
            f"Respond directly to the user."
        )

        try:
            response = self.coordinator.llm.call(prompt)
            return response if response else "✅ Hemos completado esta tarea. ¿Te gustaría que proponga el siguiente paso?"
        except Exception as e:
            return "✅ Hemos completado esta tarea. ¿Te gustaría que proponga el siguiente paso?"
