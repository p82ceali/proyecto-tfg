# 📁 tools/question_tool.py
from crewai.tools.base_tool import BaseTool
from crewai import LLM
from pydantic import PrivateAttr
import os

class QuestionTool(BaseTool):
    name: str = "question_tool"
    description: str = "Generates conversational questions for missing task parameters."
    _llm: LLM = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._llm = LLM(
            model="gemini/gemini-2.0-flash-lite",
            api_key=os.getenv("GOOGLE_API_KEY"),
            custom_llm_provider="gemini"
        )

    def _run(self, task_name: str, missing_params: list, dataset_columns: list = None) -> str:
        columns_hint = f"The dataset columns are: {', '.join(dataset_columns)}." if dataset_columns else ""
        params_text = ", ".join(missing_params)
        prompt = f"""
        The user wants to execute the task '{task_name}', but the following parameters are missing: {params_text}.
        {columns_hint}

        Write a single, natural and friendly question to the user asking them for these parameters.
        If multiple parameters are missing, ask them together in one question.
        """
        question = self._llm.call(prompt)
        return question.strip() if question else f"Could you provide values for: {params_text}?"
