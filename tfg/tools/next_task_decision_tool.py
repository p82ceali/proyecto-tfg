from crewai.tools import BaseTool
from crewai import LLM
import os
from tools.context_summary_tool import SummarizeContextTool

class NextTaskDecisionTool(BaseTool):
    name: str = "Next Task Decision Tool"
    description: str = "Decides the next ML task based on context and user input."

    def _run(self, user_request: str) -> str:
        context_summary = SummarizeContextTool()._run()
        prompt = f"""
        You are an ML Project Coordinator.
        Current context:
        {context_summary}

        User request: "{user_request}"

        Decide the next step: eda, data_cleaning, instance_selection, feature_selection, model_training.
        Respond ONLY with the task name.
        """
        llm = LLM(
            model="gemini/gemini-2.5-flash-lite",
            api_key=os.getenv("GOOGLE_API_KEY"),
            custom_llm_provider="gemini"
        )
        try:
            result = llm.call(prompt)
            return result.strip().lower()
        except Exception:
            return "eda"
