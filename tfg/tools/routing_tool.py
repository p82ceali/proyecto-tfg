# tools/routing_tool.py
from crewai.tools import BaseTool
from crewai import LLM
import os
from pydantic import ConfigDict

class RoutingTool(BaseTool):
    name: str = "Routing Tool"
    description: str = "Determines which ML task to run based on the user's input"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, user_query: str):
        llm = LLM(
            model="gemini/gemini-2.0-flash-exp",
            api_key=os.getenv("GOOGLE_API_KEY"),
            custom_llm_provider="gemini"
        )

        prompt = f"""
        A user said:
        \"{user_query}\"

        Based on the following options, return ONLY the one that best matches:
        - data_cleaning
        - eda
        - feature_selection
        - instance_selection
        - model_training

        Respond with only one word from the list.
        """

        response = llm.call(prompt)
        return response.strip().lower()
