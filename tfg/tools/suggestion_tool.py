# tools/suggestion_tool.py
from crewai.tools import BaseTool
from crewai import LLM
import os
from pydantic import ConfigDict

class SuggestionTool(BaseTool):
    name: str = "Suggestion Tool"
    description: str = "Suggests the next ML step based on conversation history"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, conversation_history: list[str]):
        llm = LLM(
            model="gemini/gemini-2.0-flash-exp",
            api_key=os.getenv("GOOGLE_API_KEY"),
            custom_llm_provider="gemini"
        )

        history = "\n".join(conversation_history)

        prompt = f"""
        Here's the conversation so far:
        {history}

        Suggest one helpful next step in the ML pipeline the user might want to perform.
        Respond in one sentence.
        """

        response = llm.call(prompt)
        return response.strip()
