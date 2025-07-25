from crewai.tools import BaseTool

class SummarizeContextTool(BaseTool):
    name: str = "Context Summary Tool"
    description: str = "Summarizes the current ML pipeline state (target variable and steps history)."

    def _run(self) -> str:
        from shared_context import SharedContext
        shared_context = SharedContext()
        history = shared_context.get_history()
        target = shared_context.get_target_variable()
        if not history:
            return "No tasks have been executed yet."
        summary = f"Current target: {target or 'Not defined yet'}\nTask history:\n"
        for s in history:
            summary += f"- {s['step']} ({s['notes']})\n"
        return summary
