# 📁 tools/memory_tool.py
from crewai.tools.base_tool import BaseTool
from shared_context import SharedContext
from pydantic import PrivateAttr

class MemoryTool(BaseTool):
    name: str = "memory_tool"
    description: str = "Stores user-provided parameters in the shared context for future tasks."
    shared_context: SharedContext

    def __init__(self, shared_context: SharedContext):
        super().__init__()
        self.shared_context = shared_context

    def _run(self, param_name: str, value: str) -> str:
        param_name = param_name.strip().lower()
        value = value.strip()
        self.shared_context.set(param_name, value)
        return f"Stored {param_name} = {value}"
