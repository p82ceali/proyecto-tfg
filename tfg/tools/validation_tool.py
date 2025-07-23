# 📁 tools/validation_tool.py
from crewai.tools.base_tool import BaseTool
from shared_context import SharedContext
from pydantic import PrivateAttr

class ValidationTool(BaseTool):
    name: str = "validation_tool"
    description: str = "Checks required parameters for a pipeline task."
    _shared_context: SharedContext = PrivateAttr()

    @property
    def shared_context(self):
        return self._shared_context

    def __init__(self, shared_context: SharedContext):
        super().__init__()
        self._shared_context = shared_context

    def _run(self, task_name: str) -> str:
        required_params = {
            "eda": ["file_path"],
            "data_cleaning": ["file_path","imputation_strategy"],
            "instance_selection": ["file_path", "target_variable", "sample_size"],
            "feature_selection": ["target_variable", "file_path", "k"],
            "model_training": ["target_variable", "file_path", "n_estimators", "max_depth"]
        }
        missing = [p for p in required_params.get(task_name, []) if not self._shared_context.get(p)]
        return f"MISSING_PARAMS: {','.join(missing)}" if missing else "ALL_OK"




