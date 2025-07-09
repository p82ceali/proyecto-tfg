# 📁 shared_context.py

class SharedContext:
    def __init__(self):
        self.target_variable = None
        self.dataset_columns = []
        self.current_file_path = None
        self.history = []  # Para trazabilidad del pipeline

    def set_target_variable(self, target):
        self.target_variable = target

    def get_target_variable(self):
        return self.target_variable

    def set_columns(self, columns):
        self.dataset_columns = columns

    def get_columns(self):
        return self.dataset_columns

    def is_valid_target(self):
        return self.target_variable in self.dataset_columns if self.target_variable else False

    def set_current_file(self, file_path):
        self.current_file_path = file_path

    def get_current_file(self):
        return self.current_file_path

    def update_history(self, step_name, notes=None):
        self.history.append({
            "step": step_name,
            "file": self.current_file_path,
            "target": self.target_variable,
            "notes": notes,
        })

    def get_history(self):
        return self.history

    def reset(self):
        self.target_variable = None
        self.dataset_columns = []
        self.current_file_path = None
        self.history.clear()

    def ensure_valid_target(self):
       
        if self.target_variable in self.dataset_columns:
            return self.target_variable  # sigue siendo válido

        # Elegir otro target razonable si el actual ya no es válido
        from pandas.api.types import is_numeric_dtype
        import pandas as pd
        try:
            df = pd.read_csv(self.current_file_path)
            num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if not num_cols:
                return None
            new_target = num_cols[-1]
            self.set_target_variable(new_target)
            return new_target
        except Exception:
            return None
