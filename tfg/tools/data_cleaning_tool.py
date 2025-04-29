import os
from crewai.tools import BaseTool
import pandas as pd


class DataCleaningTool(BaseTool):
    name: str = "Data Cleaning Tool"
    description: str = (
        "Cleans datasets by removing duplicates, handling missing values, "
        "and converting categorical variables into dummy variables. Returns a cleaned dataset."
    )

    def _run(self, file_path: str, imputation_strategy: str = "ffill") -> str:
        try:
            # Leer el archivo
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path)

            # Limpieza: eliminar duplicados
            df.drop_duplicates(inplace=True)
            
            numeric_cols = df.select_dtypes(include=["number"]).columns

            # Solo imputar en columnas numéricas
            if imputation_strategy == "mean":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif imputation_strategy == "median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif imputation_strategy == "mode":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
            elif imputation_strategy == "ffill":
                df.ffill(inplace=True)
            elif imputation_strategy == "bfill":
                df.bfill(inplace=True)
            else:
                raise ValueError(f"Invalid imputation strategy: {imputation_strategy}")


            # Identificar columnas categóricas
            categorical_columns = df.select_dtypes(include=["object"]).columns

            # Validar y limpiar valores categóricos
            for col in categorical_columns:
                unique_values = df[col].dropna().unique()
                if set(unique_values).issubset({"yes", "no"}):
                    # Codificación binaria para columnas con valores 'yes' y 'no'
                    df[col] = df[col].map({"yes": 1, "no": 0})
                else:
                    # Aplicar one-hot encoding para otras columnas categóricas
                    df = pd.get_dummies(df, columns=[col], drop_first=True)

            # Crear directorio si no existe
            os.makedirs("processed_data", exist_ok=True)

            # Crear nombre del archivo limpio
            filename = os.path.basename(file_path)
            clean_filename = f"cleaned_{filename}"
            clean_path = os.path.join("processed_data", clean_filename)

            # Guardar dataset limpio
            df.to_csv(clean_path, index=False)

            return f"✅ Cleaned dataset saved successfully at '{clean_path}'"
        except Exception as e:
            return f"❌ Error during data cleaning: {e}"