import csv
from crewai.tools import BaseTool
import pandas as pd
import os

class DataCleaningTool(BaseTool):
    name: str = "Data Cleaning Tool"
    description: str = (
        "Cleans datasets by removing duplicates, handling missing values, "
        "and converting categorical variables into dummy variables. Saves cleaned data to 'pipeline_data/dataset.csv'."
    )

    def _detect_delimiter(self, file_path):
        with open(file_path, newline='', encoding='utf-8') as f:
            sample = f.read(2048)
            f.seek(0)
            return csv.Sniffer().sniff(sample).delimiter

    def _run(self, file_path: str, imputation_strategy: str = "ffill") -> str:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            delimiter = self._detect_delimiter(file_path)
            df = pd.read_csv(file_path, sep=delimiter)

            df.drop_duplicates(inplace=True)

            numeric_cols = df.select_dtypes(include=["number"]).columns
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

            categorical_columns = df.select_dtypes(include=["object"]).columns
            for col in categorical_columns:
                unique_values = df[col].dropna().unique()
                if set(unique_values).issubset({"yes", "no"}):
                    df[col] = df[col].map({"yes": 1, "no": 0})
                else:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)

            os.makedirs("pipeline_data", exist_ok=True)
            final_path = "pipeline_data/dataset.csv"
            df.to_csv(final_path, index=False, sep=delimiter)  # <-- fijar el mismo separador

            return f"✅ Cleaned dataset saved successfully at '{final_path}'"
        except Exception as e:
            return f"❌ Error during data cleaning: {e}"