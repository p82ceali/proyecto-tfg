from crewai.tools import BaseTool
import pandas as pd
import os
import csv
from typing import Optional

class EDATool(BaseTool):
    name: str = "EDA Tool"
    description: str = (
        "Performs exploratory data analysis on a dataset: shows shape, missing values, data types, "
        "basic statistics, and top correlations with the target variable if provided."
    )

    def detect_delimiter(self, file_path):
        with open(file_path, newline='') as f:
            sample = f.read(2048)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter

    def _run(self, file_path: str, target_variable: Optional[str] = None) -> str:
        try:
            if not os.path.exists(file_path):
                return f"❌ File not found: {file_path}"

            delimiter = self.detect_delimiter(file_path)
            df = pd.read_csv(file_path, sep=delimiter)

            report = ["📊 **Exploratory Data Analysis Report**\n"]
            report.append(f"- Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

            # Tipos de columnas
            dtypes_str = df.dtypes.astype(str)
            if len(dtypes_str) > 50:
                dtypes_str = dtypes_str.head(50)
                report.append("- Columns and types (first 50):\n")
            else:
                report.append("- Columns and types:\n")
            report.append(dtypes_str.to_string())

            # Valores nulos
            report.append("\n\n- Missing values per column:\n")
            report.append(df.isnull().sum().head(50).to_string())

            # Estadísticas básicas
            report.append("\n\n- Basic statistics (numeric only):\n")
            report.append(df.describe().iloc[:, :20].round(2).to_string())  # primeras 20 cols

            # Correlaciones
            if target_variable and target_variable in df.columns:
                corr = df.corr(numeric_only=True)[target_variable].drop(target_variable).sort_values(ascending=False)
                report.append(f"\n\n🔍 Top correlations with target '{target_variable}':\n")
                report.append(corr.head(10).to_string())

            return "\n".join(report)

        except Exception as e:
            return f"❌ Error during EDA: {str(e)}"
