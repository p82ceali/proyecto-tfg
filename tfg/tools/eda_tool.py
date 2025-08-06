from crewai.tools import BaseTool
import pandas as pd
import os
import csv
import plotly.express as px
from typing import Optional

class EDATool(BaseTool):
    name: str = "EDA Tool"
    description: str = (
        "Performs exploratory data analysis on a dataset: shows shape, missing values, data types, "
        "basic statistics, top correlations with the target variable, and generates an interactive plot. Also detects outliers."
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

                # Gráfico interactivo con Plotly
                fig = px.histogram(df, x=target_variable, title=f"Distribución de '{target_variable}'")
                fig.update_layout(bargap=0.1)
                os.makedirs("pipeline_data", exist_ok=True)
                fig.write_html("pipeline_data/eda_plot.html")
                report.append("\n\n📈 Se ha generado un histograma interactivo del target en 'pipeline_data/eda_plot.html'")

            # Detección de outliers con IQR
            numeric_cols = df.select_dtypes(include='number').columns
            outlier_report = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                count = outliers.shape[0]
                if count > 0:
                    outlier_report.append(f"- '{col}': {count} valores atípicos detectados")

            if outlier_report:
                report.append("\n\n🚨 Detección de valores atípicos (outliers) con método IQR:")
                report.extend(outlier_report)
            else:
                report.append("\n\n✅ No se detectaron valores atípicos usando el método IQR.")

            return "\n".join(report)

        except Exception as e:
            return f"❌ Error during EDA: {str(e)}"
