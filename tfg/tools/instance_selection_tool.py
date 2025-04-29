from crewai.tools import BaseTool
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os

class InstanceSelectionTool(BaseTool):
    name: str = "Instance Selection Tool"
    description: str = (
        "Performs instance selection using stratified sampling or random sampling to reduce dataset size "
        "while maintaining data representativeness. Saves the reduced dataset to a specified directory."
    )

    def _run(self, file_path: str, target_variable: str, sample_size: float = 0.5, output_dir: str = "reduced_data") -> str:
        try:
            # Verificar si el archivo existe
            if not os.path.exists(file_path):
                return f"❌ File not found: {file_path}"

            # Cargar el dataset preprocesado
            df = pd.read_csv(file_path)

            # Verificar si la variable objetivo está en el dataset
            if target_variable not in df.columns:
                return f"❌ Target variable '{target_variable}' not found in dataset."

            # Separar características y objetivo
            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            # Verificar si alguna clase tiene menos de 2 muestras
            class_distribution = Counter(y)
            if any(count < 2 for count in class_distribution.values()):
                print("⚠️ Warning: At least one class has fewer than 2 samples. Stratification will not be used.")
                # Realizar muestreo aleatorio sin estratificación
                X_res, _, y_res, _ = train_test_split(X, y, train_size=sample_size, random_state=42)
            else:
                # Realizar muestreo estratificado
                X_res, _, y_res, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)

            # Crear el dataset reducido
            reduced_df = pd.concat([X_res, y_res], axis=1)

            # Crear directorio de salida si no existe
            os.makedirs(output_dir, exist_ok=True)

            # Guardar el dataset reducido
            reduced_dataset_path = os.path.join(output_dir, "reduced_dataset.csv")
            reduced_df.to_csv(reduced_dataset_path, index=False)

            return f"✅ Reduced dataset saved successfully to: {reduced_dataset_path}"
        except Exception as e:
            return f"❌ Error during instance selection: {str(e)}"