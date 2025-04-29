# ==========================
# 📁 model_training_tool.py
# ==========================
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from crewai.tools import BaseTool

class ModelTrainingTool(BaseTool):
    name: str = "Model Training Tool"
    description: str = (
        "Trains a Random Forest regression model using the specified file. "
        "Evaluates the model and reports metrics like RMSE and R2 score."
    )

    def _run(self, file_path: str, target_variable: str, n_estimators: int, max_depth: int) -> str:
        try:
            # Cargar el dataset
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path)

            if target_variable not in df.columns:
                raise ValueError(f"Target variable '{target_variable}' not found in dataset.")

            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            # Separar en entrenamiento y test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Crear y entrenar el modelo
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Predicciones
            y_pred = model.predict(X_test)

            # Evaluación
            mse = mean_squared_error(y_test, y_pred)  # 🚨 NO usamos 'squared=False'
            rmse = mse ** 0.5  # Calculamos raíz cuadrada manualmente
            r2 = r2_score(y_test, y_pred)

            # Guardar modelo si quieres (opcional)
            os.makedirs("trained_models", exist_ok=True)
            model_path = os.path.join("trained_models", "random_forest_model.pkl")
            pd.to_pickle(model, model_path)

            return (
                f"✅ Model trained and saved successfully!\n\n"
                f"📈 Evaluation Metrics:\n"
                f"- RMSE: {rmse:.2f}\n"
                f"- R2 Score: {r2:.2f}\n"
                f"📦 Model saved to: {model_path}"
            )

        except Exception as e:
            return f"❌ Error during model training: {e}"
