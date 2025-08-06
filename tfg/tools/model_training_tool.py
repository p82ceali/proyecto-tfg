import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from crewai.tools import BaseTool

class ModelTrainingTool(BaseTool):
    name: str = "Advanced Model Training Tool"
    description: str = (
        "Trains an ML model (RandomForest, SVM, GradientBoosting, MLP) with optional hyperparameter tuning. "
        "Automatically detects classification/regression tasks. Evaluates the model with appropriate metrics. "
        "Saves the trained model and metrics to the 'pipeline_data/' folder."
    )

    def _detect_task_type(self, y):
        return 'classification' if y.nunique() < 20 and y.dtype in ['int', 'object'] else 'regression'

    def _select_model(self, model_name, task_type):
        models = {
            'random_forest': (RandomForestClassifier if task_type == 'classification' else RandomForestRegressor),
            'svm': (SVC if task_type == 'classification' else SVR),
            'gradient_boosting': (GradientBoostingClassifier if task_type == 'classification' else GradientBoostingRegressor),
            'mlp': (MLPClassifier if task_type == 'classification' else MLPRegressor)
        }
        return models.get(model_name, RandomForestClassifier if task_type == 'classification' else RandomForestRegressor)

    def _param_grid(self, model_name):
        grids = {
            'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
            'gradient_boosting': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
            'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'mlp': {'hidden_layer_sizes': [(50,), (100,)], 'learning_rate_init': [0.001, 0.01]}
        }
        return grids.get(model_name, {})

    def _run(self, file_path: str, target_variable: str, model_name: str = 'random_forest', tune: bool = False) -> str:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            df = pd.read_csv(file_path)
            if target_variable not in df.columns:
                raise ValueError(f"Target variable '{target_variable}' not found in dataset.")

            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            task_type = self._detect_task_type(y)
            ModelClass = self._select_model(model_name, task_type)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = None
            if tune:
                try:
                    param_grid = self._param_grid(model_name)
                    if param_grid:
                        grid = GridSearchCV(ModelClass(), param_grid, cv=3, n_jobs=-1)
                        grid.fit(X_train, y_train)
                        model = grid.best_estimator_
                    else:
                        model = ModelClass()
                        model.fit(X_train, y_train)
                except Exception:
                    model = ModelClass()
                    model.fit(X_train, y_train)
            else:
                model = ModelClass()
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if task_type == 'regression':
                rmse = mean_squared_error(y_test, y_pred) ** 0.5
                r2 = r2_score(y_test, y_pred)
                metrics = {"RMSE": round(rmse, 3), "R2 Score": round(r2, 3)}
            else:
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                metrics = {"Accuracy": round(acc, 3), "F1 Score": round(f1, 3)}

            os.makedirs("pipeline_data", exist_ok=True)
            model_path = "pipeline_data/model.pkl"
            pd.to_pickle(model, model_path)

            metrics_path = "pipeline_data/metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "model": model_name,
                    "task_type": task_type,
                    "target": target_variable,
                    "metrics": metrics
                }, f, indent=4)

            return (
                f"✅ Model ({model_name}) trained successfully!\n"
                f"Metrics: {metrics}\n"
                f"Model saved to {model_path}\n"
                f"Evaluation metrics saved to {metrics_path}"
            )

        except Exception as e:
            return f"❌ Error during model training: {e}"
