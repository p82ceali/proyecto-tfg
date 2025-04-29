from crewai.tools import BaseTool
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.exceptions import NotFittedError
from pydantic import ConfigDict
import os

class FeatureSelector(BaseTool):
    name: str = "Feature Selector"
    description: str = (
        "Selects the best features for a target variable using mutual information or other feature selection techniques. "
        "Saves the dataset with selected features to a specified directory."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, file_path: str, target_variable: str, k: int = 5, output_dir: str = "features_data") -> str:
        try:
            # Verificar si el archivo existe
            if not os.path.exists(file_path):
                return f"❌ File not found: {file_path}"

            # Leer el archivo
            df = pd.read_csv(file_path)

            # Verificar si la variable objetivo está en el dataset
            if target_variable not in df.columns:
                return f"❌ Target variable '{target_variable}' not found in dataset."

            # Separar características (X) y variable objetivo (y)
            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            # Filtrar solo columnas numéricas
            X = X.select_dtypes(include=['number'])
            if X.empty:
                return "❌ No numeric features found in the dataset for selection."

            # Seleccionar las mejores características
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
            selector.fit(X, y)

            # Obtener las características seleccionadas
            selected_features = X.columns[selector.get_support()]
            df_selected = df[selected_features.tolist() + [target_variable]]

            # Crear directorio de salida si no existe
            os.makedirs(output_dir, exist_ok=True)

            # Crear nombre del archivo con las características seleccionadas
            filename = os.path.basename(file_path)
            feature_filename = f"selected_features_{filename}"
            feature_path = os.path.join(output_dir, feature_filename)

            # Guardar el dataset con las características seleccionadas
            df_selected.to_csv(feature_path, index=False)

            return f"✅ Selected features saved successfully to: {feature_path}"
        except NotFittedError as e:
            return f"❌ Error during feature selection: Model not fitted. Details: {str(e)}"
        except Exception as e:
            return f"❌ Error during feature selection: {str(e)}"