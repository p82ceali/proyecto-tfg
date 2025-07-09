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
        "Saves the dataset with selected features to 'pipeline_data/dataset.csv'."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, file_path: str, target_variable: str, k: int = 5, output_dir: str = "pipeline_data") -> str:
        try:
            if not os.path.exists(file_path):
                return f"❌ File not found: {file_path}"

            df = pd.read_csv(file_path)

            if target_variable not in df.columns:
                return f"❌ Target variable '{target_variable}' not found in dataset."

            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            X = X.select_dtypes(include=['number'])
            if X.empty:
                return "❌ No numeric features found in the dataset for selection."

            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
            selector.fit(X, y)

            selected_features = X.columns[selector.get_support()]
            df_selected = df[selected_features.tolist() + [target_variable]]

            os.makedirs("pipeline_data", exist_ok=True)
            final_path = "pipeline_data/dataset.csv"
            df_selected.to_csv(final_path, index=False)

            return f"✅ Selected features saved successfully to: {final_path}"

        except NotFittedError as e:
            return f"❌ Error during feature selection: Model not fitted. Details: {str(e)}"
        except Exception as e:
            return f"❌ Error during feature selection: {str(e)}"
