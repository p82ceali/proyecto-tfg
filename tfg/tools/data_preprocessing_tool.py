from crewai.tools import BaseTool
import pandas as pd
from scipy.stats import describe

class DataPreprocessor(BaseTool):
    name: str = "Data Preprocessor"
    description: str = ("Atomized data preprocessing tool: decides dynamically which preprocessing steps "
                        "to apply based on dataset characteristics.")

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            df[column] = df[column].fillna(df[column].mean())
            print(f"Filled missing values in {column} with mean.")
        for column in df.select_dtypes(exclude=['float64', 'int64']).columns:
            df[column] = df[column].fillna(df[column].mode()[0])
            print(f"Filled missing values in {column} with mode.")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_count - len(df)} duplicate rows.")
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.get_dummies(df, drop_first=True)
        print("Converted categorical variables to dummy variables.")
        return df

    def analyze_data(self, df: pd.DataFrame):
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            stats = describe(df[column].dropna())
            print(f"\nAnalysis for {column}:")
            print(f"  Mean: {stats.mean:.4f}")
            print(f"  Variance: {stats.variance:.4f}")
            print(f"  Min: {stats.minmax[0]:.4f}, Max: {stats.minmax[1]:.4f}")
            print(f"  Skewness: {stats.skewness:.4f}, Kurtosis: {stats.kurtosis:.4f}")

    def _run(self, file_path: str) -> str:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Perform statistical analysis
        self.analyze_data(df)
        
        # Atomized preprocessing steps
        df = self.fill_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.encode_categorical(df)
        
        # Save the preprocessed dataset
        processed_path = "processed_data/Processed_data.csv"
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved at: {processed_path}")
        return processed_path
