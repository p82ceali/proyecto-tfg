from crewai.tools import BaseTool  
import pandas as pd  

class EDATool(BaseTool):  
    name: str = "EDA Tool"  
    description: str = "Generates an exploratory data analysis (EDA) report including descriptive statistics, missing values, and data types."  

    def _run(self, file_path: str):  
        df = pd.read_csv(file_path)  

        # Basic analysis  
        analysis = {  
            "total_features": len(df.columns),  
            "numeric_features": len(df.select_dtypes(include=['number']).columns),  
            "categorical_features": len(df.select_dtypes(include=['object']).columns),  
            "missing_values": df.isnull().sum().sort_values(ascending=False).head(3).to_dict(),  
            "top_variability": df.std(numeric_only=True).sort_values(ascending=False).head(3).to_dict()  
        }  
    
    