import os
import re
import sqlite3
import warnings
import dotenv
import streamlit as st
import pandas as pd
import plotly.express as px

# Crew AI + Tools imports
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from langchain_openai import ChatOpenAI

# SQL Tools from langchain_community
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load Environment Variables
dotenv.load_dotenv()

# Regex to Extract Code Snippet in triple backticks
flexible_pattern = re.compile(
    r"```python\s*(.*?)(```|$)",  # match until '```' or end-of-string
    re.DOTALL | re.IGNORECASE
)

def extract_code_block(raw_text: str) -> str:
    match = flexible_pattern.search(raw_text)
    if match:
        code_part = match.group(1)
        code_part = code_part.replace("```", "").strip()
        return code_part
    return ""

# Data Visualization Agent
data_viz_agent = Agent(
    role="Data Visualization Agent",
    goal="Generate Python code using Plotly to visualize data based on user queries.",
    backstory="Expert data scientist using Plotly for visualization.",
    tools=[],
    llm=LLM(model="gemini/gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY")),
    verbose=True,
)

data_viz_task = Task(
    description="Generate a Plotly chart based on user input.",
    expected_output="A valid Plotly figure stored in a variable named 'fig'.",
    agent=data_viz_agent,
)

viz_crew = Crew(
    agents=[data_viz_agent],
    tasks=[data_viz_task],
)

# SQL / Analysis Agents and Tools
db_file = "temp_db.sqlite"

def init_database(df):
    # Si existe el archivo, lo eliminamos
    if os.path.isfile(db_file):
        try:
            conn = sqlite3.connect(db_file)
            conn.close()  # Cerrar la conexión antes de eliminar el archivo
            os.remove(db_file)
        except Exception as e:
            print(f"Error al eliminar el archivo de base de datos: {e}")
            
    # Luego, creamos la nueva base de datos
    conn = sqlite3.connect(db_file)
    table_name = "data_table"
    df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)
    conn.close()


database_uri = f"sqlite:///{db_file}"
db = SQLDatabase.from_uri(database_uri)

@tool("list_tables")
def list_tables_tool() -> str:
    """List the available tables in the database."""
    return ListSQLDatabaseTool(db=db).invoke("")


@tool("tables_schema")
def tables_schema_tool(tables: str) -> str:
    """Show schema & sample rows for the given tables (comma-separated)."""
    return InfoSQLDatabaseTool(db=db).invoke(tables)

@tool("execute_sql")
def execute_sql_tool(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result as a string."""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

@tool("check_sql")
def check_sql_tool(sql_query: str) -> str:
    """Check if the SQL query is correct. Returns suggestions/fixes or a success message."""
    try:
        llm_checker = LLM(model="gemini/gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))
        query_checker_tool = QuerySQLCheckerTool(db=db, llm=llm_checker)
        return query_checker_tool.invoke({"query": sql_query})
    except Exception as e:
        return f"Error using QuerySQLCheckerTool: {str(e)}"

sql_dev = Agent(
    role="SQL Developer",
    goal="Construct and execute SQL queries based on user requests",
    backstory="Expert database engineer creating optimized SQL queries.",
    llm=LLM(model="gemini/gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[list_tables_tool, tables_schema_tool, execute_sql_tool, check_sql_tool],
    verbose=True,
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal="Analyze SQL data and provide insights",
    backstory="Experienced analyst generating meaningful insights from data.",
    llm=LLM(model="gemini/gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY")),
    verbose=True,
)

report_writer = Agent(
    role="Report Writer",
    goal="Summarize analysis into concise executive reports",
    backstory="Expert at writing high-level summaries of data insights.",
    llm=LLM(model="gemini/gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY")),
    verbose=True,
)

extract_data = Task(
    description="Extract relevant data using SQL queries based on user requests.",
    expected_output="A SQL query result with relevant data extracted from the database.",
    agent=sql_dev,
)

analyze_data = Task(
    description="Analyze extracted data and provide insights.",
    expected_output="A summary of key insights and patterns found in the extracted data.",
    agent=data_analyst,
    context=[extract_data],
)

write_report = Task(
    description="Summarize data analysis results.",
    expected_output="A concise report summarizing the analysis findings.",
    agent=report_writer,
    context=[analyze_data],
)

main_crew = Crew(
    agents=[sql_dev, data_analyst, report_writer],
    tasks=[extract_data, analyze_data, write_report],
    process=Process.sequential,
    verbose=True,
)

# Streamlit UI
st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Upload a CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    init_database(df)
    st.success("Database created from uploaded CSV.")
    
    user_query = st.text_input("Ask a question about the data:")
    if st.button("Generate Report"):
        if user_query.strip():
            result = main_crew.kickoff(inputs={"query": user_query})
            for task_output in result.tasks_output:
                st.write(task_output.raw or "No output found.")
    
    viz_prompt = st.text_area("Enter visualization prompt:")
    if st.button("Generate Plot"):
        if viz_prompt.strip():
            data_viz_task.description = f"Generate a Plotly chart for: {viz_prompt}"
            viz_result = viz_crew.kickoff()
            generated_code = extract_code_block(viz_result.raw or "")
            if generated_code:
                env = {}
                exec(generated_code, env)
                if "fig" in env:
                    st.plotly_chart(env["fig"], use_container_width=True)