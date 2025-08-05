import os
import panel as pn
import threading
import pandas as pd
import csv
import plotly.express as px
from dotenv import load_dotenv
from shared_context import SharedContext
from data_crew import ejecutar_interaccion

# ───────────────────────────────────
# Initial Setup
# ───────────────────────────────────
pn.extension('tabulator', 'plotly', design="material")
load_dotenv()

chat_interface = pn.chat.ChatInterface()
file_input = pn.widgets.FileInput(accept=".csv")

uploaded_file_path = None
FINAL_DATA_PATH = "pipeline_data/dataset.csv"
MODEL_PATH = "pipeline_data/model.pkl"
REPORT_PATH = "pipeline_data/training_report.pdf"
MAX_EXECUTION_TIME = 90

shared_context = SharedContext()

def safe_send_to_chat(message, user="Assistant", respond=False):
    if pn.state.curdoc:
        pn.state.curdoc.add_next_tick_callback(lambda: chat_interface.send(message, user=user, respond=respond))
    else:
        chat_interface.send(message, user=user, respond=respond)

def detect_delimiter(file_path):
    with open(file_path, newline='') as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
    return dialect.delimiter

def handle_file_upload(event):
    global uploaded_file_path
    if file_input.filename:
        os.makedirs("pipeline_data", exist_ok=True)
        uploaded_file_path = "pipeline_data/dataset.csv"
        with open(uploaded_file_path, "wb") as f:
            f.write(file_input.value)
        safe_send_to_chat(f"✅ File {file_input.filename} uploaded successfully.", user="Assistant", respond=False)

        try:
            delimiter = detect_delimiter(uploaded_file_path)
            df = pd.read_csv(uploaded_file_path, sep=delimiter)
            table = pn.widgets.Tabulator(df, height=400, pagination='remote', page_size=20)
            chat_interface.send(pn.Column("### Dataset Preview", table), user="Assistant", respond=False)
        except Exception as e:
            safe_send_to_chat(f"❌ Failed to read file: {e}", user="Assistant", respond=False)

file_input.param.watch(handle_file_upload, "value")

def timeout_handler():
    safe_send_to_chat("⚠️ The process is taking longer than expected. Do you want to try another task?", user="Assistant", respond=False)

def generate_interactive_plot(df: pd.DataFrame, target: str = None):
    try:
        if target and target in df.columns:
            fig = px.histogram(df, x=target, title=f"Distribution of {target}")
        else:
            fig = px.scatter_matrix(df.select_dtypes(include='number'), title="Scatter Matrix of Numeric Features")
        return pn.pane.Plotly(fig, height=400)
    except Exception:
        return "Unable to generate plot."

def offer_downloads():
    downloads = []
    if os.path.exists(MODEL_PATH):
        downloads.append(pn.widgets.FileDownload(path=MODEL_PATH, filename="trained_model.pkl", label="Download Trained Model"))
    if os.path.exists(FINAL_DATA_PATH):
        downloads.append(pn.widgets.FileDownload(path=FINAL_DATA_PATH, filename="processed_dataset.csv", label="Download Processed Dataset"))
    if os.path.exists(REPORT_PATH):
        downloads.append(pn.widgets.FileDownload(path=REPORT_PATH, filename="training_report.pdf", label="Download Training Report"))
    if downloads:
        chat_interface.send(pn.Column("### Downloads", *downloads), user="Assistant", respond=False)

def initiate_chat(message):
    try:
        if not uploaded_file_path:
            safe_send_to_chat("❌ Please upload a CSV file first.", user="Assistant", respond=False)
            return

        timer = threading.Timer(MAX_EXECUTION_TIME, timeout_handler)
        timer.start()

        resultado = ejecutar_interaccion(message)

        safe_send_to_chat(resultado, user="Assistant", respond=False)

        if os.path.exists(FINAL_DATA_PATH):
            df = pd.read_csv(FINAL_DATA_PATH)
            plot = generate_interactive_plot(df)
            chat_interface.send(pn.Column("### Data Insights", plot), user="Assistant", respond=False)

        offer_downloads()

        timer.cancel()
    except Exception as e:
        safe_send_to_chat(f"❌ General error: {e}", user="Assistant", respond=False)

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    thread = threading.Thread(target=initiate_chat, args=(contents,))
    thread.start()

chat_interface.callback = callback

# Main Layout
layout = pn.Column(
    chat_interface,
    file_input
)

layout.servable()
