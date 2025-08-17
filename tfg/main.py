import os
import panel as pn
import threading
import time
import pandas as pd
import csv
from dotenv import load_dotenv
from shared_context import SharedContext
from data_crew import ejecutar_interaccion


# ───────────────────────────────────
# Initial Setup
# ───────────────────────────────────
pn.extension('tabulator', design="material")
load_dotenv()

chat_interface = pn.chat.ChatInterface()
file_input = pn.widgets.FileInput(accept=".csv")

user_input = None
uploaded_file_path = None
FINAL_DATA_PATH = "pipeline_data/dataset.csv"
MAX_EXECUTION_TIME = 90

shared_context = SharedContext()

status_pane = pn.pane.Markdown("⚠️ Aún no se ha iniciado el pipeline.")

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

def initiate_chat(message):
    try:
        if not uploaded_file_path:
            safe_send_to_chat("❌ Please upload a CSV file first.", user="Assistant", respond=False)
            return

        status_pane.object = "🔄 Ejecutando pipeline, por favor espera..."
        timer = threading.Timer(MAX_EXECUTION_TIME, timeout_handler)
        timer.start()

        resultado = ejecutar_interaccion(message) 


        safe_send_to_chat(resultado, user="Assistant", respond=False)
        status_pane.object = "✅ Pipeline completado"
        timer.cancel()
    except Exception as e:
        status_pane.object = "❌ Error durante la ejecución del pipeline"
        safe_send_to_chat(f"❌ General error: {e}", user="Assistant", respond=False)

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    thread = threading.Thread(target=initiate_chat, args=(contents,))
    thread.start()

chat_interface.callback = callback

# Main Layout
layout = pn.Column(
    pn.pane.Markdown("## <span style='color:#4A4A4A'>🤖 Asistente de Aprendizaje Automático</span>", sizing_mode='stretch_width'),
    pn.Row(
        pn.Card(status_pane, title="⏳ Estado del Pipeline", width=400),
        pn.Card(pn.Column("### 📂 Subir Dataset", file_input), width=400),
    ),
    pn.Card(chat_interface, title="💬 Conversación", height=500),
    sizing_mode="stretch_width",
    width=900
)

# Mensaje de bienvenida al iniciar la aplicación
safe_send_to_chat("""# 🤖 Welcome to the Automated Machine Learning Assistant

This system helps you **configure, train, and evaluate supervised learning pipelines**  
automatically, guided by your **natural language instructions**.

📂 Upload your dataset in CSV format.  
🔍 Explore and analyze your data with the EDA agent.  
🧹 Clean and preprocess features intelligently.  
🎯 Select the most relevant features for your model.  
📊 Perform sampling or train/test splits.  
⚙️ Train and evaluate supervised models with clear performance metrics.  

All interactions are powered by a **multi-agent system**, ensuring that each pipeline step is executed optimally.

✨ Start by uploading your dataset and simply asking, in natural language, what you want to do!

""", user="Assistant", respond=False)

layout.servable()
