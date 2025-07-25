import os
import panel as pn
import threading
import time
import pandas as pd
import csv
from dotenv import load_dotenv
from data_crew import MLDataCrew
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from shared_context import SharedContext

# ───────────────────────────────────
# Initial Setup
# ───────────────────────────────────
pn.extension('tabulator', design="material")
load_dotenv()

chat_interface = pn.chat.ChatInterface()
file_input = pn.widgets.FileInput(accept=".csv")

user_input = None
crew_started = False
uploaded_file_path = None
FINAL_DATA_PATH = "pipeline_data/dataset.csv"
MAX_EXECUTION_TIME = 90

shared_context = SharedContext()

def safe_send_to_chat(message, user="Assistant", respond=False):
    if pn.state.curdoc:
        pn.state.curdoc.add_next_tick_callback(lambda: chat_interface.send(message, user=user, respond=respond))
    else:
        chat_interface.send(message, user=user, respond=respond)

def custom_ask_human_input(self, final_answer: dict) -> str:
    global user_input
    safe_send_to_chat(final_answer, user="Assistant", respond=False)
    safe_send_to_chat("✋ Please provide your response:", user="System", respond=False)
    while user_input is None:
        time.sleep(0.5)
    respuesta = user_input
    user_input = None
    return respuesta

CrewAgentExecutorMixin._ask_human_input = custom_ask_human_input

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
            shared_context.set_columns(df.columns.tolist())

            # Show a preview table in the chat
            table = pn.widgets.Tabulator(df, height=400, pagination='remote', page_size=20)
            chat_interface.send(pn.Column("### Dataset Preview", table), user="Assistant", respond=False)

        except Exception as e:
            safe_send_to_chat(f"❌ Failed to read file: {e}", user="Assistant", respond=False)

file_input.param.watch(handle_file_upload, "value")

def timeout_handler():
    global crew_started
    if crew_started:
        safe_send_to_chat("⚠️ The process is taking longer than expected. Do you want to try another task?", user="Assistant", respond=False)
        crew_started = False

def initiate_chat(message):
    global crew_started
    crew_started = True
    try:
        if not uploaded_file_path:
            safe_send_to_chat("❌ Please upload a CSV file first.", user="Assistant", respond=False)
            crew_started = False
            return

        timer = threading.Timer(MAX_EXECUTION_TIME, timeout_handler)
        timer.start()

        file_to_use = FINAL_DATA_PATH if os.path.exists(FINAL_DATA_PATH) else uploaded_file_path

        crew = MLDataCrew(
            target_variable=shared_context.get_target_variable() or "",
            n_estimators=100,
            max_depth=10,
            chat_interface=chat_interface
        )

        # Full hierarchical kickoff: conversation + planning + tasks
        crew.kickoff(
            user_message=message,
            inputs={"file_path": file_to_use}
        )

        timer.cancel()
    except Exception as e:
        safe_send_to_chat(f"❌ General error: {e}", user="Assistant", respond=False)
    finally:
        crew_started = False

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global crew_started, user_input
    if not crew_started:
        thread = threading.Thread(target=initiate_chat, args=(contents,))
        thread.start()
    else:
        user_input = contents

chat_interface.callback = callback

# Main Layout
layout = pn.Column(
    chat_interface,
    file_input
)

layout.servable()
