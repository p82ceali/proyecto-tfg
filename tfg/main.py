# ==========================
# 📁 main.py
# ==========================
import os
import panel as pn
import threading
import time
from dotenv import load_dotenv
from data_crew import MLDataCrew
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin

pn.extension(design="material")
load_dotenv()

chat_interface = pn.chat.ChatInterface()
file_input = pn.widgets.FileInput(accept=".csv")

user_input = None
crew_started = False
uploaded_file_path = None
active_file_path = None  # Ruta actual usada por cada tarea
MAX_EXECUTION_TIME = 90

def safe_send_to_chat(message, user="Assistant", respond=False):
    if pn.state.curdoc:
        pn.state.curdoc.add_next_tick_callback(lambda: chat_interface.send(message, user=user, respond=respond))
    else:
        chat_interface.send(message, user=user, respond=respond)

def custom_ask_human_input(self, final_answer: dict) -> str:
    global user_input
    safe_send_to_chat(final_answer, user="Assistant", respond=False)
    safe_send_to_chat("✋ Por favor proporciona tu respuesta:", user="System", respond=False)
    while user_input is None:
        time.sleep(0.5)
    respuesta = user_input
    user_input = None
    return respuesta

CrewAgentExecutorMixin._ask_human_input = custom_ask_human_input

def handle_file_upload(event):
    global uploaded_file_path, active_file_path
    if file_input.filename:
        uploaded_file_path = f"raw_data/{file_input.filename}"
        active_file_path = None  # Se reinicia al subir un nuevo archivo
        with open(uploaded_file_path, "wb") as f:
            f.write(file_input.value)
        safe_send_to_chat(f"✅ Archivo {file_input.filename} subido correctamente.", user="Assistant", respond=False)
        safe_send_to_chat("Ahora, ¿qué tarea deseas realizar? Describe en lenguaje natural:", user="Assistant", respond=False)

file_input.param.watch(handle_file_upload, "value")

def timeout_handler():
    global crew_started
    if crew_started:
        safe_send_to_chat("⚠️ El proceso está tardando más de lo esperado. ¿Quieres intentar otra tarea?", user="Assistant", respond=False)
        crew_started = False

def initiate_chat(message):
    global crew_started, active_file_path
    crew_started = True
    try:
        if not uploaded_file_path:
            safe_send_to_chat("❌ Primero debes subir un archivo CSV para comenzar.", user="Assistant", respond=False)
            crew_started = False
            return

        timer = threading.Timer(MAX_EXECUTION_TIME, timeout_handler)
        timer.start()

        file_to_use = active_file_path or uploaded_file_path

        crew = MLDataCrew(target_variable="price", n_estimators=100, max_depth=10, chat_interface=chat_interface)

        task_decision = crew.decide_task_from_message(message)
        friendly_explanation = crew.explain_decision(task_decision)
        safe_send_to_chat(f"🤖 {friendly_explanation}", user="System", respond=False)

        crew.run_task(task_decision, inputs={"file_path": file_to_use})

        # Actualizar la ruta del archivo activo si se genera uno nuevo
        if task_decision == "data_cleaning":
            active_file_path = f"processed_data/cleaned_{os.path.basename(file_to_use)}"
        elif task_decision == "feature_selection":
            active_file_path = f"selected_features/selected_features_{os.path.basename(file_to_use)}"
        elif task_decision == "instance_selection":
            active_file_path = "sampled_data/reduced_dataset.csv"

        # Sugerencia del coordinador
        suggestion = crew.suggest_next_task_based_on_result(task_decision)
        if suggestion:
            safe_send_to_chat(f"💡 {suggestion}", user="System", respond=False)

        timer.cancel()

    except Exception as e:
        safe_send_to_chat(f"❌ Error general: {e}", user="Assistant", respond=False)
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
safe_send_to_chat("👋 ¡Bienvenido! Sube tu archivo CSV y dime qué deseas hacer.", user="Assistant", respond=False)
pn.Column(file_input, chat_interface).servable()
