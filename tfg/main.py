import os
import csv
import time
import threading
from typing import Optional

import pandas as pd
import panel as pn
from dotenv import load_dotenv

# ── Tus módulos existentes ─────────────────────────────────────────────────────
# Asegúrate de que existan en tu proyecto
from shared_context import SharedContext
from data_crew import ejecutar_interaccion
# ───────────────────────────────────────────────────────────────────────────────

# Panel setup (activar notificaciones para poder usarlas)
pn.extension("tabulator", design="material", notifications=True)
load_dotenv()

# ───────────────────────────────────
# Constantes y estado
# ───────────────────────────────────
FINAL_DATA_PATH = "pipeline_data/dataset.csv"
MAX_EXECUTION_TIME = 90  # segundos
MAX_FILE_SIZE_MB = 50
ACCEPTED_EXT = ".csv"  # si luego quieres parquet: ".csv,.parquet"

shared_context = SharedContext()
_lock = threading.Lock()            # evita ejecuciones concurrentes del pipeline
_running_flag = False               # estado de ejecución en curso
_uploaded_file_path: Optional[str] = None
_detected_delimiter: Optional[str] = None

# ───────────────────────────────────
# Utilidades UI (thread-safe)
# ───────────────────────────────────
def _notify(kind: str, msg: str, duration: int = 3000):
    """
    Envía una notificación de forma segura desde cualquier hilo.
    Si no hay área de notificaciones disponible, hace fallback al chat o status_text.
    """
    def _do():
        notif = getattr(pn.state, "notifications", None)
        if notif:
            try:
                if kind == "success":
                    notif.success(msg, duration=duration)
                elif kind == "error":
                    notif.error(msg, duration=duration)
                else:
                    notif.info(msg, duration=duration)
                return
            except Exception:
                pass
        # Fallbacks (por si no existe el área o estamos fuera de sesión)
        try:
            if "chat" in globals() and chat is not None:
                chat.send(f"**{kind.upper()}**: {msg}", user="Sistema")
                return
        except Exception:
            pass
        try:
            if "status_text" in globals() and status_text is not None:
                prefix = "✅" if kind == "success" else "❌" if kind == "error" else "ℹ️"
                status_text.object = f"{prefix} {msg}"
        except Exception:
            pass

    if pn.state.curdoc:
        pn.state.curdoc.add_next_tick_callback(_do)
    else:
        _do()

def toast_success(msg: str):
    _notify("success", msg, duration=3000)

def toast_error(msg: str):
    _notify("error", msg, duration=5000)

def toast_info(msg: str):
    _notify("info", msg, duration=2500)

def safe_send_to_chat(message, user="Assistant", respond=False):
    """Envía al chat de forma segura desde callbacks o hilos."""
    def _do():
        try:
            chat.send(message, user=user, respond=respond)
        except Exception:
            pass
    if pn.state.curdoc:
        pn.state.curdoc.add_next_tick_callback(_do)
    else:
        _do()

def detect_delimiter_from_bytes(raw: bytes) -> str:
    """Detecta delimitador leyendo una muestra de bytes como texto."""
    sample = raw[:4096].decode("utf-8", errors="ignore")
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","  # fallback

def read_csv_safely(path: str, sep: str) -> pd.DataFrame:
    """Lectura de CSV con tipos básicos y sin explosionar memoria."""
    return pd.read_csv(path, sep=sep, low_memory=False)

def summarize_schema(df: pd.DataFrame) -> str:
    lines = ["### Esquema detectado", "", "| Columna | Tipo | Nulos |", "|---|---|---|"]
    for c in df.columns[:100]:  # límite visual
        dtype = str(df[c].dtype)
        nulls = int(df[c].isna().sum())
        lines.append(f"| `{c}` | `{dtype}` | {nulls:,} |")
    if len(df.columns) > 100:
        lines.append(f"\n> Mostrando primeras 100 columnas de {len(df.columns)}.")
    return "\n".join(lines)

def set_running(is_running: bool, msg: str = ""):
    global _running_flag
    _running_flag = is_running
    def _do():
        status.value = is_running
        if msg:
            status_text.object = msg
    if pn.state.curdoc:
        pn.state.curdoc.add_next_tick_callback(_do)
    else:
        _do()

# ───────────────────────────────────
# Widgets y componentes
# ───────────────────────────────────
chat = pn.chat.ChatInterface(
    callback=None,                 # se asigna más abajo
    sizing_mode="stretch_both",
    show_avatar=True,
    show_timestamp=False,
    height=560,
    # No uses 'message_align' ni 'renderers' en versiones antiguas de Panel
)

file_input = pn.widgets.FileInput(accept=ACCEPTED_EXT, name="Archivo CSV", sizing_mode="stretch_width")
btn_clear = pn.widgets.Button(name="Limpiar chat", button_type="warning", icon="trash-2")
btn_reset = pn.widgets.Button(name="Reiniciar sesión", button_type="primary", icon="rotate-ccw")
btn_example = pn.widgets.Button(name="Ejemplo: 'Haz un EDA'", button_type="default", icon="sparkles")

status = pn.indicators.LoadingSpinner(value=False, width=36, height=36)
status_text = pn.pane.Markdown("⚠️ Aún no se ha iniciado el pipeline.", sizing_mode="stretch_width")
k_rows = pn.indicators.Number(name="Filas", value=0, format="{value:,}", title_size="12pt")
k_cols = pn.indicators.Number(name="Columnas", value=0, format="{value:,}", title_size="12pt")
k_delim_label = pn.pane.Markdown("", sizing_mode="stretch_width")

tbl = pn.widgets.Tabulator(pd.DataFrame(), height=300, pagination="remote", page_size=20, sizing_mode="stretch_both")
schema_md = pn.pane.Markdown("", sizing_mode="stretch_width")

# ───────────────────────────────────
# Carga de archivo
# ───────────────────────────────────
def on_file_upload(event):
    global _uploaded_file_path, _detected_delimiter
    if not file_input.value:
        return

    size_mb = len(file_input.value) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        toast_error(f"El archivo supera el máximo permitido ({MAX_FILE_SIZE_MB} MB).")
        return

    os.makedirs("pipeline_data", exist_ok=True)
    _uploaded_file_path = FINAL_DATA_PATH

    try:
        # Detecta delimitador y guarda a disco
        _detected_delimiter = detect_delimiter_from_bytes(file_input.value)
        with open(_uploaded_file_path, "wb") as f:
            f.write(file_input.value)

        # Lee y muestra preview
        df = read_csv_safely(_uploaded_file_path, sep=_detected_delimiter)
        tbl.value = df
        schema_md.object = summarize_schema(df)

        # KPIs
        k_rows.value = int(df.shape[0])
        k_cols.value = int(df.shape[1])
        k_delim_label.object = f"**Delimitador**: `{_detected_delimiter}`"

        safe_send_to_chat(
            pn.Column(
                pn.pane.Markdown("### ✅ Dataset cargado"),
                pn.pane.Markdown(f"- **Archivo:** `{file_input.filename}`"),
                pn.pane.Markdown(f"- **Filas:** {df.shape[0]:,} • **Columnas:** {df.shape[1]:,}"),
            ),
            user="Assistant",
            respond=False,
        )
        toast_success(f"Archivo {file_input.filename} cargado correctamente.")
        status_text.object = "📦 Dataset listo para trabajar."

        # Persistimos en SharedContext si lo usas en tu pipeline
        try:
            shared_context.dataset_path = _uploaded_file_path
        except Exception:
            pass
    except Exception as e:
        toast_error(f"Error al leer el archivo: {e}")
        safe_send_to_chat(f"❌ No se pudo leer el archivo: {e}", user="Assistant", respond=False)

file_input.param.watch(on_file_upload, "value")

# ───────────────────────────────────
# Ejecución del pipeline vía chat
# ───────────────────────────────────
def timeout_handler():
    safe_send_to_chat(
        "⏱️ El proceso está tardando más de lo esperado. ¿Quieres intentar otra tarea o ajustar los parámetros?",
        user="Assistant",
        respond=False,
    )

def run_interaction(message: str):
    """Ejecuta la interacción con bloqueo, timeout y feedback visual (thread-safe)."""
    global _uploaded_file_path

    if not _uploaded_file_path:
        safe_send_to_chat("❌ Primero sube un archivo CSV.", user="Assistant", respond=False)
        return

    if _running_flag:
        safe_send_to_chat("⏳ Ya hay una tarea en ejecución. Espera a que finalice.", user="Assistant", respond=False)
        return

    with _lock:
        set_running(True, "🔄 Ejecutando pipeline, por favor espera...")
        timer = threading.Timer(MAX_EXECUTION_TIME, timeout_handler)
        timer.start()
        t0 = time.time()

        try:
            # Llamada a tu función principal
            result = ejecutar_interaccion(message)
            safe_send_to_chat(result, user="Assistant", respond=False)

            elapsed = time.time() - t0
            set_running(False, f"✅ Pipeline completado en {elapsed:0.1f}s")
            toast_success("Pipeline completado")
        except Exception as e:
            set_running(False, "❌ Error durante la ejecución del pipeline")
            safe_send_to_chat(f"❌ Error general: {e}", user="Assistant", respond=False)
            toast_error(str(e))
        finally:
            try:
                timer.cancel()
            except Exception:
                pass

def on_chat_callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    # Ejecutamos en hilo para no bloquear el servidor
    thread = threading.Thread(target=run_interaction, args=(contents,))
    thread.daemon = True
    thread.start()

chat.callback = on_chat_callback

# ───────────────────────────────────
# Acciones rápidas
# ───────────────────────────────────
def clear_chat(_):
    chat.clear()
    safe_send_to_chat("🧹 Chat limpiado.", user="Assistant", respond=False)

def reset_session(_):
    global _uploaded_file_path, _detected_delimiter
    _uploaded_file_path = None
    _detected_delimiter = None
    file_input.value = None
    tbl.value = pd.DataFrame()
    schema_md.object = ""
    k_rows.value = 0
    k_cols.value = 0
    k_delim_label.object = ""
    chat.clear()
    status_text.object = "🔁 Sesión reiniciada. Sube un dataset para empezar."
    toast_info("Sesión reiniciada.")

def send_example(_):
    prompt = "Haz un EDA inicial del dataset y muéstrame las 10 columnas con más nulos."
    chat.send(prompt, user="User")
    on_chat_callback(prompt, "User", chat)

btn_clear.on_click(clear_chat)
btn_reset.on_click(reset_session)
btn_example.on_click(send_example)

# ───────────────────────────────────
# Layout moderno con FastListTemplate
# ───────────────────────────────────
sidebar = pn.Column(
    pn.pane.Markdown("### 📂 Datos"),
    pn.Card(pn.Column(file_input, sizing_mode="stretch_width"), title="Subir dataset", collapsed=False),
    pn.Spacer(height=10),
    pn.pane.Markdown("### ⚙️ Acciones"),
    pn.Row(btn_reset, btn_clear),
    pn.Row(btn_example),
    pn.Spacer(height=10),
    pn.pane.Markdown("### 📊 Estado"),
    pn.Row(k_rows, k_cols, sizing_mode="stretch_width"),
    k_delim_label,
    pn.Spacer(height=5),
    pn.Row(status, status_text, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

dataset_section = pn.Accordion(
    (
        "📈 Vista rápida del dataset",
        pn.Column(
            pn.pane.Markdown("#### Preview"),
            tbl,
            pn.layout.Divider(),
            schema_md,
            sizing_mode="stretch_both",
        ),
    ),
    active=[0],
    sizing_mode="stretch_both",
)

main = pn.Column(
    pn.Card(chat, title="💬 Conversación", sizing_mode="stretch_both"),
    dataset_section,
    sizing_mode="stretch_both",
)

template = pn.template.FastListTemplate(
    title="Asistente ML — Multiagente",
    header=[pn.pane.Markdown("Filtrado, EDA y entrenamiento guiado por conversación")],
    theme_toggle=True,
    theme="dark",
    sidebar_width=360,
    main=[main],
    sidebar=[sidebar],
    accent_base_color="#4F46E5",
)

# Mensaje de bienvenida
safe_send_to_chat(
    """# 🤖 Bienvenido/a
Este asistente te ayuda a **configurar, analizar y entrenar pipelines de ML** guiado por **lenguaje natural**.

1) **Sube tu dataset CSV** en la barra lateral.  
2) Pide lo que necesites en el chat: “haz un EDA”, “limpia nulos”, “entrena un modelo…”.

💡 Prueba: *"Haz un EDA inicial del dataset y sugiere la variable objetivo."*
""",
    user="Assistant",
    respond=False,
)

template.servable()
