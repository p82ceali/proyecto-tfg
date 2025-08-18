import os
import csv
import time
import threading
from typing import Optional

import pandas as pd
import panel as pn
from dotenv import load_dotenv

# ── Your existing modules ──────────────────────────────────────────────────────
# Make sure they exist in your project
from shared_context import SharedContext
from data_crew import ejecutar_interaccion
# ──────────────────────────────────────────────────────────────────────────────

# Panel setup (enable notifications)
pn.extension("tabulator", design="material", notifications=True)
load_dotenv()

# ───────────────────────────────────
# Constants and state
# ───────────────────────────────────
FINAL_DATA_PATH = "pipeline_data/dataset.csv"
MAX_EXECUTION_TIME = 90  # seconds
MAX_FILE_SIZE_MB = 50
ACCEPTED_EXT = ".csv"  # if later you want parquet: ".csv,.parquet"

shared_context = SharedContext()
_lock = threading.Lock()            # avoid concurrent pipeline runs
_running_flag = False               # execution state
_uploaded_file_path: Optional[str] = None
_detected_delimiter: Optional[str] = None

# ───────────────────────────────────
# UI Utilities (thread-safe)
# ───────────────────────────────────
def _notify(kind: str, msg: str, duration: int = 3000):
    """
    Sends a notification safely from any thread.
    If no notification area is available, falls back to chat or status_text.
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
        # Fallbacks (if area does not exist or we are outside a session)
        try:
            if "chat" in globals() and chat is not None:
                chat.send(f"**{kind.upper()}**: {msg}", user="System")
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
    """Safely sends to the chat from callbacks or threads."""
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
    """Detects delimiter by reading a sample of bytes as text."""
    sample = raw[:4096].decode("utf-8", errors="ignore")
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","  # fallback

def read_csv_safely(path: str, sep: str) -> pd.DataFrame:
    """CSV reader with basic types and memory safe."""
    return pd.read_csv(path, sep=sep, low_memory=False)


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
# Widgets and components
# ───────────────────────────────────
chat = pn.chat.ChatInterface(
    callback=None,                 # assigned later
    sizing_mode="stretch_both",
    show_avatar=True,
    show_timestamp=False,
    min_height=560,
)

file_input = pn.widgets.FileInput(accept=ACCEPTED_EXT, name="CSV File", sizing_mode="stretch_width")
btn_clear = pn.widgets.Button(name="Clear chat", button_type="warning", icon="trash-2")
btn_reset = pn.widgets.Button(name="Reset session", button_type="primary", icon="rotate-ccw")
btn_example = pn.widgets.Button(name="Example: 'Do an EDA'", button_type="default", icon="sparkles")

status = pn.indicators.LoadingSpinner(value=False, width=36, height=36)
status_text = pn.pane.Markdown("⚠️ The pipeline has not yet been started.", sizing_mode="stretch_width")
k_rows = pn.indicators.Number(name="Rows", value=0, format="{value:,}", title_size="12pt")
k_cols = pn.indicators.Number(name="Columns", value=0, format="{value:,}", title_size="12pt")
k_delim_label = pn.pane.Markdown("", sizing_mode="stretch_width")

tbl = pn.widgets.Tabulator(pd.DataFrame(), height=300, pagination="remote", page_size=20, sizing_mode="stretch_both")
schema_md = pn.pane.Markdown("", sizing_mode="stretch_width")

# ───────────────────────────────────
# File upload
# ───────────────────────────────────
def on_file_upload(event):
    global _uploaded_file_path, _detected_delimiter
    if not file_input.value:
        return

    size_mb = len(file_input.value) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        toast_error(f"The file exceeds the maximum allowed size ({MAX_FILE_SIZE_MB} MB).")
        return

    os.makedirs("pipeline_data", exist_ok=True)
    _uploaded_file_path = FINAL_DATA_PATH

    try:
        # Detect delimiter and save to disk
        _detected_delimiter = detect_delimiter_from_bytes(file_input.value)
        with open(_uploaded_file_path, "wb") as f:
            f.write(file_input.value)

        # Read and show preview
        df = read_csv_safely(_uploaded_file_path, sep=_detected_delimiter)
        tbl.value = df

        # KPIs
        k_rows.value = int(df.shape[0])
        k_cols.value = int(df.shape[1])
        k_delim_label.object = f"**Delimiter**: `{_detected_delimiter}`"

        safe_send_to_chat(
            pn.Column(
                pn.pane.Markdown("### ✅ Dataset loaded"),
                pn.pane.Markdown(f"- **File:** `{file_input.filename}`"),
                pn.pane.Markdown(f"- **Rows:** {df.shape[0]:,} • **Columns:** {df.shape[1]:,}"),
            ),
            user="Assistant",
            respond=False,
        )
        toast_success(f"File {file_input.filename} loaded successfully.")
        status_text.object = "📦 Dataset ready to work with."

        # Persist in SharedContext if used in your pipeline
        try:
            shared_context.dataset_path = _uploaded_file_path
        except Exception:
            pass
    except Exception as e:
        toast_error(f"Error reading file: {e}")
        safe_send_to_chat(f"❌ Could not read file: {e}", user="Assistant", respond=False)

file_input.param.watch(on_file_upload, "value")

# ───────────────────────────────────
# Pipeline execution via chat
# ───────────────────────────────────
def timeout_handler():
    safe_send_to_chat(
        "⏱️ The process is taking longer than expected. Do you want to try another task or adjust the parameters?",
        user="Assistant",
        respond=False,
    )

def run_interaction(message: str):
    """Runs the interaction with lock, timeout and visual feedback (thread-safe)."""
    global _uploaded_file_path

    if not _uploaded_file_path:
        safe_send_to_chat("❌ Please upload a CSV file first.", user="Assistant", respond=False)
        return

    if _running_flag:
        safe_send_to_chat("⏳ A task is already running. Please wait for it to finish.", user="Assistant", respond=False)
        return

    with _lock:
        set_running(True, "🔄 Running pipeline, please wait...")
        timer = threading.Timer(MAX_EXECUTION_TIME, timeout_handler)
        timer.start()
        t0 = time.time()

        try:
            # Call your main function
            result = ejecutar_interaccion(message)
            safe_send_to_chat(result, user="Assistant", respond=False)

            elapsed = time.time() - t0
            set_running(False, f"✅ Pipeline completed in {elapsed:0.1f}s")
            toast_success("Pipeline completed")
        except Exception as e:
            set_running(False, "❌ Error during pipeline execution")
            safe_send_to_chat(f"❌ General error: {e}", user="Assistant", respond=False)
            toast_error(str(e))
        finally:
            try:
                timer.cancel()
            except Exception:
                pass

def on_chat_callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    # Run in a thread to avoid blocking the server
    thread = threading.Thread(target=run_interaction, args=(contents,))
    thread.daemon = True
    thread.start()

chat.callback = on_chat_callback

# ───────────────────────────────────
# Quick actions
# ───────────────────────────────────
def clear_chat(_):
    chat.clear()
    safe_send_to_chat("🧹 Chat cleared.", user="Assistant", respond=False)

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
    status_text.object = "🔁 Session reset. Upload a dataset to start."
    toast_info("Session reset.")

def send_example(_):
    prompt = "Perform an initial EDA of the dataset and show me the 10 columns with the most missing values."
    chat.send(prompt, user="User")
    on_chat_callback(prompt, "User", chat)

btn_clear.on_click(clear_chat)
btn_reset.on_click(reset_session)
btn_example.on_click(send_example)

# ───────────────────────────────────
# Modern layout with FastListTemplate
# ───────────────────────────────────
sidebar = pn.Column(
    pn.pane.Markdown("### 📂 Data"),
    pn.Card(pn.Column(file_input, sizing_mode="stretch_width"), title="Upload dataset", collapsed=False),
    pn.Spacer(height=10),
    pn.pane.Markdown("### 📊 State"),
    pn.Row(k_rows, k_cols, sizing_mode="stretch_width"),
    k_delim_label,
    pn.Spacer(height=5),
    pn.Row(status, status_text, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

dataset_section = pn.Accordion(
    (
        "📈 Dataset view",
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
    pn.Card(chat, title="💬 Conversation", sizing_mode="stretch_both"),
    dataset_section,
    sizing_mode="stretch_both",
)

template = pn.template.FastListTemplate(
    title="Assistant ML — Multiagent",
    header=[pn.pane.Markdown("Filtering, EDA and training guided by conversation")],
    theme_toggle=True,
    theme="dark",
    sidebar_width=360,
    main=[main],
    sidebar=[sidebar],
    accent_base_color="#4F46E5",
)

# Welcome message
safe_send_to_chat(
    """# 🤖 Welcome to the Automated Machine Learning Assistant
    This system helps you configure, train, and evaluate supervised learning pipelines  
    automatically, guided by your natural language instructions.

📂 Upload your dataset in CSV format.  
🔍 Explore and analyze your data with the EDA agent.  
🧹 Clean and preprocess features intelligently.  
🎯 Select the most relevant features for your model.  
📊 Perform sampling or train/test splits.  
⚙️ Train and evaluate supervised models with clear performance metrics.

All interactions are powered by a **multi-agent system**, ensuring that each pipeline step is executed optimally.

✨ Start by uploading your dataset and simply asking, in natural language, what you want to do!
""",
    user="Assistant",
    respond=False,
)

template.servable()
