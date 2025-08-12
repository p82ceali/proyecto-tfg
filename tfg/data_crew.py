# crew_manager.py
from crewai import Agent, Task, Crew, LLM
from crewai_tools import DirectoryReadTool
import os
from dotenv import load_dotenv
from agents.data_agents import DataAgents
from shared_context import SharedContext
from tools.data_cleaning_tool import DataCleaningTool
from tools.eda_tool import EDATool
from tools.feature_selection_tool import FeatureSelector
from collections import deque

# --- NUEVO: arriba con el resto de imports ---
import pandas as pd
from crewai import Crew
from agents_v2.coordinator_agent import build_coordinator_agent, build_coordinator_task

load_dotenv()

memoria = SharedContext()
agents = DataAgents()


# ===== Model base =====
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini"
)



# ===== Task dinámico =====
def crear_tarea(pregunta_usuario):
    contexto = memoria.resumen_historial()
    ultimo_tema = memoria.get("ultimo_tema")
    datos_pendientes = memoria.get("datos_pendientes", {})

    return Task(
        description=f"""
        Historial reciente de la conversación:
        {contexto if contexto else "No hay interacciones previas."}

        Nueva pregunta: "{pregunta_usuario}"

        Estado previo:
        - Último tema tratado: {ultimo_tema if ultimo_tema else "Ninguno"}
        - Datos pendientes: {datos_pendientes if datos_pendientes else "Ninguno"}

        Instrucciones:
        - Usa el historial para entender el contexto.
        - Si la pregunta pertenece a limpieza de datos: 
            - Si NO tienes la estrategia de imputación (mean, median, mode, ffill, bfill) o la estrategia de escalado (normalization, standardization) PREGÚNTALOS al usuario y espera a que responda.
            - Si ya tienes la estrategia de imputacion (mean, median, mode, ffill, bfill) y la estrategia de escalado(normalization, standardization), delega al agente de limpieza de datos pasandole la entrada EXACTA que necesita para la estrategia de imputacion y de escalado y ejecuta sus tools.
        - Si la pregunta pertenece a análisis de datos:
            - Si NO tienes la variable objetivo, PREGÚNTALA al usuario y espera a que responda.
            - Si ya la tienes, delega al agente de análisis de datos y ejecuta sus tools.
        - Si la pregunta pertenece a selección de características:
            - Si NO tienes la variable objetivo o el número de características, PREGÚNTALOS y espera a que el usuario responda.
            - Si ya los tienes, delega al agente correspondiente y ejecuta sus tools.
        - Si la pregunta pertenece a selección de instancias:
            - Si NO tienes la variable objetivo o el número de instancias, PREGÚNTALOS y espera a que el usuario responda.
            - Si ya los tienes, delega al agente correspondiente y ejecuta sus tools.
        - Si la pregunta pertenece a entrenamiento de modelos:
            - Si NO tienes la variable objetivo, el modelo a entrenar (random_forest, svm, gradient_boosting, mlp) o si desea tuning, PREGÚNTALOS y espera a que el usuario responda.
            - Si ya los tienes, delega al agente correspondiente y ejecuta sus tools.
        - Si no requiere delegación o es muy general, responde directamente.
        - Guarda el tema principal tratado en 'ultimo_tema'.
        - Guarda cualquier decisión relevante en 'decisiones'.
        - Da una sugerencia del siguiente paso.

        IMPORTANTE:
        Cuando uses la herramienta 'Delegate work to coworker', debes pasar SIEMPRE texto plano:
        {{
            "task": "Describe aqui la tarea de forma simple",
            "context": "Describe aqui el contexto necesario",
            "coworker": "ml data quality engineer / Senior Data Analyst / lead feature engineer / instance selection expert / model training expert"
        }}
        """,
        agent=agents.coordinator_agent(),
        expected_output="Una respuesta al usuario, ya sea para pedir la información faltante o para ejecutar la acción correspondiente."
    )

# ===== Función principal =====
def ejecutar_interaccion(pregunta):
    tarea = crear_tarea(pregunta)    
    crew = Crew(
        agents=[agents.coordinator_agent(), agents.data_cleaning_agent(), agents.eda_agent(), agents.feature_selection_agent(), agents.instance_selection_agent(), agents.model_training_agent()],
        tasks=[tarea],
        verbose=True
    )
    resultado = crew.kickoff()
    memoria.add_interaccion(pregunta, resultado)
    memoria.set("ultimo_tema", pregunta)  # Guardamos el último tema
    return str(resultado)

# --- NUEVO: función v2 que usa el coordinador de delegación explícita ---
def ejecutar_interaccion_v2(pregunta: str) -> str:
    # 1) Carga el dataset activo
    df = pd.read_csv("pipeline_data/dataset.csv")

    # 2) Crea coordinador v2 (adjunta dataset a tools de delegación)
    coordinator = build_coordinator_agent(df)
    task = build_coordinator_task(coordinator)

    # 3) Ejecuta crew con la petición del usuario
    crew = Crew(agents=[coordinator], tasks=[task], verbose=True)
    result = crew.kickoff(inputs={"user_request": pregunta})

    # 4) Devuelve respuesta del sub-agente (ya formateada)
    return str(result)