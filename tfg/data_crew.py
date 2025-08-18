# crew_manager.py
from crewai import Agent, Task, Crew, LLM
from crewai_tools import DirectoryReadTool
import os
from dotenv import load_dotenv
from shared_context import SharedContext

from collections import deque

import pandas as pd
from crewai import Crew
from agents.coordinator_agent import build_coordinator_agent, build_coordinator_task

from shared_context import CTX 


memoria = SharedContext()



def ejecutar_interaccion(pregunta: str) -> str:
    # 1) Carga el dataset activo
    df = pd.read_csv("pipeline_data/dataset.csv")

    chat_context = CTX.resumen_historial(n=20)

    # 2) Crea coordinador v2 (adjunta dataset a tools de delegación)
    coordinator = build_coordinator_agent(df)
    task = build_coordinator_task(coordinator)

    # 3) Ejecuta crew con la petición del usuario
    crew = Crew(agents=[coordinator], tasks=[task], verbose=True)
    result = crew.kickoff(inputs={
        "user_request": pregunta,
        "chat_context": chat_context,     # <-- CONTEXTO AL COORDINADOR
    })
    
    respuesta = str(result)
    CTX.add_interaccion(pregunta, respuesta)

    # 4) Devuelve respuesta del sub-agente (ya formateada)
    return respuesta