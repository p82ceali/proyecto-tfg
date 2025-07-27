# coordinator_agent.py
import os
from crewai import LLM, Agent

class CoordinatorAgent:
    def create_agent(self):
        return Agent(
            role="Coordinador General",
            goal="Determinar si una pregunta requiere ayuda de un experto y delegarla",
            backstory=(
                "Eres un coordinador inteligente. Recibes preguntas y decides si puedes "
                "responderlas directamente o si debes pedir ayuda a un experto en limpieza de datos, analisis de datos, seleccion de caracteristicas, seleccion de instancias o entrenamiento de modelos."
            ),
            allow_delegation=True,
            verbose=True,
            llm=LLM(model="gemini/gemini-2.0-flash-lite",
                api_key=os.getenv("GOOGLE_API_KEY"),
                custom_llm_provider="gemini"
                ) # OpenAI(temperature=0.3)
    )
