class SharedContext:
    def __init__(self):
        self.data = {
            "historial": [],  # lista de {pregunta, respuesta}
            "ultimo_tema": None,
            "decisiones": []
        }

    def add_interaccion(self, pregunta, respuesta):
        self.data["historial"].append({"pregunta": pregunta, "respuesta": respuesta})

    def set(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def resumen_historial(self, n=20):
        # Devuelve las últimas N interacciones como texto
        historial = self.data["historial"][-n:]
        return "\n".join([f"Usuario: {h['pregunta']}\nRespuesta: {h['respuesta']}" for h in historial])
