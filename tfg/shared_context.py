# shared_context.py  (añadir/editar)

class SharedContext:
    def __init__(self):
        self.data = {
            "historial": [],      # lista de {pregunta, respuesta}
            "ultimo_tema": None,
            "decisiones": []      # lista de {stage, text, meta}
        }

    def add_interaccion(self, pregunta, respuesta):
        self.data["historial"].append({"pregunta": pregunta, "respuesta": respuesta})

    def add_decision(self, stage: str, text: str, meta: dict | None = None):
        self.data["decisiones"].append({
            "stage": stage,
            "text": text,
            "meta": (meta or {})
        })

    def set(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def resumen_historial(self, n=20):
        historial = self.data["historial"][-n:]
        return "\n".join([f"Usuario: {h['pregunta']}\nRespuesta: {h['respuesta']}" for h in historial])

    # resumen compacto de decisiones
    def summary(self, n: int = 10) -> str:
        decs = self.data["decisiones"][-n:]
        return "\n".join([f"- [{d['stage']}] {d['text']}" for d in decs])

# ✅ Singleton compartido por todos los módulos
CTX = SharedContext()
