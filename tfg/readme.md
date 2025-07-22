# Proyecto TFG - Sistema Inteligente de Análisis y Modelado de Datos con CrewAI

Este proyecto es un sistema automatizado de predicción basado en agentes inteligentes, utilizando la librería [CrewAI](https://docs.crewai.com). Permite cargar un dataset, realizar análisis exploratorio, limpieza, selección de características, muestreo y entrenamiento de modelos de forma modular e interactiva.

## Requisitos

- Python 3.10 o superior
- pip

### Crear y activar entorno virtual

#### En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### En Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Configurar entorno

Crear un archivo `.env` con las siguientes variables:
```env
GEMINI_API_KEY="tu_api_key"
```

## Uso del sistema

### Interfaz interactiva (Panel)

Lanzar el dashboard interactivo:
```bash
panel serve main_panel.py --autoreload --show
```

Esto abrirá una interfaz web donde podrás subir tu dataset `.csv` y comunicarte en lenguaje natural con los agentes inteligentes.

### Ejecución clásica

Si prefieres ejecutar el sistema paso a paso desde consola:
```bash
python main.py
```

## Estructura del Proyecto

- `main.py`: ejecución secuencial del sistema
- `agents/`: definición de los agentes CrewAI por tarea
- `tools/`: herramientas funcionales usadas por los agentes
- `shared_context.py`: contexto compartido entre agentes
- `pipeline_data/`: dataset original

## Autor
Isaac Cejudo Alfaro

## Licencia
MIT

