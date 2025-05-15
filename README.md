# Asistente de Machine Learning Interactivo (TFG)

Este proyecto implementa un asistente inteligente para tareas de machine learning, que guía al usuario paso a paso a través del pipeline típico: desde el análisis exploratorio hasta el entrenamiento de modelos, todo mediante interacción en lenguaje natural.

## ¿Qué puede hacer?

- Recibe datasets `.csv` subidos por el usuario.
- Interpreta instrucciones en lenguaje natural (ej. "Quiero limpiar los datos").
- Ejecuta tareas como:
  - Limpieza de datos
  - Selección de variables
  - Reducción de instancias
  - Entrenamiento de modelos (Random Forest)
- Usa agentes inteligentes (`CrewAI`) y herramientas personalizadas para cada tarea.
- Responde de forma explicativa y propone siguientes pasos sin imponer un flujo fijo.

---

## Requisitos

- Python 3.10+
- Cuenta y API Key de [Gemini](https://makersuite.google.com/)
- Git (para clonar el proyecto)

---

## Instalación paso a paso

### 1. Clonar el repositorio

git clone https://github.com/p82ceali/proyecto-tfg.git
cd proyecto-tfg

### 2. Activar el entorno virtual

python -m venv venv

#### Activar el entorno:

##### En Windows (cmd):

venv\Scripts\activate

##### En PowerShell:

.\venv\Scripts\Activate.ps1

##### En macOS-/Linux:

source venv/bin/activate

### 3. Instalar dependencias

pip install -r requirements.txt

### 4. Configurar las variables de entorno

#### Crea un archivo .env en la raíz con este contenido:

GOOGLE_API_KEY=tu_api_key

### 5. Ejecutar la aplicacion

#### Desde el entorno virtual activo, lanza el sistema con:

python -m panel serve main.py --autoreload --show

##### Esto abrirá una interfaz web donde podrás:

-Subir un dataset .csv
-Escribir lo que quieres hacer (EDA, limpieza, etc.)


