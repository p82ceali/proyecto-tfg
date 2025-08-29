# TFG-ML: Assistant for configuring Supervised Learning Pipelines using LLMs

This project is an automated prediction system based on intelligent agents, using the [CrewAI](https://docs.crewai.com) library. It allows you to upload a dataset, perform exploratory analysis, cleaning, feature selection, sampling, and model training in a modular and interactive way.

## Requirements

- Python 3.10 or higher
- pip

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/tfg-ml.git
    cd tfg-ml
    ```

2.  **Create and activate a virtual environment:**

    -   **On Windows:**
        ```bash
        python -m venv venv
        venv\\Scripts\\activate
        ```

    -   **On Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the dependencies:**
    ```bash
    pip install ".[ui,agents,xgb]"
    ```
    This will install all the required dependencies, including the optional ones for the UI, agents, and XGBoost.

4.  **Configure the environment:**

    Create a `.env` file in the root of the project and add your Google API key:
    ```env
    GOOGLE_API_KEY="your_api_key"
    ```

## Usage

### Interactive Interface (Panel)

To launch the interactive dashboard, run the following command:
```bash
panel serve src/tfg_ml/interface/ui_panel.py --autoreload --show
```
This will open a web interface where you can upload your `.csv` dataset and interact with the intelligent agents using natural language.

## Project Structure

-   `src/tfg_ml/`: Main source code for the project.
    -   `adapters/`: Tools and utilities used by the agents.
    -   `agents/`: Definitions of the CrewAI agents for each task.
    -   `interface/`: User interface components (Panel UI).
    -   `pipelines/`: Core logic for the machine learning workflows.
    -   `context.py`: Shared context between agents.
-   `data/`: Directory for datasets.
-   `.env.example`: Example environment file.
-   `pyproject.toml`: Project metadata and dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

-   **Isaac Cejudo Alfaro**
