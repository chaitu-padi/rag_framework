# RAG Framework

A modular, configurable Retrieval-Augmented Generation (RAG) framework for Python. Supports pluggable data sources, vector databases, embedding models, and LLMs—all configurable via `config/config.yaml`.

# Modular RAG Framework Documentation

## Overview
This project is a modular Retrieval-Augmented Generation (RAG) framework built in Python. It allows users to flexibly select and configure data sources, embedding models, vector databases, and large language models (LLMs) for building and querying RAG pipelines. The framework is designed for extensibility, configurability, and ease of use, supporting both command-line and Streamlit web UI interfaces.

---

## Architecture

### Main Components

- **Data Sources (`datasources/`)**: Pluggable modules for loading data from CSV, RDBMS, or Hive.
- **Embeddings (`embeddings/`)**: Pluggable embedding model modules (e.g., OpenAI, more can be added).
- **Vector Databases (`vectordb/`)**: Pluggable vector DB modules (e.g., ChromaDB, more can be added).
- **LLMs (`llms/`)**: Pluggable LLM modules (e.g., OpenAI, more can be added).
- **RAG Pipeline (`rag/rag_pipeline.py`)**: Orchestrates the flow: loads data, generates embeddings, stores/retrieves vectors, and queries the LLM.
- **Configuration (`config/config.yaml`)**: Stores all user-selected options and parameters for reproducibility and automation.
- **Streamlit UI (`app.py`)**: Provides a user-friendly web interface for configuring, building, and querying the RAG pipeline.
- **Main Script (`main.py`)**: CLI entry point for running the pipeline using the configuration file.

---

## Code Flow

### 1. Configuration
- All options (data source, embedding model, vector DB, LLM, and their parameters) are stored in `config/config.yaml`.
- The Streamlit UI (`app.py`) allows users to select and configure these options interactively. When the user clicks "Build Vector Store", the config file is updated.

### 2. Building the Vector Store
- The pipeline loads documents from the selected data source.
- Embeddings are generated for each document using the selected embedding model.
- Embeddings and documents are stored in the selected vector database.

### 3. Querying
- The user enters a question.
- The question is embedded using the selected embedding model.
- Relevant documents are retrieved from the vector database (all or top-k, depending on configuration).
- The context is constructed from the retrieved documents.
- The LLM is prompted with the context and the user question, and generates an answer.

---

## How to Use

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Launch the Streamlit UI
```
streamlit run app.py
```

### 3. Configure the Pipeline
- Use the sidebar to select:
  - Data source type and parameters (CSV path, RDBMS/Hive connection string)
  - Embedding model and parameters (e.g., OpenAI model and API key)
  - Vector DB and parameters (e.g., ChromaDB directory)
  - LLM and parameters (e.g., OpenAI model and API key)
- Click **Build Vector Store** to process and store your data.

### 4. Ask Questions
- Enter your question in the main input box.
- Click **Get Answer** to receive a response from the LLM, grounded in your data.

### 5. Configuration File
- All selections are saved to `config/config.yaml` for reproducibility and CLI use.

### 6. Command-Line Usage
- You can also run the pipeline from the command line using `main.py`, which reads from `config/config.yaml`:
```
python main.py
```

---

## Extending the Framework
- To add new data sources, embedding models, vector DBs, or LLMs, implement a new class in the appropriate module and register it in the UI and config logic.
- The framework is designed for easy extension and configuration.

---

## Troubleshooting
- Ensure all required API keys are set (either in the UI or as environment variables).
- If you encounter version or schema errors with ChromaDB or NumPy, follow the compatibility instructions in `requirements.txt`.
- For large datasets, consider context window management or chunking strategies (see `rag/rag_pipeline.py`).

---

## Directory Structure
```
├── app.py                # Streamlit UI
├── main.py               # CLI entry point
├── config/
│   └── config.yaml       # Configuration file
├── datasources/          # Data source modules
├── embeddings/           # Embedding model modules
├── vectordb/             # Vector DB modules
├── llms/                 # LLM modules
├── rag/
│   └── rag_pipeline.py   # RAG pipeline logic
├── requirements.txt      # Python dependencies
└── ...
```

---

## Contact
For questions or contributions, please refer to the project repository or contact the maintainer.
