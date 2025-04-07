# Chat With Your Data: Natural Language Tabular Data Assistant

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This Streamlit web app lets you upload tabular data (Excel files or published Google Sheet CSV URLs) and query it in plain language. It uses LLMs for SQL generation, semantic search, and natural language summarization.

## Key Features

- **Natural Language Queries:** Ask questions about your data in everyday language.
- **Multiple Data Sources:** Supports Excel uploads and CSV URLs from published Google Sheets.
- **Automated Data Processing:**  
  - Converts data to an SQLite database.  
  - Uses an LLM to select text columns for embedding.  
  - Embeds text with FastEmbed and indexes them in an in-memory Qdrant vector DB.
- **Intelligent Query Routing:**  
  - **SQL Mode:** Generates and executes SQL queries against SQLite.
  - **Semantic Mode:** Performs vector similarity searches on embedded data.
- **LLM-Powered SQL Generation:**  
  - Uses a specialized GGUF model (e.g., `PipableAI/pip-sql-1.3b`) via `llama-cpp-python`.  
  - Includes prompt engineering and a retry loop for error correction.
- **Dual LLM Backend Modes:**  
  - **Development:** Uses OpenRouter API with JSON outputs for routing and summarization.  
  - **Production/Local:** Fully offline with locally loaded GGUF models.
- **Optional GPU Acceleration:** For faster local GGUF inference via CUDA/ROCm/Metal.

## Technology Stack

- **Backend:** Python 3.10+, Pandas, SQLite  
- **Web Framework:** Streamlit  
- **LLMs & Embeddings:** llama-cpp-python, FastEmbed (`paraphrase-multilingual-mpnet-base-v2`), OpenRouter API  
- **Vector Database:** Qdrant (in-memory)  
- **Configuration:** python-dotenv, huggingface_hub

## Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-name>

2. **Create and Activate a Virtual Environment:**

python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate     # Windows

3. **Install Dependencies:**

    pip install -r requirements.txt

    For GPU acceleration, install llama-cpp-python with appropriate compilation flags (see llama-cpp-python docs).

4. **Configure Environment Variables:**
Copy and edit .env.example to create your .env file. Key variables include:

        DEVELOPMENT_MODE (true for OpenRouter, false for local GGUF)

        OPENROUTER_API_KEY & OPENROUTER_MODEL_NAME (for development)

        GGUF_MODEL_PATH, SQL_GGUF_REPO_ID, SQL_GGUF_FILENAME (for production/local)

    Download Models:
    The SQL GGUF model downloads automatically via huggingface_hub. For local mode, download the main GGUF model manually and set its path in .env.

## Running the Application

# Start the App:

    streamlit run app.py

    Open your browser at the URL provided by Streamlit (typically http://localhost:8501).

## Usage

1. **Load Data:**
    Use the sidebar to upload an Excel file or paste a published Google Sheet CSV URL. Enter a table name and process the data to build the SQLite database and vector index.

2. **Chat:**
    Type your natural language query into the chat box. The app routes the query to either SQL or semantic search, retrieves results, and provides a natural language summary. Expanders reveal raw SQL or data if needed.

3. **Modes: Development vs. Production**

    Development Mode (DEVELOPMENT_MODE=true):
    Leverages the OpenRouter API for quick iterations.

    Production/Local Mode (DEVELOPMENT_MODE=false):
    Runs entirely locally using GGUF models for full offline operation.

4. **GPU Acceleration (Optional)**

    Requirements:

        NVIDIA/AMD drivers, CUDA/ROCm toolkit, build tools (cmake, etc.)

    Installation:
    Uninstall and reinstall llama-cpp-python with the proper CMAKE_ARGS for your GPU.

    Configuration:
    Set GGUF_N_GPU_LAYERS in your .env to enable GPU offloading.

5. **Troubleshooting**

    Model/Download Issues: Verify your internet connection, HF_TOKEN, and cache permissions.

    GGUF Loading Errors: Check your model paths, compatibility, and resource availability.

    Database Locks: Ensure the database isn’t accessed concurrently (use local DB or adjust timeouts).

    OpenRouter Errors: Confirm API keys and rate limits.

## License

Licensed under the MIT License.