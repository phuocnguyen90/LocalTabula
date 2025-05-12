# scripts/download_models.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, set_key
from huggingface_hub import hf_hub_download
from fastembed import TextEmbedding 


# Ensure the project root is in sys.path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Define MODELS_BASE_DIR here if not importing from utils
MODELS_BASE_DIR = PROJECT_ROOT / "models"

# Define specific cache subdirectories
MAIN_LLM_CACHE_DIR = MODELS_BASE_DIR / "main_llm"
SQL_GGUF_CACHE_DIR = MODELS_BASE_DIR / "auxiliary" / "sql_gguf"
EMBEDDING_CACHE_DIR = MODELS_BASE_DIR / "auxiliary" / "embedding_cache"

# Path to the .env file that stores final model paths
# This could be config/.env or a separate config/paths.env
ENV_PATH = PROJECT_ROOT / "config" / ".env" # Or "config/paths.env"

def update_env_file(key, value):
    """Updates or adds a key-value pair in the .env file."""
    print(f"Updating .env at {ENV_PATH}: {key} = {value}")
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), key, str(value), quote_mode="always") # Use "always" for consistency


def download_all_models():
    print(f"Using base directory for models: {MODELS_BASE_DIR}")
    MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SQL_GGUF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load REPO_IDs and FILENAMES from the central .env file
    load_dotenv(ENV_PATH, override=True) # Override to ensure we get download configs

    hf_token = os.getenv("HF_TOKEN")

    # 1. Main LLM
    main_repo = os.getenv("MAIN_LLM_REPO_ID")
    main_file = os.getenv("MAIN_LLM_FILENAME")
    if main_repo and main_file:
        print(f"Downloading Main LLM: {main_repo}/{main_file} to {MODELS_BASE_DIR}...")
        try:
            downloaded_main_llm_path = hf_hub_download(
                repo_id=main_repo,
                filename=main_file,
                local_dir=MAIN_LLM_CACHE_DIR,
                token=hf_token
            )
            print(f"Main LLM downloaded to: {downloaded_main_llm_path}")
            update_env_file("LOCAL_MAIN_LLM_PATH", downloaded_main_llm_path)
            os.environ["LOCAL_MAIN_LLM_PATH"] = str(downloaded_main_llm_path) # For current session
        except Exception as e:
            print(f"ERROR downloading Main LLM '{main_file}': {e}")
    else:
        print("Skipping Main LLM download: MAIN_LLM_REPO_ID or MAIN_LLM_FILENAME not set in .env.")

    # 2. SQL LLM
    sql_repo = os.getenv("SQL_LLM_REPO_ID")
    sql_file = os.getenv("SQL_LLM_FILENAME")
    if sql_repo and sql_file:
        print(f"Downloading SQL LLM: {sql_repo}/{sql_file} to {SQL_GGUF_CACHE_DIR}...")
        try:
            downloaded_sql_llm_path = hf_hub_download(
                repo_id=sql_repo,
                filename=sql_file,
                local_dir=SQL_GGUF_CACHE_DIR,
                token=hf_token
            )
            print(f"SQL LLM downloaded to: {downloaded_sql_llm_path}")
            update_env_file("LOCAL_SQL_LLM_PATH", downloaded_sql_llm_path)
            os.environ["LOCAL_SQL_LLM_PATH"] = str(downloaded_sql_llm_path) # For current session
        except Exception as e:
            print(f"ERROR downloading SQL LLM '{sql_file}': {e}")
    else:
        print("Skipping SQL LLM download: SQL_LLM_REPO_ID or SQL_LLM_FILENAME not set in .env.")

    # 3. Embedding Model
    embed_repo_name = os.getenv("EMBEDDING_MODEL_REPO_NAME")
    if embed_repo_name:
        print(f"Ensuring Embedding Model ({embed_repo_name}) is cached in {EMBEDDING_CACHE_DIR}...")
        try:
            # TextEmbedding will download to its own structure within the specified cache_dir
            TextEmbedding(model_name=embed_repo_name, cache_dir=str(EMBEDDING_CACHE_DIR))
            print(f"Embedding model '{embed_repo_name}' cached in '{EMBEDDING_CACHE_DIR}'.")
            update_env_file("LOCAL_EMBEDDING_CACHE_DIR", str(EMBEDDING_CACHE_DIR))
            os.environ["LOCAL_EMBEDDING_CACHE_DIR"] = str(EMBEDDING_CACHE_DIR) # For current session
        except Exception as e:
            print(f"ERROR caching Embedding Model '{embed_repo_name}': {e}")
    else:
        print("Skipping Embedding Model download: EMBEDDING_MODEL_REPO_NAME not set in .env.")

    print("Model download process finished.")

if __name__ == "__main__":
    download_all_models()